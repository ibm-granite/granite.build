#!/usr/bin/env python3

# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Admin-DB reconciliation for centralized lineage recording.

The admin DB already holds the complete lineage graph: every successful target
run and its input/output artifacts are persisted to admin storage during the
build. ``WandBLineageStore.add_jobstats_for_build_target`` reconstructs a
target's lineage purely from ``storage.target_storage`` — no build events are
involved. So the full lineage for granite.build is recoverable by re-reading the
admin DB alone.

This module makes that reconstruction the *central* recording mechanism, rather
than driving recording off the (in-memory, restart-blind) event stream:

- ``record_target_lineage`` is the single idempotent leaf: "record this one
  (build, target)". Everything that records lineage goes through it — the
  reconciliation scan below, and (later) a manual/CLI selector for pushing
  selected build lineage to the store, with no rework.
- ``reconcile_once`` is the central selector: it scans the admin DB for
  successful target runs and feeds each through the leaf.

Idempotency is what makes a full rescan safe: the underlying store records with
deterministic runIds + ``resume="allow"`` + content-dedupe, so re-recording an
already-recorded target is harmless. Because the scan re-derives the recordable
set from the DB on every pass, a target that succeeded while the recorder was
down is picked up on the next scan — there is no restart blind spot.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Iterable, NamedTuple, Optional

from gbserver.lineage.jobstats import ILineageStore
from gbserver.storage.singleton_storage import SingletonAdminStorage
from gbserver.storage.storage import Pagination, QueryControl, SortOrder
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.status import Status
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

# gb_status key under which the LineageWatcher persists its checkpoint, so a
# restart resumes from the last successfully-recorded target instead of
# rescanning the whole admin DB. Value shape: {"build_id": str, "finished_at":
# <ISO 8601 str>}.
LINEAGE_WATCHER_CHECKPOINT_KEY = "lineage_store_latest_build_id"

# Column the reconciliation scan sorts/paginates successful targets by. A target
# gets finished_at set when it succeeds, so it is the moment the target becomes
# recordable — the correct watermark for "finished since I last scanned" (unlike
# created_time, which is set at build start and would skip a long-running target
# that started before the watermark but finished after it).
_FINISHED_AT_FIELD = "finished_at"

# Rows fetched per admin-DB page. The scan sorts newest-finished-first and stops
# at the caller's watermark, so a steady-state scan reads only newly-finished
# targets (typically a partial first page); the page size just bounds how many
# rows a single query materializes when catching up a backlog.
_SCAN_PAGE_SIZE = 200


def record_target_lineage(
    store: ILineageStore,
    storage: SingletonAdminStorage,
    build_id: str,
    target_id: str,
) -> None:
    """Record lineage for a single (build, target) — the one recording leaf.

    Idempotent: the underlying store dedupes by deterministic runId, so calling
    this for an already-recorded target is a harmless no-op on the backend. Both
    the reconciliation scan and any future manual/selective push feed this same
    leaf.

    Args:
        store: The lineage store to record into.
        storage: Admin storage the store reads the target's lineage from.
        build_id: Build the target belongs to.
        target_id: Target run to record lineage for.
    """
    store.add_jobstats_for_build_target(storage, build_id=build_id, target_id=target_id)


def _successful_targets_page(
    storage: SingletonAdminStorage, page_index: int
) -> list[StoredTargetRun]:
    """Fetch one newest-finished-first page of successful target runs.

    ``status`` is a queryable column, so this filters server-side; results are
    ordered by ``finished_at`` descending and paginated so the caller can walk
    from the newest completion down and stop at its watermark, rather than
    materializing the whole successful-target set.
    """
    query_control = QueryControl(
        pagination=Pagination(index=page_index, size=_SCAN_PAGE_SIZE),
        sort_orders=[SortOrder(column=_FINISHED_AT_FIELD, ascending=False)],
    )
    targets = storage.target_storage.get_by_where(
        {"status": Status.SUCCESS.name}, query_control=query_control
    )
    return [t for t in targets if isinstance(t, StoredTargetRun)]


def _as_utc_naive(value: datetime) -> datetime:
    """Normalize a ``finished_at`` to naive UTC for safe comparison.

    ``finished_at`` values are written naive (``datetime.now()`` / event
    timestamps), but a storage backend or DB driver may hand some rows back
    timezone-aware. Comparing a naive and an aware ``datetime`` raises
    ``TypeError``, which would abort the whole scan. Coercing both sides to
    naive UTC before comparing keeps the watermark walk robust regardless of
    which awareness the read path yields; naive values are assumed UTC.
    """
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def select_recordable_targets(
    storage: SingletonAdminStorage,
    finished_after: Optional[datetime] = None,
) -> list[StoredTargetRun]:
    """Select successful target runs whose lineage should be recorded.

    A target is recordable once it has completed successfully; its lineage is
    fully persisted in admin storage at that point. The successful-target set
    grows without bound over the platform's lifetime, so this never materializes
    all of it in steady state. Targets are fetched newest-``finished_at``-first
    and the walk stops as soon as it crosses ``finished_after``:

    - ``finished_after=None`` (startup / full catch-up): page through every
      successful target so anything that finished while the recorder was down is
      picked up. Recording is idempotent, so re-reading already-recorded targets
      on this pass is harmless.
    - ``finished_after=<watermark>`` (steady state): return only targets that
      finished at or after the watermark, i.e. the newly-completed ones. Because
      results are sorted by ``finished_at`` descending, the walk stops at the
      first target that is not newer than the watermark — so a steady-state scan
      reads only the new rows (typically a partial first page), never the whole
      table, regardless of how many builds have accumulated.

    The comparison is ``>=`` (not ``>``) so the boundary target is re-included
    rather than dropped; idempotent recording makes the re-read harmless and the
    caller's watermark advances past it. Targets with no ``finished_at`` are
    skipped (they are not yet complete) but do not stop the walk — a NULL row
    interleaved among finished ones must not truncate the scan.

    Returns:
        The selected successful target runs, newest-finished first.
    """
    selected: list[StoredTargetRun] = []
    page_index = 0
    while True:
        page = _successful_targets_page(storage, page_index)
        for target in page:
            if target.finished_at is None:
                # Not yet finished. NULL finished_at rows may be interleaved
                # rather than sorted last, so this is a skip-and-continue — never
                # an early return — regardless of whether a watermark is set.
                continue
            # Sorted newest-finished-first: once we reach a target that finished
            # before the watermark, every later one is older too — stop early.
            if finished_after is not None and _as_utc_naive(
                target.finished_at
            ) < _as_utc_naive(finished_after):
                return selected
            selected.append(target)
        if len(page) < _SCAN_PAGE_SIZE:
            break
        page_index += 1
    return selected


def get_most_recent_successful_target(
    storage: SingletonAdminStorage,
) -> Optional[StoredTargetRun]:
    """Return the single newest-finished successful target, or ``None``.

    Used to seed the LineageWatcher's checkpoint on a fresh deployment (no
    persisted checkpoint yet): a single-page, newest-first query rather than
    the full ``select_recordable_targets`` walk, since only the first result is
    needed. Targets with no ``finished_at`` are skipped, mirroring
    ``select_recordable_targets``.
    """
    for target in _successful_targets_page(storage, page_index=0):
        if target.finished_at is not None:
            return target
    return None


def _expected_run_count(target: StoredTargetRun) -> int:
    """Number of lineage runs a fully-recorded ``target`` should have in a sink.

    Must mirror how ``WandBLineageStore._build_events_for_target`` emits events:
    one run per output artifact (summed across every output-artifact list), or a
    single "no-output" run when the target produced no outputs. Inputs do not add
    runs — they are attached to each output's run — so only outputs are counted.
    This is derived from the in-memory ``StoredTargetRun`` (already loaded by the
    scan) to avoid any extra storage read. Keep this in lockstep with
    ``_build_events_for_target``; the count-vs-events coherence test guards drift.
    """
    n = sum(len(uuids) for uuids in target.output_artifacts.values())
    return n if n > 0 else 1


class ReconcileResult(NamedTuple):
    """Outcome of one ``reconcile_once`` pass.

    ``max_finished_at``/``max_finished_at_build_id`` describe the last target
    that was *actually recorded successfully* this pass (not merely the newest
    one considered) — a failed or skipped target must not advance a persisted
    checkpoint past it. Both are ``None`` when nothing was recorded and no
    watermark existed coming in.
    """

    max_finished_at: Optional[datetime]
    max_finished_at_build_id: Optional[str]


def reconcile_once(
    store: ILineageStore,
    storage: SingletonAdminStorage,
    finished_after: Optional[datetime] = None,
    on_error: Optional[Callable[[str, str, Exception], None]] = None,
    on_success: Optional[Callable[[str, str], None]] = None,
    on_checkpoint_advance: Optional[Callable[[str, datetime], None]] = None,
    skip: Optional[set[str]] = None,
) -> ReconcileResult:
    """Reconcile admin-DB lineage into the store once (the central mechanism).

    Selects successful target runs that finished at or after ``finished_after``
    (or every successful target when it is ``None``), asks the store which of
    those it has not yet recorded, and records each of those through the single
    leaf.

    Two independent mechanisms bound the work and keep it sink-neutral:

    - ``finished_after`` is a *time watermark* on the target itself (not on any
      sink), so a steady-state scan reads only newly-finished targets from the
      admin DB regardless of how many builds have accumulated. It says nothing
      about whether a given sink has recorded a target.
    - ``store.filter_unrecorded`` is the *per-sink* recorded-state check: each
      sink owns its own record of what it has already recorded, so the same
      admin DB can feed W&B and other sinks independently. It never raises; on
      failure it returns the full candidate set, degrading to re-recording
      (harmless — recording is idempotent). It is given each candidate's expected
      run count (``_expected_run_count``) so a target whose runs were only
      partially emitted on a prior crashed scan is reported unrecorded and
      re-recorded, rather than masked by its already-present runs.

    Args:
        store: The lineage store to record into.
        storage: Admin storage to reconcile from.
        finished_after: Only consider targets that finished at or after this
            time. ``None`` (startup / restart) considers every successful target
            so anything that finished while the recorder was down is re-driven.
        on_error: Optional callback ``(build_id, target_id, exc)`` invoked when
            recording a single target raises, so the caller can queue a retry.
            When omitted, a failure is logged and the target is simply retried on
            the next scan.
        on_success: Optional callback ``(build_id, target_id)`` invoked when a
            target records successfully, so the caller can clear any retry state
            it was tracking for that target (a target that failed a prior scan
            and then succeeds is only reported here — it drops out of the
            unrecorded set, so ``on_error`` is never called for it again).
        on_checkpoint_advance: Optional callback ``(build_id, finished_at)``
            invoked immediately after each individual target records
            successfully (oldest-``finished_at``-first). Lets the caller persist
            a checkpoint per-target rather than once per batch, so a crash
            mid-scan leaves a durable checkpoint at the last target actually
            recorded — not at the newest one merely considered, and not stuck at
            the pre-scan watermark until the whole batch finishes.
        skip: Target uuids the caller has given up on (e.g. dropped after
            exhausting retries). These are excluded from recording so a
            persistently failing target — which still falls within the watermark
            window every scan — cannot wedge the scan. They do NOT advance the
            returned watermark or fire ``on_checkpoint_advance`` themselves
            (only an actually-recorded target does that); the caller's
            in-memory dedup — e.g. ``LineageWatcher._dropped`` — is what keeps a
            skipped target from being reconsidered forever, independent of the
            checkpoint.

    Returns:
        A ``ReconcileResult`` describing the last target *actually recorded
        successfully* this pass (``finished_after`` unchanged, with no
        build_id, when nothing recorded). Callers that need per-target
        granularity (e.g. persisting a checkpoint immediately as each target is
        recorded) should use ``on_checkpoint_advance`` instead — this return
        value is a coarser end-of-scan summary.
    """
    # Selection is newest-finished-first (bounds the DB walk: it can stop as
    # soon as it crosses the watermark). Recording order is the reverse —
    # oldest-first — so finished_at advances monotonically as each target
    # records, letting a checkpoint be persisted safely after every single
    # target rather than only once at the end of the batch.
    targets = list(reversed(select_recordable_targets(storage, finished_after=finished_after)))

    skip = skip or set()
    by_uuid = {t.uuid: t for t in targets if t.uuid not in skip}
    # No candidates → nothing to record and nothing to check. Skip the
    # per-sink filter_unrecorded query entirely so an idle scan (or one where
    # only the watermark-overlap boundary targets are all skipped) does not fire
    # a backend query (e.g. a wandb api.runs call) that would return nothing.
    if not by_uuid:
        return ReconcileResult(finished_after, None)
    # Expected run count per candidate, so the sink can tell a fully-recorded
    # target from one whose runs were only partially emitted on a prior crashed
    # scan (see ILineageStore.filter_unrecorded). Derived in memory from the
    # already-loaded targets — no extra storage read. A skipped-for-prerun target
    # records the *original* target's outputs, not its own, so its in-memory
    # output_artifacts would give the wrong count; omit it and let it fall back to
    # the presence check (a rare case; re-recording is a harmless idempotent no-op).
    expected_counts = {
        uuid: _expected_run_count(target)
        for uuid, target in by_uuid.items()
        if not target.skipped_for_prerun_target_id
    }
    unrecorded = store.filter_unrecorded(set(by_uuid), expected_counts)

    max_finished_at = finished_after
    max_finished_at_build_id = None
    newly_recorded = 0
    # Iterate in oldest-first order (the `targets` order), not the (unordered)
    # `unrecorded` set, so the checkpoint advances monotonically.
    for target in targets:
        if target.uuid not in unrecorded:
            continue
        try:
            record_target_lineage(
                store, storage, build_id=target.build_id, target_id=target.uuid
            )
            newly_recorded += 1
            if target.finished_at is not None:
                max_finished_at = target.finished_at
                max_finished_at_build_id = target.build_id
                if on_checkpoint_advance is not None:
                    on_checkpoint_advance(target.build_id, target.finished_at)
            if on_success is not None:
                on_success(target.build_id, target.uuid)
        except Exception as exc:  # noqa: BLE001 - reconciliation must not abort
            if on_error is not None:
                on_error(target.build_id, target.uuid, exc)
            else:
                logger.warning(
                    "Failed to record lineage for target %s in build %s; "
                    "will retry on next scan: %s",
                    target.uuid,
                    target.build_id,
                    exc,
                )

    if newly_recorded:
        logger.info("Reconciled lineage for %d target(s)", newly_recorded)
    return ReconcileResult(max_finished_at, max_finished_at_build_id)


def record_selected_targets(
    store: ILineageStore,
    storage: SingletonAdminStorage,
    targets: Iterable[tuple[str, str]],
) -> None:
    """Record lineage for an explicitly selected set of (build_id, target_id).

    The seam for a future manual/selective push (e.g. a standalone user recording
    a few important builds to a centralized store): a selector supplies the pairs
    and they flow through the same idempotent leaf the reconciliation scan uses.

    Args:
        store: The lineage store to record into.
        storage: Admin storage the store reads lineage from.
        targets: Iterable of (build_id, target_id) pairs to record.
    """
    for build_id, target_id in targets:
        record_target_lineage(store, storage, build_id=build_id, target_id=target_id)
