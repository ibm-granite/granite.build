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

from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, Optional

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

# gb_status key under which the LineageWatcher persists the target uuids it has
# permanently given up on (after _MAX_RECORD_ATTEMPTS failed attempts). This must
# be durable, not in-memory: the checkpoint deliberately refuses to advance past
# an unrecorded target, so a dropped target that came back after a restart would
# block the watermark again, fail its attempts again, and repeat forever —
# wedging all later lineage behind a target that will never record. Value shape:
# {"target_ids": [str, ...]}.
LINEAGE_WATCHER_DROPPED_KEY = "lineage_store_dropped_target_ids"

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
    storage: SingletonAdminStorage,
    page_index: int,
    build_id: Optional[str] = None,
) -> list[StoredTargetRun]:
    """Fetch one newest-finished-first page of successful target runs.

    ``status`` is a queryable column, so this filters server-side; results are
    ordered by ``finished_at`` descending and paginated so the caller can walk
    from the newest completion down and stop at its watermark, rather than
    materializing the whole successful-target set.

    ``build_id`` narrows the scan to a single build (used by the watcher's
    start-time checkpoint verification, which only cares about the checkpoint's
    own build); it is likewise a queryable column, so this also filters
    server-side.
    """
    where: dict = {"status": Status.SUCCESS.name}
    if build_id is not None:
        where["build_id"] = build_id
    query_control = QueryControl(
        pagination=Pagination(index=page_index, size=_SCAN_PAGE_SIZE),
        sort_orders=[SortOrder(column=_FINISHED_AT_FIELD, ascending=False)],
    )
    targets = storage.target_storage.get_by_where(where, query_control=query_control)
    return [t for t in targets if isinstance(t, StoredTargetRun)]


def as_utc_naive(value: datetime) -> datetime:
    """Normalize a ``finished_at`` to naive UTC for safe comparison.

    ``finished_at`` values are written naive (``datetime.now()`` / event
    timestamps), but a storage backend or DB driver may hand some rows back
    timezone-aware. Comparing a naive and an aware ``datetime`` raises
    ``TypeError``, which would abort the whole scan. Coercing both sides to
    naive UTC before comparing keeps the watermark walk robust regardless of
    which awareness the read path yields; naive values are assumed UTC.

    Shifting to UTC can itself raise ``OverflowError``, when the shift would
    carry the value outside ``datetime.min``/``datetime.max`` — reachable here
    because the ``--base-build-id all`` backfill anchor *is* ``datetime.min``, and a
    backend may hand it back aware with a positive UTC offset. That would abort
    the whole scan, so it is clamped to the bound it overflowed past instead. The
    clamp cannot distort an ordering decision: an overflow means the true UTC
    instant lies beyond the bound, and the bound is already older (or newer) than
    every representable ``finished_at``, so every comparison against it yields
    the same answer the unrepresentable value would have.
    """
    if value.tzinfo is None:
        return value
    try:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    except OverflowError:
        # Overflow is only reachable within one UTC offset (<24h) of a bound, so
        # the sign of the offset says which bound was crossed: a positive offset
        # shifts backwards past datetime.min, a negative one forwards past max.
        offset = value.utcoffset()
        if offset is not None and offset > timedelta(0):
            return datetime.min
        return datetime.max


def select_recordable_targets(
    storage: SingletonAdminStorage,
    finished_after: datetime,
    build_id: Optional[str] = None,
) -> list[StoredTargetRun]:
    """Select successful target runs whose lineage should be recorded.

    A target is recordable once it has completed successfully; its lineage is
    fully persisted in admin storage at that point. The successful-target set
    grows without bound over the platform's lifetime, so this never materializes
    all of it in steady state: targets are fetched newest-``finished_at``-first
    and the walk stops as soon as it crosses ``finished_after``, so a
    steady-state scan reads only the newly-finished rows, never the whole table,
    regardless of how many builds have accumulated.

    One caveat on that bound: rows with ``finished_at`` NULL (successful targets
    written before ``finished_at`` stamping existed) sort *first* under
    PostgreSQL's ``DESC``, so the walk pages through that backlog before reaching
    any real timestamp. It is bounded and correct — NULLs are skipped, never
    treated as a stopping point — but a deployment with a large pre-stamping
    backlog re-reads it on every scan. Pushing an ``IS NOT NULL`` filter
    server-side would remove it; the storage layer's ``where`` currently supports
    only equality/IN, so that needs a storage-layer change.

    ``finished_after`` is required rather than defaulting to "no lower bound".
    An omitted watermark would silently page through every successful target the
    platform has ever run — a full historical backfill — which is a deliberate
    operation, never a default. A caller that genuinely wants that (e.g. an
    explicit backfill command) passes ``datetime.min``.

    The comparison is ``>=`` (not ``>``) so the boundary target is re-included
    rather than dropped; idempotent recording makes the re-read harmless and the
    caller's watermark advances past it. Targets with no ``finished_at`` are
    skipped (they are not yet complete) but do not stop the walk — a NULL row
    interleaved among finished ones must not truncate the scan.

    ``build_id`` restricts the selection to a single build, for callers that
    want the same selection semantics over one build rather than the whole DB
    (e.g. verifying a checkpoint's own build on startup).

    Returns:
        The selected successful target runs, newest-finished first.
    """
    selected: list[StoredTargetRun] = []
    page_index = 0
    cutoff = as_utc_naive(finished_after)
    while True:
        page = _successful_targets_page(storage, page_index, build_id=build_id)
        for target in page:
            if target.finished_at is None:
                # Not yet finished. NULL finished_at rows may be interleaved
                # rather than sorted last, so this is a skip-and-continue — never
                # an early return.
                continue
            # Sorted newest-finished-first: once we reach a target that finished
            # before the watermark, every later one is older too — stop early.
            if as_utc_naive(target.finished_at) < cutoff:
                return selected
            selected.append(target)
        if len(page) < _SCAN_PAGE_SIZE:
            break
        page_index += 1
    return selected


def get_most_recent_successful_target(
    storage: SingletonAdminStorage,
    build_id: Optional[str] = None,
) -> Optional[StoredTargetRun]:
    """Return the single newest-finished successful target, or ``None``.

    Used by ``lineage_seeding`` (``gbserver lineage-watch --base-build-id``) to
    place the LineageWatcher's checkpoint: a single-page, newest-first query
    rather than the full ``select_recordable_targets`` walk, since only the first
    result is needed.
    Targets with no ``finished_at`` are skipped, mirroring
    ``select_recordable_targets``.

    ``build_id`` restricts the search to one build, so a caller can anchor the
    checkpoint at a chosen build rather than at whatever finished most recently.

    This pages rather than reading only the first page. ``finished_at`` stamping
    was added after rows were already being written, so a real deployment holds
    successful targets with ``finished_at`` NULL — and PostgreSQL sorts NULLs
    *first* under ``DESC`` (the sort is a bare ``desc()``, with no
    ``NULLS LAST``). A single-page read would therefore return ``None`` whenever
    the NULL backlog fills the first page, making ``--base-build-id`` raise
    ``LineageSeedError`` on exactly the deployments that have history to anchor
    against — and since ``--base-build-id`` is meant to live permanently in the
    pod spec, that is a crashloop rather than a one-off error.
    """
    page_index = 0
    while True:
        page = _successful_targets_page(storage, page_index, build_id=build_id)
        for target in page:
            if target.finished_at is not None:
                return target
        # A short (or empty) page is the last one: no non-NULL row exists.
        if len(page) < _SCAN_PAGE_SIZE:
            return None
        page_index += 1


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


def reconcile_once(
    store: ILineageStore,
    storage: SingletonAdminStorage,
    finished_after: datetime,
    on_error: Optional[Callable[[str, str, Exception], None]] = None,
    on_success: Optional[Callable[[str, str], None]] = None,
    on_checkpoint_advance: Optional[Callable[[str, datetime], None]] = None,
    on_scan_complete: Optional[Callable[[bool], None]] = None,
    skip: Optional[set[str]] = None,
    build_id: Optional[str] = None,
) -> int:
    """Reconcile admin-DB lineage into the store once (the central mechanism).

    Selects successful target runs that finished at or after ``finished_after``,
    asks the store which of those it has not yet recorded, and records each of
    those through the single leaf.

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
            time. Required: an implicit "no lower bound" would make a full
            historical backfill the default. Pass ``datetime.min`` to request
            one deliberately.
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

            It stops firing for the rest of the pass as soon as one target
            fails, so the checkpoint only ever advances over a *contiguous*
            oldest-first run of recorded targets. Otherwise a target that failed
            mid-batch would be passed by the newer targets that succeed after
            it, durably advancing the checkpoint beyond a target that was never
            recorded — retry would then rest solely on the caller's in-memory
            state and a restart would drop it permanently.
        on_scan_complete: Optional callback ``(watermark_untouched)`` invoked once
            when the pass finishes. ``True`` means the pass completed having left
            the watermark exactly where it found it *and* with no unrecorded
            lineage behind it: nothing recorded (so no ``on_checkpoint_advance``
            fired), nothing failed, and nothing dropped via ``skip``. That is the
            only condition under which a caller may move its watermark on its own
            — notably to retire a ``datetime.min`` backfill anchor that would
            otherwise re-walk the whole table on every scan.

            ``newly_recorded == 0`` alone does NOT imply it: the count is also 0
            when every candidate failed or was skipped, and moving the watermark
            then would strand that lineage permanently, since the steady-state
            scan never looks behind its watermark.
        skip: Target uuids the caller has given up on (e.g. dropped after
            exhausting retries). These are excluded from recording so a
            persistently failing target — which still falls within the watermark
            window every scan — cannot wedge the scan. They do NOT fire
            ``on_checkpoint_advance`` themselves (only an actually-recorded
            target does that); the caller's in-memory dedup — e.g.
            ``LineageWatcher._dropped`` — is what keeps a skipped target from
            being reconsidered forever, independent of the checkpoint.
        build_id: Restrict the pass to a single build. Used by the watcher's
            start-time checkpoint verification, which needs exactly this
            selection/filter/record behaviour over the checkpoint's own build.

    Returns:
        How many targets were newly recorded this pass. Where the watermark
        reached is reported through ``on_checkpoint_advance``, per-target, as
        each one records: that is the only granularity a caller can persist
        safely, so there is no coarser end-of-scan watermark to return.
    """
    # Selection is newest-finished-first (bounds the DB walk: it can stop as
    # soon as it crosses the watermark). Recording order is the reverse —
    # oldest-first — so finished_at advances monotonically as each target
    # records, letting a checkpoint be persisted safely after every single
    # target rather than only once at the end of the batch.
    targets = list(
        reversed(
            select_recordable_targets(
                storage, finished_after=finished_after, build_id=build_id
            )
        )
    )

    skip = skip or set()
    by_uuid = {t.uuid: t for t in targets if t.uuid not in skip}
    # No candidates → nothing to record and nothing to check. Skip the
    # per-sink filter_unrecorded query entirely so an idle scan (or one where
    # only the watermark-overlap boundary targets are all skipped) does not fire
    # a backend query (e.g. a wandb api.runs call) that would return nothing.
    if not by_uuid:
        # A pass with nothing to do — report it so a caller anchored at
        # datetime.min can retire the backfill anchor instead of re-walking the
        # whole table on every scan. But it is only *clean* if there were no
        # candidates at all: when `targets` is non-empty, every one of them was
        # dropped via `skip`, i.e. lineage that will never be recorded is being
        # left behind, and the anchor must stay put.
        if on_scan_complete is not None:
            on_scan_complete(not targets)
        return 0
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

    newly_recorded = 0
    # Set once a target in this pass fails to record. From that point on the
    # checkpoint must not advance any further, even though newer targets keep
    # recording successfully: advancing past the failed target would durably
    # move the watermark beyond lineage that was never written, and the next
    # scan would no longer re-surface it.
    checkpoint_blocked = False
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
            if target.finished_at is None:
                # Unreachable via select_recordable_targets, which skips NULL
                # finished_at rows. Guarded anyway because the failure mode is
                # silent: with no timestamp there is nothing to advance the
                # watermark to, and merely *not* advancing would let the next
                # target advance past this one. Block instead, so a future
                # caller that bypasses the selector cannot move the checkpoint
                # beyond a target the watermark can no longer re-surface.
                checkpoint_blocked = True
            elif not checkpoint_blocked and on_checkpoint_advance is not None:
                on_checkpoint_advance(target.build_id, target.finished_at)
            if on_success is not None:
                on_success(target.build_id, target.uuid)
        except Exception as exc:  # noqa: BLE001 - reconciliation must not abort
            # Freeze the checkpoint here: later targets in this pass may still
            # record, but the watermark must not move past this one or the next
            # scan will not re-surface it (and a restart would lose it entirely,
            # since retry state is only in memory).
            checkpoint_blocked = True
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
    if on_scan_complete is not None:
        # "Left the watermark alone": nothing recorded (so no per-target advance
        # fired) and nothing blocked. Either a record or a block means the
        # watermark is already where this pass wants it, and a caller must not
        # move it further.
        on_scan_complete(newly_recorded == 0 and not checkpoint_blocked)
    return newly_recorded


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
