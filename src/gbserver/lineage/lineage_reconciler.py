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

import logging
from datetime import datetime, timezone, tzinfo
from typing import Callable, Iterable, Optional

from gbserver.lineage.jobstats import ILineageStore
from gbserver.storage.singleton_storage import SingletonAdminStorage
from gbserver.storage.storage import Pagination, QueryControl, SortOrder
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.status import Status
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

# Aware equivalent of datetime.min, used as the backfill anchor. Every timestamp
# in the watermark comparison is aware (see as_aware), so the anchor must be too
# — a naive datetime.min compared against an aware instant raises TypeError. The
# offset is UTC only because the anchor is a sentinel with no source row of its
# own; it compares below every real finished_at whatever their offsets.
UTC_MIN = datetime.min.replace(tzinfo=timezone.utc)

# gb_kv_pairs key under which the LineageWatcher persists its checkpoint, so a
# restart resumes from the last successfully-recorded target instead of
# rescanning the whole admin DB. Value shape: {"build_id": str, "finished_at":
# <ISO 8601 str>}.
LINEAGE_WATCHER_CHECKPOINT_KEY = "lineage_store_latest_build_id"

# gb_kv_pairs key under which the LineageWatcher persists the target uuids it has
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


def local_tzinfo() -> Optional[tzinfo]:
    """Return the local UTC offset, used to interpret naive ``finished_at`` values.

    Split out so the assumption has one home and the tests can pin it: a naive
    timestamp in this data is local, not UTC (see ``as_aware``).
    """
    return datetime.now().astimezone().tzinfo


def as_aware(value: datetime) -> datetime:
    """Ensure a ``finished_at`` is timezone-aware, preserving its own offset.

    Every timestamp is made aware before any comparison, so the watermark walk
    compares *instants* rather than wall-clock readings. Mixing the two is what
    this exists to prevent: comparing a naive and an aware ``datetime`` raises
    ``TypeError``, and — more insidiously — treating a naive local reading as UTC
    silently shifts it by the local offset, which can put a target on the wrong
    side of the watermark and truncate the scan.

    The offset is deliberately *not* rewritten to UTC. Two aware datetimes
    compare as instants regardless of their offsets, so converting buys nothing
    for the comparison — while it does make the value written to ``gb_kv_pairs``
    disagree textually with the ``gb_targets`` row it came from, which is exactly
    the confusion that made the same target read three hours apart depending on
    which table it was loaded from. Keeping the offset means the checkpoint holds
    the target's own timestamp verbatim (see ``get_time()``, aware local).

    A naive value is interpreted as **local**, not UTC. ``finished_at`` originates
    from ``utils.get_time()`` (``datetime.now().astimezone()``), which is aware
    local, so the offset to put back is the local one. Assuming UTC here would
    re-introduce exactly the skew described above.

    Note where the offset *can* be lost, because it is not this path. A
    ``StoredTargetRun`` is reconstructed from the JSON column, whose ISO string
    carries the offset and round-trips losslessly on **both** SQLite and
    Postgres — so ``finished_at`` arrives aware either way and the naive branch
    below is defensive rather than routinely taken. The backend that drops an
    offset is SQLite's typed ``DateTime(timezone=True)`` column, which stores
    wall-clock text; that only affects the ``ORDER BY finished_at`` sort (see
    ``select_recordable_targets``), not the value read back here.

    Two caveats on the naive branch, for whoever does reach it. It is exact only
    when the reader shares the writer's offset — true for a single-timezone
    deployment, a best-effort guess otherwise. And ``local_tzinfo()`` reports the
    offset in effect *now*, not the one in effect when the row was written, so
    across a DST transition the interpreted instant shifts by an hour and a
    target can land on the wrong side of the watermark. Keeping ``finished_at``
    aware at rest is what removes both guesses; until then this is the closest
    correct reading of a naive value.
    """
    if value.tzinfo is None:
        value = value.replace(tzinfo=local_tzinfo())
    return value


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
    explicit backfill command) passes ``UTC_MIN``.

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
    cutoff = as_aware(finished_after)
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
            if as_aware(target.finished_at) < cutoff:
                # Log where the walk stopped and what it was holding. The early
                # return is load-bearing *and* it is the one place a mis-sorted
                # page silently truncates the scan: it trusts the backend's
                # ORDER BY finished_at DESC, and a backend that does not order
                # instants (SQLite stores these as strings, so rows with mixed
                # UTC offsets sort by their text) can put an old row first and
                # end the walk before any newer one is seen — reporting "nothing
                # to process" while real targets sit unread behind it.
                logger.debug(
                    "Lineage SQL walk stopped at %s(%s, build %s, finished %s): "
                    "older than the %s cutoff. Holding %d selected target(s) so "
                    "far; rows after this one on the page were not examined.",
                    target.name or "<unnamed>",
                    target.uuid[:8],
                    target.build_id,
                    target.finished_at,
                    cutoff,
                    len(selected),
                )
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


def get_oldest_successful_target(
    storage: SingletonAdminStorage,
    build_id: Optional[str] = None,
) -> Optional[StoredTargetRun]:
    """Return the single oldest-finished successful target, or ``None``.

    Used by ``lineage_seeding`` to anchor a checkpoint at a *build*: a build has
    many targets finishing at different times, and the watermark is inclusive
    (``finished_at >= cutoff``), so anchoring at the build's newest target would
    exclude every earlier target of that same build. Anchoring at its oldest
    includes the whole build.

    That matters beyond the anchored build itself. ``_verify_checkpoint`` does
    re-scan the checkpoint's own build with no lower bound, so the anchored
    build's earlier targets would be recovered there — but a *concurrent* build
    whose targets finished inside the skipped window is covered by neither that
    sweep (scoped to one build) nor the steady-state scan (which never looks
    behind the watermark), and ``_WATERMARK_OVERLAP`` spans only a minute. Those
    targets would be lost permanently, so the anchor is placed low enough that
    they are never skipped in the first place.

    Pages to the end rather than reading one page, because the ordering is
    newest-first: the oldest row is on the *last* page. NULL-``finished_at`` rows
    are skipped (mirroring ``select_recordable_targets``), which also means the
    NULL backlog PostgreSQL sorts first under ``DESC`` cannot be mistaken for the
    oldest target.
    """
    oldest: Optional[StoredTargetRun] = None
    # Tracked alongside `oldest` rather than re-derived from it: `finished_at` is
    # Optional on the model, so reaching back through `oldest` to compare would
    # need a redundant None check that the loop above has already done.
    oldest_at: Optional[datetime] = None
    page_index = 0
    while True:
        page = _successful_targets_page(storage, page_index, build_id=build_id)
        for target in page:
            if target.finished_at is None:
                continue
            # Compare as instants: finished_at may arrive aware or naive
            # depending on the backend, and mixing the two raises TypeError.
            finished_at = as_aware(target.finished_at)
            if oldest_at is None or finished_at < oldest_at:
                oldest, oldest_at = target, finished_at
        if len(page) < _SCAN_PAGE_SIZE:
            return oldest
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
    watermark_build_id: Optional[str] = None,
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
    # build_id scans (the watcher's start-time checkpoint verification) always
    # pass finished_after=UTC_MIN, since the whole point is to ignore the
    # watermark and re-check that one build regardless of it. Printing UTC_MIN
    # there reads as a suspiciously stuck watermark, so name the build instead
    # of the epoch it is scanning from.
    # The watermark alone does not say *why* the scan starts where it does. Naming
    # the build whose target set it — the checkpoint's own build_id — makes a
    # steady-state log line self-explanatory: the timestamp can be traced back to
    # the row it came from instead of looking like an arbitrary epoch, which is
    # what makes a stuck watermark recognizable as stuck.
    if build_id is not None:
        scope_desc = f"for build {build_id}"
    elif watermark_build_id is not None:
        scope_desc = (
            f"at or after {finished_after} (the watermark from build "
            f"{watermark_build_id})"
        )
    else:
        scope_desc = f"at or after {finished_after}"
    # Dump what the SQL walk actually returned, oldest-first. The counts alone
    # cannot distinguish the states that matter when a scan looks stuck: a query
    # returning nothing, a query returning rows that the watermark then excluded,
    # and rows excluded because they were permanently dropped all print as some
    # flavour of "nothing to process". Naming each row — and whether it survived
    # the skip set — is what makes "the same target every iteration" diagnosable
    # instead of inferred. DEBUG, because this is per-row output on every scan.
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Lineage SQL walk %s returned %d successful target(s) with a finish "
            "time, oldest-first: %s",
            scope_desc,
            len(targets),
            ", ".join(
                f"{t.name or '<unnamed>'}({t.uuid[:8]}, build {t.build_id}, "
                f"finished {t.finished_at}"
                f"{', SKIPPED as permanently dropped' if t.uuid in skip else ''})"
                for t in targets
            )
            or "<none>",
        )
    # No candidates → nothing to record and nothing to check. Skip the
    # per-sink filter_unrecorded query entirely so an idle scan (or one where
    # only the watermark-overlap boundary targets are all skipped) does not fire
    # a backend query (e.g. a wandb api.runs call) that would return nothing.
    if not by_uuid:
        # Say so explicitly. This is the overwhelmingly common steady-state
        # outcome, and without a line here an idle scan and a wedged watcher are
        # indistinguishable in the log.
        logger.info(
            "Lineage scan found nothing new to process (%d successful target(s) "
            "%s, %d skipped as permanently dropped); waiting for the "
            "next iteration.",
            len(targets),
            scope_desc,
            len(targets) - len(by_uuid),
        )
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
    # Log the selection *and* the sink's verdict on it, since "found candidates
    # but recorded none" (all already in the sink) and "found no candidates at
    # all" are very different states that otherwise look the same.
    if unrecorded:
        logger.info(
            "Lineage scan selected %d candidate target(s) %s; %d "
            "already recorded in the sink, recording %d: %s",
            len(by_uuid),
            scope_desc,
            len(by_uuid) - len(unrecorded),
            len(unrecorded),
            ", ".join(
                f"{by_uuid[uuid].name or '<unnamed>'}({uuid[:8]}, "
                f"build {by_uuid[uuid].build_id}, "
                f"finished {by_uuid[uuid].finished_at})"
                for uuid in sorted(unrecorded)
                if uuid in by_uuid
            ),
        )
    else:
        # When every candidate belongs to the watermark's own build, the scan did
        # not find pending work at all: it re-selected the target the watermark
        # already sits on, which the _WATERMARK_OVERLAP window deliberately keeps
        # in range so a target finishing in the same second as the checkpoint is
        # never skipped. Say so, because "1 candidate, already recorded" otherwise
        # reads as a target the sink keeps refusing to accept.
        candidate_builds = {t.build_id for t in by_uuid.values()}
        if watermark_build_id is not None and candidate_builds == {watermark_build_id}:
            reprocessed_note = " (the watermark's own build, within the overlap window)"
        else:
            reprocessed_note = ""
        logger.info(
            "Lineage up to date: all %d candidate target(s) %s already recorded%s.",
            len(by_uuid),
            scope_desc,
            reprocessed_note,
        )

    newly_recorded = 0
    # Set once a target in this pass fails to record. From that point on the
    # checkpoint must not advance any further, even though newer targets keep
    # recording successfully: advancing past the failed target would durably
    # move the watermark beyond lineage that was never written, and the next
    # scan would no longer re-surface it.
    checkpoint_blocked = False
    # Iterate in oldest-first order (the `targets` order), not the (unordered)
    # `unrecorded` set, so the checkpoint advances monotonically.
    # Position within the batch, so a slow or stuck pass is readable as progress
    # ("3/7") rather than as an unexplained gap between log lines.
    to_record = [t for t in targets if t.uuid in unrecorded]
    # Walk *every* candidate, not just the unrecorded ones, so an already-recorded
    # target still advances the watermark past itself. Otherwise it stays inside
    # the window forever: each scan re-selects it, asks the sink about it, and
    # reports "already recorded" without the watermark ever clearing it — the
    # per-scan cost is permanent and the log line is indistinguishable from a
    # wedged watcher. Ordering is the `targets` order (oldest-first), which is
    # what keeps the advance monotonic across the mixed set.
    position = 0
    # `targets` still holds the skip set; iterate the filtered candidates instead.
    # A skipped target is NOT in the sink, so it must never advance the watermark
    # — doing so would move it past lineage that was deliberately given up on and
    # can no longer be re-surfaced. It is excluded here rather than treated as
    # already-recorded (it is absent from `unrecorded` for the opposite reason).
    for target in (t for t in targets if t.uuid in by_uuid):
        if target.uuid not in unrecorded:
            # Already in the sink: nothing to write, but the watermark may still
            # move past it — subject to the same contiguity rule as a recorded
            # one. `checkpoint_blocked` means an *earlier* target in this pass
            # failed, so advancing past this one would strand that failure behind
            # the watermark, where no later scan looks.
            if target.finished_at is None:
                # No timestamp to advance to; block rather than let a later
                # target step over it (see the same guard below).
                checkpoint_blocked = True
            elif not checkpoint_blocked and on_checkpoint_advance is not None:
                logger.debug(
                    "Advancing the lineage watermark past already-recorded "
                    "target %s (%s) of build %s, finished %s.",
                    target.name or "<unnamed>",
                    target.uuid,
                    target.build_id,
                    target.finished_at,
                )
                on_checkpoint_advance(target.build_id, target.finished_at)
            continue
        position += 1
        # Logged *before* the write, not only after: this is the call that reaches
        # the sink (a wandb api round-trip), so it is where a pass stalls. Logging
        # only on success would leave the target that hung invisible — the very
        # one worth naming.
        logger.info(
            "Recording lineage %d/%d: target %s (%s) of build %s, finished %s",
            position,
            len(to_record),
            target.name or "<unnamed>",
            target.uuid,
            target.build_id,
            target.finished_at,
        )
        try:
            record_target_lineage(
                store, storage, build_id=target.build_id, target_id=target.uuid
            )
            newly_recorded += 1
            logger.info(
                "Recorded lineage for target %s (%s) of build %s",
                target.name or "<unnamed>",
                target.uuid,
                target.build_id,
            )
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
