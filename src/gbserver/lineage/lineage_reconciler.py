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

- ``record_target_lineage`` is the single leaf: "record this one (build,
  target)". Everything that records lineage goes through it — the reconciliation
  scan below, and (later) a manual/CLI selector for pushing selected build
  lineage to the store, with no rework.
- Selection is *build-scoped* and runs in two steps:
  ``select_builds_from_checkpoint`` picks the checkpoint's build and everything
  created at or after it, and ``reconcile_build`` reconciles one of those builds
  by reading its successful targets and feeding the unrecorded ones through the
  leaf. The build is the unit of iteration and of checkpoint advancement.

What makes a rescan safe is the sink's dedup query, *not* idempotent writes. Run
ids are random uuids, so re-recording a target the sink already has writes a
second set of runs rather than resuming the first — dedup is the only thing
preventing duplicates, and it identifies a run solely by its ``target_id`` tag.
That is why ``ILineageStore.filter_unrecorded`` fails CLOSED (an unanswered query
records nothing and the pass aborts) and why every emitted event must carry that
tag. Because selection re-derives the recordable set from the DB on every pass, a
target that succeeded while the recorder was down is picked up on the next scan —
there is no restart blind spot.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, tzinfo
from typing import Callable, Iterable, Optional

from gbserver.lineage.jobstats import ILineageStore
from gbserver.storage.singleton_storage import SingletonAdminStorage
from gbserver.storage.storage import (
    CREATED_TIME_FIELD_NAME,
    Pagination,
    QueryControl,
    SortOrder,
)
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_target_run import (
    StoredTargetRun,
    latest_success_per_target,
)
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
# restart resumes from the last fully-recorded build instead of rescanning the
# whole admin DB. Value shape (v2): {"build_id": str, "created_time": <ISO 8601
# str>, "version": 2}.
#
# The key name is unchanged from the v1 (target-shaped) checkpoint because
# operators have it in pod specs and runbooks; the *value* migrates in place, see
# LineageWatcher._read_checkpoint.
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

# Builds fetched per admin-DB page by select_builds_from_checkpoint. The walk
# sorts newest-created-first and stops at the anchor's created_time, so a
# steady-state scan reads only builds at or after the checkpoint; the page size
# just bounds how many rows one query materializes while catching up.
_BUILD_SCAN_PAGE_SIZE = 200

# Version stamped into the checkpoint value. v1 was target-shaped
# ({"build_id", "finished_at"}, a *target*'s finish time); v2 is build-shaped
# ({"build_id", "created_time", "version"}). The field name had to change with
# the ordering key: keeping "finished_at" while storing a build's creation time
# would make every log line and hand-seeded value a lie.
LINEAGE_WATCHER_CHECKPOINT_VERSION = 2

# Substrings identifying a dedup-query failure that no retry can clear: the sink
# is reachable but will never answer this query as configured (bad project or
# entity, invalid or unauthorized credentials). Recording is switched off rather
# than retried, because retrying forever would leave the watcher aborting every
# pass in silence with the checkpoint pinned.
#
# Matched on the message rather than the exception type, for the same reason the
# deleted-run rejection is: wandb raises CommError for ordinary transient network
# failures AND for permanent refusals, so the type alone cannot separate them
# (production logs show every one of them arriving as
# wandb.errors.errors.CommError).
#
# Kept SHORT and loose on purpose. A substring that is too specific stops
# matching the moment wandb rewords its error, and the failure then reads as
# transient — which is the safe direction, but silently loses the classification.
# Add to this list only causes an operator must fix; anything unrecognized is
# treated as transient by is_permanent_sink_failure below.
_PERMANENT_SINK_FAILURES = (
    "permission denied",
    "unauthorized",
    "invalid api key",
    "could not find project",
    "does not exist",
    "not a member of",
)


def is_permanent_sink_failure(exc: BaseException) -> bool:
    """Whether a dedup-query failure is permanent (operator must intervene).

    Default is False — "transient" — for anything unrecognized. That is the safe
    direction: retrying a permanent failure only costs queries, while treating a
    transient one as permanent would switch recording off over a network blip.

    Walks the exception chain because the real cause often arrives wrapped by
    wandb's own error handling, the same way the deleted-run rejection did.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).lower()
        if any(marker in message for marker in _PERMANENT_SINK_FAILURES):
            return True
        current = current.__cause__ or current.__context__
    return False


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


def _builds_page(storage: SingletonAdminStorage, page_index: int) -> list[StoredBuild]:
    """Fetch one newest-created-first page of builds.

    Ordered by ``created_time`` descending so the caller can walk down from the
    newest build and stop once it passes its lower bound, rather than
    materializing every build the platform has ever run.
    """
    query_control = QueryControl(
        pagination=Pagination(index=page_index, size=_BUILD_SCAN_PAGE_SIZE),
        sort_orders=[SortOrder(column=CREATED_TIME_FIELD_NAME, ascending=False)],
    )
    builds = storage.build_storage.get_by_where(None, query_control=query_control)
    return [b for b in builds if isinstance(b, StoredBuild)]


def select_builds_from_checkpoint(
    storage: SingletonAdminStorage,
    anchor_created_time: datetime,
    exclude_ids: Optional[set[str]] = None,
) -> list[StoredBuild]:
    """Select the checkpoint's build and every build created at or after it.

    Two different orders are at work here, and conflating them is the easy
    mistake. The *query* pages newest-created-first, purely so the walk can stop
    at the anchor instead of reading the whole table. The *return* is
    oldest-created-first, because that is the order the caller must process in:
    the checkpoint advances contiguously and has to stop at the first unfinished
    build, which cannot be determined walking backwards.

    The bound is ``>=``, not ``>``, so the anchor build is itself included. That
    is deliberate: a pass that crashed partway through the anchor build leaves
    some of its targets unrecorded, and re-selecting it every scan is what
    recovers them -- it replaces the separate start-up verification sweep the
    target-anchored design needed.

    Three caveats on the bound, all inherited from how ``created_time`` is stored:

    - It is a Python-side clock read in ``BaseItemStorage.__add_items``, which
      overwrites whatever the model's ``default_factory`` put there. It lands
      after validation, schema init and the uniqueness check, and the insert then
      runs in its own transaction doing nothing but the INSERT -- so
      stamp-to-visibility is short (a model_dump, a log line and one round trip),
      but not zero. Build creation is not serialized: ``submit_build`` is a sync
      endpoint, so submissions overlap in the threadpool, and
      ``GBSERVER_REST_SERVER_WORKERS`` above 1 adds cross-process concurrency
      with no shared lock. Two builds created concurrently can therefore become
      visible in a different order than their timestamps imply, and a build whose
      ``created_time`` is fractionally behind the anchor's can appear only after
      the checkpoint advanced -- permanently missed, since nothing sweeps behind
      the anchor. Accepted deliberately in exchange for an exact cutoff with no
      re-processed overlap window: the window is a few milliseconds wide and the
      anchor itself is included, so it takes a build created at virtually the same
      instant as the anchor. Note the single-replica *watcher* deployment does not
      shrink this -- the race is on the creation side.
    - The stamp is per ``add()`` call, not per row: a batched add gives every row
      one identical ``t``. Build submission always adds one build, but it does
      mean ``created_time`` is not unique and cannot order two builds against each
      other. ``_advance_checkpoint`` handles that by stepping to the successor *in
      this list's order* rather than to the first non-anchor row.
    - SQLite stores ``DateTime(timezone=True)`` as text and drops the offset, so
      ``ORDER BY created_time`` is a string sort there. Every writer goes through
      ``get_utc_time()``, so all rows share one offset and the text sort is
      correct -- but a row written with a differently-formatted timestamp (e.g. by
      a seeding script) can sort wrongly and truncate this walk early. A second
      source of the same disagreement: ``__fill_missing_times`` substitutes
      ``BEGINNING_OF_TIME`` when the *JSON blob* lacks the field, so a
      pre-migration row can read far-past in Python while its physical column
      sorts elsewhere. Both are why the walk reads every page and re-sorts on
      aware instants below.

    Args:
        storage: Admin storage to read builds from.
        anchor_created_time: ``created_time`` of the checkpoint build; builds at or
            after this instant are selected.
        exclude_ids: Build ids to leave out of the result entirely.

    Returns:
        The selected builds, oldest-created first.
    """
    skip = exclude_ids or set()
    cutoff = as_aware(anchor_created_time)
    selected: list[StoredBuild] = []
    page_index = 0
    while True:
        page = _builds_page(storage, page_index)
        if not page:
            break
        for build in page:
            if build.created_time is None:
                # No creation time to place it against the cutoff. Skip the row
                # but keep walking: like a NULL finished_at among targets, it must
                # not truncate the scan.
                continue
            if as_aware(build.created_time) < cutoff:
                # Behind the cutoff: skip the row, but keep walking. Deliberately
                # NOT an early return, even though the query asks for
                # newest-created-first and a trustworthy ordering would make
                # everything past this row older too.
                #
                # The ordering cannot be trusted. SQLite stores
                # DateTime(timezone=True) as TEXT and orders it as TEXT, and this
                # column has been written in two different spellings over time:
                # SQLAlchemy's own "YYYY-MM-DD HH:MM:SS.ffffff" and an ISO
                # "YYYY-MM-DDTHH:MM:SS.mmmZ" form. ' ' (0x20) sorts before 'T'
                # (0x54), so with both present the sort interleaves rows whose real
                # instants are months apart -- a genuinely newer build can appear
                # below an older one. An early return there stops the walk before
                # reaching it and reports "nothing to record" while real builds sit
                # unread, which is silent data loss rather than a slow scan.
                #
                # The cost is reading every page instead of stopping early. That is
                # the right trade for a table with one row per build: correctness
                # cannot rest on the backend's collation matching our comparison.
                continue
            if build.uuid in skip:
                continue
            selected.append(build)
        if len(page) < _BUILD_SCAN_PAGE_SIZE:
            break
        page_index += 1
    # Sorted here, on aware instants, so the caller's contiguous walk is ordered by
    # real time regardless of how the backend collated the text.
    return sorted(selected, key=lambda b: as_aware(b.created_time))


def get_most_recent_build(storage: SingletonAdminStorage) -> Optional[StoredBuild]:
    """Return the newest build by ``created_time``, or None if there are none.

    Used by seeding to anchor "start from the latest build".

    Pages through every build and takes the maximum by aware instant rather than
    trusting the query's ordering to put the newest first. Same reason as
    ``select_builds_from_checkpoint``: SQLite orders this column as TEXT and the
    column holds two different spellings, so the first row of page 0 is not
    reliably the newest. Getting this wrong would anchor the checkpoint at an
    arbitrary older build and silently re-drive history from there.

    A NULL ``created_time`` row is skipped, so the anchor always has a usable
    timestamp.
    """
    newest: Optional[StoredBuild] = None
    page_index = 0
    while True:
        page = _builds_page(storage, page_index)
        if not page:
            break
        for build in page:
            if build.created_time is None:
                continue
            if newest is None or as_aware(build.created_time) > as_aware(
                newest.created_time
            ):
                newest = build
        if len(page) < _BUILD_SCAN_PAGE_SIZE:
            break
        page_index += 1
    return newest


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
    build_id: str,
) -> list[StoredTargetRun]:
    """Select one build's successful target runs whose lineage should be recorded.

    A target is recordable once it has completed successfully; its lineage is
    fully persisted in admin storage at that point. Scoped to a single build,
    which is the unit the watcher iterates: builds are selected first (see
    ``select_builds_from_checkpoint``) and their targets read per build, so there
    is no unbounded walk over the whole successful-target set and therefore no
    need for a timestamp watermark or a row anchor here.

    Targets with no ``finished_at`` are skipped: a successful target is expected
    to carry the instant it finished, and one without it is not yet complete. That
    is a skip, never an early stop -- NULL rows can be interleaved rather than
    sorted last.

    Targets with no input artifacts AND no output artifacts are also skipped: the
    standalone UI reads target nodes straight from admin storage regardless of
    artifacts (see ``__build_target_records`` in ``api/builds.py``), so wandb no
    longer needs to carry a run for a fully artifact-less target just to make it
    appear as a node. A target with only inputs, or only outputs, is still
    recorded -- it still has real lineage (an edge) to represent. An output-artifact
    name mapped to an empty list does not count as an output, which is why the
    check reads the dict's *values*; ``_build_events_for_target`` emits nothing for
    such a name, and a selector that kept the target would leave it permanently
    unconfirmable.

    Note what the artifact-less skip does to state keyed on targets this function
    no longer returns. Such a target never reaches ``candidates``, so it is also
    absent from ``ReconcileResult.dropped`` -- meaning the watcher's
    "permanently-dropped lineage" ERROR line stops naming an artifact-less target
    that is still sitting in the durable drop set from before this skip existed.
    The gap that line reports is therefore not exhaustive; an operator reading
    only the logs would conclude the target's gap had been resolved. Nothing is
    lost or mis-recorded (the target has no lineage to record, which is the whole
    point of the skip), and the checkpoint is unaffected -- the build reports
    ``all_confirmed`` with ``sink_unqueried`` and advances normally. The stale
    ``gb_kv_pairs`` drop-set entry is dead weight only, clearable with
    ``lineage-init --clear-dropped-targets``. Same for runs such a target already
    has in wandb from before the skip: nothing selects it, so nothing revisits
    them.

    Args:
        storage: Admin storage to read targets from.
        build_id: Build whose successful targets to select.

    Returns:
        The build's recordable target runs, newest-finished first.
    """
    selected: list[StoredTargetRun] = []
    artifact_less = 0
    page_index = 0
    while True:
        page = _successful_targets_page(storage, page_index, build_id=build_id)
        if not page:
            break
        for target in page:
            if target.finished_at is None:
                continue
            if not target.input_artifacts and not any(target.output_artifacts.values()):
                artifact_less += 1
                logger.debug(
                    "Skipping wandb lineage recording for target %s (build %s): "
                    "no input or output artifacts.",
                    target.uuid,
                    build_id,
                )
                continue
            selected.append(target)
        if len(page) < _SCAN_PAGE_SIZE:
            break
        page_index += 1
    # One aggregate line per build rather than one per skipped target, and only
    # when something was actually skipped. This function re-runs for the same
    # build on every scan -- the watcher's build cutoff is inclusive, so the
    # anchor build stays in range indefinitely in steady state -- so a per-target
    # info line means a build with many artifact-less targets reprints its whole
    # skip list every monitoring interval, forever. The per-target detail is
    # still available at debug, same tradeoff as the recorded-cache hit log in
    # WandBLineageStore.filter_unrecorded.
    if artifact_less:
        logger.info(
            "Skipped wandb lineage recording for %d target(s) of build %s: "
            "no input or output artifacts.",
            artifact_less,
            build_id,
        )
    # In-place retry reuses one build id, so a target can hold more than one
    # SUCCESS run (a prior success with unregistered artifacts is re-run; a
    # reuse-disabled build re-runs every target). Record only the latest per
    # target — the pages are newest-finished first, so the winner stays first.
    return latest_success_per_target(selected)


def expected_run_count(target: StoredTargetRun) -> int:
    """Number of lineage runs a fully-recorded ``target`` should have in a sink.

    Must mirror how ``WandBLineageStore._build_events_for_target`` emits events:
    one run per output artifact (summed across every output-artifact list), or a
    single "no-output" run when the target has inputs but no outputs. Inputs
    otherwise do not add runs — they are attached to each output's run — so only
    outputs are counted when there are any. ``select_recordable_targets`` excludes
    fully artifact-less targets, so every target reaching here has at least one
    real edge. This is derived from the in-memory
    ``StoredTargetRun`` (already loaded by the scan) to avoid any extra storage
    read. Keep this in lockstep with ``_build_events_for_target``; the
    count-vs-events coherence test guards drift.
    """
    n = sum(len(uuids) for uuids in target.output_artifacts.values())
    return n if n > 0 else 1


@dataclass
class ReconcileResult:
    """Outcome of reconciling one build's lineage into the sink.

    A value rather than a set of callbacks: the watcher's checkpoint now advances
    per *build*, and "was this build fully confirmed?" is the question it has to
    answer -- which no per-target callback expresses well.

    Attributes:
        newly_recorded: How many targets this pass actually wrote.
        all_confirmed: Every recordable target of the build is either recorded now,
            already in the sink, or permanently dropped. This is what makes the
            build eligible for the checkpoint to advance past it.
        dedup_query_failed: The sink could not answer whether targets were already
            recorded. Nothing was written; the caller must abort the pass rather
            than read an empty candidate set as "nothing to do".
        query_failure: The exception behind ``dedup_query_failed``, so the caller
            can classify it as permanent or transient.
        dropped: Targets skipped because they are in the caller's permanent-drop
            set. Present so the caller can log the resulting known gap.
        sink_unqueried: ``all_confirmed`` was reached without asking the sink
            anything -- ``filter_unrecorded`` was never called, because there was no
            candidate to ask about. Two passes end that way: no recordable target at
            all, and every recordable target in the caller's permanent-drop set. Both
            are reported ``all_confirmed`` so the checkpoint can still pass a finished
            build, but the confirmation rests on an *empty* candidate set rather than
            a sink answer. The caller must not cache that as "this build is done,
            stop re-reading its targets" while the build can still gain one.

            The build row and its target rows are separate, non-transactional
            writes, and a scan lands at an arbitrary point between them: a build
            read while still RUNNING has no target rows yet, or only ones that
            happen to be dropped. Its targets are persisted before it reaches
            SUCCESS (buildrunner drains one FIFO with a single consumer, and
            finalize_build_status commits children before the parent), so they are
            there on a later scan -- but only if the caller still looks.
    """

    newly_recorded: int = 0
    all_confirmed: bool = False
    dedup_query_failed: bool = False
    query_failure: Optional[Exception] = None
    dropped: set[str] = field(default_factory=set)
    sink_unqueried: bool = False


def reconcile_build(
    store: ILineageStore,
    storage: SingletonAdminStorage,
    build_id: str,
    on_error: Optional[Callable[[str, str, Exception], None]] = None,
    on_success: Optional[Callable[[str, str], None]] = None,
    skip: Optional[set[str]] = None,
) -> ReconcileResult:
    """Reconcile one build's lineage into the store (the central mechanism).

    Selects the build's successful targets, asks the store which of them it has
    not yet recorded, and records each of those through the single leaf. Scoped to
    one build because that is the watcher's unit of iteration and of checkpoint
    advancement.

    Sink-neutral: which targets to write comes from
    ``ILineageStore.filter_unrecorded``, so a store that already has a target says
    so and nothing is re-emitted. That matters more than it used to -- run ids are
    random now, so re-recording a target the sink already has writes duplicate runs
    instead of resuming the existing ones. There is no idempotency underneath, and
    the dedup query is the only thing standing between a re-selected target and a
    duplicate.

    Hence the failure contract. ``filter_unrecorded`` fails CLOSED: it returns an
    empty set on error, which by value is indistinguishable from "everything is
    already recorded". This function therefore reports the failure explicitly in
    ``ReconcileResult.dedup_query_failed`` and records nothing, and the caller must
    abort rather than treat the empty set as success.

    Args:
        store: The lineage store to record into.
        storage: Admin storage to read targets and lineage from.
        build_id: Build to reconcile.
        on_error: Called as ``(build_id, target_id, exc)`` when recording a target
            raises. The scan continues to the build's other targets.
        on_success: Called as ``(build_id, target_id)`` after a target records.
        skip: Target uuids to leave alone permanently (the caller's dropped set).

    Returns:
        A ``ReconcileResult`` describing what happened for this build.
    """
    dropped = skip or set()
    targets = select_recordable_targets(storage, build_id=build_id)
    if not targets:
        # No recordable target: nothing to write and nothing outstanding, so the
        # build is trivially confirmed and the checkpoint may pass it.
        #
        # Flagged unqueried, though, because this confirmation is not a sink answer:
        # with no target to ask about, filter_unrecorded was never called. A build
        # read while still RUNNING lands here (its target rows are written before it
        # reaches SUCCESS, but this scan ran before that). If the caller caches this
        # as complete, its finished-and-complete skip gate stops re-reading the
        # targets and the lineage is never recorded.
        return ReconcileResult(all_confirmed=True, sink_unqueried=True)

    skipped = {t.uuid for t in targets if t.uuid in dropped}
    candidates = {t.uuid for t in targets if t.uuid not in dropped}
    if not candidates:
        # Every target is permanently dropped. Confirmed *with a known gap*: the
        # alternative is pinning the checkpoint forever behind targets that will
        # never record, which is what the durable dropped set exists to prevent.
        #
        # Unqueried for the same reason as the no-target case above: every target
        # was filtered out of `candidates`, so filter_unrecorded was never called
        # and this confirmation is not a sink answer either. A RUNNING build whose
        # only targets so far happen to all be dropped lands here, and the ones
        # written between this scan and its terminal status are still to come --
        # so the caller must not cache it as done while it can still gain them.
        return ReconcileResult(all_confirmed=True, dropped=skipped, sink_unqueried=True)

    # Expected run count per candidate, so ``filter_unrecorded`` can tell a
    # fully-recorded target from one whose runs were only partially emitted by a
    # prior crashed scan. Derived in memory from the already-loaded targets — no
    # extra storage read. Every candidate is a real run whose own output_artifacts
    # give the correct count (there is no skip concept: an in-place retry keeps
    # both the FAILED and the SUCCESS run in one build).
    expected = {t.uuid: expected_run_count(t) for t in targets if t.uuid in candidates}

    failure: Optional[Exception] = None

    def _note_failure(exc: Exception) -> None:
        nonlocal failure
        failure = exc

    unrecorded = store.filter_unrecorded(
        candidates, expected, on_query_error=_note_failure
    )
    if failure is not None:
        logger.error(
            "Lineage dedup query failed for build %s; recording nothing for it: %s",
            build_id,
            failure,
        )
        return ReconcileResult(dedup_query_failed=True, query_failure=failure)

    recorded = 0
    all_confirmed = True
    for target in targets:
        if target.uuid not in unrecorded:
            continue
        try:
            record_target_lineage(
                store, storage, build_id=build_id, target_id=target.uuid
            )
        except Exception as exc:  # noqa: BLE001 — one target must not end the pass
            logger.warning(
                "Failed to record lineage for target %s of build %s: %s",
                target.uuid,
                build_id,
                exc,
            )
            all_confirmed = False
            if on_error is not None:
                on_error(build_id, target.uuid, exc)
            continue
        recorded += 1
        if on_success is not None:
            on_success(build_id, target.uuid)

    if recorded:
        logger.info(
            "Recorded lineage for %d target(s) of build %s.", recorded, build_id
        )
    return ReconcileResult(
        newly_recorded=recorded,
        all_confirmed=all_confirmed,
        dropped=skipped,
    )


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
