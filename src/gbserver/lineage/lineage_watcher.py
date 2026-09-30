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

"""Async lineage-recording agent driven by admin-DB reconciliation."""

import threading
from datetime import datetime
from typing import Optional

from gbserver.lineage.jobstats import ILineageStore, get_lineage_store
from gbserver.lineage.lineage_reconciler import (
    LINEAGE_WATCHER_CHECKPOINT_KEY,
    LINEAGE_WATCHER_CHECKPOINT_VERSION,
    LINEAGE_WATCHER_DROPPED_KEY,
    UTC_MIN,
    as_aware,
    is_permanent_sink_failure,
    reconcile_build,
    select_builds_from_checkpoint,
)
from gbserver.lineage.lineage_seeding import BACKFILL_BUILD_ID
from gbserver.storage.singleton_storage import SingletonAdminStorage, get_admin_storage
from gbserver.storage.stored_build import StoredBuild
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class LineageWatcher:
    """Periodically reconciles admin-DB lineage into the configured sink.

    Selection is build-driven, in two nested bounds:

    1. **Which builds.** Each scan reads the checkpoint, resolves the build it
       names, and selects that build plus every build created at or after it
       (``select_builds_from_checkpoint``). The list is rebuilt every scan, so a
       build created since the last one is picked up without any separate
       registration step. The checkpoint build is deliberately re-included: a scan
       that crashed partway through it left targets unrecorded, and re-selecting it
       is what recovers them.
    2. **Which targets.** Within a build, its successful targets with a
       ``finished_at`` (``reconcile_build``).

    The checkpoint advances one build per scan, and the gate is on the build it
    *leaves*: the mark moves off a build only once that build is finished and its
    lineage is confirmed in the sink. Where it lands is the next build by creation
    order, whatever its state. Given A(finished), B(running), C(finished) the mark
    steps onto B and stays there until B finishes and is confirmed — C's targets
    are still recorded meanwhile, they just do not move the mark. So the mark reads
    as "everything created strictly before this build is confirmed"; the build it
    names may still be running, which is safe because the base stays in range and
    is re-reconciled every scan. No build ever falls out of range while it still
    has lineage to produce.

    That is also the one unbounded shape here: a permanently stuck build pins the
    cutoff, and the selected list grows with every newer build, all of which get
    re-examined each scan. The sink side stays cheap (the store's TTL cache
    answers most dedup queries), and finished-and-confirmed builds are skipped
    without re-reading their targets, but a build wedged forever needs operator
    action (``--force-build-id``).

    **Duplicates, not idempotency, are the hazard.** Run ids are random uuids, so
    re-recording a target the sink already has writes a *second* set of runs
    rather than resuming the first. The dedup query is the only thing preventing
    that, so it fails CLOSED: an unanswered query records nothing and aborts the
    whole pass (a sink that cannot answer for one build cannot answer for the
    next), leaving the checkpoint where it was for the next scan to retry.

    A dedup failure is classified. A *permanent* one — bad project or entity,
    invalid or unauthorized credentials — switches recording off and logs CRITICAL
    each scan, because retrying it forever would wedge the watcher in silence. A
    *transient* one just aborts the pass. Anything unrecognized counts as
    transient, which is the safe direction.

    Single-writer by design: the deployment runs one replica
    (``k8s/chart/templates/dep-lineage-watcher.yaml``). Two watchers would not
    corrupt the checkpoint (its advance is monotonic) but would race the dedup
    query and duplicate runs.
    """

    # Attempts a single target gets before its lineage is dropped. Bounded because
    # the checkpoint refuses to advance past a build with unrecorded lineage, so an
    # unbounded retry would pin it forever.
    _MAX_RECORD_ATTEMPTS = 3

    def __init__(self, monitoring_interval: float = 30.0) -> None:
        """Initialize the LineageWatcher.

        Args:
            monitoring_interval: Sleep duration between reconciliation scans
                (seconds).
        """
        self.monitoring_interval = monitoring_interval
        self.stop_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None
        self._store: Optional[ILineageStore] = None
        # Target uuids dropped after exhausting retries; skipped on later scans so a
        # persistently failing target cannot wedge every scan. Persisted to
        # gb_kv_pairs because the checkpoint refuses to advance past a build with
        # unrecorded lineage: an in-memory-only drop set would let a dropped target
        # return after a restart and block the checkpoint forever.
        self._dropped: set[str] = set()
        # Drops this process decided but could not persist (_persist_dropped is
        # non-raising, so a kv failure leaves the decision in memory only). Tracked
        # separately because _load_dropped re-reads the durable row every scan and
        # takes it as authoritative: without this, an unpersisted drop would be
        # dropped from the set on the next scan and the target re-recorded forever,
        # while a *persisted* id must follow the row so an operator's clear lands.
        self._dropped_unpersisted: set[str] = set()
        # target_uuid -> attempts so far, for targets to retry on a later scan.
        self._failed_attempts: dict[str, int] = {}
        # Builds whose lineage the sink has confirmed complete this process. Two
        # jobs: it gates the checkpoint's contiguous advance, and it lets a scan
        # skip re-reading the targets of a build already known finished and
        # recorded — the mitigation for the growing selected list described in the
        # class docstring.
        #
        # NOT persisted: after a restart the first scan re-asks the sink for the
        # whole range, which is correct and merely more expensive that once.
        #
        # Bounded, not append-only: every scan prunes it to the builds still in the
        # selection range (see ``_reconcile``), so it tracks the live window rather
        # than every build ever confirmed.
        self._complete_builds: set[str] = set()
        # Set when the sink reports a failure no retry can clear. Recording stops
        # rather than looping in silence; the process stays alive and says so at
        # CRITICAL every scan.
        self._recording_disabled = False
        self._disabled_reason: Optional[str] = None
        # Whether the "no checkpoint yet" notice has been logged, so a one-time
        # operator notice does not repeat every scan forever.
        self._missing_checkpoint_logged = False

    def start(self) -> None:
        """Start the watcher thread (daemon=True, does not keep process alive)."""
        if self.worker_thread is not None and self.worker_thread.is_alive():
            logger.error("lineage watcher thread is already running")
            return

        self._store = get_lineage_store()
        storage = get_admin_storage()
        # Load the durable drop set up front so a target already given up on stays
        # skipped from the first scan. _reconcile re-reads it every scan too; this
        # only makes the state correct before the thread is running, and resets it
        # when start() runs again on an instance a previous stop() left populated.
        self._dropped = set()
        self._dropped_unpersisted = set()
        self._load_dropped(storage)
        self.worker_thread = threading.Thread(
            target=self._run, name="lineage-watcher", daemon=True
        )
        self.worker_thread.start()
        logger.info("LineageWatcher started")

    def _load_dropped(self, storage: SingletonAdminStorage) -> None:
        """Re-read the durable set of permanently-given-up-on target uuids.

        A dropped target is one that failed ``_MAX_RECORD_ATTEMPTS`` times; the
        decision to stop trying is permanent, so it must outlive the process.
        Without this the target would return on the next start(), block the
        checkpoint (which never advances past a build with unrecorded lineage),
        exhaust its attempts again, and repeat every restart — wedging all newer
        lineage behind it.

        Called before every scan, not once in ``start()``, for the same reason the
        checkpoint is: the row is the single source of truth, so an operator who
        clears it with ``lineage-init --clear-dropped-targets`` has the targets
        retried on the next scan instead of having to restart the service. Loading
        once also made the clear *unsafe* rather than merely slow -- the next drop
        persisted the whole stale in-memory set, restoring every id just cleared.

        The row is authoritative, plus any drop this process could not persist. A
        plain union would never let a clear through (the cleared ids are still in
        memory); a plain assignment would resurrect a drop whose persist failed
        (``_persist_dropped`` is deliberately non-raising), re-recording a hopeless
        target every scan. Taking the row and adding back only
        ``_dropped_unpersisted`` gets both: the clear lands, the unpersisted drop
        survives.

        A read failure or an unusable shape is logged and leaves the in-memory set
        as it is: raising would abort ``start()`` (no watcher at all) or, per scan,
        kill the loop. Keeping the current set is the conservative direction -- it
        skips what this process already knows is hopeless rather than retrying it --
        and it is logged at ERROR with the offending value, not swallowed. The row is
        left for an operator to inspect; the next ``_persist_dropped`` overwrites it,
        which is why the value goes in the log while it still exists.
        """
        try:
            value = storage.kv_pair_storage.get_value(LINEAGE_WATCHER_DROPPED_KEY)
        except Exception:
            logger.exception(
                "Failed to read the lineage drop set from %s; keeping the current "
                "in-memory set (%d target(s)) for this scan.",
                LINEAGE_WATCHER_DROPPED_KEY,
                len(self._dropped),
            )
            return

        if value is None:
            # No row at all: never seeded, or cleared by deleting it rather than
            # emptying it. Treated the same as an empty set -- honouring that is the
            # point of re-reading -- while unpersisted drops are added back below.
            value = {}

        target_ids = value.get("target_ids", []) if isinstance(value, dict) else None
        # Element types matter as much as the container's: a mixed list loads fine
        # here but makes ``_persist_dropped``'s sorted() raise on the next drop, and
        # that unwinds through reconcile_build (on_error is called outside its
        # try/except) into _run, failing every scan forever.
        if isinstance(target_ids, list) and not all(
            isinstance(t, str) for t in target_ids
        ):
            target_ids = None
        if not isinstance(target_ids, list):
            logger.error(
                "Lineage drop set under %s is unusable (%r); keeping the current "
                "in-memory set (%d target(s)). Expected {'target_ids': [...]}.",
                LINEAGE_WATCHER_DROPPED_KEY,
                value,
                len(self._dropped),
            )
            return

        self._dropped = set(target_ids) | self._dropped_unpersisted

    def _persist_dropped(self, storage: SingletonAdminStorage) -> None:
        """Persist the drop set so the decision survives a restart.

        Never raises. ``_on_record_error`` calls this from inside ``reconcile_build``,
        where ``on_error`` runs outside the try/except guarding the record call, so an
        exception here would unwind into ``_run`` and fail the whole scan -- and then
        every later scan the same way. Losing durability for one drop only costs a
        re-attempt after a restart; losing the scan loop stops all lineage.

        On failure the set is remembered in ``_dropped_unpersisted`` so the next
        ``_load_dropped`` -- which treats the durable row as authoritative -- adds it
        back instead of resurrecting the target. A success clears that, handing those
        ids over to the row, where an operator's clear can then reach them.
        """
        try:
            storage.kv_pair_storage.set_value(
                LINEAGE_WATCHER_DROPPED_KEY, {"target_ids": sorted(self._dropped)}
            )
        except Exception:
            self._dropped_unpersisted |= self._dropped
            logger.exception(
                "Failed to persist the lineage drop set to %s; the in-memory set "
                "still applies for this process, but the drop decision will not "
                "survive a restart.",
                LINEAGE_WATCHER_DROPPED_KEY,
            )
            return

        self._dropped_unpersisted.clear()

    def _run(self) -> None:
        """Main monitoring loop (runs in daemon thread).

        Sleeps between scans only when a scan made no progress. Because the
        checkpoint advances at most one build per scan
        (see ``_advance_checkpoint``), a watcher with a backlog of N complete
        builds would otherwise need N full intervals to catch up. A scan that
        moved the mark means there is more to do right now, so the next one
        starts immediately and the backlog drains at the speed of the work
        rather than of the clock.

        The loop still yields on every iteration: the wait is on ``stop_event``
        rather than a bare sleep, so a stop() lands promptly during catch-up
        instead of after the whole backlog. Progress is bounded by the builds
        that exist and each step persists the mark, so this terminates -- the
        first scan that cannot advance falls back to the interval.
        """
        while not self.stop_event.is_set():
            advanced = False
            try:
                advanced = self._reconcile()
            except Exception:
                # Treat a crashed scan as no progress: sleep before retrying
                # rather than spinning on a reproducible failure.
                logger.exception("LineageWatcher iteration failed")

            if advanced:
                # Yield without waiting the interval, so stop() is still honored.
                if self.stop_event.wait(0):
                    return
                continue
            self.stop_event.wait(self.monitoring_interval)

    def _reconcile(self) -> bool:
        """Run one reconciliation scan over the admin DB.

        Returns whether the checkpoint advanced, i.e. whether this scan made
        progress worth chasing immediately instead of sleeping the interval.

        Selects the checkpoint's build and everything created at or after it,
        reconciles each oldest-first, then advances the checkpoint contiguously
        over the leading run of finished-and-confirmed builds.

        The checkpoint is re-read every scan rather than cached: it is the single
        source of truth, so a key seeded or corrected mid-run takes effect on the
        next scan instead of the next restart. A missing key is a no-op — "record
        nothing" until seeded — and must never fall back to scanning the whole admin
        DB, which would turn an unseeded deployment into a full backfill.

        The checkpoint is written once, after the build loop, rather than per
        recorded target: it now names a *build* that is fully confirmed, and a
        mid-scan crash simply leaves it where it was for the next scan to redo.
        """
        if self._store is None:
            logger.error("lineage store not initialized; start() must run first")
            return False
        if self._recording_disabled:
            logger.critical(
                "Lineage recording is disabled: the sink reported a failure no "
                "retry can clear, so no lineage is being recorded and the "
                "checkpoint is frozen. An operator must fix the sink "
                "configuration (project, entity, credentials) and restart the "
                "watcher. Underlying error: %s",
                self._disabled_reason,
            )
            return False

        # Heartbeat. An idle watcher is otherwise indistinguishable from a dead
        # one: every path below returns without logging when there is nothing new,
        # so a healthy scan over an up-to-date DB produced total silence. Logged
        # before the reads so a scan that dies inside one still leaves a mark.
        logger.info("Lineage scan: reading checkpoint and builds...")

        storage = get_admin_storage()
        # Re-read the drop set every scan, like the checkpoint below: it is the
        # single source of truth, so `lineage-init --clear-dropped-targets` retries
        # those targets on the next scan rather than requiring a service restart.
        self._load_dropped(storage)
        checkpoint = self._read_checkpoint(storage)
        if checkpoint is None:
            # No checkpoint: recording is off until seeded. Return before selecting
            # builds or touching the sink.
            return False
        anchor_build_id, anchor_created_time = checkpoint

        builds = select_builds_from_checkpoint(storage, anchor_created_time)
        if not builds:
            logger.info(
                "Lineage scan: no builds created at or after anchor %s; nothing to "
                "record.",
                anchor_build_id,
            )
            return False
        logger.info(
            "Lineage scan: %d build(s) selected from anchor %s.",
            len(builds),
            anchor_build_id,
        )

        for build in builds:
            # A build already finished and confirmed cannot gain lineage, so skip
            # the per-build target read entirely. This is what keeps a pinned
            # cutoff from re-reading every newer build's targets each scan.
            #
            # Deliberately never re-verified: this is a cache of a past verdict,
            # not a live query, and by design a build confirmed once is not
            # checked against the sink again. That rests on two invariants of the
            # platform, both intended and neither enforced here:
            #
            #   1. Once a build is finished, no new target appears for it. So a
            #      confirmed build's recordable set cannot grow. (is_finished()
            #      excludes RUNNING -- a build re-running in place for a retry is
            #      RUNNING, not finished, so it fails this gate and is
            #      re-reconciled.)
            #   2. Lineage already in the sink is not deleted out from under us.
            #
            # Invariant 1 holds by design but is NOT enforced: the buildrunner
            # drops post-finish events on a *cached* StoredBuild status
            # (buildrunner.__process_event), refreshed only when the runner itself
            # writes a status and otherwise re-read at most once per its
            # monitoring_interval. A terminal status written from outside -- an
            # external cancel, or a concurrent stop_and_fail() from another thread
            # -- leaves that cache stale for up to one interval, and a target event
            # arriving in that window still inserts a gb_targets row. Enforcing it
            # would mean a status read from storage next to the insert; that
            # belongs in the buildrunner, not here.
            #
            # Note what the confirmation in _complete_builds does and does not
            # claim. It is a real sink query, not a status inference: the uuid is
            # added only after filter_unrecorded answered for every candidate and
            # every needed write succeeded. But it is accurate as of *that pass's*
            # target read. A target inserted after select_recordable_targets
            # returned was never in `candidates`, so the sink was never asked about
            # it, and this gate then skips the build without re-reading it.
            #
            # If either invariant stops holding -- a target added to a finished
            # build, or a retention policy pruning wandb runs -- a build cached
            # complete here keeps the checkpoint advancing over lineage that is
            # missing or stale, and nothing detects it. The bound on that is
            # process lifetime: _complete_builds is in-memory (see __init__), so a
            # restart re-asks the sink for whatever is still in range. Recording is
            # not affected either way, only the decision to skip re-reading.
            if build.uuid in self._complete_builds and build.status.is_finished():
                continue

            result = reconcile_build(
                self._store,
                storage,
                build_id=build.uuid,
                on_error=lambda build_id, target_id, exc: (
                    self._on_record_error(storage, build_id, target_id, exc)
                ),
                on_success=self._on_record_success,
                skip=self._dropped,
            )

            if result.dedup_query_failed:
                # Abort the whole pass, not just this build: the sink could not
                # answer here and will not answer for the builds behind it either,
                # and with random run ids proceeding on an unanswered query is what
                # writes duplicates. The checkpoint stays put; the next scan retries
                # everything. Builds already recorded earlier in this pass keep
                # their work — nothing is undone, the walk simply stops.
                failure = result.query_failure
                if failure is not None and is_permanent_sink_failure(failure):
                    self._recording_disabled = True
                    self._disabled_reason = str(failure)
                    logger.critical(
                        "The lineage sink rejected the dedup query for build %s "
                        "with a failure no retry can clear; disabling lineage "
                        "recording. Nothing further will be recorded and the "
                        "checkpoint will not advance until an operator fixes the "
                        "sink configuration and restarts the watcher. Underlying "
                        "error: %s",
                        build.uuid,
                        failure,
                    )
                else:
                    logger.error(
                        "Aborting this lineage scan: the sink could not answer "
                        "whether build %s's targets are already recorded, and "
                        "recording on an unanswered query would duplicate runs. "
                        "The checkpoint stays at %s; retrying next scan. "
                        "Underlying error: %s",
                        build.uuid,
                        anchor_build_id,
                        failure,
                    )
                # Not progress: the pass aborted and the mark stayed put. Sleeping
                # before the retry is the point -- an immediate retry against a
                # sink that just failed to answer is a hot loop.
                return False

            if result.dropped:
                # A build advanced past with knowingly-missing lineage. Logged at
                # ERROR naming the ids, because this gap is otherwise invisible.
                logger.error(
                    "Build %s has %d target(s) whose lineage was permanently "
                    "dropped and will never reach the sink: %s. Its lineage in "
                    "the sink is knowingly incomplete.",
                    build.uuid,
                    len(result.dropped),
                    ", ".join(sorted(result.dropped)),
                )

            if result.all_confirmed and (
                not result.sink_unqueried or build.status.is_finished()
            ):
                self._complete_builds.add(build.uuid)
            else:
                # A confirmation the sink was never asked for is cached only once the
                # build is finished, and the build state is what separates the two
                # reasons such a pass can come back with an empty candidate set.
                # Either reason reaches here: no target rows at all, or target rows
                # that are all in the durable drop set.
                #
                # Still RUNNING: the targets do not exist *yet*. The build row and
                # its target rows are separate, non-transactional writes, so a scan
                # lands between them and reads none. Caching that would arm the skip
                # gate below ("cached complete AND finished") on a build whose
                # targets appear moments later, and it would then skip re-reading
                # them forever -- only a restart cleared this set, which is exactly
                # how the bug surfaced (lineage recorded only after a service
                # restart).
                #
                # Finished: the emptiness is treated as final and cached above. It
                # has to be: _advance_checkpoint requires the anchor in this set, so
                # discarding a finished build here would wedge the mark on it
                # forever and block every newer build's lineage behind it. That
                # includes every FAILED build, which select_builds_from_checkpoint
                # does not filter out and which therefore does become an anchor.
                #
                # "Finished" is very nearly "will never gain a *recordable* target",
                # and the gap is narrower than the bare build/target write ordering
                # suggests. Two things close most of it:
                #
                #   - finalize_build_status (buildrunner/build_utils.py) finalizes
                #     targets before it writes the build's terminal status, and
                #     writes that status only when the build is not already
                #     finished. So on the ordinary path the targets are committed
                #     first, by construction rather than by luck.
                #   - select_recordable_targets asks for status==SUCCESS with a
                #     non-NULL finished_at, so a merely-present row is not enough to
                #     lose. Notably the re-run entity finalization does NOT
                #     manufacture one: _finalize_target_or_step_status leaves a
                #     PENDING target PENDING when the build succeeded (it logs a
                #     warning and treats it as a platform bug), so it never
                #     back-fills a target into SUCCESS behind us.
                #
                # What remains is a genuine target event -- carrying its own SUCCESS
                # and finished_at -- drained from the buildrunner's FIFO after a
                # terminal status was written from *outside* that flow: an external
                # cancel, or a concurrent stop_and_fail() landing on a cached
                # StoredBuild status (see the skip gate's note on that stale
                # window). When that happens the skip gate never re-reads this build
                # and the checkpoint advances past it, losing that lineage until a
                # restart clears this in-memory set.
                #
                # That is the accepted side of the trade, and the asymmetry is what
                # settles it: the loss is rare and bounded by process lifetime,
                # while the alternative wedges the mark on every finished build with
                # no recordable targets -- the common case, not the race. If it ever
                # needs closing, the cheap fix lives here (age out this set, or
                # withhold caching when the terminal status did not come from the
                # runner) rather than in a status re-read next to the target insert.
                self._complete_builds.discard(build.uuid)

        advanced = self._advance_checkpoint(storage, anchor_build_id, builds)
        # Drop confirmations for builds this scan no longer selects. Selection is
        # ">= the anchor", so a build older than the anchor is never read again and
        # its entry can only be dead weight -- otherwise the set grows by one uuid
        # per build ever confirmed and is cleared only by a restart. Intersecting
        # against *this* scan's list is the safe cut: it contains every build the
        # cutoff still reaches, including the one just advanced onto.
        self._complete_builds &= {b.uuid for b in builds}
        if not advanced:
            # The steady state once everything is recorded: the anchor is the
            # newest build, so the mark has nowhere to step. Say so rather than
            # returning in silence -- "held" is a healthy scan, and it reads
            # identically to a stalled watcher without this line.
            logger.info(
                "Lineage scan: checkpoint held at %s; sleeping %.0fs.",
                anchor_build_id,
                self.monitoring_interval,
            )
        return advanced

    def _advance_checkpoint(
        self,
        storage: SingletonAdminStorage,
        anchor_build_id: str,
        builds: list[StoredBuild],
    ) -> bool:
        """Move the checkpoint to the build after the anchor, once the anchor is done.

        Returns whether the mark moved. The caller uses that to decide between
        stepping again immediately and sleeping the interval.

        The condition is on the **anchor**, not on the build being moved to: the
        mark leaves A only once A is finished and its lineage is confirmed in the
        sink. Where it lands is simply the next build by creation order, whatever
        its state — a running B becomes the new base, and the same condition then
        governs the step off B. So the walk is A -> B -> C, one build per scan,
        each step gated by the build it is leaving.

        This is safe because the base stays in range: selection is "the checkpoint
        build and everything created at or after it"
        (``select_builds_from_checkpoint``), so a running base is re-reconciled
        every scan and any target it produces later is still picked up. Nothing
        falls out of the cutoff with lineage pending — the step off a build is what
        requires that build to be finished and confirmed.

        The mark therefore reads as "everything created strictly before this build
        is confirmed in the sink"; the build it names may still be running.

        Stepping one build per scan (rather than jumping to the far end of a run of
        complete builds) keeps the durable mark close to the work: a process that
        dies mid-catch-up resumes one build back instead of redoing the whole run.

        Note this bounds only the *mark*, not the recording: builds after the base
        still have their lineage written on this same pass (see ``_reconcile``).
        Only the checkpoint waits.

        ``builds`` must be oldest-created-first, which is what
        ``select_builds_from_checkpoint`` returns.
        """
        anchor_index = next(
            (i for i, b in enumerate(builds) if b.uuid == anchor_build_id), None
        )
        # Branch on the index rather than on a separate `anchor is None` check: the
        # two are equivalent at runtime, but narrowing the index here is what lets
        # the successor slice below use it as an int without a second guard.
        if anchor_index is None:
            # No anchor row to gate on. Two very different reasons:
            #
            # The backfill sentinel names no build by construction, so it is never
            # in the list. It resolves to a UTC_MIN cutoff, so leaving the mark on
            # it re-selects the platform's whole history every scan -- it must be
            # stepped off, and there is no anchor state to require. Move onto the
            # oldest selected build unconditionally; the gate applies from there on.
            #
            # Otherwise the checkpoint names a build whose row is gone or
            # unreadable. Nothing to reason about, and guessing could step over
            # unrecorded lineage, so hold the mark and let an operator intervene.
            if anchor_build_id != BACKFILL_BUILD_ID:
                return False
            return self._write_checkpoint(storage, builds[0])
        anchor = builds[anchor_index]
        # The gate: the mark may leave the anchor only once the anchor can no
        # longer gain lineage and everything it has is in the sink. A running
        # anchor, or one whose targets failed to record, holds the mark — stepping
        # off it would put it behind the cutoff with lineage still missing and no
        # later scan able to reach it.
        if not anchor.status.is_finished():
            return False
        # Two distinct questions, both required. is_finished() above is build
        # state from the admin DB ("did it stop running?"); this is sink state
        # ("did its lineage arrive?"). Replacing this with another status check
        # would advance the mark on a finished build whose lineage never reached
        # the sink -- and since advancing moves it out of the selection range,
        # that lineage would be unreachable by any later scan. The cache is read
        # rather than re-queried on purpose; see the invariants noted at the
        # equivalent skip in _reconcile.
        if anchor.uuid not in self._complete_builds:
            return False
        # Step to the build immediately after the anchor *in this order*, rather
        # than to the first non-anchor entry. Builds created at virtually the same
        # instant as the anchor have no defined order between them (see
        # ``select_builds_from_checkpoint`` on the ``>=`` bound), so one can sort
        # ahead of it; treating that as the destination would advance the mark past
        # a build the new cutoff then excludes. Anything ahead of the anchor here
        # is reconciled on this same pass either way -- it is only ineligible as a
        # destination.
        successors = builds[anchor_index + 1 :]
        if not successors:
            # The anchor is the newest build: complete, but nothing to move to yet.
            return False
        # Whatever its status: it becomes the new base and stays in range until it
        # too is finished and confirmed.
        return self._write_checkpoint(storage, successors[0])

    def _read_checkpoint_value(self, storage: SingletonAdminStorage) -> Optional[dict]:
        """Read the raw checkpoint value, or None when there is nothing usable.

        Split from ``_read_checkpoint`` so that "is there a checkpoint at all"
        (including the one-time absent notice) is separate from interpreting its
        shape. A read failure is logged and treated as absent: recording nothing
        for one scan is safe, whereas raising would abort the loop iteration.
        """
        try:
            value = storage.kv_pair_storage.get_value(LINEAGE_WATCHER_CHECKPOINT_KEY)
        except Exception:
            logger.exception(
                "Failed to read the lineage checkpoint from %s; recording nothing "
                "this scan.",
                LINEAGE_WATCHER_CHECKPOINT_KEY,
            )
            return None

        if not value:
            if not self._missing_checkpoint_logged:
                logger.info(
                    "No lineage checkpoint under %s; recording nothing until one "
                    "is seeded (see the lineage-watch command's --base-build-id).",
                    LINEAGE_WATCHER_CHECKPOINT_KEY,
                )
                self._missing_checkpoint_logged = True
            return None
        self._missing_checkpoint_logged = False
        return value

    def _read_checkpoint(
        self, storage: SingletonAdminStorage
    ) -> Optional[tuple[str, datetime]]:
        """Read the checkpoint as ``(build_id, created_time)``.

        Accepts both value shapes, so an existing deployment keeps its place:

        - v2 (``{"build_id", "created_time", "version"}``) is used directly.
        - v1 (``{"build_id", "finished_at"}``, where the timestamp was a *target*'s
          finish time) contributes only its ``build_id``; the build's own
          ``created_time`` is re-read from storage and the key is rewritten in v2
          form. The v1 timestamp is deliberately not reused — it measured a
          different thing and would place the cutoff at the wrong instant.
        - The backfill sentinel keeps its "reach everything" meaning by resolving
          to ``UTC_MIN`` without needing a build row.

        Returns None when nothing is seeded, when the value is unusable, or when a
        v1 build no longer exists — all meaning "record nothing", never "scan
        everything".
        """
        value = self._read_checkpoint_value(storage)
        if value is None:
            return None

        build_id = value.get("build_id")
        if not build_id:
            logger.error(
                "Lineage checkpoint under %s has no build_id (%s); recording "
                "nothing this scan.",
                LINEAGE_WATCHER_CHECKPOINT_KEY,
                value,
            )
            return None

        if build_id == BACKFILL_BUILD_ID:
            # The backfill anchor names no real build: everything is in range.
            return build_id, UTC_MIN

        raw_created = value.get("created_time")
        if raw_created is not None:
            try:
                return build_id, as_aware(datetime.fromisoformat(raw_created))
            except (TypeError, ValueError) as exc:
                logger.error(
                    "Lineage checkpoint under %s has an unparseable created_time "
                    "(%r): %s. Recording nothing this scan.",
                    LINEAGE_WATCHER_CHECKPOINT_KEY,
                    raw_created,
                    exc,
                )
                return None

        # v1 shape: re-resolve the anchor from the build itself.
        return self._migrate_v1_checkpoint(storage, build_id)

    def _migrate_v1_checkpoint(
        self, storage: SingletonAdminStorage, build_id: str
    ) -> Optional[tuple[str, datetime]]:
        """Re-anchor an old target-shaped checkpoint on its build and rewrite it.

        Only the ``build_id`` carries over. The v1 timestamp was a *target*'s
        finish time, so reusing it would place the build cutoff at an unrelated
        instant; the build's own ``created_time`` is read fresh instead.

        Returns None if the build is gone — "record nothing" until re-seeded,
        never "no cutoff", which would turn a stale checkpoint into a full backfill.
        """
        build = storage.build_storage.get_by_uuid(build_id)
        if not isinstance(build, StoredBuild) or build.created_time is None:
            logger.error(
                "Lineage checkpoint under %s names build %s in the old "
                "target-shaped form, but that build has no readable creation "
                "time now. Recording nothing this scan; re-seed the checkpoint to "
                "continue.",
                LINEAGE_WATCHER_CHECKPOINT_KEY,
                build_id,
            )
            return None
        created = as_aware(build.created_time)
        logger.info(
            "Migrating the lineage checkpoint under %s from the target-shaped "
            "form to the build-shaped one: build %s, created %s.",
            LINEAGE_WATCHER_CHECKPOINT_KEY,
            build_id,
            created,
        )
        self._write_checkpoint(storage, build)
        return build_id, created

    def _write_checkpoint(
        self, storage: SingletonAdminStorage, build: StoredBuild
    ) -> bool:
        """Persist ``build`` as the checkpoint anchor.

        Returns whether the mark actually moved, which is what lets the loop
        catch up without sleeping between steps. A swallowed failure below
        returns False: the mark did not move, so there is no progress to chase.

        The timestamp is written verbatim from the build row rather than converted
        to UTC, so the checkpoint reads identically to the ``created_time`` it came
        from (see ``as_aware`` on why rewriting offsets is what made the same row
        appear to differ by hours depending on where it was read).

        Failures are logged and swallowed: losing an advance costs a repeated scan,
        which the dedup query makes harmless, while raising here would abort a scan
        that has already recorded successfully.
        """
        if build.created_time is None:
            logger.error(
                "Refusing to checkpoint build %s: it has no creation time to "
                "anchor on.",
                build.uuid,
            )
            return False
        payload = {
            "build_id": build.uuid,
            "created_time": build.created_time.isoformat(),
            "version": LINEAGE_WATCHER_CHECKPOINT_VERSION,
        }
        try:
            storage.kv_pair_storage.set_value(LINEAGE_WATCHER_CHECKPOINT_KEY, payload)
        except Exception:
            logger.exception(
                "Failed to persist the lineage checkpoint at build %s; it will be "
                "retried next scan.",
                build.uuid,
            )
            return False
        logger.info(
            "Lineage checkpoint advanced to build %s (created %s).",
            build.uuid,
            build.created_time,
        )
        return True

    # pylint: disable=unused-argument  # build_id is part of the on_success contract
    def _on_record_success(self, build_id: str, target_id: str) -> None:
        """Clear any retry state for a target that recorded successfully.

        ``build_id`` is unused but kept: this is passed as ``reconcile_build``'s
        ``on_success`` callback, whose signature is ``(build_id, target_id)``.

        A target that failed a prior scan and then succeeds is reported only
        here — it drops out of the unrecorded set on the next scan, so
        ``_on_record_error`` is never called for it again. Without this, its
        ``_failed_attempts`` entry would linger for the process lifetime and a
        much-later re-failure would resume from a nonzero count.
        """
        self._failed_attempts.pop(target_id, None)

    def _on_record_error(
        self,
        storage: SingletonAdminStorage,
        build_id: str,
        target_id: str,
        exc: Exception,
    ) -> None:
        """Handle a recording failure for one target: retry or drop.

        Keeps a per-target attempt count so a transient failure is retried on the
        next scan, while a persistently failing target is dropped after
        ``_MAX_RECORD_ATTEMPTS`` (added to ``_dropped``, which ``_reconcile`` passes
        as ``skip``) so it stops being re-recorded — it still falls within the
        selected range each scan, so the skip set is what keeps it from wedging the
        checkpoint.

        The drop is persisted: giving up is permanent, and since the checkpoint
        never advances past a build with unrecorded lineage, a drop forgotten on
        restart would block it forever. The attempt *counts* stay in memory — a
        within-process backoff, and a restart legitimately retries from zero (the
        failure may have been the crash itself). Only the terminal decision is
        durable.

        Every failure is retryable here. The one rejection that used to be
        permanent — a run id the sink had seen and deleted — cannot occur now that
        ids are fresh random uuids, so there is no special case left to make.
        """
        attempts = self._failed_attempts.get(target_id, 0) + 1
        if attempts >= self._MAX_RECORD_ATTEMPTS:
            self._failed_attempts.pop(target_id, None)
            # Mark dropped so a persistent failure does not wedge every scan,
            # and persist it so a restart does not resurrect the target.
            self._dropped.add(target_id)
            self._persist_dropped(storage)
            logger.exception(
                "Dropping lineage for target %s in build %s after %d attempts: %s",
                target_id,
                build_id,
                attempts,
                exc,
            )
        else:
            self._failed_attempts[target_id] = attempts
            logger.warning(
                "Failed to record lineage for target %s in build %s "
                "(attempt %d/%d); will retry on next scan: %s",
                target_id,
                build_id,
                attempts,
                self._MAX_RECORD_ATTEMPTS,
                exc,
            )

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the watcher thread to stop and wait for it to exit.

        Joins the worker thread (bounded by ``timeout``) so shutdown does not
        race an in-flight scan, and resets state so the watcher can be started
        again.

        Args:
            timeout: Maximum seconds to wait for the worker thread to exit.
        """
        logger.info("Stopping LineageWatcher")
        self.stop_event.set()
        thread = self.worker_thread
        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning(
                    "LineageWatcher thread did not stop within %.1fs", timeout
                )
        self.worker_thread = None
        self.stop_event.clear()
        # Nothing checkpoint-related to reset: it lives only in the gb_kv_pairs
        # checkpoint, which every scan re-reads.
