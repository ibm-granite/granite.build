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
import time
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

    The checkpoint advances **contiguously** and only over builds that are both
    finished and confirmed in the sink: the walk stops at the first build that is
    still running, or finished but not fully recorded. Given A(finished),
    B(running), C(finished) it lands on A and stays there — C's targets are still
    recorded, they just do not move the mark. It never skips a live build, so no
    build ever falls out of range while it still has lineage to produce.

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
        # Load the durable drop set before the first scan, so a target already
        # given up on stays skipped instead of blocking the checkpoint again.
        self._load_dropped(storage)
        self.worker_thread = threading.Thread(
            target=self._run, name="lineage-watcher", daemon=True
        )
        self.worker_thread.start()
        logger.info("LineageWatcher started")

    def _load_dropped(self, storage: SingletonAdminStorage) -> None:
        """Load the durable set of permanently-given-up-on target uuids.

        A dropped target is one that failed ``_MAX_RECORD_ATTEMPTS`` times; the
        decision to stop trying is permanent, so it must outlive the process.
        Without this the target would return on the next start(), block the
        checkpoint (which never advances past a build with unrecorded lineage),
        exhaust its attempts again, and repeat every restart — wedging all newer
        lineage behind it.
        """
        value = storage.kv_pair_storage.get_value(LINEAGE_WATCHER_DROPPED_KEY)
        if value:
            self._dropped = set(value.get("target_ids", []))

    def _persist_dropped(self, storage: SingletonAdminStorage) -> None:
        """Persist the drop set so the decision survives a restart."""
        storage.kv_pair_storage.set_value(
            LINEAGE_WATCHER_DROPPED_KEY, {"target_ids": sorted(self._dropped)}
        )

    def _run(self) -> None:
        """Main monitoring loop (runs in daemon thread)."""
        while not self.stop_event.is_set():
            try:
                self._reconcile()
            except Exception:
                logger.exception("LineageWatcher iteration failed")

            time.sleep(self.monitoring_interval)

    def _reconcile(self) -> None:
        """Run one reconciliation scan over the admin DB.

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
            return
        if self._recording_disabled:
            logger.critical(
                "Lineage recording is disabled: the sink reported a failure no "
                "retry can clear, so no lineage is being recorded and the "
                "checkpoint is frozen. An operator must fix the sink "
                "configuration (project, entity, credentials) and restart the "
                "watcher. Underlying error: %s",
                self._disabled_reason,
            )
            return

        storage = get_admin_storage()
        checkpoint = self._read_checkpoint(storage)
        if checkpoint is None:
            # No checkpoint: recording is off until seeded. Return before selecting
            # builds or touching the sink.
            return
        anchor_build_id, anchor_created_time = checkpoint

        builds = select_builds_from_checkpoint(storage, anchor_created_time)
        if not builds:
            return

        for build in builds:
            # A build already finished and confirmed cannot gain lineage, so skip
            # the per-build target read entirely. This is what keeps a pinned
            # cutoff from re-reading every newer build's targets each scan.
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
                return

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

            if result.all_confirmed:
                self._complete_builds.add(build.uuid)
            else:
                self._complete_builds.discard(build.uuid)

        self._advance_checkpoint(storage, anchor_build_id, builds)

    def _advance_checkpoint(
        self,
        storage: SingletonAdminStorage,
        anchor_build_id: str,
        builds: list[StoredBuild],
    ) -> None:
        """Move the checkpoint one build forward, if the next one is complete.

        One build per scan, deliberately. The mark walks the sequence step by step
        — base -> next -> next — rather than jumping to the far end of a run of
        already-complete builds. Both reach the same place eventually and neither
        loses lineage (a build behind the mark was confirmed before the mark
        passed it), but stepping keeps the durable mark closer to the work: a
        process that dies mid-catch-up resumes one build back instead of redoing
        the whole run.

        "Complete" is two conditions, and the *first* is what makes the walk safe:
        a build still running must never be stepped over, because it can still
        produce targets and the selection cutoff would then exclude it forever.
        The second is that the sink has confirmed its lineage — a finished build
        whose targets failed to record must not be passed either.

        Note this bounds only the *mark*, not the recording: builds after an
        unfinished one still have their lineage written on this same pass (see
        ``_reconcile``). Only the checkpoint waits.

        ``builds`` must be oldest-created-first, which is what
        ``select_builds_from_checkpoint`` returns.
        """
        for build in builds:
            if build.uuid == anchor_build_id:
                # The anchor itself: already the mark, so it is not a candidate to
                # move to. Skip rather than stop — the build to advance to is the
                # one after it.
                continue
            if not build.status.is_finished():
                return
            if build.uuid not in self._complete_builds:
                return
            # First build past the anchor that is finished and confirmed. Take it
            # and stop: the next scan takes the one after, and so on.
            self._write_checkpoint(storage, build)
            return

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
    ) -> None:
        """Persist ``build`` as the checkpoint anchor.

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
            return
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
            return
        logger.info(
            "Lineage checkpoint advanced to build %s (created %s).",
            build.uuid,
            build.created_time,
        )

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
