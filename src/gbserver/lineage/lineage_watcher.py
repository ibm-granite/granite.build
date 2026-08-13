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
from datetime import datetime, timedelta
from typing import Optional

from gbserver.lineage.jobstats import ILineageStore, get_lineage_store
from gbserver.lineage.lineage_reconciler import (
    LINEAGE_WATCHER_CHECKPOINT_KEY,
    _expected_run_count,
    get_most_recent_successful_target,
    reconcile_once,
    record_target_lineage,
)
from gbserver.storage.singleton_storage import SingletonAdminStorage, get_admin_storage
from gbserver.types.status import Status
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class LineageWatcher:
    """Async background thread that reconciles lineage from the admin DB.

    Runs a single background daemon thread that periodically calls
    ``reconcile_once`` (see ``lineage_reconciler``), which scans the admin DB for
    successful target runs and records their lineage into the configured store,
    off the build's hot path.

    Reconciliation — not the event stream — is the authoritative mechanism: the
    admin DB persists the complete lineage graph, so the full lineage is
    recoverable by re-reading it alone. Because each scan re-derives the
    recordable set from the DB, a target that succeeded while this process was
    down is picked up on the next scan; there is no restart blind spot. Recording
    is idempotent (deterministic runIds + resume="allow" + content-dedupe), so a
    re-recorded target is a harmless backend no-op.

    Single-writer guarantee: the watcher is deployed as its own single-replica
    ``lineage-watch`` command/pod (see ``command_lineage_watch.py`` and
    ``dep-lineage-watcher.yaml``), so exactly one process reconciles lineage. It
    must not be wired into any other entrypoint. Even if that were violated,
    idempotent recording means a duplicate watcher would waste I/O but not
    corrupt lineage.

    Steady state uses a ``finished_at`` *time watermark* (``_last_seen``): each
    scan asks the admin DB only for targets that finished at or after the
    watermark, so per-scan work stays bounded no matter how many builds have
    accumulated. The watermark is persisted to ``gb_status`` (see
    ``lineage_reconciler.LINEAGE_WATCHER_CHECKPOINT_KEY``) after each
    individually-recorded target, so a restart resumes from the last
    successfully-recorded target rather than rescanning the whole admin DB.
    ``start()`` loads this checkpoint; if none exists yet (fresh deployment), it
    seeds one from the single most-recent successful target — the watcher
    starts "from now," not from a full historical backfill. Either way, the
    checkpoint's target is verified against the store's own recorded-state
    (``filter_unrecorded``) and re-recorded if missing, closing any gap left by
    a crash between recording and persisting the checkpoint (or vice versa) —
    this is also what actually records a freshly-seeded checkpoint's target, so
    seeding never leaves a target silently unrecorded. A small
    ``_WATERMARK_OVERLAP`` is subtracted when querying so a target that
    finished in the same instant as the watermark boundary is never skipped;
    idempotent recording makes the resulting re-reads harmless.

    Which of those newly-finished targets actually get recorded is decided
    per-sink by ``store.filter_unrecorded`` (see ``reconcile_once``): the time
    watermark is sink-neutral, and each sink owns its own recorded-state, so the
    same admin DB can feed W&B and other sinks independently.

    A target whose recording raises is retried on the next scan (the watermark
    does not advance past a completion just because recording failed, and the
    overlap guard re-surfaces it); a target that keeps failing is dropped after
    ``_MAX_RECORD_ATTEMPTS`` so a persistent failure cannot wedge later scans.
    """

    # A target whose lineage recording keeps failing is retried this many times
    # on subsequent scans before being dropped, so a transient failure (e.g. a
    # network blip) is recovered without a persistent failure wedging the scan.
    _MAX_RECORD_ATTEMPTS = 3

    # Subtracted from the watermark when querying so a target that finished at (or
    # a hair before) the boundary is re-surfaced rather than skipped — guards
    # against equal-timestamp / clock-resolution races at the watermark edge.
    # Re-reads are harmless because recording is idempotent.
    _WATERMARK_OVERLAP = timedelta(seconds=5)

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
        # Target uuids dropped after exhausting retries; skipped on later scans
        # so a persistently failing target cannot wedge every scan.
        self._dropped: set[str] = set()
        # target_uuid -> attempts so far, for targets whose recording failed and
        # should be retried on a subsequent scan.
        self._failed_attempts: dict[str, int] = {}
        # Watermark: the newest target ``finished_at`` actually recorded so far.
        # Loaded from the persisted ``gb_status`` checkpoint on start() (or
        # seeded from the most recent successful target if none exists yet);
        # later scans query only targets that finished at/after this, keeping
        # per-scan work bounded regardless of how many builds have accumulated.
        self._last_seen: Optional[datetime] = None

    def start(self) -> None:
        """Start the watcher thread (daemon=True, does not keep process alive)."""
        if self.worker_thread is not None and self.worker_thread.is_alive():
            logger.error("lineage watcher thread is already running")
            return

        self._store = get_lineage_store()
        self._load_or_seed_checkpoint(get_admin_storage())
        self.worker_thread = threading.Thread(
            target=self._run, name="lineage-watcher", daemon=True
        )
        self.worker_thread.start()
        logger.info("LineageWatcher started")

    def _load_or_seed_checkpoint(self, storage: SingletonAdminStorage) -> None:
        """Establish ``self._last_seen`` from a persisted or freshly-seeded checkpoint.

        Loads the checkpoint from ``gb_status`` if present. If absent (fresh
        deployment, or a store predating this feature), seeds one from the
        single most-recent successful target instead of defaulting to a full
        historical catch-up. Either way, the checkpoint's target is then
        verified against the store's own recorded-state and re-recorded if
        missing — closing a gap from a crash between recording and persisting
        the checkpoint (or vice versa), and, for a freshly-seeded checkpoint,
        actually recording its target (seeding is not itself a "don't record"
        special case).

        Leaves ``self._last_seen`` as ``None`` if there is no checkpoint to
        load and nothing to seed from (no successful target exists yet); the
        next scan then has nothing to do until one succeeds.
        """
        checkpoint = storage.status_storage.get_value(LINEAGE_WATCHER_CHECKPOINT_KEY)
        if checkpoint is None:
            latest = get_most_recent_successful_target(storage)
            if latest is None:
                return
            checkpoint = {
                "build_id": latest.build_id,
                "finished_at": latest.finished_at.isoformat(),
            }
            storage.status_storage.set_value(LINEAGE_WATCHER_CHECKPOINT_KEY, checkpoint)

        checkpoint_build_id = checkpoint["build_id"]
        checkpoint_finished_at = datetime.fromisoformat(checkpoint["finished_at"])
        self._verify_checkpoint_target_recorded(storage, checkpoint_build_id)
        self._last_seen = checkpoint_finished_at

    def _verify_checkpoint_target_recorded(
        self, storage: SingletonAdminStorage, build_id: str
    ) -> None:
        """Re-record the checkpoint's target(s) if the store doesn't have them.

        A crash could have persisted the checkpoint without the store write
        actually landing (or the checkpoint could be freshly seeded and never
        recorded at all). ``filter_unrecorded`` is the same per-sink check
        ``reconcile_once`` uses, so this is a harmless idempotent no-op when
        the target really is already recorded.
        """
        if self._store is None:
            return
        targets = [
            t
            for t in storage.target_storage.get_by_where({"build_id": build_id})
            if t.status == Status.SUCCESS
        ]
        if not targets:
            return
        expected_counts = {
            t.uuid: _expected_run_count(t)
            for t in targets
            if not t.skipped_for_prerun_target_id
        }
        by_uuid = {t.uuid: t for t in targets}
        unrecorded = self._store.filter_unrecorded(set(by_uuid), expected_counts)
        for uuid in unrecorded:
            target = by_uuid[uuid]
            try:
                record_target_lineage(
                    self._store, storage, build_id=target.build_id, target_id=target.uuid
                )
            except Exception:
                logger.exception(
                    "Failed to re-record checkpoint target %s in build %s on start()",
                    target.uuid,
                    target.build_id,
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

        Delegates target selection and recording to ``reconcile_once`` (the
        central mechanism), passing the ``finished_at`` watermark so steady-state
        scans read only newly-finished targets. ``start()`` has already
        established ``self._last_seen`` from a persisted or freshly-seeded
        checkpoint, so ``_last_seen`` is only ``None`` here in the (harmless)
        case where the admin DB had no successful target at all when the
        watcher started. Recording failures are routed to ``_on_record_error`` to drive the
        bounded per-target retry. The watermark advances, and the checkpoint is
        persisted to ``gb_status``, immediately after each individually-recorded
        target (``_on_checkpoint_advance``) rather than once at the end of the
        scan, so a mid-scan crash leaves the checkpoint at the last target
        actually recorded.
        """
        if self._store is None:
            logger.error("lineage store not initialized; start() must run first")
            return
        storage = get_admin_storage()
        # self._last_seen is None only when start() found no checkpoint to load
        # and nothing to seed from (no successful target exists yet at all) —
        # a full-DB scan in that state finds nothing, so it is harmless and
        # will establish a watermark as soon as one target succeeds.
        # Otherwise, query slightly before the watermark so a
        # boundary-timestamp completion is not skipped; idempotent recording
        # makes the overlap re-reads harmless.
        finished_after = (
            None
            if self._last_seen is None
            else self._last_seen - self._WATERMARK_OVERLAP
        )
        reconcile_once(
            self._store,
            storage,
            finished_after=finished_after,
            on_error=self._on_record_error,
            on_success=self._on_record_success,
            on_checkpoint_advance=lambda build_id, finished_at: (
                self._on_checkpoint_advance(storage, build_id, finished_at)
            ),
            skip=self._dropped,
        )

    def _on_checkpoint_advance(
        self, storage: SingletonAdminStorage, build_id: str, finished_at: datetime
    ) -> None:
        """Persist the checkpoint and advance the in-memory watermark.

        Called once per successfully-recorded target (oldest-first), so the
        durable checkpoint always reflects the last target actually recorded —
        never a target merely considered or one that failed to record.
        """
        storage.status_storage.set_value(
            LINEAGE_WATCHER_CHECKPOINT_KEY,
            {"build_id": build_id, "finished_at": finished_at.isoformat()},
        )
        self._last_seen = finished_at

    def _on_record_success(self, build_id: str, target_id: str) -> None:
        """Clear any retry state for a target that recorded successfully.

        A target that failed a prior scan and then succeeds is reported only
        here — it drops out of the unrecorded set on the next scan, so
        ``_on_record_error`` is never called for it again. Without this, its
        ``_failed_attempts`` entry would linger for the process lifetime and a
        much-later re-failure would resume from a nonzero count.
        """
        self._failed_attempts.pop(target_id, None)

    def _on_record_error(self, build_id: str, target_id: str, exc: Exception) -> None:
        """Handle a recording failure for one target: retry or drop.

        Keeps a per-target attempt count so a transient failure is retried on the
        next scan, while a persistently failing target is dropped after
        ``_MAX_RECORD_ATTEMPTS`` (added to ``_dropped``, which ``_reconcile``
        passes as ``skip`` to ``reconcile_once``) so it stops being re-recorded —
        it still falls within the watermark window each scan, so the skip set is
        what keeps it from wedging the scan.
        """
        attempts = self._failed_attempts.get(target_id, 0) + 1
        if attempts >= self._MAX_RECORD_ATTEMPTS:
            self._failed_attempts.pop(target_id, None)
            # Mark dropped so a persistent failure does not wedge every scan.
            self._dropped.add(target_id)
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
        # Do NOT reset self._last_seen here: the checkpoint is persisted to
        # gb_status, so the next start() reloads it (or re-verifies/re-seeds)
        # rather than needing in-memory state to survive a restart.
