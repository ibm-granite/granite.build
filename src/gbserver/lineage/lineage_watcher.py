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
    get_most_recent_successful_target,
    reconcile_once,
)
from gbserver.storage.singleton_storage import SingletonAdminStorage, get_admin_storage
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
    (``filter_unrecorded``, via a build-scoped ``reconcile_once``) and
    re-recorded if missing, closing any gap left by a crash between recording
    and persisting the checkpoint (or vice versa) — this is also what actually
    records a freshly-seeded checkpoint's target, so seeding never leaves a
    target silently unrecorded. A freshly-seeded checkpoint is persisted only
    *after* that recording succeeds, so a failed seed-time recording is retried
    on the next start() rather than being stranded behind an already-advanced
    watermark. A small
    ``_WATERMARK_OVERLAP`` is subtracted when querying so a target that
    finished in the same instant as the watermark boundary is never skipped;
    idempotent recording makes the resulting re-reads harmless.

    Which of those newly-finished targets actually get recorded is decided
    per-sink by ``store.filter_unrecorded`` (see ``reconcile_once``): the time
    watermark is sink-neutral, and each sink owns its own recorded-state, so the
    same admin DB can feed W&B and other sinks independently.

    A target whose recording raises is retried on the next scan: the checkpoint
    stops advancing at the failed target for the remainder of that pass (even
    though newer targets in the same pass may still record), so the durable
    watermark never moves past unrecorded lineage and the failed target is
    re-surfaced by the next scan — including after a restart, which is what makes
    retry survive process death rather than depending on in-memory state. A
    target that keeps failing is dropped after ``_MAX_RECORD_ATTEMPTS`` (into
    ``_dropped``, passed as ``skip``) so a persistent failure cannot wedge later
    scans or hold the checkpoint back forever.
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
        historical catch-up. Either way, the checkpoint's build is then run
        through ``reconcile_once`` so any of its unrecorded targets are recorded
        — closing a gap from a crash between recording and persisting the
        checkpoint (or vice versa), and, for a freshly-seeded checkpoint,
        actually recording its target (seeding is not itself a "don't record"
        special case).

        Ordering matters when seeding: the seeded checkpoint is written only
        after its target is confirmed recorded. Writing first would durably
        advance the watermark past a target whose recording then failed —
        recording failures here are logged and swallowed — leaving it
        permanently unrecorded with nothing to retry it.

        Leaves ``self._last_seen`` as ``None`` if there is no checkpoint to
        load and nothing to seed from (no successful target exists yet), or if
        a freshly-seeded checkpoint's target failed to record; the next scan
        then has nothing to do until one succeeds, and the next ``start()``
        re-seeds and retries.
        """
        checkpoint = storage.status_storage.get_value(LINEAGE_WATCHER_CHECKPOINT_KEY)
        seeding = checkpoint is None
        if seeding:
            latest = get_most_recent_successful_target(storage)
            if latest is None:
                return
            checkpoint = {
                "build_id": latest.build_id,
                "finished_at": latest.finished_at.isoformat(),
            }

        # Verify/re-record via reconcile_once — the same central selector the
        # steady-state scan uses — scoped to the checkpoint's own build, so the
        # per-sink filter_unrecorded check, expected-run-count derivation and
        # prerun-skip handling are shared rather than reimplemented here. Its
        # returned build_id is non-None only if a target actually recorded, and
        # on_error reports any that failed.
        failed = False

        def _on_error(build_id: str, target_id: str, exc: Exception) -> None:
            nonlocal failed
            failed = True
            logger.exception(
                "Failed to re-record checkpoint target %s in build %s on start(): %s",
                target_id,
                build_id,
                exc,
            )

        if self._store is not None:
            reconcile_once(
                self._store, storage, build_id=checkpoint["build_id"], on_error=_on_error
            )
        # A freshly-seeded checkpoint is persisted only once its target is known
        # recorded: writing it up-front would durably advance the watermark past
        # a target whose recording then failed (failures here are logged and
        # swallowed), leaving nothing to retry it. Leaving it unwritten means the
        # next start() re-seeds and retries.
        if seeding:
            if failed:
                return
            storage.status_storage.set_value(LINEAGE_WATCHER_CHECKPOINT_KEY, checkpoint)
        self._last_seen = datetime.fromisoformat(checkpoint["finished_at"])

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
