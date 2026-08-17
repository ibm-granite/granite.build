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
from gbserver.lineage.lineage_reconciler import reconcile_once
from gbserver.storage.singleton_storage import get_admin_storage
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
    accumulated. The first scan after start() runs with no watermark (a full
    catch-up over the whole admin DB) so anything that finished while the process
    was down is re-driven; the watermark then advances to the newest completion
    seen. A small ``_WATERMARK_OVERLAP`` is subtracted when querying so a target
    that finished in the same instant as the watermark boundary is never skipped;
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
        # Watermark: the newest target ``finished_at`` seen so far. None until the
        # first scan, which runs as a full catch-up over the whole admin DB;
        # later scans query only targets that finished at/after this, keeping
        # per-scan work bounded regardless of how many builds have accumulated.
        self._last_seen: Optional[datetime] = None

    def start(self) -> None:
        """Start the watcher thread (daemon=True, does not keep process alive)."""
        if self.worker_thread is not None and self.worker_thread.is_alive():
            logger.error("lineage watcher thread is already running")
            return

        self._store = get_lineage_store()
        self.worker_thread = threading.Thread(
            target=self._run, name="lineage-watcher", daemon=True
        )
        self.worker_thread.start()
        logger.info("LineageWatcher started")

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
        scans read only newly-finished targets. The first scan passes no
        watermark (full catch-up); the watermark then advances to the newest
        completion seen. Recording failures are routed to ``_on_record_error`` to
        drive the bounded per-target retry.
        """
        if self._store is None:
            logger.error("lineage store not initialized; start() must run first")
            return
        storage = get_admin_storage()
        # First scan: no watermark → full catch-up. Later scans query slightly
        # before the watermark so a boundary-timestamp completion is not skipped;
        # idempotent recording makes the overlap re-reads harmless.
        finished_after = (
            None
            if self._last_seen is None
            else self._last_seen - self._WATERMARK_OVERLAP
        )
        max_finished_at = reconcile_once(
            self._store,
            storage,
            finished_after=finished_after,
            on_error=self._on_record_error,
            on_success=self._on_record_success,
            skip=self._dropped,
        )
        # Advance the watermark to the newest completion seen this pass so the
        # next scan reads only targets that finish after it. reconcile_once
        # returns the max finished_at it considered (or finished_after unchanged
        # when nothing newer was found), never moving the watermark backwards.
        if max_finished_at is not None and (
            self._last_seen is None or max_finished_at > self._last_seen
        ):
            self._last_seen = max_finished_at

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
        # Reset the watermark so the next start() runs a full catch-up over the
        # whole admin DB, re-driving anything that finished while stopped.
        self._last_seen = None
