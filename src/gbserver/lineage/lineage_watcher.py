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

    ``already_recorded`` tracks target uuids recorded this process lifetime to
    skip re-recording in steady state (an optimization; correctness does not
    depend on it). It is seeded empty, so the first scan is a full rescan that
    re-drives anything missed while the process was down.

    A target whose recording raises is not marked recorded and is retried on the
    next scan; a target that keeps failing is dropped after
    ``_MAX_RECORD_ATTEMPTS`` so a persistent failure cannot wedge later scans
    (it is then re-attempted only after a full-rescan reset).
    """

    # A target whose lineage recording keeps failing is retried this many times
    # on subsequent scans before being dropped, so a transient failure (e.g. a
    # network blip) is recovered without a persistent failure wedging the scan.
    _MAX_RECORD_ATTEMPTS = 3

    def __init__(self, monitoring_interval: float = 2.0) -> None:
        """Initialize the LineageWatcher.

        Args:
            monitoring_interval: Sleep duration between reconciliation scans
                (seconds).
        """
        self.monitoring_interval = monitoring_interval
        self.stop_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None
        self._store: Optional[ILineageStore] = None
        # Target uuids already recorded this process lifetime; skipped on later
        # scans in steady state (idempotency makes this an optimization only).
        self._recorded: set[str] = set()
        # Target uuids dropped after exhausting retries; skipped on later scans
        # so a persistently failing target cannot wedge every scan.
        self._dropped: set[str] = set()
        # target_uuid -> attempts so far, for targets whose recording failed and
        # should be retried on a subsequent scan.
        self._failed_attempts: dict[str, int] = {}

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
        central mechanism), threading ``_recorded`` through so steady-state scans
        skip already-recorded targets. Recording failures are routed to
        ``_on_record_error`` to drive the bounded per-target retry.
        """
        if self._store is None:
            logger.error("lineage store not initialized; start() must run first")
            return
        storage = get_admin_storage()
        recorded = reconcile_once(
            self._store,
            storage,
            already_recorded=self._recorded,
            on_error=self._on_record_error,
        )
        # reconcile_once returns targets recorded successfully; union in any
        # targets dropped after exhausting retries (tracked by _on_record_error)
        # so a persistent failure stays skipped and cannot wedge every scan.
        self._recorded = recorded | self._dropped
        # A target that recorded successfully is now in _recorded; clear any
        # retry bookkeeping so its attempt count does not linger.
        for uuid in list(self._failed_attempts):
            if uuid in self._recorded:
                self._failed_attempts.pop(uuid, None)

    def _on_record_error(self, build_id: str, target_id: str, exc: Exception) -> None:
        """Handle a recording failure for one target: retry or drop.

        Keeps a per-target attempt count so a transient failure is retried on the
        next scan, while a persistently failing target is dropped after
        ``_MAX_RECORD_ATTEMPTS`` (added to ``_dropped``, which ``_reconcile``
        unions into ``_recorded``) so it stops being re-scanned — it is only
        re-attempted after a full-rescan reset of ``_recorded``.
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
