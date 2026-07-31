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

"""Async in-process lineage recording agent."""

import threading
import time
from typing import Any, Optional

from gbserver.lineage.jobstats import ILineageStore, get_lineage_store
from gbserver.storage.singleton_storage import get_admin_storage
from gbserver.types.buildevent import BuildEventStatusPayload, BuildEventType
from gbserver.types.status import Status
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class LineageWatcher:
    """Async background thread that records lineage from gb_events.

    Mirrors the BuildWatcher daemon-thread pattern. Runs as a single background
    thread in the build-watch process (single-replica deployment), picking up
    target-SUCCESS events from gb_events and persisting lineage asynchronously.

    The watermark is in-memory (seeded to max gb_events.index at start), so
    restart blind spot is accepted: events while watcher was down are skipped.
    Replay is safe by construction (deterministic runIds + resume="allow" +
    content-dedupe).

    A target whose recording fails is not lost behind the watermark: it is
    queued and retried on later polls up to ``_MAX_RECORD_ATTEMPTS`` before
    being dropped, so a transient backend failure is recovered while a
    persistent one cannot wedge the watcher.
    """

    # A target whose lineage recording keeps failing is retried this many times
    # on subsequent polls before being dropped, so a transient failure (e.g. a
    # network blip) is recovered without a persistent failure wedging the queue.
    _MAX_RECORD_ATTEMPTS = 3

    def __init__(self, monitoring_interval: float = 2.0) -> None:
        """Initialize the LineageWatcher.

        Args:
            monitoring_interval: Sleep duration between iterations (seconds).
        """
        self.monitoring_interval = monitoring_interval
        self.stop_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None
        self._watermark: int = 0
        self._store: Optional[ILineageStore] = None
        # (build_id, target_id) -> attempts so far, for targets whose recording
        # failed and should be retried on a subsequent poll.
        self._pending_retries: dict[tuple[str, str], int] = {}

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
        """Main monitoring loop (runs in daemon thread).

        Seed watermark once, then poll for new events and record lineage.
        """
        self._watermark = self._get_max_event_index()
        logger.debug("LineageWatcher seeded watermark to %d", self._watermark)

        while not self.stop_event.is_set():
            try:
                self._process_new_events()
            except Exception:
                logger.exception("LineageWatcher iteration failed")

            time.sleep(self.monitoring_interval)

    def _get_max_event_index(self) -> int:
        """Get the current maximum gb_events.index."""
        try:
            storage = get_admin_storage()
            return storage.event_storage.get_max_index()
        except Exception as e:
            logger.warning("Failed to get max event index: %s", e)
            return 0

    @staticmethod
    def _is_target_success(build_event: Any) -> bool:
        """Return True if the event marks a target that completed successfully."""
        if build_event.type != BuildEventType.STATUS_EVENT:
            return False
        if build_event.run_metadata.type != "Target":
            return False
        payload = build_event.payload
        return (
            isinstance(payload, BuildEventStatusPayload)
            and payload.status == Status.SUCCESS
        )

    def _process_new_events(self) -> None:
        """Process successful target-completion events since the watermark and
        record lineage.

        A target that completed successfully is a STATUS_EVENT whose run
        metadata type is "Target" and whose payload status is SUCCESS. This
        mirrors how BuildRunner detected target completion before lineage
        recording moved to this watcher (see buildrunner.py __process_build_
        target_info_type_event).
        """
        if self._store is None:
            logger.error("lineage store not initialized; start() must run first")
            return
        storage = get_admin_storage()

        # Retry previously-failed targets first so a transient failure recovers
        # promptly once the backend is healthy again.
        self._retry_pending(storage)

        events = storage.event_storage.get_events_after_index(self._watermark)
        if not events:
            return

        for index, event in events:
            # Advance the watermark before attempting to record, so a single
            # event that fails to record does not stall the whole batch behind
            # it. A failed target is not left behind the watermark but queued in
            # _pending_retries and retried on a later poll (up to
            # _MAX_RECORD_ATTEMPTS); after that it is dropped rather than
            # wedging the watcher. Recording is idempotent/replay-safe
            # (deterministic runIds + resume="allow" + content-dedupe), so a
            # retry of an already-recorded target is harmless.
            self._watermark = index
            if not self._is_target_success(event.build_event):
                continue

            build_id = event.build_event.run_metadata.build_id
            target_id = event.build_event.run_metadata.targetrun_id
            self._record_lineage(storage, build_id, target_id, attempts=0)

    def _record_lineage(
        self, storage: Any, build_id: str, target_id: str, attempts: int
    ) -> None:
        """Record lineage for one target; queue for retry on failure.

        Args:
            storage: Admin storage passed through to the lineage store.
            build_id: Build the target belongs to.
            target_id: Target run to record lineage for.
            attempts: How many times recording has already been attempted for
                this target (0 on the first, watermark-driven attempt).
        """
        assert self._store is not None  # guaranteed by _process_new_events
        key = (build_id, target_id)
        try:
            logger.debug(
                "Recording lineage for target %s in build %s", target_id, build_id
            )
            self._store.add_jobstats_for_build_target(
                storage, build_id=build_id, target_id=target_id
            )
            self._pending_retries.pop(key, None)
        except Exception:
            attempts += 1
            if attempts >= self._MAX_RECORD_ATTEMPTS:
                self._pending_retries.pop(key, None)
                logger.exception(
                    "Dropping lineage for target %s in build %s after %d attempts",
                    target_id,
                    build_id,
                    attempts,
                )
            else:
                self._pending_retries[key] = attempts
                logger.warning(
                    "Failed to record lineage for target %s in build %s "
                    "(attempt %d/%d); will retry",
                    target_id,
                    build_id,
                    attempts,
                    self._MAX_RECORD_ATTEMPTS,
                )

    def _retry_pending(self, storage: Any) -> None:
        """Re-attempt recording for targets that failed on a previous poll."""
        for (build_id, target_id), attempts in list(self._pending_retries.items()):
            self._record_lineage(storage, build_id, target_id, attempts=attempts)

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the watcher thread to stop and wait for it to exit.

        Joins the worker thread (bounded by ``timeout``) so shutdown does not
        race an in-flight iteration, and resets state so the watcher can be
        started again.

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
