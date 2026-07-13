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
from typing import Optional

from gbserver.lineage.jobstats import ILineageStore, get_lineage_store
from gbserver.storage.singleton_storage import get_admin_storage
from gbserver.types.buildevent import BuildEventType
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
    """

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

    def start(self) -> None:
        """Start the watcher thread (daemon=True, does not keep process alive)."""
        if self.worker_thread is not None:
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
            except Exception as e:
                logger.exception("LineageWatcher iteration failed: %s", e)

            time.sleep(self.monitoring_interval)

    def _get_max_event_index(self) -> int:
        """Get the current maximum gb_events.index."""
        try:
            storage = get_admin_storage()
            return storage.event_storage.get_max_index()
        except Exception as e:
            logger.warning("Failed to get max event index: %s", e)
            return 0

    def _process_new_events(self) -> None:
        """Process target-SUCCESS events since the watermark and record lineage."""
        storage = get_admin_storage()

        events = storage.event_storage.get_events_after_index(self._watermark)
        if not events:
            return

        for event in events:
            self._watermark = event.index
            if event.build_event.type != BuildEventType.TARGET_SUCCESS:
                continue

            try:
                build_id = event.build_event.run_metadata.build_id
                target_id = event.build_event.run_metadata.targetrun_id

                logger.debug(
                    "Recording lineage for target %s in build %s", target_id, build_id
                )
                self._store.add_jobstats_for_build_target(
                    storage, build_id=build_id, target_id=target_id
                )
            except Exception as e:
                logger.exception("Failed to record lineage for event: %s", e)

    def stop(self) -> None:
        """Signal the watcher thread to stop."""
        logger.info("Stopping LineageWatcher")
        self.stop_event.set()
