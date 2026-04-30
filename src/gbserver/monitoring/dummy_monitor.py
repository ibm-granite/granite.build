# process_monitors.py
"""Dummy monitor module."""

import asyncio
from typing import Optional, Self

from gbserver.monitoring.monitor_base import MonitorBase
from gbserver.types.buildevent import EntityRunMetadata
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


# ------------------- Dummy Monitor -------------------------
class DummyMonitor(MonitorBase):
    """
    Sets `stop_event` after `delay_sec`, emulating a finished workload.
    Useful in unit-tests or local runs where no real process/k8s object exists.
    """

    def __init__(
        self: Self,
        delay_sec: float,
        launch_id: str = "",
        entityrun_metadata: Optional[EntityRunMetadata] = None,
        event_queue: Optional[asyncio.Queue] = None,
        stop_event: Optional[asyncio.Event] = None,
    ) -> None:
        super().__init__(
            launch_id=launch_id,
            entityrun_metadata=entityrun_metadata,
            event_queue=event_queue,
            stop_event=stop_event,
        )
        self.delay = delay_sec

    async def monitor(self: Self) -> None:
        logger.info(f"[DummyMon] Will stop after {self.delay}s ...")
        await asyncio.sleep(self.delay)
        self.stop()
        logger.info("[DummyMon] stop_event set; exiting")
