# process_monitors.py
import asyncio
import os
import time
from typing import Optional, Self

import psutil

from gbserver.monitoring.monitor_base import MonitorBase
from gbserver.types.buildevent import EntityRunMetadata
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


# -------------------- psutil-based (local PIDs) -------------------------
class CmdlineMonitor(MonitorBase):
    """
    Repeatedly scans `psutil.process_iter()` for a command-line substring.
    When *no* matching process is found, sets stop_event.
    """

    def __init__(
        self: Self,
        cmd_substring: str,
        poll: float = 10,  # 10 seconds
        log_path: str = "/logs/output.log",
        launch_id: str = "",
        entityrun_metadata: Optional[EntityRunMetadata] = None,
        event_queue: Optional[asyncio.Queue] = None,
        stop_event: Optional[asyncio.Event] = None,
        fast_poll: float = 0.1,  # 100 ms
        startup_window: int = 30,  # seconds
        max_wait: int = 3600,  # seconds (1 hour)
        **kwargs,
    ) -> None:
        super().__init__(
            launch_id=launch_id,
            entityrun_metadata=entityrun_metadata,
            event_queue=event_queue,
            stop_event=stop_event,
        )
        self.cmd_sub = cmd_substring
        self.polling_interval = poll
        self.fast_polling_interval = fast_poll
        self.startup_window = startup_window
        self.max_wait_for_active_processes = max_wait
        self.log_path = log_path
        self.targetsteprun_id = kwargs.get("targetsteprun_id", "n/a")

    def _find_targets(self) -> list[int]:
        """Return PIDs of processes whose cmdline contains cmd_sub, excluding self."""
        pids = []
        for p in psutil.process_iter(["cmdline", "pid"]):
            try:
                cmdline = " ".join(p.info["cmdline"] or [])
                if self.cmd_sub in cmdline and p.info["pid"] != os.getpid():
                    pids.append(p.info["pid"])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return pids

    def _log_has_content(self) -> bool:
        """This function checks the log and determines whether it has output or not"""
        try:
            return os.path.getsize(self.log_path) > 0
        except OSError:
            return False  # file not found or inaccessible

    async def monitor(self):
        logger.info(
            "[CmdlineMon %s] Waiting for process '%s' to exit ...",
            self.targetsteprun_id,
            self.cmd_sub,
        )
        start_time = time.monotonic()
        detected_running_processes = False

        # --- Startup check: log file ---
        if self._log_has_content():
            logger.info(
                "[CmdlineMon %s] startup: log file %s already has content",
                self.targetsteprun_id,
                self.log_path,
            )
            detected_running_processes = True

        # Track running PIDs and previous snapshot (for change detection)
        running: set[int] = set()
        previous_running: set[int] = set()

        # --- Startup phase (fast polling) ---
        while time.monotonic() - start_time < self.startup_window:
            running = set(self._find_targets())
            # Print on change during startup
            if running != previous_running:
                logger.info(
                    "[CmdlineMon %s] running changed during startup: %s",
                    self.targetsteprun_id,
                    sorted(running),
                )
                previous_running = set(running)
            if running:
                detected_running_processes = True
                break
            await asyncio.sleep(self.fast_polling_interval)

        # --- Normal phase (slow polling) ---
        num_polls = 0
        while not self.stop_event.is_set():
            running = set(self._find_targets())
            # Print on change during normal phase
            if running != previous_running:
                logger.info(
                    "[CmdlineMon %s] running changed during normal phase: %s",
                    self.targetsteprun_id,
                    sorted(running),
                )
                previous_running = set(running)
            num_polls += 1
            # Also print every 12 polling intervals regardless of change
            if num_polls % 12 == 0:
                logger.info(
                    "[CmdlineMon %s] (periodic) normal phase running: %s",
                    self.targetsteprun_id,
                    sorted(running),
                )

            if running:
                detected_running_processes = True
            else:
                # Check if log file has content - if so, a process must have run
                if not detected_running_processes and self._log_has_content():
                    logger.info(
                        "[CmdlineMon %s] log file has content, process must have run",
                        self.targetsteprun_id,
                    )
                    detected_running_processes = True

                if detected_running_processes:
                    logger.info(
                        "[CmdlineMon %s] monitored processes terminated, stopping.",
                        self.targetsteprun_id,
                    )
                    await asyncio.sleep(
                        20
                    )  # give time to the tailer and publisher to finish processing the log before exiting
                    self.stop_event.set()
                    break
                elif (
                    time.monotonic() - start_time >= self.max_wait_for_active_processes
                ):
                    logger.info(
                        "[CmdlineMon %s] no process detected within max wait, stopping.",
                        self.targetsteprun_id,
                    )
                    self.stop_event.set()
                    break

            await asyncio.sleep(self.polling_interval)
        if self.stop_event.is_set():
            logger.warning(
                "[CmdlineMon %s] stop event has been set, stopping cmd line monitoring...",
                self.targetsteprun_id,
            )
