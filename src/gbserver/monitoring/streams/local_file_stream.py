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

"""
Stream from a local file (pure Python tail).
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Optional, Self

from gbserver.monitoring.streams.log_stream_base import LogStreamSource
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class LocalFileStream(LogStreamSource):
    """Stream from a local file (pure Python tail)."""

    def __init__(
        self: Self,
        path: str | Path,
        targetsteprun_id: str = "",
        launch_id: str = "",
        wait_for_file_to_exist: bool = True,
        timeout: int = 600,
    ) -> None:
        self.path = Path(path)
        self.wait_for_file_to_exist = wait_for_file_to_exist
        self.timeout = timeout
        self.targetsteprun_id = targetsteprun_id
        self.launch_id = launch_id
        assert timeout >= 0, "Negative timeout"

    async def stream_lines(  # type: ignore[override]
        self: Self,
        stop_event: Optional[asyncio.Event] = None,
        abort_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[str]:
        # Read existing content first
        if stop_event is None:
            stop_event = asyncio.Event()
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    yield line.rstrip()

        start_time = datetime.now()
        current_time = datetime.now()
        while not self.path.exists():
            logger.warning(
                "[Sidecar/LocalFileStream %s] Log file %s not found. Sleeping for 5 seconds",
                self.targetsteprun_id,
                self.path,
            )
            await asyncio.sleep(5)
            current_time = datetime.now()
            if (current_time - start_time).total_seconds() > self.timeout:
                raise RuntimeError(
                    f"[Sidecar/LocalFileStream {self.targetsteprun_id}.{self.launch_id}] Timeout occurred waiting for the {self.path} file to exist"
                )

        # Tail new lines (Phase 1: real-time following while job is running)
        with self.path.open("r", encoding="utf-8") as f:
            f.seek(0, os.SEEK_END)
            while not stop_event.is_set():
                line = f.readline()
                if not line:
                    # sleep or break early if stop_event set
                    if stop_event:
                        try:
                            await asyncio.wait_for(stop_event.wait(), timeout=0.5)
                            break  # stop_event was set
                        except asyncio.TimeoutError:
                            continue
                    else:
                        await asyncio.sleep(0.5)
                        continue
                yield line.rstrip()
            if stop_event.is_set():
                logger.warning(
                    "[LocalFileStream %s] stop event has been set, stopping local file stream phase 1",
                    self.launch_id,
                )
            # Phase 2: Drain any remaining content after stop_event is set
            # The file is complete (job finished), but there may be buffered content
            # that wasn't read yet. Read until true EOF.
            if abort_event is not None and abort_event.is_set():
                logger.info(
                    "[LocalFileStream] Phase 2 drain aborted for %s (abort_event set)",
                    self.path,
                )
                return
            logger.info(
                "[LocalFileStream] Stop event set, draining remaining content from %s",
                self.path,
            )
            remaining_lines = 0
            while True:
                line = f.readline()
                if not line:
                    # True EOF - no more content
                    break
                yield line.rstrip()
                remaining_lines += 1

            if remaining_lines > 0:
                logger.info(
                    "[LocalFileStream] Drained %d remaining lines from %s",
                    remaining_lines,
                    self.path,
                )
            else:
                logger.info(
                    "[LocalFileStream] No remaining lines to drain from %s",
                    self.path,
                )

    def __str__(self: Self) -> str:
        return f"LocalFileStream(path = '{self.path}')"
