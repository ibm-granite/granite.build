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
Read and monitor a log file, line by line.
"""

import asyncio
from collections import deque
from typing import Dict, List, Optional, Self

from gbserver.environment.environment import Environment, EventLogLineParserConfig
from gbserver.messaging.messaging_base import MessagingBase
from gbserver.monitoring.monitor_base import MonitorBase
from gbserver.monitoring.streams.log_stream_base import LogStreamSource
from gbserver.types.buildevent import EntityRunMetadata
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class LogFileMonitor(MonitorBase):
    """
    Read and monitor a log file, line by line.
    """

    DEFAULT_MAX_LINE_LEN = 4096

    def __init__(
        self: Self,
        step_id: str,  # a step id to tag all logs from the same step
        stream_source: LogStreamSource,
        event_configs: Optional[List[EventLogLineParserConfig]] = None,
        messenger: Optional[MessagingBase] = None,
        launch_id: str = "",
        entityrun_metadata: Optional[EntityRunMetadata] = None,
        event_queue: Optional[asyncio.Queue] = None,
        stop_event: Optional[asyncio.Event] = None,
        max_line_len: Optional[int] = None,
    ) -> None:
        super().__init__(
            launch_id=launch_id,
            entityrun_metadata=entityrun_metadata,
            event_queue=event_queue,
            stop_event=stop_event,
        )
        self.stream_source = stream_source
        self.line_num = 0
        self.last_read_line = -1
        self.messenger = messenger
        self.step_id = step_id
        if event_configs is None:
            event_configs = []
        self._event_configs = event_configs
        self.max_line_len = max_line_len or self.DEFAULT_MAX_LINE_LEN
        self.monitoring_interval = 5

    async def monitor(self: Self) -> None:
        buffer = deque(maxlen=10)  # type: ignore[var-annotated]
        logger.info("[LogFileMon %s] started using stream %s", self.step_id, self.stream_source)
        async for log_line in self.stream_source.stream_lines(stop_event=self.stop_event):  # type: ignore[attr-defined]
            buffer.append(log_line)
            # Note: We don't check stop_event here anymore!
            # The stream (ssh_file_stream.py) handles stop_event by entering Phase 2
            # draining mode, where it reads remaining lines from the complete file.
            # We must continue consuming all yielded lines, including those from Phase 2,
            # until the stream naturally stops yielding (reaches EOF).
            if len(log_line) < self.max_line_len:
                logger.debug(
                    "[LogFileMon %s] log line %d = %s",
                    self.step_id,
                    self.line_num,
                    log_line,
                )
                await self.get_events_from_log_line(log_line=log_line)
            else:
                logger.warning(
                    "[LogFileMon %s] skipping log line %d because it is too long (%d characters)",
                    self.step_id,
                    self.line_num,
                    len(log_line),
                )
            self.line_num += 1
        logger.info(
            "[LogFileMon %s] stopped monitoring stream %s",
            self.step_id,
            self.stream_source,
        )
        logger.info(
            "[LogFileMon %s] the last 10 lines in the monitored file: %s",
            self.step_id,
            "\n".join(buffer),
        )

    async def get_events_from_log_line(
        self: Self,
        log_line: str,
    ) -> List[Dict]:
        """Parse each log line; generate and publish a build event for the
        log lines that follow a pattern specified in the configuration
        """
        return await Environment.get_events_from_log_line(  # type: ignore[return-value]
            log_line=log_line,
            event_configs=self._event_configs,
            event_q=self.event_queue,
            entityrun_metadata=self.entityrun_metadata,
            messenger=self.messenger,
            line_num=self.line_num,
        )
