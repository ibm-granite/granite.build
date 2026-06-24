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
Stream log lines from a SkyPilot job via sky.tail_logs API.
"""

import asyncio
from typing import AsyncIterator, Optional, Self

from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

_SENTINEL = object()
_NEXT_TIMEOUT_SECONDS = 90


class SkyPilotLogStreamSource:
    """Stream log lines from a running SkyPilot job in real-time.

    Wraps sky.tail_logs(follow=True, preload_content=False) which returns a
    synchronous Iterator[str | None].  Each next() call blocks on an HTTP
    streaming response, so consumption is offloaded to a thread via
    asyncio.to_thread.
    """

    def __init__(
        self: Self,
        cluster_name: str,
        job_id: int,
        start_line: int = 0,
        abort_event: Optional[asyncio.Event] = None,
        log_file=None,
    ) -> None:
        self.cluster_name = cluster_name
        self.job_id = job_id
        self.start_line = start_line
        self._abort_event = abort_event
        self.lines_consumed = 0
        self._log_file = log_file

    def __repr__(self: Self) -> str:
        return (
            f"SkyPilotLogStreamSource(cluster={self.cluster_name}, "
            f"job_id={self.job_id}, start_line={self.start_line})"
        )

    async def stream_lines(
        self: Self,
        stop_event: Optional[asyncio.Event] = None,
        abort_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[str]:
        """Yield log lines from the SkyPilot job.

        Lines before self.start_line are consumed but not yielded (for resume).

        stop_event: when set, exit the stream gracefully.
        abort_event: ignored (constructor abort_event is used instead so that
            LogFileMonitor — which only passes stop_event — doesn't need changes).
        """
        effective_abort = self._abort_event

        try:
            import sky  # noqa: F401
        except ImportError:
            logger.error("[SkyPilotLogStream] sky package not available")
            return

        def _start_tail():
            return sky.tail_logs(
                cluster_name=self.cluster_name,
                job_id=self.job_id,
                follow=True,
                tail=0,
                preload_content=False,
            )

        try:
            iterator = await asyncio.to_thread(_start_tail)
        except Exception as e:
            logger.warning(
                "[SkyPilotLogStream] Failed to start tail_logs for %s job %s: %s",
                self.cluster_name,
                self.job_id,
                e,
            )
            return

        def _next_line():
            return next(iterator, _SENTINEL)

        try:
            while True:
                if stop_event and stop_event.is_set():
                    logger.info(
                        "[SkyPilotLogStream] stop_event set after %d lines, exiting",
                        self.lines_consumed,
                    )
                    return
                if effective_abort and effective_abort.is_set():
                    logger.info(
                        "[SkyPilotLogStream] abort_event set after %d lines, exiting",
                        self.lines_consumed,
                    )
                    return

                try:
                    line = await asyncio.wait_for(
                        asyncio.to_thread(_next_line),
                        timeout=_NEXT_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    logger.debug(
                        "[SkyPilotLogStream] Timeout waiting for next line "
                        "(consumed=%d), retrying",
                        self.lines_consumed,
                    )
                    continue
                except asyncio.CancelledError:
                    raise

                if line is _SENTINEL:
                    logger.info(
                        "[SkyPilotLogStream] Iterator exhausted after %d lines",
                        self.lines_consumed,
                    )
                    return

                if line is None:
                    continue

                self.lines_consumed += 1
                if self.lines_consumed <= self.start_line:
                    continue

                stripped = line.rstrip("\n").rstrip("\r")
                if self._log_file:
                    try:
                        self._log_file.write(stripped + "\n")
                        self._log_file.flush()
                    except (OSError, ValueError):
                        pass
                yield stripped

        except asyncio.CancelledError:
            logger.info(
                "[SkyPilotLogStream] Cancelled after %d lines", self.lines_consumed
            )
            raise
        except Exception as e:
            logger.warning(
                "[SkyPilotLogStream] Stream error after %d lines for %s job %s: %s",
                self.lines_consumed,
                self.cluster_name,
                self.job_id,
                e,
            )
            raise
        finally:
            if self._log_file:
                try:
                    self._log_file.close()
                except (OSError, ValueError):
                    pass
