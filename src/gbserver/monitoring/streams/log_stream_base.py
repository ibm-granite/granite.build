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
This file contains the base LogStreamSource protocol and retry mixin
"""

import asyncio
import random
from typing import AsyncIterator, Callable, Optional, Protocol, Self, TypeVar

from gbserver.utils.logger import get_logger

logger = get_logger(__name__)
T = TypeVar(
    "T"
)  # generic type for items yielded by an iterator (e.g. str for log lines)


class LogStreamSource(Protocol):
    """Protocol for objects that asynchronously yield log lines.
    Any class that implements the stream_lines(self) -> AsyncIterator[str] function below can be
    passed as a LogStreamSource argument to a function"""

    async def stream_lines(
        self: Self,
        stop_event: Optional[asyncio.Event] = None,
        abort_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[str]:
        """Yield decoded log lines as they arrive.

        stop_event: when set, transitions the stream from phase 1 (real-time
            tail) to phase 2 (drain remaining content).
        abort_event: when set before or during phase 2, the drain is skipped
            entirely.  Use this to suppress stale artifact events from a job
            that is being retried.
        """


class RetryMixin:
    """Provides exponential backoff + retry logic for reconnectable streams."""

    def __init__(
        self: Self,
        reconnect_delay: int = 2,
        max_retries: int = 5,
        max_backoff: int = 60,
    ) -> None:
        self.reconnect_delay = reconnect_delay
        self.max_retries = max_retries
        self.max_backoff = max_backoff

    async def _retry_loop(
        self: Self,
        task_coro: Callable[[Optional[asyncio.Event]], AsyncIterator[T]],
        stop_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[T]:
        """
        DEPRECATED: This is deprecated in favor of the 2-phase log draining.
        Repeatedly runs an async generator task with exponential backoff and jitter on failure.
        Args:
            task_coro: A one-argument async callable that returns an AsyncIterator[T].
                       For example, a method like `self._stream_once`.
        Yields:
            Items produced by the async iterator, as long as retries are allowed.
        Raises:
            asyncio.CancelledError: if the outer task is cancelled.
        """
        if stop_event is None:
            stop_event = asyncio.Event()
        attempt = 0
        while not stop_event.is_set():
            try:
                async for item in task_coro(stop_event):
                    attempt = 0  # reset on any successful yield
                    yield item
                return  # normal end of stream
            except asyncio.CancelledError:
                raise
            except Exception as e:
                attempt += 1
                if attempt >= self.max_retries:
                    logger.error(
                        "[RetryMixin] max retries reached (%d attempts): %s",
                        attempt,
                        e,
                    )
                    return
                delay = min(
                    self.reconnect_delay * (2 ** (attempt - 1)), self.max_backoff
                )
                delay += random.uniform(0, delay * 0.1)
                logger.warning(
                    "[RetryMixin] reconnect attempt %d/%d after %.1fs (%s)",
                    attempt,
                    self.max_retries,
                    delay,
                    e,
                )
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=delay)
                    logger.info("[RetryMixin] stopping retry loop due to stop_event")
                    return
                except asyncio.TimeoutError:
                    pass
