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
The abstract base class for all types of monitoring.
"""

import abc
import asyncio
from typing import Any, Optional, Self

from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventType,
    EntityRunMetadata,
    EventPayload,
)
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class MonitorBase(abc.ABC):
    """
    The abstract base class for all types of monitoring.
    """

    def __init__(
        self: Self,
        launch_id: str,
        entityrun_metadata: Optional[EntityRunMetadata] = None,
        event_queue: Optional[asyncio.Queue] = None,
        stop_event: Optional[asyncio.Event] = None,
    ) -> None:
        self.launch_id = launch_id
        if entityrun_metadata is None:
            logger.warning("entityrun_metadata is None, using default")
            entityrun_metadata = EntityRunMetadata()
        self.entityrun_metadata = entityrun_metadata
        self.event_queue = event_queue
        if stop_event is None:
            stop_event = asyncio.Event()
        self.stop_event = stop_event

    def stop(self: Self) -> None:
        """Stop the monitoring."""
        logger.info(
            "Stopping launch_id %s; stop_event = %s", self.launch_id, self.stop_event
        )
        if self.stop_event is not None:
            self.stop_event.set()

    async def create_event(
        self: Self, build_event_type: BuildEventType, payload_data: Any
    ) -> BuildEvent:
        """Create and add an event to the event queue."""
        event = BuildEvent(
            run_metadata=self.entityrun_metadata,
            type=build_event_type,
            payload=EventPayload.payload_parser(
                event_type=build_event_type,
                data=payload_data,
            ),
        )
        logger.info("[MonitorBase] Built event %s", event)
        if self.event_queue is not None:
            await self.event_queue.put(event)
        return event

    @abc.abstractmethod
    async def monitor(self: Self) -> None:
        """The actual monitoring function."""
