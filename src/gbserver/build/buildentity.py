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
The entity. All entities like Build, BuildTarget etc inherity this
"""

from asyncio import Event, Queue
from pathlib import Path
from typing import Self

from gbserver.build.entity import Entity
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class BuildEntity(Entity):
    """A base class for entities related to a build."""

    # instance attributes
    build_id: str
    event_q: Queue
    build_workspace_dir: Path
    username: str = ""

    def __init__(
        self: Self,
        build_id: str,
        event_q: Queue,
        build_workspace_dir: Path,
        username: str = "",
        **kwargs,
    ) -> None:
        """Loads a entity"""

        self.build_id = build_id
        self.event_q = event_q
        self.build_workspace_dir = build_workspace_dir
        self.username = username
        super().__init__(**kwargs)
        logger.info(
            "build %s after super().__init__ self.event_q: %s %s",
            self.build_id,
            id(self.event_q),
            self.event_q,
        )

    def dispatch_event(self: Self, event: Event) -> None:
        """Dispatch an event if the event queue exists."""
        logger.debug("BuildEntity.dispatch_event %s start", self.build_id)
        if self.event_q is None:
            logger.debug("Run.dispatch_event no event_q end")
            return
        self.event_q.put_nowait(event)
        logger.debug("BuildEntity.dispatch_event %s end", self.build_id)
