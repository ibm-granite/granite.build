#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""BuildEventPublishLogger — publishes status events to RabbitMQ via the logger framework."""

from __future__ import annotations

import asyncio
from typing import Optional, Self

from gbserver.buildrunner.buildlogger import AbstractBuildLogger
from gbserver.messaging.build_event_publisher import BuildEventPublisher
from gbserver.storage.stored_build import StoredBuild
from gbserver.types.buildevent import BuildEvent, BuildEventType, BuildLogLevel
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class BuildEventPublishLogger(AbstractBuildLogger):
    """Publishes status events to RabbitMQ via BuildEventPublisher.

    Only STATUS_EVENT types are forwarded; all other event types are ignored.
    Publishing is fire-and-forget — failures are logged but never raised.
    """

    def __init__(
        self: Self, stored_build: StoredBuild, publisher: BuildEventPublisher
    ) -> None:
        super().__init__(stored_build=stored_build)
        self._publisher = publisher
        self._setup_done = False
        self._setup_lock = asyncio.Lock()

    def _log(
        self: Self,
        level: BuildLogLevel,
        markdown: str,
        triggering_event: Optional[BuildEvent] = None,
    ) -> None:
        if triggering_event is None:
            return
        if triggering_event.type != BuildEventType.STATUS_EVENT:
            return

        asyncio.ensure_future(self._publish(triggering_event))

    async def _publish(self: Self, event: BuildEvent) -> None:
        try:
            async with self._setup_lock:
                if not self._setup_done:
                    await self._publisher.setup()
                    self._setup_done = True
            await self._publisher.publish_event(event)
        except Exception as exc:
            logger.warning(
                "BuildEventPublishLogger failed to publish event build_id=%s: %s",
                event.run_metadata.build_id,
                exc,
            )
