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
BuildEventPublisher — publishes BuildEvents to a messaging backend.

Supports RabbitMQ (topic exchange) and NATS (standalone mode).
Routing key format: build.<build_id>.<event_type>
Exchange name: build-events (RabbitMQ only)
"""

from __future__ import annotations

import asyncio
import os
import socket
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from gbserver.messaging.messaging_base import Address, MessagingBase
from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventStatusPayload,
)
from gbserver.types.constants import GBSERVER_BUILD_EVENTS_EXCHANGE
from gbserver.utils.logger import get_logger
from gbserver.utils.optional_imports import HAS_NATS

logger = get_logger(__name__)


from gbcommon.types.gbenvconfig import is_standalone as _is_standalone


def _is_nats_reachable(nats_url: str, timeout: float = 0.5) -> bool:
    """Quick socket probe to check if a NATS server is accepting connections.

    This probe runs once per build phase (not per event) due to lru_cache in
    get_message_logger. A local NATS server responds in <1ms; 0.5s timeout is
    generous while avoiding long hangs when the server is down.
    """
    parsed = urlparse(nats_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 4222
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class BuildEventPublisher:
    """
    Publishes BuildEvents to a messaging backend (RabbitMQ or NATS).

    Routing key format: build.<build_id>.<event_type>
    Internal events (TERMINATE, NEWARTIFACT_IN_ENVIRONMENT, NEW_MULTIARTIFACT_IN_ENVIRONMENT)
    are silently skipped.
    """

    def __init__(self, backend: MessagingBase) -> None:
        self._backend = backend
        self._publish_lock = asyncio.Lock()

    @classmethod
    def from_env(
        cls,
        messaging_secret: Optional[Any] = None,
    ) -> "Optional[BuildEventPublisher]":
        """
        Factory that creates the publisher from environment variables.

        Backend selection:
        - If RABBITMQ_HOST is set: use RabbitMQ
        - Elif standalone mode, nats-py available, and server reachable: use NATS
        - Else: return None (no backend available)
        """
        if os.getenv("RABBITMQ_HOST"):
            from gbserver.messaging.rabbitmq_base import RabbitMQBase

            backend = RabbitMQBase.from_env_and_args(
                exchange_name=GBSERVER_BUILD_EVENTS_EXCHANGE,
                queue_name="build",
                routing_key=None,
                messaging_secret=messaging_secret,
            )
            return cls(backend=backend)

        if _is_standalone() and HAS_NATS:
            from gbserver.types.constants import GBSERVER_NATS_URL

            if _is_nats_reachable(GBSERVER_NATS_URL):
                from gbserver.messaging.nats_messaging import NATSMessaging

                logger.info(
                    "NATS server reachable at %s — event publishing enabled",
                    GBSERVER_NATS_URL,
                )
                addr = Address(exchange=None, queue="build", routing_key=None)
                backend = NATSMessaging(addr=addr, nats_url=GBSERVER_NATS_URL)
                return cls(backend=backend)

            logger.warning(
                "NATS server not reachable at %s — event publishing disabled",
                GBSERVER_NATS_URL,
            )

        return None

    @classmethod
    def is_configured(cls) -> bool:
        """Return True if a messaging backend is available."""
        if os.getenv("RABBITMQ_HOST"):
            return True
        if _is_standalone() and HAS_NATS:
            from gbserver.types.constants import GBSERVER_NATS_URL

            return _is_nats_reachable(GBSERVER_NATS_URL)
        return False

    async def setup(self) -> None:
        """Initialize the messaging backend connection."""
        await self._backend.setup()

    async def close(self) -> None:
        """Close the messaging backend connection."""
        await self._backend.close()

    async def publish_event(self, event: BuildEvent) -> None:
        """
        Publish a BuildEvent to the messaging backend.

        Internal events are silently skipped.
        If the backend is unavailable, a warning is logged and the error is swallowed
        so that build processing is not interrupted.
        """
        # Skip internal events
        if event.type.is_internal_event():
            logger.debug(
                "Skipping internal event type=%s build_id=%s",
                event.type.value,
                event.run_metadata.build_id,
            )
            return

        build_id = event.run_metadata.build_id or "unknown"
        event_type = event.type.value  # e.g. "status_event"

        payload = self._serialize_event(event)

        # The Address for this particular publish uses queue="build.<build_id>"
        # so that Address.rk(suffix=event_type) produces "build.<build_id>.<event_type>"
        # We temporarily override the address on the backend instance for this publish.
        # A lock is required because the address swap is not coroutine-safe.
        async with self._publish_lock:
            original_addr = self._backend.addr
            publish_addr = Address(
                exchange=(
                    GBSERVER_BUILD_EVENTS_EXCHANGE if original_addr.exchange else None
                ),
                queue=f"build.{build_id}",
                routing_key=None,
            )
            try:
                self._backend.addr = publish_addr  # type: ignore[misc]
                await self._backend.publish(payload=payload, suffix=event_type)
                logger.info(
                    "Published event type=%s build_id=%s subject=%s",
                    event_type,
                    build_id,
                    publish_addr.rk(event_type),
                )
            except Exception as exc:
                logger.warning(
                    "Failed to publish event type=%s build_id=%s: %s",
                    event_type,
                    build_id,
                    exc,
                )
            finally:
                self._backend.addr = original_addr  # type: ignore[misc]

    @staticmethod
    def _serialize_event(event: BuildEvent) -> Dict[str, Any]:
        """
        Serialize a BuildEvent to a JSON-compatible dict suitable for publishing.
        """
        payload: Dict[str, Any] = {
            "build_id": event.run_metadata.build_id or "unknown",
            "event_type": event.type.value,
            "timestamp": int(event.timestamp.timestamp()),
            "target_name": event.run_metadata.target_name or "",
            "step_name": event.run_metadata.targetstep_uri or "",
            "source": event.source,
        }

        # Add status-specific fields
        if isinstance(event.payload, BuildEventStatusPayload):
            payload["status"] = event.payload.status.value
            payload["message"] = event.payload.msg

        return payload
