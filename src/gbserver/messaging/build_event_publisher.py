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
BuildEventPublisher — publishes BuildEvents to a RabbitMQ topic exchange.

Routing key format: build.<build_id>.<event_type>
Exchange name: build-events
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from gbserver.messaging.messaging_base import Address
from gbserver.messaging.rabbitmq_base import RabbitMQBase
from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventStatusPayload,
)
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

EXCHANGE_NAME = "build-events"


class BuildEventPublisher:
    """
    Publishes BuildEvents to RabbitMQ using a topic exchange named 'build-events'.

    Routing key format: build.<build_id>.<event_type>
    Internal events (TERMINATE, NEWARTIFACT_IN_ENVIRONMENT, NEW_MULTIARTIFACT_IN_ENVIRONMENT)
    are silently skipped.
    """

    def __init__(self, rabbitmq: RabbitMQBase) -> None:
        self._rabbitmq = rabbitmq

    @classmethod
    def from_env(
        cls,
        messaging_secret: Optional[Any] = None,
    ) -> "BuildEventPublisher":
        """
        Factory that creates the publisher from environment variables.
        The exchange is always 'build-events'; queue and routing_key are set per-publish.
        """
        rabbitmq = RabbitMQBase.from_env_and_args(
            exchange_name=EXCHANGE_NAME,
            queue_name="build",
            routing_key=None,
            messaging_secret=messaging_secret,
        )
        return cls(rabbitmq=rabbitmq)

    async def setup(self) -> None:
        """Initialize the RabbitMQ connection and channel."""
        await self._rabbitmq.setup()

    async def close(self) -> None:
        """Close the RabbitMQ connection."""
        await self._rabbitmq.close()

    async def publish_event(self, event: BuildEvent) -> None:
        """
        Publish a BuildEvent to the build-events exchange.

        Internal events are silently skipped.
        If RabbitMQ is unavailable, a warning is logged and the error is swallowed
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
        # We temporarily override the address on the rabbitmq instance for this publish.
        original_addr = self._rabbitmq.addr
        publish_addr = Address(
            exchange=EXCHANGE_NAME,
            queue=f"build.{build_id}",
            routing_key=None,
        )
        try:
            self._rabbitmq.addr = publish_addr  # type: ignore[misc]
            await self._rabbitmq.publish(payload=payload, suffix=event_type)
            logger.info(
                "Published event type=%s build_id=%s routing_key=%s",
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
            self._rabbitmq.addr = original_addr  # type: ignore[misc]

    @staticmethod
    def _serialize_event(event: BuildEvent) -> Dict[str, Any]:
        """
        Serialize a BuildEvent to a JSON-compatible dict suitable for publishing.
        """
        payload: Dict[str, Any] = {
            "build_id": event.run_metadata.build_id or "",
            "event_type": event.type.value,
            "timestamp": event.timestamp.isoformat(),
            "target_name": event.run_metadata.target_name or "",
            "step_name": event.run_metadata.targetstep_uri or "",
            "source": event.source,
        }

        # Add status-specific fields
        if isinstance(event.payload, BuildEventStatusPayload):
            payload["status"] = event.payload.status.value
            payload["message"] = event.payload.msg

        return payload
