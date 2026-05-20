"""Webhook batch dispatcher. Orchestrates batching, log scanning, and delivery.

Provides the WebhookDispatcher class which manages a single build's webhook
lifecycle: accepting events, buffering them per subscription, scanning logs
for pattern matches, and flushing batches to subscriber endpoints.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventStatusPayload,
    BuildEventType,
)
from gbserver.utils.logger import get_logger
from gbserver.webhooks.batch_buffer import WebhookBatchBuffer
from gbserver.webhooks.delivery import WebhookDelivery
from gbserver.webhooks.log_scanner import scan_log_lines
from gbserver.webhooks.models import StoredWebhookSubscription
from gbserver.webhooks.storage import IWebhookStorage

logger = get_logger(__name__)


class WebhookDispatcher:
    """Orchestrates batched webhook delivery for a single build.

    One instance per active build. Holds the batch buffer, accepts events,
    and flushes batches to subscriber endpoints on a per-subscription basis.

    Args:
        webhook_storage: Storage backend for subscriptions.
        build_id: UUID of the build being monitored.
        space_name: Space the build belongs to.
        build_name: Name of the build.
        username: User who created the build.
        build_start_time: ISO timestamp of build start.
    """

    def __init__(
        self,
        webhook_storage: IWebhookStorage,
        build_id: str,
        space_name: str,
        build_name: str,
        username: str,
        build_start_time: str,
    ) -> None:
        self.webhook_storage = webhook_storage
        self.build_id = build_id
        self.space_name = space_name
        self.build_name = build_name
        self.username = username
        self.build_start_time = build_start_time
        self.buffer = WebhookBatchBuffer()
        self._subscriptions: Dict[str, StoredWebhookSubscription] = {}

    def start(self, subscriptions: List[StoredWebhookSubscription]) -> None:
        """Register subscriptions for buffering.

        Args:
            subscriptions: List of active subscriptions to register.
        """
        for sub in subscriptions:
            self._subscriptions[sub.uuid] = sub
            self.buffer.register_subscription(sub)

    def accept_event(self, event: BuildEvent) -> None:
        """Accept a build event and buffer it for matching subscriptions.

        Internal events (TERMINATE, NEWARTIFACT_IN_ENVIRONMENT, etc.) are
        skipped. For each registered subscription, the event is checked
        against the subscription's include/exclude filters before buffering.

        Args:
            event: The build event to process.
        """
        if event.type.is_internal_event():
            return

        event_type_name = event.type.value
        event_data = self._serialize_event(event)

        for sub_id, sub in self._subscriptions.items():
            if sub.should_include_event(event_type_name):
                self.buffer.add_event(sub_id, event_data)

    def _serialize_event(self, event: BuildEvent) -> Dict[str, Any]:
        """Serialize a BuildEvent to a dict for the batch payload.

        Args:
            event: The build event to serialize.

        Returns:
            A dictionary representation suitable for JSON delivery.
        """
        meta = event.run_metadata
        data: Dict[str, Any] = {
            "event_id": str(uuid.uuid4()),
            "event_type": event.type.value,
            "timestamp": event.timestamp.isoformat()
            if hasattr(event, "timestamp")
            else datetime.now(timezone.utc).isoformat(),
            "target_name": meta.target_name,
            "step_name": meta.targetstep_uri,
        }
        if isinstance(event.payload, BuildEventStatusPayload):
            data["status"] = event.payload.status.value
            data["message"] = {"text": event.payload.msg}
        else:
            payload_data = getattr(event.payload, "data", None)
            data["message"] = payload_data if payload_data else {}
        return data

    async def flush_subscription(
        self,
        subscription: StoredWebhookSubscription,
        log_lines: Optional[List[str]] = None,
    ) -> None:
        """Flush buffer for a subscription and deliver the batch.

        Collects all buffered events, appends any log pattern matches,
        constructs the delivery payload, and sends it via WebhookDelivery.
        Does nothing if there are no events to deliver.

        Args:
            subscription: The subscription to flush.
            log_lines: Optional log lines to scan for pattern matches.
        """
        events = self.buffer.flush(subscription.uuid)

        # Scan logs for pattern matches
        if subscription.log_pattern and log_lines:
            matches = scan_log_lines(log_lines, subscription.log_pattern)
            for match in matches:
                events.append(
                    {
                        "event_id": str(uuid.uuid4()),
                        "event_type": "LOG_EVENT",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "target_name": None,
                        "step_name": None,
                        "message": match,
                    }
                )

        if not events:
            return

        now = datetime.now(timezone.utc).isoformat()
        payload = {
            "delivery_id": str(uuid.uuid4()),
            "build_id": self.build_id,
            "build_name": self.build_name,
            "space_name": self.space_name,
            "user": self.username,
            "build_start_time": self.build_start_time,
            "batch_start": now,
            "batch_end": now,
            "events": events,
        }

        delivery = WebhookDelivery(
            webhook_url=subscription.webhook_url,
            secret=subscription.secret,
        )
        await delivery.deliver(payload)

    async def flush_all_ready(
        self, log_lines: Optional[List[str]] = None
    ) -> None:
        """Flush all subscriptions whose batch interval has elapsed.

        Errors during individual subscription flushes are logged but do
        not prevent other subscriptions from being flushed.

        Args:
            log_lines: Optional log lines to pass to flush_subscription.
        """
        ready_ids = self.buffer.get_ready_subscriptions()
        for sub_id in ready_ids:
            sub = self._subscriptions.get(sub_id)
            if sub:
                try:
                    await self.flush_subscription(sub, log_lines)
                except Exception as e:
                    logger.error(
                        "[WebhookDispatcher] Error flushing %s: %s", sub_id, e
                    )

    async def flush_final(
        self, log_lines: Optional[List[str]] = None
    ) -> None:
        """Force-flush all subscriptions (on build completion).

        Unlike flush_all_ready, this ignores batch timers and flushes
        every subscription unconditionally. Errors are logged but do
        not propagate.

        Args:
            log_lines: Optional log lines to pass to flush_subscription.
        """
        for sub_id, sub in self._subscriptions.items():
            try:
                await self.flush_subscription(sub, log_lines)
            except Exception as e:
                logger.error(
                    "[WebhookDispatcher] Final flush error for %s: %s",
                    sub_id,
                    e,
                )

    def stop(self) -> None:
        """Stop dispatcher, clean up buffers and subscription state."""
        for sub_id in list(self._subscriptions.keys()):
            self.buffer.unregister_subscription(sub_id)
        self._subscriptions.clear()
