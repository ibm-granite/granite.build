"""Webhook event writer — persists build events for later delivery.

Replaces the old WebhookDispatcher + BatchBuffer approach. Instead of
buffering in memory and delivering inline, events are written to the
database immediately. A separate delivery worker (Phase 2) handles
batching and HTTP delivery.
"""

import uuid as uuid_mod
from typing import Any, Dict, List, Optional

from gbserver.storage.sql.webhook_event_storage import create_webhook_event_storage
from gbserver.storage.sql.webhook_subscription_storage import create_webhook_storage
from gbserver.storage.webhook_event_storage import IWebhookEventStorage
from gbserver.storage.webhook_subscription_storage import IWebhookStorage
from gbserver.types.buildevent import BuildEvent, BuildEventStatusPayload
from gbserver.utils.logger import get_logger
from gbserver.webhooks.event_models import StoredWebhookEvent
from gbserver.webhooks.models import StoredWebhookSubscription

logger = get_logger(__name__)


class WebhookEventWriter:
    """Persists build events to webhook event storage for later delivery.

    One instance per active build. Queries subscriptions on start(),
    then for each event, persists a StoredWebhookEvent per matching
    subscription.

    Args:
        build_id: UUID of the build being monitored.
        space_name: Space the build belongs to.
    """

    _REFRESH_INTERVAL = 50  # Re-read subscriptions every N events

    def __init__(self, build_id: str, space_name: str) -> None:
        self.build_id = build_id
        self.space_name = space_name
        self._subscriptions: List[StoredWebhookSubscription] = []
        self._webhook_storage: Optional[IWebhookStorage] = None
        self._event_storage: Optional[IWebhookEventStorage] = None
        self._events_since_refresh: int = 0

    def start(self) -> List[StoredWebhookSubscription]:
        """Query and register active subscriptions for this build.

        Returns:
            List of active subscriptions found.
        """
        self._webhook_storage = create_webhook_storage()
        self._event_storage = create_webhook_event_storage()

        # Get per-build subscriptions (legacy build_id + new build_filter)
        subs = self._webhook_storage.get_active_for_build(self.build_id)
        filter_subs = self._webhook_storage.get_active_for_build_filter(self.build_id)
        # Get space-wide subscriptions
        space_subs = self._webhook_storage.get_active_for_space(self.space_name)

        # Deduplicate by UUID
        seen: set = set()
        all_subs: List[StoredWebhookSubscription] = []
        for s in subs + filter_subs + space_subs:
            if s.uuid not in seen and s.status == "active":
                seen.add(s.uuid)
                all_subs.append(s)
        self._subscriptions = all_subs

        logger.info(
            "[WebhookEventWriter] Found %d active subscription(s) for build %s",
            len(self._subscriptions),
            self.build_id,
        )
        return self._subscriptions

    def _refresh_subscriptions(self) -> None:
        """Re-read active subscriptions from DB to pick up late arrivals."""
        self._events_since_refresh = 0
        if self._webhook_storage is None:
            return
        subs = self._webhook_storage.get_active_for_build(self.build_id)
        filter_subs = self._webhook_storage.get_active_for_build_filter(self.build_id)
        space_subs = self._webhook_storage.get_active_for_space(self.space_name)

        seen: set = set()
        all_subs: List[StoredWebhookSubscription] = []
        for s in subs + filter_subs + space_subs:
            if s.uuid not in seen and s.status == "active":
                seen.add(s.uuid)
                all_subs.append(s)
        self._subscriptions = all_subs

    def accept_event(self, event: BuildEvent) -> None:
        """Persist event to DB for each matching subscription.

        Internal events are skipped. For each active subscription whose
        event_types filter matches, a StoredWebhookEvent is written.

        Args:
            event: The build event to persist.
        """
        if event.type.is_internal_event():
            return

        if not self._subscriptions or self._event_storage is None:
            return

        # Periodic refresh of subscription list
        self._events_since_refresh += 1
        if self._events_since_refresh >= self._REFRESH_INTERVAL:
            self._refresh_subscriptions()

        event_type_name = event.type.value
        payload = self._serialize_event(event)

        for sub in self._subscriptions:
            if sub.should_include_event(event_type_name):
                stored_event = StoredWebhookEvent(
                    subscription_id=sub.uuid,
                    build_id=self.build_id,
                    event_type=event_type_name,
                    payload=payload,
                )
                try:
                    self._event_storage.add(stored_event)
                except Exception as e:
                    logger.warning(
                        "[WebhookEventWriter] Failed to persist event for sub %s: %s",
                        sub.uuid,
                        e,
                    )

    def _serialize_event(self, event: BuildEvent) -> Dict[str, Any]:
        """Serialize a BuildEvent to a dict for storage."""
        meta = event.run_metadata
        data: Dict[str, Any] = {
            "event_id": str(uuid_mod.uuid4()),
            "event_type": event.type.value,
            "timestamp": event.timestamp.isoformat(),
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
