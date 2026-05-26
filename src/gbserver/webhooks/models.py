"""Webhook subscription storage model.

Defines the StoredWebhookSubscription model used to persist webhook
registrations. Subscriptions specify which events to deliver, how
frequently to batch them, and the HMAC secret used for signing payloads.
"""

import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field

from gbserver.storage.storage import BaseStoredItem
from gbserver.utils.utils import get_utc_time

WEBHOOK_DEFAULT_FREQUENCY = 30
"""Default batch flush interval in seconds."""

WEBHOOK_MIN_FREQUENCY = 15
"""Minimum allowed batch flush interval in seconds."""


class StoredWebhookSubscription(BaseStoredItem):
    """Persistent webhook subscription record.

    Inherits uuid from BaseStoredItem. Each subscription ties a space
    (and optionally a specific build) to a webhook endpoint that will
    receive batched event notifications.

    Args:
        space_name: The space this subscription belongs to.
        build_id: Optional build ID to scope notifications to a single build.
            None means space-wide (all builds in the space).
        webhook_url: The URL to POST event payloads to.
        secret: HMAC signing key used to sign outgoing payloads.
        event_types: List of event type strings to include. ["*"] means all.
        excluded_types: List of event type strings to always exclude.
        frequency: Batch flush interval in seconds.
        log_pattern: Optional regex pattern for log line scanning.
        created_by: Username of the subscription creator.
        active: Whether this subscription is currently active.
        metadata: Arbitrary metadata dict (stored as JSONB).
        created_time: Timestamp when the subscription was created.
        updated_time: Timestamp of last modification.
    """

    space_name: str
    build_id: Optional[str] = None
    webhook_url: str
    secret: str
    event_types: List[str] = Field(default_factory=lambda: ["*"])
    excluded_types: List[str] = Field(default_factory=list)
    frequency: int = WEBHOOK_DEFAULT_FREQUENCY
    log_pattern: Optional[str] = None
    created_by: str
    active: bool = True
    scope: Literal["space", "server"] = "space"
    status: Literal["pending", "active", "suspended", "disabled"] = "pending"
    build_filter: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_time: datetime.datetime = Field(
        default_factory=get_utc_time, description="Time at which it was created"
    )
    updated_time: datetime.datetime = Field(
        default_factory=get_utc_time, description="Time at which it was last updated"
    )

    def should_include_event(self, event_type: str) -> bool:
        """Determine whether a given event type should be delivered.

        Exclusions take priority over inclusions. If the event is not
        excluded, a wildcard in event_types matches everything; otherwise
        the event must appear in the explicit list.

        Args:
            event_type: The event type string to check.

        Returns:
            True if this subscription should receive the event, False otherwise.
        """
        if event_type in self.excluded_types:
            return False
        if "*" in self.event_types:
            return True
        return event_type in self.event_types

    def effective_frequency(self) -> int:
        """Return the effective batch flush interval, enforcing the minimum.

        Returns:
            The frequency clamped to at least WEBHOOK_MIN_FREQUENCY seconds.
        """
        return max(self.frequency, WEBHOOK_MIN_FREQUENCY)
