"""Webhook event persistence model (write-ahead log).

Events are written to the database as soon as they are produced by
BuildRunner. A separate delivery worker (Phase 2) reads pending events,
batches them, and delivers to subscriber endpoints.
"""

import datetime
from typing import Any, Dict

from pydantic import Field

from gbserver.storage.storage import BaseStoredItem
from gbserver.utils.utils import get_utc_time


class StoredWebhookEvent(BaseStoredItem):
    """A single webhook event persisted for later delivery.

    Args:
        subscription_id: UUID of the target subscription.
        build_id: UUID of the build that produced this event.
        event_type: The event type string (e.g. STATUS_EVENT).
        payload: Serialized event data dict.
        delivered: Whether this event has been successfully delivered.
        created_time: When the event was persisted.
    """

    subscription_id: str
    build_id: str
    event_type: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    delivered: bool = False
    created_time: datetime.datetime = Field(default_factory=get_utc_time)
