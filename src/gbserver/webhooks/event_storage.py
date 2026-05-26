"""Storage layer for webhook events (write-ahead log).

Provides CRUD and query methods for persisted webhook events.
Events are written by BuildRunner and read by the delivery worker.
"""

from typing import List

from gbserver.storage.storage import CREATED_TIME_FIELD_NAME, BaseItemStorage, IItemStorage
from gbserver.storage.sql.sql_storage import BaseSQLItemStorage
from gbserver.storage.sqlite.sqlite_storage import SqliteStorageOverrides
from gbserver.webhooks.event_models import StoredWebhookEvent

GB_WEBHOOK_EVENTS_TABLE_NAME = "gb_webhook_events"

_PAGE_SIZE = 100


class IWebhookEventStorage(IItemStorage[StoredWebhookEvent]):
    """Interface for webhook event storage."""

    def get_pending_for_subscription(
        self, subscription_id: str
    ) -> List[StoredWebhookEvent]:
        """Get undelivered events for a subscription."""
        raise NotImplementedError

    def get_pending_for_build(self, build_id: str) -> List[StoredWebhookEvent]:
        """Get undelivered events for a build."""
        raise NotImplementedError

    def mark_delivered(self, event_ids: List[str]) -> None:
        """Mark events as delivered."""
        raise NotImplementedError


class BaseWebhookEventStorage(
    BaseItemStorage[StoredWebhookEvent], IWebhookEventStorage
):
    """Base webhook event storage implementation."""

    def __init__(self, **kwargs) -> None:
        kwargs["item_class"] = StoredWebhookEvent
        if kwargs.get("table_name") is None:
            kwargs["table_name"] = GB_WEBHOOK_EVENTS_TABLE_NAME
        super().__init__(**kwargs)

    def _get_column_values(self, item: StoredWebhookEvent) -> dict:
        return {
            "subscription_id": item.subscription_id,
            "build_id": item.build_id,
            "event_type": item.event_type,
            "delivered": item.delivered,
            CREATED_TIME_FIELD_NAME: item.created_time,
        }

    @classmethod
    def _get_sample_item(cls) -> StoredWebhookEvent:
        return StoredWebhookEvent(
            subscription_id="sample-sub-id",
            build_id="sample-build-id",
            event_type="STATUS_EVENT",
            payload={"status": "running"},
        )

    def get_pending_for_subscription(
        self, subscription_id: str
    ) -> List[StoredWebhookEvent]:
        result: List[StoredWebhookEvent] = []
        for page in self.get_paged(
            {"subscription_id": subscription_id, "delivered": False},
            page_size=_PAGE_SIZE,
        ):
            result.extend(page)
        return result

    def get_pending_for_build(self, build_id: str) -> List[StoredWebhookEvent]:
        result: List[StoredWebhookEvent] = []
        for page in self.get_paged(
            {"build_id": build_id, "delivered": False},
            page_size=_PAGE_SIZE,
        ):
            result.extend(page)
        return result

    def mark_delivered(self, event_ids: List[str]) -> None:
        for event_id in event_ids:
            self.update_fields(event_id, {"delivered": True})


class SQLWebhookEventStorage(
    BaseSQLItemStorage[StoredWebhookEvent],
    BaseWebhookEventStorage,
    IWebhookEventStorage,
):
    """PostgreSQL-backed webhook event storage."""

    def __init__(self, **kwargs) -> None:
        kwargs["indexed_columns"] = ["subscription_id", "build_id", "delivered"]
        kwargs["autoincr_column"] = "index"
        kwargs["default_pagination_sort_by_column"] = "index"
        super().__init__(**kwargs)


class SqliteWebhookEventStorage(
    SqliteStorageOverrides[StoredWebhookEvent],
    SQLWebhookEventStorage,
):
    """SQLite-backed webhook event storage."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


def create_webhook_event_storage(**kwargs) -> IWebhookEventStorage:
    """Create the appropriate webhook event storage backend."""
    from gbserver.storage.singleton_storage import get_storage_factory
    from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory

    factory = get_storage_factory()
    if isinstance(factory, SqliteStorageFactory):
        return SqliteWebhookEventStorage(**kwargs)
    return SQLWebhookEventStorage(**kwargs)
