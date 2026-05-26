"""SQL storage implementation for webhook events.

Provides both PostgreSQL and SQLite backends. Use `create_webhook_event_storage()`
to get the correct backend based on the current GBSERVER_METADATA_STORAGE setting.
"""

from gbserver.storage.sql.sql_storage import BaseSQLItemStorage
from gbserver.storage.sqlite.sqlite_storage import SqliteStorageOverrides
from gbserver.storage.webhook_event_storage import (
    BaseWebhookEventStorage,
    IWebhookEventStorage,
)
from gbserver.webhooks.event_models import StoredWebhookEvent


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
