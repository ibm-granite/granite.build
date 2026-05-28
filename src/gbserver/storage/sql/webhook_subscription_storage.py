"""SQL storage implementation for webhook subscriptions.

Provides both PostgreSQL and SQLite backends. Use `create_webhook_storage()`
to get the correct backend based on the current GBSERVER_METADATA_STORAGE setting.
"""

from gbserver.storage.sql.sql_storage import BaseSQLItemStorage
from gbserver.storage.sqlite.sqlite_storage import SqliteStorageOverrides
from gbserver.storage.webhook_subscription_storage import (
    BaseWebhookStorage,
    IWebhookStorage,
)
from gbserver.webhooks.models import StoredWebhookSubscription


class SQLWebhookStorage(
    BaseSQLItemStorage[StoredWebhookSubscription],
    BaseWebhookStorage,
    IWebhookStorage,
):
    """PostgreSQL-backed webhook subscription storage."""

    def __init__(self, **kwargs) -> None:
        kwargs["indexed_columns"] = [
            "build_filter",
            "space_name",
            "active",
            "status",
        ]
        kwargs["autoincr_column"] = "index"
        kwargs["default_pagination_sort_by_column"] = "index"
        super().__init__(**kwargs)


class SqliteWebhookStorage(
    SqliteStorageOverrides[StoredWebhookSubscription],
    SQLWebhookStorage,
):
    """SQLite-backed webhook subscription storage (for standalone/local dev)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


def create_webhook_storage(**kwargs) -> IWebhookStorage:
    """Create the appropriate webhook storage backend.

    Uses the same storage backend as the rest of gbserver (determined by
    GBSERVER_METADATA_STORAGE env var). Returns SQLite storage in standalone
    mode, PostgreSQL storage otherwise.

    Returns:
        IWebhookStorage implementation matching the configured backend
    """
    from gbserver.storage.singleton_storage import get_storage_factory
    from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory

    factory = get_storage_factory()
    if isinstance(factory, SqliteStorageFactory):
        return SqliteWebhookStorage(**kwargs)
    return SQLWebhookStorage(**kwargs)
