"""SQL storage implementation for webhook subscriptions."""

from gbserver.storage.sql.sql_storage import BaseSQLItemStorage
from gbserver.webhooks.models import StoredWebhookSubscription
from gbserver.webhooks.storage import BaseWebhookStorage, IWebhookStorage


class SQLWebhookStorage(
    BaseSQLItemStorage[StoredWebhookSubscription],
    BaseWebhookStorage,
    IWebhookStorage,
):
    """SQL-based storage implementation for webhook subscriptions.

    Uses PostgreSQL (or SQLite for local dev) to persist webhook subscription
    records. Indexes the build_id, space_name, and active columns for
    efficient filtered queries by the webhook dispatcher.
    """

    def __init__(self, **kwargs) -> None:
        kwargs["indexed_columns"] = ["build_id", "space_name", "active"]
        kwargs["autoincr_column"] = "index"
        kwargs["default_pagination_sort_by_column"] = "index"
        super().__init__(**kwargs)
