"""Backward-compatible re-exports. Storage moved to gbserver.storage."""
from gbserver.storage.webhook_subscription_storage import (  # noqa: F401
    BaseWebhookStorage,
    GB_WEBHOOK_SUBSCRIPTIONS_TABLE_NAME,
    IWebhookStorage,
)
