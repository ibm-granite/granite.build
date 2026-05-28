"""Base storage interface and implementation for webhook subscriptions.

Provides query methods for webhook subscription data access. These methods
are the primary data access layer used by the REST API and the webhook
dispatcher to find active subscriptions that should receive event notifications.
"""

from typing import List

from gbserver.storage.storage import (
    CREATED_TIME_FIELD_NAME,
    BaseItemStorage,
    IItemStorage,
)
from gbserver.webhooks.models import StoredWebhookSubscription

GB_WEBHOOK_SUBSCRIPTIONS_TABLE_NAME = "gb_webhook_subscriptions"
"""Default table name for webhook subscription storage."""

_PAGE_SIZE = 100


class IWebhookStorage(IItemStorage[StoredWebhookSubscription]):
    """Interface for webhook subscription storage implementations.

    Extends base storage with domain-specific query methods for
    finding active subscriptions and managing subscription lifecycle.
    """

    def get_active_for_build_filter(
        self, build_filter: str
    ) -> List[StoredWebhookSubscription]:
        """Get active subscriptions with matching build_filter.

        Args:
            build_filter: The build_filter value to match.

        Returns:
            List of active subscriptions with the given build_filter.
        """
        raise NotImplementedError

    def get_by_space(self, space_name: str) -> List[StoredWebhookSubscription]:
        """Get all subscriptions belonging to a space.

        Args:
            space_name: The space name to filter by.

        Returns:
            List of subscriptions (active and inactive) for the space.
        """
        raise NotImplementedError

    def get_active_for_space(self, space_name: str) -> List[StoredWebhookSubscription]:
        """Get active space-wide subscriptions (build_filter is empty/None).

        Space-wide subscriptions are those where build_filter is not set,
        meaning they receive events for ALL builds in the space.

        Args:
            space_name: The space to find subscriptions for.

        Returns:
            List of active subscriptions where build_filter is empty (space-wide).
        """
        raise NotImplementedError

    def deactivate(self, subscription_id: str) -> None:
        """Deactivate a single subscription by its UUID.

        Sets the active field to False. Does not delete the record.

        Args:
            subscription_id: The UUID of the subscription to deactivate.
        """
        raise NotImplementedError

    def deactivate_for_build(self, build_id: str) -> int:
        """Deactivate all active subscriptions for a given build.

        Sets active=False on all active subscriptions whose build_filter
        matches the given build ID.

        Args:
            build_id: The build ID whose subscriptions should be deactivated.

        Returns:
            The number of subscriptions that were deactivated.
        """
        raise NotImplementedError


class BaseWebhookStorage(  # pylint: disable=abstract-method
    BaseItemStorage[StoredWebhookSubscription], IWebhookStorage
):
    """Base storage implementation for webhook subscriptions.

    Provides common functionality for storing and querying webhook
    subscription data across different storage backends (SQL, SQLite, etc.).
    Low-level storage methods (_add_item_dicts, _count, etc.) are left
    abstract for concrete backends (SQL, SQLite) to implement.
    """

    def __init__(self, **kwargs) -> None:
        kwargs["item_class"] = StoredWebhookSubscription
        if kwargs.get("table_name") is None:
            kwargs["table_name"] = GB_WEBHOOK_SUBSCRIPTIONS_TABLE_NAME
        super().__init__(**kwargs)

    def _get_column_values(self, item: StoredWebhookSubscription) -> dict:
        """Extract column values for storage from a StoredWebhookSubscription.

        Exposes key fields for querying:
        - space_name: For filtering by space
        - build_filter: For filtering by build (empty string if None)
        - active: For filtering active vs inactive subscriptions
        - created_by: For audit and ownership queries
        - created_time: For time-based queries

        Args:
            item: The subscription to extract column values from.

        Returns:
            Dict of column name to value for the searchable columns.
        """
        return {
            "space_name": item.space_name,
            "active": item.active,
            "status": item.status,
            "build_filter": item.build_filter or "",
            "created_by": item.created_by,
            CREATED_TIME_FIELD_NAME: item.created_time,
        }

    @classmethod
    def _get_sample_item(cls) -> StoredWebhookSubscription:
        """Return a sample item for use by BaseItemStorage schema initialization.

        Returns:
            A StoredWebhookSubscription with representative field values.
        """
        return StoredWebhookSubscription(
            space_name="sample-space",
            webhook_url="https://example.com/webhook",
            secret="sample-secret",
            event_types=["*"],
            created_by="system",
            status="active",
            build_filter=None,
        )

    # ── Query methods ────────────────────────────────────────────────

    def get_active_for_build_filter(
        self, build_filter: str
    ) -> List[StoredWebhookSubscription]:
        """Get active subscriptions with matching build_filter.

        Args:
            build_filter: The build_filter value to match.

        Returns:
            List of active subscriptions with the given build_filter.
        """
        result: List[StoredWebhookSubscription] = []
        for page in self.get_paged(
            {"build_filter": build_filter, "active": True}, page_size=_PAGE_SIZE
        ):
            result.extend(page)
        return result

    def get_active_for_space(self, space_name: str) -> List[StoredWebhookSubscription]:
        """Get active space-wide subscriptions (build_filter is empty string = None).

        Space-wide subscriptions have build_filter stored as empty string (the
        storage layer maps None to "" in _get_column_values). Only active
        subscriptions are returned.

        Args:
            space_name: The space to find subscriptions for.

        Returns:
            List of active space-wide subscriptions for the given space.
        """
        result: List[StoredWebhookSubscription] = []
        for page in self.get_paged(
            {"space_name": space_name, "build_filter": "", "active": True},
            page_size=_PAGE_SIZE,
        ):
            result.extend(page)
        return result

    def get_by_space(self, space_name: str) -> List[StoredWebhookSubscription]:
        """Get all subscriptions belonging to a space.

        Args:
            space_name: The space name to filter by.

        Returns:
            List of subscriptions (active and inactive) for the space.
        """
        result: List[StoredWebhookSubscription] = []
        for page in self.get_paged({"space_name": space_name}, page_size=_PAGE_SIZE):
            result.extend(page)
        return result

    def deactivate(self, subscription_id: str) -> None:
        """Deactivate a single subscription by its UUID.

        Sets the active field to False. Does not delete the record.

        Args:
            subscription_id: The UUID of the subscription to deactivate.
        """
        self.update_fields(subscription_id, {"active": False})

    def deactivate_for_build(self, build_id: str) -> int:
        """Deactivate all active subscriptions for a given build.

        Sets active=False on all active subscriptions whose build_filter
        matches the given build ID.

        Args:
            build_id: The build ID whose subscriptions should be deactivated.

        Returns:
            The number of subscriptions that were deactivated.
        """
        count = 0
        for page in self.get_paged(
            {"build_filter": build_id, "active": True}, page_size=_PAGE_SIZE
        ):
            for item in page:
                self.update_fields(item.uuid, {"active": False})
                count += 1
        return count
