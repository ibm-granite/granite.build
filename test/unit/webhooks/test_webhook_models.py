"""Tests for webhook subscription storage model."""

import datetime

from gbserver.webhooks.models import (
    WEBHOOK_DEFAULT_FREQUENCY,
    WEBHOOK_MIN_FREQUENCY,
    StoredWebhookSubscription,
)


class TestStoredWebhookSubscription:
    """Tests for StoredWebhookSubscription model."""

    def test_create_subscription_with_defaults(self):
        """Verify default values: active=True, frequency=30, excluded_types=[], event_types=["*"], metadata={}."""
        sub = StoredWebhookSubscription(
            space_name="test-space",
            webhook_url="https://example.com/hook",
            secret="my-secret-key",
            created_by="testuser",
        )

        assert sub.uuid is not None and len(sub.uuid) > 0
        assert sub.space_name == "test-space"
        assert sub.build_filter is None
        assert sub.webhook_url == "https://example.com/hook"
        assert sub.secret == "my-secret-key"
        assert sub.event_types == ["*"]
        assert sub.excluded_types == []
        assert sub.frequency == WEBHOOK_DEFAULT_FREQUENCY
        assert sub.log_pattern is None
        assert sub.created_by == "testuser"
        assert sub.active is True
        assert sub.metadata == {}
        assert isinstance(sub.created_time, datetime.datetime)
        assert isinstance(sub.updated_time, datetime.datetime)

    def test_subscription_with_all_fields(self):
        """Create with all fields explicitly set, verify each."""
        now = datetime.datetime.now(datetime.timezone.utc)
        sub = StoredWebhookSubscription(
            space_name="prod-space",
            build_filter="build-123",
            webhook_url="https://hooks.example.com/notify",
            secret="super-secret",
            event_types=["STATUS_EVENT", "ARTIFACT_EVENT"],
            excluded_types=["METRICS_EVENT"],
            frequency=60,
            log_pattern=r"ERROR|FATAL",
            created_by="admin",
            active=False,
            metadata={"team": "ml-ops", "priority": "high"},
            created_time=now,
            updated_time=now,
        )

        assert sub.space_name == "prod-space"
        assert sub.build_filter == "build-123"
        assert sub.webhook_url == "https://hooks.example.com/notify"
        assert sub.secret == "super-secret"
        assert sub.event_types == ["STATUS_EVENT", "ARTIFACT_EVENT"]
        assert sub.excluded_types == ["METRICS_EVENT"]
        assert sub.frequency == 60
        assert sub.log_pattern == r"ERROR|FATAL"
        assert sub.created_by == "admin"
        assert sub.active is False
        assert sub.metadata == {"team": "ml-ops", "priority": "high"}
        assert sub.created_time == now
        assert sub.updated_time == now

    def test_should_include_event_wildcard(self):
        """Wildcard event_types includes all except excluded."""
        sub = StoredWebhookSubscription(
            space_name="test-space",
            webhook_url="https://example.com/hook",
            secret="key",
            event_types=["*"],
            excluded_types=["METRICS_EVENT"],
            created_by="testuser",
        )

        assert sub.should_include_event("STATUS_EVENT") is True
        assert sub.should_include_event("ARTIFACT_EVENT") is True
        assert sub.should_include_event("METRICS_EVENT") is False

    def test_should_include_event_explicit_list(self):
        """Explicit event_types list only includes listed types."""
        sub = StoredWebhookSubscription(
            space_name="test-space",
            webhook_url="https://example.com/hook",
            secret="key",
            event_types=["STATUS_EVENT", "ARTIFACT_EVENT"],
            excluded_types=[],
            created_by="testuser",
        )

        assert sub.should_include_event("STATUS_EVENT") is True
        assert sub.should_include_event("ARTIFACT_EVENT") is True
        assert sub.should_include_event("METRICS_EVENT") is False

    def test_frequency_minimum_enforced(self):
        """Frequency below minimum is clamped by effective_frequency()."""
        sub = StoredWebhookSubscription(
            space_name="test-space",
            webhook_url="https://example.com/hook",
            secret="key",
            frequency=5,
            created_by="testuser",
        )

        assert sub.frequency == 5  # raw value stored as-is
        assert sub.effective_frequency() >= WEBHOOK_MIN_FREQUENCY
        assert sub.effective_frequency() == WEBHOOK_MIN_FREQUENCY


class TestSubscriptionStatusAndBuildFilter:
    """Tests for status and build_filter fields."""

    def test_default_status_is_pending(self):
        sub = StoredWebhookSubscription(
            space_name="my-space",
            webhook_url="https://example.com/hook",
            secret="s",
            created_by="user",
        )
        assert sub.status == "pending"

    def test_status_active(self):
        sub = StoredWebhookSubscription(
            space_name="my-space",
            webhook_url="https://example.com/hook",
            secret="s",
            created_by="user",
            status="active",
        )
        assert sub.status == "active"

    def test_build_filter_none_means_space_wide(self):
        sub = StoredWebhookSubscription(
            space_name="my-space",
            webhook_url="https://example.com/hook",
            secret="s",
            created_by="user",
        )
        assert sub.build_filter is None

    def test_build_filter_with_uuid_means_per_build(self):
        sub = StoredWebhookSubscription(
            space_name="my-space",
            webhook_url="https://example.com/hook",
            secret="s",
            created_by="user",
            build_filter="build-123",
        )
        assert sub.build_filter == "build-123"
