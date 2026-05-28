"""Unit tests for webhook event writer and BuildRunner integration."""

from unittest.mock import MagicMock, patch

from gbserver.webhooks.event_models import StoredWebhookEvent
from gbserver.webhooks.event_writer import WebhookEventWriter
from gbserver.webhooks.models import StoredWebhookSubscription


class TestWebhookEventWriter:
    """Tests for WebhookEventWriter."""

    def _make_subscription(self, **overrides) -> StoredWebhookSubscription:
        defaults = {
            "space_name": "test-space",
            "build_id": "build-001",
            "webhook_url": "https://example.com/hook",
            "secret": "s",
            "created_by": "user",
            "event_types": ["*"],
            "status": "active",
            "scope": "space",
        }
        defaults.update(overrides)
        return StoredWebhookSubscription(**defaults)

    @patch("gbserver.webhooks.event_writer.create_webhook_event_storage")
    @patch("gbserver.webhooks.event_writer.create_webhook_storage")
    def test_start_finds_subscriptions(self, mock_sub_factory, mock_evt_factory):
        """start() queries active subscriptions for the build."""
        sub = self._make_subscription()
        mock_sub_storage = MagicMock()
        mock_sub_storage.get_active_for_build.return_value = [sub]
        mock_sub_storage.get_active_for_build_filter.return_value = []
        mock_sub_storage.get_active_for_space.return_value = []
        mock_sub_factory.return_value = mock_sub_storage
        mock_evt_factory.return_value = MagicMock()

        writer = WebhookEventWriter(build_id="build-001", space_name="test-space")
        subs = writer.start()

        assert len(subs) == 1
        assert subs[0].uuid == sub.uuid

    @patch("gbserver.webhooks.event_writer.create_webhook_event_storage")
    @patch("gbserver.webhooks.event_writer.create_webhook_storage")
    def test_accept_event_persists_to_storage(self, mock_sub_factory, mock_evt_factory):
        """accept_event() writes a StoredWebhookEvent per matching subscription."""
        sub = self._make_subscription()
        mock_sub_storage = MagicMock()
        mock_sub_storage.get_active_for_build.return_value = [sub]
        mock_sub_storage.get_active_for_build_filter.return_value = []
        mock_sub_storage.get_active_for_space.return_value = []
        mock_sub_factory.return_value = mock_sub_storage

        mock_evt_storage = MagicMock()
        mock_evt_factory.return_value = mock_evt_storage

        writer = WebhookEventWriter(build_id="build-001", space_name="test-space")
        writer.start()

        from gbserver.types.buildevent import (
            BuildEvent,
            BuildEventStatusPayload,
            BuildEventType,
            EntityRunMetadata,
        )
        from gbserver.types.status import Status

        event = BuildEvent(
            type=BuildEventType.STATUS_EVENT,
            payload=BuildEventStatusPayload(status=Status.RUNNING, msg="Build started"),
            run_metadata=EntityRunMetadata(
                build_id="build-001",
                target_name="target-1",
                targetstep_uri="step/uri",
                targetrun_id="tr-1",
                targetsteprun_id="tsr-1",
                username="user",
            ),
        )

        writer.accept_event(event)

        mock_evt_storage.add.assert_called_once()
        persisted = mock_evt_storage.add.call_args[0][0]
        assert isinstance(persisted, StoredWebhookEvent)
        assert persisted.subscription_id == sub.uuid
        assert persisted.build_id == "build-001"
        assert persisted.event_type == "status_event"
        assert persisted.delivered is False

    @patch("gbserver.webhooks.event_writer.create_webhook_event_storage")
    @patch("gbserver.webhooks.event_writer.create_webhook_storage")
    def test_filters_by_event_type(self, mock_sub_factory, mock_evt_factory):
        """accept_event() respects subscription event_types filter."""
        sub = self._make_subscription(event_types=["artifact_event"])
        mock_sub_storage = MagicMock()
        mock_sub_storage.get_active_for_build.return_value = [sub]
        mock_sub_storage.get_active_for_build_filter.return_value = []
        mock_sub_storage.get_active_for_space.return_value = []
        mock_sub_factory.return_value = mock_sub_storage

        mock_evt_storage = MagicMock()
        mock_evt_factory.return_value = mock_evt_storage

        writer = WebhookEventWriter(build_id="build-001", space_name="test-space")
        writer.start()

        from gbserver.types.buildevent import (
            BuildEvent,
            BuildEventStatusPayload,
            BuildEventType,
            EntityRunMetadata,
        )
        from gbserver.types.status import Status

        # STATUS_EVENT should be filtered out
        event = BuildEvent(
            type=BuildEventType.STATUS_EVENT,
            payload=BuildEventStatusPayload(status=Status.RUNNING, msg="Build started"),
            run_metadata=EntityRunMetadata(
                build_id="build-001",
                target_name="target-1",
                targetstep_uri="step/uri",
                targetrun_id="tr-1",
                targetsteprun_id="tsr-1",
                username="user",
            ),
        )

        writer.accept_event(event)

        # Should NOT have been persisted (filtered out)
        mock_evt_storage.add.assert_not_called()

    @patch("gbserver.webhooks.event_writer.create_webhook_event_storage")
    @patch("gbserver.webhooks.event_writer.create_webhook_storage")
    def test_skips_pending_subscriptions(self, mock_sub_factory, mock_evt_factory):
        """start() filters out non-active subscriptions."""
        active_sub = self._make_subscription(status="active")
        pending_sub = self._make_subscription(status="pending")
        mock_sub_storage = MagicMock()
        mock_sub_storage.get_active_for_build.return_value = [active_sub, pending_sub]
        mock_sub_storage.get_active_for_build_filter.return_value = []
        mock_sub_storage.get_active_for_space.return_value = []
        mock_sub_factory.return_value = mock_sub_storage
        mock_evt_factory.return_value = MagicMock()

        writer = WebhookEventWriter(build_id="build-001", space_name="test-space")
        subs = writer.start()

        # Only the active subscription should be returned
        assert len(subs) == 1
        assert subs[0].status == "active"


class TestFinalizeDeactivation:
    """Tests that finalize_build_status deactivates webhook subscriptions."""

    @patch("gbserver.buildwatcher.build_utils.get_admin_storage")
    @patch("gbserver.buildwatcher.build_utils.create_webhook_storage")
    @patch("gbserver.types.constants.GBSERVER_WEBHOOKS_ENABLED", True)
    def test_finalize_deactivates_subscriptions(self, mock_create_storage, mock_admin):
        """finalize_build_status deactivates webhook subscriptions for the build."""
        from gbserver.buildwatcher.build_utils import finalize_build_status
        from gbserver.types.status import Status

        # Mock admin storage with a build that exists and is still running
        mock_build = MagicMock()
        mock_build.status = Status.RUNNING
        mock_build.uuid = "build-001"
        mock_admin.return_value.build_storage.get_by_uuid.return_value = mock_build
        mock_admin.return_value.build_storage.update_fields.return_value = mock_build
        mock_admin.return_value.target_storage.get_by_where.return_value = []
        mock_admin.return_value.step_storage.get_by_where.return_value = []
        mock_admin.return_value.artifact_registry.get_by_where.return_value = []

        # Mock webhook storage
        mock_wh_storage = MagicMock()
        mock_wh_storage.deactivate_for_build.return_value = 2
        mock_create_storage.return_value = mock_wh_storage

        finalize_build_status("build-001", Status.SUCCESS)

        mock_wh_storage.deactivate_for_build.assert_called_once_with("build-001")

    @patch("gbserver.buildwatcher.build_utils.get_admin_storage")
    @patch("gbserver.storage.sql.webhook_subscription_storage.create_webhook_storage")
    @patch("gbserver.types.constants.GBSERVER_WEBHOOKS_ENABLED", False)
    def test_finalize_skips_when_webhooks_disabled(
        self, mock_create_storage, mock_admin
    ):
        """finalize_build_status does NOT deactivate when webhooks disabled."""
        from gbserver.buildwatcher.build_utils import finalize_build_status
        from gbserver.types.status import Status

        mock_build = MagicMock()
        mock_build.status = Status.RUNNING
        mock_build.uuid = "build-001"
        mock_admin.return_value.build_storage.get_by_uuid.return_value = mock_build
        mock_admin.return_value.build_storage.update_fields.return_value = mock_build
        mock_admin.return_value.target_storage.get_by_where.return_value = []
        mock_admin.return_value.step_storage.get_by_where.return_value = []
        mock_admin.return_value.artifact_registry.get_by_where.return_value = []

        finalize_build_status("build-001", Status.SUCCESS)

        # create_webhook_storage should never have been called
        mock_create_storage.assert_not_called()


class TestBuildRunnerDispatch:
    """Tests for BuildRunner __dispatch_to_webhooks via WebhookEventWriter."""

    @patch("gbserver.webhooks.event_writer.create_webhook_event_storage")
    @patch("gbserver.webhooks.event_writer.create_webhook_storage")
    def test_dispatch_via_build_filter_lookup(self, mock_sub_factory, mock_evt_factory):
        """WebhookEventWriter finds subscriptions via build_filter and persists events."""
        sub = StoredWebhookSubscription(
            space_name="test-space",
            build_id=None,
            build_filter="build-001",
            webhook_url="https://example.com/hook",
            secret="test-secret-key",
            created_by="user",
            event_types=["*"],
            status="active",
            scope="space",
        )
        mock_sub_storage = MagicMock()
        mock_sub_storage.get_active_for_build.return_value = []
        mock_sub_storage.get_active_for_build_filter.return_value = [sub]
        mock_sub_storage.get_active_for_space.return_value = []
        mock_sub_factory.return_value = mock_sub_storage

        mock_evt_storage = MagicMock()
        mock_evt_factory.return_value = mock_evt_storage

        writer = WebhookEventWriter(build_id="build-001", space_name="test-space")
        subs = writer.start()
        assert len(subs) == 1
        assert subs[0].build_filter == "build-001"

        from gbserver.types.buildevent import (
            BuildEvent,
            BuildEventStatusPayload,
            BuildEventType,
            EntityRunMetadata,
        )
        from gbserver.types.status import Status

        event = BuildEvent(
            type=BuildEventType.STATUS_EVENT,
            payload=BuildEventStatusPayload(
                status=Status.RUNNING, msg="Build started"
            ),
            run_metadata=EntityRunMetadata(
                build_id="build-001",
                target_name="target-1",
                targetstep_uri="step/uri",
                targetrun_id="tr-1",
                targetsteprun_id="tsr-1",
                username="user",
            ),
        )

        writer.accept_event(event)
        mock_evt_storage.add.assert_called_once()
        persisted = mock_evt_storage.add.call_args[0][0]
        assert isinstance(persisted, StoredWebhookEvent)
        assert persisted.subscription_id == sub.uuid
        assert persisted.build_id == "build-001"

    @patch("gbserver.webhooks.event_writer.create_webhook_event_storage")
    @patch("gbserver.webhooks.event_writer.create_webhook_storage")
    def test_dispatch_deduplicates_across_sources(
        self, mock_sub_factory, mock_evt_factory
    ):
        """WebhookEventWriter deduplicates subscriptions found via multiple lookup paths."""
        sub = StoredWebhookSubscription(
            space_name="test-space",
            build_id="build-001",
            build_filter="build-001",
            webhook_url="https://example.com/hook",
            secret="test-secret-key",
            created_by="user",
            event_types=["*"],
            status="active",
            scope="space",
        )
        # Same subscription returned from multiple lookups
        mock_sub_storage = MagicMock()
        mock_sub_storage.get_active_for_build.return_value = [sub]
        mock_sub_storage.get_active_for_build_filter.return_value = [sub]
        mock_sub_storage.get_active_for_space.return_value = [sub]
        mock_sub_factory.return_value = mock_sub_storage
        mock_evt_factory.return_value = MagicMock()

        writer = WebhookEventWriter(build_id="build-001", space_name="test-space")
        subs = writer.start()

        # Should be deduplicated to a single subscription
        assert len(subs) == 1
