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
