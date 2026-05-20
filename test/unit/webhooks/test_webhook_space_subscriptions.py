"""Tests for Phase 2: space-wide webhook subscriptions.

Covers the API endpoints for creating and listing space-wide subscriptions,
and verifies that the dispatcher correctly buffers events for space-wide
subscribers alongside per-build subscribers.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from gbserver.webhooks.api import webhooks_api
from gbserver.webhooks.models import StoredWebhookSubscription


class TestSpaceWideSubscriptionAPI:
    """Tests for the space-wide webhook subscription API routes."""

    def setup_method(self):
        """Create a fresh test client for each test."""
        self.client = TestClient(webhooks_api)

    @patch("gbserver.webhooks.api.get_webhook_storage")
    @patch("gbserver.webhooks.api.get_admin_storage")
    def test_create_space_subscription(self, mock_admin, mock_get_storage):
        """POST to /spaces/{name}/subscriptions returns 201 with build_id=None."""
        mock_storage = MagicMock()
        mock_get_storage.return_value = mock_storage

        # Mock space exists
        mock_space_storage = MagicMock()
        mock_space_storage.get_by_where.return_value = [MagicMock()]
        admin = MagicMock()
        admin.space_storage = mock_space_storage
        mock_admin.return_value = admin

        response = self.client.post(
            "/spaces/my-space/subscriptions",
            json={
                "webhook_url": "https://dashboard.example.com/hooks",
                "secret": "dashboard-secret",
                "event_types": ["*"],
                "frequency": 15,
            },
            headers={"X-Forwarded-User": "dashboard-service"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["space_name"] == "my-space"
        assert data["build_id"] is None
        assert data["webhook_url"] == "https://dashboard.example.com/hooks"
        assert data["active"] is True
        assert data["created_by"] == "dashboard-service"
        # Secret must NEVER be returned
        assert "secret" not in data
        # Verify storage.add was called
        mock_storage.add.assert_called_once()

    @patch("gbserver.webhooks.api.get_webhook_storage")
    @patch("gbserver.webhooks.api.get_admin_storage")
    def test_create_space_subscription_space_not_found(
        self, mock_admin, mock_get_storage
    ):
        """POST for nonexistent space returns 404."""
        mock_space_storage = MagicMock()
        mock_space_storage.get_by_where.return_value = []
        admin = MagicMock()
        admin.space_storage = mock_space_storage
        mock_admin.return_value = admin

        response = self.client.post(
            "/spaces/nonexistent/subscriptions",
            json={
                "webhook_url": "https://x.com/hook",
                "secret": "s",
                "event_types": ["*"],
            },
            headers={"X-Forwarded-User": "user"},
        )

        assert response.status_code == 404
        assert "nonexistent" in response.json()["detail"]

    @patch("gbserver.webhooks.api.get_webhook_storage")
    @patch("gbserver.webhooks.api.get_admin_storage")
    def test_create_space_subscription_enforces_min_frequency(
        self, mock_admin, mock_get_storage
    ):
        """POST with frequency below minimum returns 400."""
        mock_space_storage = MagicMock()
        mock_space_storage.get_by_where.return_value = [MagicMock()]
        admin = MagicMock()
        admin.space_storage = mock_space_storage
        mock_admin.return_value = admin

        response = self.client.post(
            "/spaces/my-space/subscriptions",
            json={
                "webhook_url": "https://example.com/hook",
                "secret": "my-secret",
                "frequency": 5,
            },
            headers={"X-Forwarded-User": "testuser"},
        )

        assert response.status_code == 400
        assert "minimum" in response.json()["detail"].lower()

    def test_create_space_subscription_no_auth(self):
        """POST without auth header returns 401."""
        response = self.client.post(
            "/spaces/my-space/subscriptions",
            json={
                "webhook_url": "https://example.com/hook",
                "secret": "s",
                "event_types": ["*"],
            },
        )

        assert response.status_code == 401

    @patch("gbserver.webhooks.api.get_webhook_storage")
    def test_list_space_subscriptions(self, mock_get_storage):
        """GET returns list of active space-wide subscriptions."""
        mock_storage = MagicMock()
        sub = StoredWebhookSubscription(
            space_name="my-space",
            build_id=None,
            webhook_url="https://dashboard.example.com/hooks",
            secret="s",
            event_types=["*"],
            created_by="dashboard-service",
        )
        mock_storage.get_active_for_space.return_value = [sub]
        mock_get_storage.return_value = mock_storage

        response = self.client.get(
            "/spaces/my-space/subscriptions",
            headers={"X-Forwarded-User": "user"},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["subscriptions"]) == 1
        assert data["subscriptions"][0]["build_id"] is None
        assert data["subscriptions"][0]["space_name"] == "my-space"
        assert data["subscriptions"][0]["webhook_url"] == (
            "https://dashboard.example.com/hooks"
        )
        # Secret must NEVER be returned
        assert "secret" not in data["subscriptions"][0]

    @patch("gbserver.webhooks.api.get_webhook_storage")
    def test_list_space_subscriptions_empty(self, mock_get_storage):
        """GET returns empty list when no space-wide subscriptions exist."""
        mock_storage = MagicMock()
        mock_storage.get_active_for_space.return_value = []
        mock_get_storage.return_value = mock_storage

        response = self.client.get(
            "/spaces/my-space/subscriptions",
            headers={"X-Forwarded-User": "user"},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["subscriptions"]) == 0

    def test_list_space_subscriptions_no_auth(self):
        """GET without auth header returns 401."""
        response = self.client.get("/spaces/my-space/subscriptions")

        assert response.status_code == 401


class TestSpaceWideDispatcherIntegration:
    """Verify dispatcher loads and buffers events for space-wide subscriptions."""

    def test_space_subscription_receives_build_events(self):
        """A space-wide subscription should buffer events from any build in that space."""
        from gbserver.types.buildevent import (
            BuildEvent,
            BuildEventStatusPayload,
            BuildEventType,
            EntityRunMetadata,
        )
        from gbserver.types.status import Status
        from gbserver.webhooks.dispatcher import WebhookDispatcher

        mock_storage = MagicMock()
        # Space-wide sub (no build_id)
        space_sub = StoredWebhookSubscription(
            space_name="shared-space",
            build_id=None,
            webhook_url="https://dashboard.example.com/hooks",
            secret="s",
            event_types=["*"],
            created_by="dashboard",
        )

        dispatcher = WebhookDispatcher(
            webhook_storage=mock_storage,
            build_id="any-build-123",
            space_name="shared-space",
            build_name="some-build",
            username="user",
            build_start_time="2026-05-20T11:00:00Z",
        )
        dispatcher.start([space_sub])

        event = BuildEvent(
            type=BuildEventType.STATUS_EVENT,
            run_metadata=EntityRunMetadata(
                build_id="any-build-123",
                username="user",
                target_name="t",
                targetrun_id="tr",
                targetstep_uri="s",
                targetsteprun_id="tsr",
            ),
            payload=BuildEventStatusPayload(status=Status.RUNNING, msg="go"),
        )
        dispatcher.accept_event(event)
        assert dispatcher.buffer.pending_count(space_sub.uuid) == 1

    def test_space_and_build_subscriptions_both_receive_events(self):
        """Both space-wide and per-build subscriptions receive the same event."""
        from gbserver.types.buildevent import (
            BuildEvent,
            BuildEventStatusPayload,
            BuildEventType,
            EntityRunMetadata,
        )
        from gbserver.types.status import Status
        from gbserver.webhooks.dispatcher import WebhookDispatcher

        mock_storage = MagicMock()

        # Space-wide subscription
        space_sub = StoredWebhookSubscription(
            space_name="shared-space",
            build_id=None,
            webhook_url="https://dashboard.example.com/hooks",
            secret="s1",
            event_types=["*"],
            created_by="dashboard",
        )
        # Per-build subscription
        build_sub = StoredWebhookSubscription(
            space_name="shared-space",
            build_id="build-xyz",
            webhook_url="https://ci.example.com/hooks",
            secret="s2",
            event_types=["*"],
            created_by="ci-bot",
        )

        dispatcher = WebhookDispatcher(
            webhook_storage=mock_storage,
            build_id="build-xyz",
            space_name="shared-space",
            build_name="test-build",
            username="user",
            build_start_time="2026-05-20T11:00:00Z",
        )
        dispatcher.start([build_sub, space_sub])

        event = BuildEvent(
            type=BuildEventType.STATUS_EVENT,
            run_metadata=EntityRunMetadata(
                build_id="build-xyz",
                username="user",
                target_name="t",
                targetrun_id="tr",
                targetstep_uri="s",
                targetsteprun_id="tsr",
            ),
            payload=BuildEventStatusPayload(status=Status.SUCCESS, msg="done"),
        )
        dispatcher.accept_event(event)

        # Both subscriptions should have buffered the event
        assert dispatcher.buffer.pending_count(space_sub.uuid) == 1
        assert dispatcher.buffer.pending_count(build_sub.uuid) == 1
