"""Unit tests for webhook subscription REST API endpoints.

Tests the FastAPI routes for creating, listing, and deleting webhook
subscriptions, including authentication, authorization, and validation.
"""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from gbserver.webhooks.api import webhooks_api
from gbserver.webhooks.models import StoredWebhookSubscription
from gbserver.webhooks.url_validator import WebhookURLError


class TestWebhookAPI:
    """Tests for webhook subscription API endpoints."""

    def setup_method(self):
        """Create a fresh test client for each test."""
        self.client = TestClient(webhooks_api)

    def _make_subscription(self, **overrides) -> StoredWebhookSubscription:
        """Create a StoredWebhookSubscription with sensible defaults.

        Args:
            **overrides: Fields to override on the default subscription.

        Returns:
            A StoredWebhookSubscription instance.
        """
        defaults = {
            "space_name": "test-space",
            "build_filter": "build-001",
            "webhook_url": "https://example.com/hook",
            "secret": "super-secret-key",
            "event_types": ["*"],
            "created_by": "testuser",
            "active": True,
            "frequency": 30,
        }
        defaults.update(overrides)
        return StoredWebhookSubscription(**defaults)

    @patch("gbserver.webhooks.api.validate_webhook_url")
    @patch("gbserver.webhooks.api.get_webhook_storage")
    @patch("gbserver.webhooks.api.get_admin_storage")
    def test_create_subscription(self, mock_admin, mock_get_storage, mock_validate):
        """POST with valid body returns 201 with all fields except secret."""
        mock_validate.return_value = None
        # Mock build exists
        mock_build = MagicMock()
        mock_build.space_name = "test-space"
        mock_admin.return_value.build_storage.get_by_uuid.return_value = mock_build

        # Mock webhook storage add
        mock_storage = MagicMock()
        mock_storage.get_by_space.return_value = []
        mock_get_storage.return_value = mock_storage

        response = self.client.post(
            "/build-001/subscriptions",
            json={
                "webhook_url": "https://example.com/hook",
                "secret": "test-secret-key",
                "event_types": ["build.started", "build.completed"],
                "frequency": 30,
            },
            headers={"X-Forwarded-User": "testuser"},
        )

        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["build_filter"] == "build-001"
        assert data["space_name"] == "test-space"
        assert data["webhook_url"] == "https://example.com/hook"
        assert data["event_types"] == ["build.started", "build.completed"]
        assert data["frequency"] == 30
        assert data["active"] is True
        assert data["status"] == "pending"
        assert data["created_by"] == "testuser"
        assert "created_time" in data
        # Secret must NEVER be returned
        assert "secret" not in data

        # Verify storage.add was called
        mock_storage.add.assert_called_once()

    @patch("gbserver.webhooks.api.get_webhook_storage")
    @patch("gbserver.webhooks.api.get_admin_storage")
    def test_create_subscription_enforces_min_frequency(
        self, mock_admin, mock_get_storage
    ):
        """POST with frequency below minimum returns 400."""
        mock_build = MagicMock()
        mock_build.space_name = "test-space"
        mock_admin.return_value.build_storage.get_by_uuid.return_value = mock_build

        response = self.client.post(
            "/build-001/subscriptions",
            json={
                "webhook_url": "https://example.com/hook",
                "secret": "test-secret-key",
                "frequency": 5,
            },
            headers={"X-Forwarded-User": "testuser"},
        )

        assert response.status_code == 400
        assert "minimum" in response.json()["detail"].lower()

    @patch("gbserver.webhooks.api.validate_webhook_url")
    @patch("gbserver.webhooks.api.get_webhook_storage")
    @patch("gbserver.webhooks.api.get_admin_storage")
    def test_create_subscription_build_not_found(
        self, mock_admin, mock_get_storage, mock_validate
    ):
        """POST for nonexistent build returns 404."""
        mock_validate.return_value = None
        mock_admin.return_value.build_storage.get_by_uuid.return_value = None

        response = self.client.post(
            "/build-001/subscriptions",
            json={
                "webhook_url": "https://example.com/hook",
                "secret": "test-secret-key",
            },
            headers={"X-Forwarded-User": "testuser"},
        )

        assert response.status_code == 404

    @patch("gbserver.webhooks.api.get_webhook_storage")
    @patch("gbserver.webhooks.api.get_admin_storage")
    def test_list_subscriptions(self, mock_admin, mock_get_storage):
        """GET returns list of active subscriptions without secrets."""
        mock_build = MagicMock()
        mock_build.space_name = "test-space"
        mock_admin.return_value.build_storage.get_by_uuid.return_value = mock_build

        sub = self._make_subscription()
        mock_storage = MagicMock()
        mock_storage.get_active_for_build_filter.return_value = [sub]
        mock_get_storage.return_value = mock_storage

        response = self.client.get(
            "/build-001/subscriptions",
            headers={"X-Forwarded-User": "testuser"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "subscriptions" in data
        assert len(data["subscriptions"]) == 1
        item = data["subscriptions"][0]
        assert item["webhook_url"] == "https://example.com/hook"
        assert item["space_name"] == "test-space"
        # Secret must NEVER be returned
        assert "secret" not in item

    @patch("gbserver.webhooks.api.get_webhook_storage")
    def test_delete_subscription(self, mock_get_storage):
        """DELETE by owner returns 204."""
        sub = self._make_subscription(created_by="testuser")
        mock_storage = MagicMock()
        mock_storage.get_by_uuid.return_value = sub
        mock_get_storage.return_value = mock_storage

        response = self.client.delete(
            f"/{sub.uuid}",
            headers={"X-Forwarded-User": "testuser"},
        )

        assert response.status_code == 204
        mock_storage.deactivate.assert_called_once_with(sub.uuid)

    @patch("gbserver.webhooks.api.get_webhook_storage")
    def test_delete_subscription_forbidden(self, mock_get_storage):
        """DELETE by non-owner returns 403."""
        sub = self._make_subscription(created_by="owner-user")
        mock_storage = MagicMock()
        mock_storage.get_by_uuid.return_value = sub
        mock_get_storage.return_value = mock_storage

        response = self.client.delete(
            f"/{sub.uuid}",
            headers={"X-Forwarded-User": "other-user"},
        )

        assert response.status_code == 403
        mock_storage.deactivate.assert_not_called()

    @patch("gbserver.webhooks.api.get_webhook_storage")
    def test_delete_subscription_not_found(self, mock_get_storage):
        """DELETE for nonexistent subscription returns 404."""
        mock_storage = MagicMock()
        mock_storage.get_by_uuid.return_value = None
        mock_get_storage.return_value = mock_storage

        response = self.client.delete(
            "/nonexistent-id",
            headers={"X-Forwarded-User": "testuser"},
        )

        assert response.status_code == 404

    def test_create_no_auth_header(self):
        """POST without X-Forwarded-User header returns 401."""
        response = self.client.post(
            "/build-001/subscriptions",
            json={
                "webhook_url": "https://example.com/hook",
                "secret": "test-secret-key",
            },
        )

        assert response.status_code == 401

    def test_list_no_auth_header(self):
        """GET without X-Forwarded-User header returns 401."""
        response = self.client.get("/build-001/subscriptions")

        assert response.status_code == 401

    def test_delete_no_auth_header(self):
        """DELETE without X-Forwarded-User header returns 401."""
        response = self.client.delete("/some-webhook-id")

        assert response.status_code == 401

    @patch("gbserver.webhooks.api.validate_webhook_url")
    @patch("gbserver.webhooks.api.get_webhook_storage")
    @patch("gbserver.webhooks.api.get_admin_storage")
    def test_create_subscription_default_event_types(
        self, mock_admin, mock_get_storage, mock_validate
    ):
        """POST without event_types defaults to wildcard."""
        mock_validate.return_value = None
        mock_build = MagicMock()
        mock_build.space_name = "test-space"
        mock_admin.return_value.build_storage.get_by_uuid.return_value = mock_build

        mock_storage = MagicMock()
        mock_storage.get_by_space.return_value = []
        mock_get_storage.return_value = mock_storage

        response = self.client.post(
            "/build-001/subscriptions",
            json={
                "webhook_url": "https://example.com/hook",
                "secret": "test-secret-key",
            },
            headers={"X-Forwarded-User": "testuser"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["event_types"] == ["*"]

    @patch("gbserver.webhooks.api.validate_webhook_url")
    @patch("gbserver.webhooks.api.get_webhook_storage")
    @patch("gbserver.webhooks.api.get_admin_storage")
    def test_create_subscription_with_log_pattern(
        self, mock_admin, mock_get_storage, mock_validate
    ):
        """POST with log_pattern stores and returns it."""
        mock_validate.return_value = None
        mock_build = MagicMock()
        mock_build.space_name = "test-space"
        mock_admin.return_value.build_storage.get_by_uuid.return_value = mock_build

        mock_storage = MagicMock()
        mock_storage.get_by_space.return_value = []
        mock_get_storage.return_value = mock_storage

        response = self.client.post(
            "/build-001/subscriptions",
            json={
                "webhook_url": "https://example.com/hook",
                "secret": "test-secret-key",
                "log_pattern": "ERROR|WARN",
            },
            headers={"X-Forwarded-User": "testuser"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["log_pattern"] == "ERROR|WARN"

    @patch("gbserver.webhooks.api.get_webhook_storage")
    @patch("gbserver.webhooks.api.get_admin_storage")
    @patch("gbserver.webhooks.api.validate_webhook_url")
    def test_create_subscription_rejects_private_ip(
        self, mock_validate, mock_admin, mock_get_storage
    ):
        """POST with private IP webhook URL returns 400."""
        mock_validate.side_effect = WebhookURLError(
            "Webhook URL blocked (private/blocked IP): 10.0.0.1"
        )
        mock_build = MagicMock()
        mock_build.space_name = "test-space"
        mock_admin.return_value.build_storage.get_by_uuid.return_value = mock_build

        response = self.client.post(
            "/build-001/subscriptions",
            json={
                "webhook_url": "https://10.0.0.1/hook",
                "secret": "test-secret-key",
            },
            headers={"X-Forwarded-User": "testuser"},
        )

        assert response.status_code == 400
        assert "Invalid webhook URL" in response.json()["detail"]

    @patch("gbserver.webhooks.api.get_webhook_storage")
    @patch("gbserver.webhooks.api.get_admin_storage")
    @patch("gbserver.webhooks.api.validate_webhook_url")
    def test_create_subscription_rate_limited(
        self, mock_validate, mock_admin, mock_get_storage
    ):
        """POST when space has max subscriptions returns 429."""
        mock_validate.return_value = None  # URL is valid
        mock_build = MagicMock()
        mock_build.space_name = "test-space"
        mock_admin.return_value.build_storage.get_by_uuid.return_value = mock_build

        # Mock storage with max subscriptions already
        mock_storage = MagicMock()
        mock_storage.get_by_space.return_value = [
            MagicMock(active=True) for _ in range(20)
        ]
        mock_get_storage.return_value = mock_storage

        response = self.client.post(
            "/build-001/subscriptions",
            json={
                "webhook_url": "https://example.com/hook",
                "secret": "test-secret-key",
            },
            headers={"X-Forwarded-User": "testuser"},
        )

        assert response.status_code == 429
