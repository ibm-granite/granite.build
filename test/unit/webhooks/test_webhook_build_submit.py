"""Tests for webhook auto-subscription on build submit."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from gbserver.api.builds import BuildSubmitRequest, BuildSubmitResponse, builds_api


class TestBuildSubmitWithWebhook:
    """Verify that providing webhook_url during build submission auto-creates a subscription."""

    def setup_method(self):
        self.client = TestClient(builds_api)

    @patch("gbserver.api.builds.get_admin_storage")
    @patch("gbserver.webhooks.sql_storage.SQLWebhookStorage.add")
    def test_submit_with_webhook_url_creates_subscription(
        self, mock_ws_add, mock_admin
    ):
        """When webhook_url is provided, a subscription is auto-created."""
        mock_space = MagicMock()
        mock_space.name = "test-space"
        mock_build_storage = MagicMock()
        mock_build_storage.add.return_value = "build-uuid-123"
        mock_space_storage = MagicMock()
        mock_space_storage.get_by_name.return_value = mock_space
        admin = MagicMock()
        admin.space_storage = mock_space_storage
        admin.build_storage = mock_build_storage
        mock_admin.return_value = admin

        with patch("gbserver.types.constants.GBSERVER_WEBHOOKS_ENABLED", True):
            response = self.client.post(
                "/",
                json={
                    "name": "test-build",
                    "build_archive": "base64data",
                    "space_name": "test-space",
                    "username": "testuser",
                    "webhook_url": "https://example.com/hook",
                    "webhook_secret": "test-secret-key",
                },
                headers={"X-Forwarded-User": "testuser"},
            )
            assert response.status_code == 200
            data = response.json()
            assert "build_id" in data
            assert data.get("webhook_subscription_id") is not None
            mock_ws_add.assert_called_once()

    @patch("gbserver.api.builds.get_admin_storage")
    def test_submit_without_webhook_url_no_subscription(self, mock_admin):
        """Without webhook_url, no subscription is created."""
        mock_space = MagicMock()
        mock_space.name = "test-space"
        mock_space_storage = MagicMock()
        mock_space_storage.get_by_name.return_value = mock_space
        mock_build_storage = MagicMock()
        mock_build_storage.add.return_value = "build-uuid-456"
        admin = MagicMock()
        admin.space_storage = mock_space_storage
        admin.build_storage = mock_build_storage
        mock_admin.return_value = admin

        response = self.client.post(
            "/",
            json={
                "name": "test-build",
                "build_archive": "base64data",
                "space_name": "test-space",
                "username": "testuser",
            },
            headers={"X-Forwarded-User": "testuser"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("webhook_subscription_id") is None

    @patch("gbserver.api.builds.get_admin_storage")
    def test_submit_with_webhook_url_but_webhooks_disabled(self, mock_admin):
        """When webhooks are disabled, no subscription is created even if URL is provided."""
        mock_space = MagicMock()
        mock_space.name = "test-space"
        mock_build_storage = MagicMock()
        mock_build_storage.add.return_value = "build-uuid-789"
        mock_space_storage = MagicMock()
        mock_space_storage.get_by_name.return_value = mock_space
        admin = MagicMock()
        admin.space_storage = mock_space_storage
        admin.build_storage = mock_build_storage
        mock_admin.return_value = admin

        with patch("gbserver.types.constants.GBSERVER_WEBHOOKS_ENABLED", False):
            response = self.client.post(
                "/",
                json={
                    "name": "test-build",
                    "build_archive": "base64data",
                    "space_name": "test-space",
                    "username": "testuser",
                    "webhook_url": "https://example.com/hook",
                },
                headers={"X-Forwarded-User": "testuser"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data.get("webhook_subscription_id") is None

    @patch("gbserver.api.builds.get_admin_storage")
    @patch("gbserver.webhooks.sql_storage.SQLWebhookStorage.add")
    def test_submit_webhook_failure_does_not_block_build(self, mock_ws_add, mock_admin):
        """If webhook subscription creation fails, the build is still created."""
        mock_space = MagicMock()
        mock_space.name = "test-space"
        mock_build_storage = MagicMock()
        mock_build_storage.add.return_value = "build-uuid-err"
        mock_space_storage = MagicMock()
        mock_space_storage.get_by_name.return_value = mock_space
        admin = MagicMock()
        admin.space_storage = mock_space_storage
        admin.build_storage = mock_build_storage
        mock_admin.return_value = admin
        mock_ws_add.side_effect = RuntimeError("DB connection lost")

        with patch("gbserver.types.constants.GBSERVER_WEBHOOKS_ENABLED", True):
            response = self.client.post(
                "/",
                json={
                    "name": "test-build",
                    "build_archive": "base64data",
                    "space_name": "test-space",
                    "username": "testuser",
                    "webhook_url": "https://example.com/hook",
                },
                headers={"X-Forwarded-User": "testuser"},
            )
            assert response.status_code == 200
            data = response.json()
            assert "build_id" in data
            # Subscription ID is None because creation failed
            assert data.get("webhook_subscription_id") is None

    @patch("gbserver.api.builds.get_admin_storage")
    @patch(
        "gbserver.webhooks.url_validator.validate_webhook_url",
        side_effect=__import__(
            "gbserver.webhooks.url_validator", fromlist=["WebhookURLError"]
        ).WebhookURLError("HTTPS required for webhook URLs"),
    )
    def test_submit_with_invalid_webhook_url_returns_no_subscription(
        self, mock_validate, mock_admin
    ):
        """When webhook URL validation fails, no subscription is created."""
        mock_space = MagicMock()
        mock_space.name = "test-space"
        mock_build_storage = MagicMock()
        mock_build_storage.add.return_value = "build-uuid-invalid-url"
        mock_space_storage = MagicMock()
        mock_space_storage.get_by_name.return_value = mock_space
        admin = MagicMock()
        admin.space_storage = mock_space_storage
        admin.build_storage = mock_build_storage
        mock_admin.return_value = admin

        with patch("gbserver.types.constants.GBSERVER_WEBHOOKS_ENABLED", True):
            response = self.client.post(
                "/",
                json={
                    "name": "test-build",
                    "build_archive": "base64data",
                    "space_name": "test-space",
                    "username": "testuser",
                    "webhook_url": "http://localhost/hook",
                },
                headers={"X-Forwarded-User": "testuser"},
            )
            assert response.status_code == 200
            data = response.json()
            assert "build_id" in data
            # Subscription ID is None because URL validation failed
            assert data.get("webhook_subscription_id") is None

    def test_build_submit_request_model_accepts_webhook_fields(self):
        """BuildSubmitRequest model accepts webhook fields."""
        req = BuildSubmitRequest(
            name="test",
            build_archive="data",
            space_name="space",
            username="user",
            webhook_url="https://example.com/hook",
            webhook_secret="secret123",
            webhook_event_types=["build.status_changed"],
            webhook_frequency=60,
        )
        assert req.webhook_url == "https://example.com/hook"
        assert req.webhook_secret == "secret123"
        assert req.webhook_event_types == ["build.status_changed"]
        assert req.webhook_frequency == 60

    def test_build_submit_request_model_webhook_fields_optional(self):
        """BuildSubmitRequest model works without webhook fields."""
        req = BuildSubmitRequest(
            name="test",
            build_archive="data",
            space_name="space",
            username="user",
        )
        assert req.webhook_url is None
        assert req.webhook_secret is None
        assert req.webhook_event_types is None
        assert req.webhook_frequency is None

    def test_build_submit_response_model_includes_webhook_subscription_id(self):
        """BuildSubmitResponse includes optional webhook_subscription_id."""
        resp = BuildSubmitResponse(build_id="abc-123")
        assert resp.webhook_subscription_id is None

        resp_with = BuildSubmitResponse(
            build_id="abc-123", webhook_subscription_id="sub-456"
        )
        assert resp_with.webhook_subscription_id == "sub-456"
