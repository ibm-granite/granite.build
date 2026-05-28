"""Unit tests for webhook subscription REST API endpoints.

Tests the APIRouter-based routes for creating, listing, and deleting webhook
subscriptions, including authentication, authorization, and validation.
"""

from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from gbserver.api.webhooks import webhooks_router
from gbserver.webhooks.models import StoredWebhookSubscription
from gbserver.webhooks.url_validator import WebhookURLError


class _FakeAuthMiddleware(BaseHTTPMiddleware):
    """Middleware that mimics AuthMiddleware by reading X-Forwarded-User."""

    async def dispatch(self, request, call_next):
        from gbserver.types.auth import User

        username = request.headers.get("X-Forwarded-User")
        if not username:
            from fastapi.responses import JSONResponse

            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
        user = User(
            login=username,
            id=0,
            url="",
            html_url="",
            name=username,
            email=f"{username}@test",
        )
        request.state.data = {"user": user}
        return await call_next(request)


def _make_test_app():
    """Create a minimal test app with the webhooks router."""
    app = FastAPI()
    app.add_middleware(_FakeAuthMiddleware)
    app.include_router(webhooks_router, prefix="/api/v1")
    return app


class TestWebhookAPI:
    """Tests for webhook subscription API endpoints."""

    def setup_method(self):
        """Create a fresh test client for each test."""
        self.client = TestClient(_make_test_app())

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

    @patch("gbserver.api.webhooks.validate_webhook_url")
    @patch("gbserver.api.webhooks._get_storage")
    @patch("gbserver.api.webhooks.get_admin_storage")
    def test_create_subscription_with_build_filter(
        self, mock_admin, mock_get_storage, mock_validate
    ):
        """POST with build_filter in body returns 201 with build_filter set."""
        mock_validate.return_value = None
        # Mock space exists
        mock_space_storage = MagicMock()
        mock_space_storage.get_by_where.return_value = [MagicMock()]
        # Mock build exists
        mock_build = MagicMock()
        mock_build.space_name = "test-space"
        admin = MagicMock()
        admin.space_storage = mock_space_storage
        admin.build_storage.get_by_uuid.return_value = mock_build
        mock_admin.return_value = admin

        # Mock webhook storage add
        mock_storage = MagicMock()
        mock_storage.get_by_space.return_value = []
        mock_get_storage.return_value = mock_storage

        response = self.client.post(
            "/api/v1/webhooks/spaces/test-space/subscriptions",
            json={
                "webhook_url": "https://example.com/hook",
                "secret": "test-secret-key",
                "event_types": ["build.started", "build.completed"],
                "frequency": 30,
                "build_filter": "build-001",
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

    @patch("gbserver.api.webhooks._get_storage")
    @patch("gbserver.api.webhooks.get_admin_storage")
    def test_create_subscription_enforces_min_frequency(
        self, mock_admin, mock_get_storage
    ):
        """POST with frequency below minimum returns 400."""
        mock_space_storage = MagicMock()
        mock_space_storage.get_by_where.return_value = [MagicMock()]
        admin = MagicMock()
        admin.space_storage = mock_space_storage
        mock_admin.return_value = admin

        response = self.client.post(
            "/api/v1/webhooks/spaces/test-space/subscriptions",
            json={
                "webhook_url": "https://example.com/hook",
                "secret": "test-secret-key",
                "frequency": 5,
            },
            headers={"X-Forwarded-User": "testuser"},
        )

        assert response.status_code == 400
        assert "minimum" in response.json()["detail"].lower()

    @patch("gbserver.api.webhooks.validate_webhook_url")
    @patch("gbserver.api.webhooks._get_storage")
    @patch("gbserver.api.webhooks.get_admin_storage")
    def test_create_subscription_build_not_found(
        self, mock_admin, mock_get_storage, mock_validate
    ):
        """POST with build_filter for nonexistent build returns 404."""
        mock_validate.return_value = None
        mock_space_storage = MagicMock()
        mock_space_storage.get_by_where.return_value = [MagicMock()]
        admin = MagicMock()
        admin.space_storage = mock_space_storage
        admin.build_storage.get_by_uuid.return_value = None
        mock_admin.return_value = admin

        response = self.client.post(
            "/api/v1/webhooks/spaces/test-space/subscriptions",
            json={
                "webhook_url": "https://example.com/hook",
                "secret": "test-secret-key",
                "build_filter": "build-001",
            },
            headers={"X-Forwarded-User": "testuser"},
        )

        assert response.status_code == 404

    @patch("gbserver.api.webhooks._get_storage")
    def test_list_subscriptions_with_build_filter(self, mock_get_storage):
        """GET with build_filter query param returns matching subscriptions."""
        sub = self._make_subscription()
        mock_storage = MagicMock()
        mock_storage.get_active_for_build_filter.return_value = [sub]
        mock_get_storage.return_value = mock_storage

        response = self.client.get(
            "/api/v1/webhooks/spaces/test-space/subscriptions?build_filter=build-001",
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

    @patch("gbserver.api.webhooks._get_storage")
    def test_delete_subscription(self, mock_get_storage):
        """DELETE by owner returns 204."""
        sub = self._make_subscription(created_by="testuser")
        mock_storage = MagicMock()
        mock_storage.get_by_uuid.return_value = sub
        mock_get_storage.return_value = mock_storage

        response = self.client.delete(
            f"/api/v1/webhooks/{sub.uuid}",
            headers={"X-Forwarded-User": "testuser"},
        )

        assert response.status_code == 204
        mock_storage.deactivate.assert_called_once_with(sub.uuid)

    @patch("gbserver.api.webhooks._get_storage")
    def test_delete_subscription_forbidden(self, mock_get_storage):
        """DELETE by non-owner returns 403."""
        sub = self._make_subscription(created_by="owner-user")
        mock_storage = MagicMock()
        mock_storage.get_by_uuid.return_value = sub
        mock_get_storage.return_value = mock_storage

        response = self.client.delete(
            f"/api/v1/webhooks/{sub.uuid}",
            headers={"X-Forwarded-User": "other-user"},
        )

        assert response.status_code == 403
        mock_storage.deactivate.assert_not_called()

    @patch("gbserver.api.webhooks._get_storage")
    def test_delete_subscription_not_found(self, mock_get_storage):
        """DELETE for nonexistent subscription returns 404."""
        mock_storage = MagicMock()
        mock_storage.get_by_uuid.return_value = None
        mock_get_storage.return_value = mock_storage

        response = self.client.delete(
            "/api/v1/webhooks/nonexistent-id",
            headers={"X-Forwarded-User": "testuser"},
        )

        assert response.status_code == 404

    def test_create_no_auth_header(self):
        """POST without X-Forwarded-User header returns 401."""
        response = self.client.post(
            "/api/v1/webhooks/spaces/test-space/subscriptions",
            json={
                "webhook_url": "https://example.com/hook",
                "secret": "test-secret-key",
            },
        )

        assert response.status_code == 401

    def test_list_no_auth_header(self):
        """GET without X-Forwarded-User header returns 401."""
        response = self.client.get(
            "/api/v1/webhooks/spaces/test-space/subscriptions"
        )

        assert response.status_code == 401

    def test_delete_no_auth_header(self):
        """DELETE without X-Forwarded-User header returns 401."""
        response = self.client.delete("/api/v1/webhooks/some-webhook-id")

        assert response.status_code == 401

    @patch("gbserver.api.webhooks.validate_webhook_url")
    @patch("gbserver.api.webhooks._get_storage")
    @patch("gbserver.api.webhooks.get_admin_storage")
    def test_create_subscription_default_event_types(
        self, mock_admin, mock_get_storage, mock_validate
    ):
        """POST without event_types defaults to wildcard."""
        mock_validate.return_value = None
        mock_space_storage = MagicMock()
        mock_space_storage.get_by_where.return_value = [MagicMock()]
        admin = MagicMock()
        admin.space_storage = mock_space_storage
        mock_admin.return_value = admin

        mock_storage = MagicMock()
        mock_storage.get_by_space.return_value = []
        mock_get_storage.return_value = mock_storage

        response = self.client.post(
            "/api/v1/webhooks/spaces/test-space/subscriptions",
            json={
                "webhook_url": "https://example.com/hook",
                "secret": "test-secret-key",
            },
            headers={"X-Forwarded-User": "testuser"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["event_types"] == ["*"]

    @patch("gbserver.api.webhooks.validate_webhook_url")
    @patch("gbserver.api.webhooks._get_storage")
    @patch("gbserver.api.webhooks.get_admin_storage")
    def test_create_subscription_with_log_pattern(
        self, mock_admin, mock_get_storage, mock_validate
    ):
        """POST with log_pattern stores and returns it."""
        mock_validate.return_value = None
        mock_space_storage = MagicMock()
        mock_space_storage.get_by_where.return_value = [MagicMock()]
        admin = MagicMock()
        admin.space_storage = mock_space_storage
        mock_admin.return_value = admin

        mock_storage = MagicMock()
        mock_storage.get_by_space.return_value = []
        mock_get_storage.return_value = mock_storage

        response = self.client.post(
            "/api/v1/webhooks/spaces/test-space/subscriptions",
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

    @patch("gbserver.api.webhooks._get_storage")
    @patch("gbserver.api.webhooks.get_admin_storage")
    @patch("gbserver.api.webhooks.validate_webhook_url")
    def test_create_subscription_rejects_private_ip(
        self, mock_validate, mock_admin, mock_get_storage
    ):
        """POST with private IP webhook URL returns 400."""
        mock_validate.side_effect = WebhookURLError(
            "Webhook URL blocked (private/blocked IP): 10.0.0.1"
        )
        mock_space_storage = MagicMock()
        mock_space_storage.get_by_where.return_value = [MagicMock()]
        admin = MagicMock()
        admin.space_storage = mock_space_storage
        mock_admin.return_value = admin

        response = self.client.post(
            "/api/v1/webhooks/spaces/test-space/subscriptions",
            json={
                "webhook_url": "https://10.0.0.1/hook",
                "secret": "test-secret-key",
            },
            headers={"X-Forwarded-User": "testuser"},
        )

        assert response.status_code == 400
        assert "Invalid webhook URL" in response.json()["detail"]

    @patch("gbserver.api.webhooks._get_storage")
    @patch("gbserver.api.webhooks.get_admin_storage")
    @patch("gbserver.api.webhooks.validate_webhook_url")
    def test_create_subscription_rate_limited(
        self, mock_validate, mock_admin, mock_get_storage
    ):
        """POST when space has max subscriptions returns 429."""
        mock_validate.return_value = None  # URL is valid
        mock_space_storage = MagicMock()
        mock_space_storage.get_by_where.return_value = [MagicMock()]
        admin = MagicMock()
        admin.space_storage = mock_space_storage
        mock_admin.return_value = admin

        # Mock storage with max subscriptions already
        mock_storage = MagicMock()
        mock_storage.get_by_space.return_value = [
            MagicMock(active=True) for _ in range(20)
        ]
        mock_get_storage.return_value = mock_storage

        response = self.client.post(
            "/api/v1/webhooks/spaces/test-space/subscriptions",
            json={
                "webhook_url": "https://example.com/hook",
                "secret": "test-secret-key",
            },
            headers={"X-Forwarded-User": "testuser"},
        )

        assert response.status_code == 429
