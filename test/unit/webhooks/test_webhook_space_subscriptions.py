"""Tests for space-wide webhook subscriptions.

Covers the API endpoints for creating and listing space-wide subscriptions
using the consolidated APIRouter.
"""

from unittest.mock import MagicMock, patch

import pytest
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


class TestSpaceWideSubscriptionAPI:
    """Tests for the space-wide webhook subscription API routes."""

    def setup_method(self):
        """Create a fresh test client for each test."""
        self.client = TestClient(_make_test_app())

    @patch("gbserver.api.webhooks.validate_webhook_url")
    @patch("gbserver.api.webhooks._get_storage")
    @patch("gbserver.api.webhooks.get_admin_storage")
    def test_create_space_subscription(
        self, mock_admin, mock_get_storage, mock_validate
    ):
        """POST to /spaces/{name}/subscriptions returns 201 with build_filter=None."""
        mock_validate.return_value = None
        mock_storage = MagicMock()
        mock_storage.get_by_space.return_value = []
        mock_get_storage.return_value = mock_storage

        # Mock space exists
        mock_space_storage = MagicMock()
        mock_space_storage.get_by_where.return_value = [MagicMock()]
        admin = MagicMock()
        admin.space_storage = mock_space_storage
        mock_admin.return_value = admin

        response = self.client.post(
            "/api/v1/webhooks/spaces/my-space/subscriptions",
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
        assert data["build_filter"] is None
        assert data["webhook_url"] == "https://dashboard.example.com/hooks"
        assert data["active"] is True
        assert data["status"] == "pending"
        assert data["created_by"] == "dashboard-service"
        # Secret must NEVER be returned
        assert "secret" not in data
        # Verify storage.add was called
        mock_storage.add.assert_called_once()

    @patch("gbserver.api.webhooks.validate_webhook_url")
    @patch("gbserver.api.webhooks._get_storage")
    @patch("gbserver.api.webhooks.get_admin_storage")
    def test_create_space_subscription_space_not_found(
        self, mock_admin, mock_get_storage, mock_validate
    ):
        """POST for nonexistent space returns 404."""
        mock_validate.return_value = None
        mock_space_storage = MagicMock()
        mock_space_storage.get_by_where.return_value = []
        admin = MagicMock()
        admin.space_storage = mock_space_storage
        mock_admin.return_value = admin

        response = self.client.post(
            "/api/v1/webhooks/spaces/nonexistent/subscriptions",
            json={
                "webhook_url": "https://x.com/hook",
                "secret": "test-secret-key",
                "event_types": ["*"],
            },
            headers={"X-Forwarded-User": "user"},
        )

        assert response.status_code == 404
        assert "nonexistent" in response.json()["detail"]

    @patch("gbserver.api.webhooks._get_storage")
    @patch("gbserver.api.webhooks.get_admin_storage")
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
            "/api/v1/webhooks/spaces/my-space/subscriptions",
            json={
                "webhook_url": "https://example.com/hook",
                "secret": "test-secret-key",
                "frequency": 5,
            },
            headers={"X-Forwarded-User": "testuser"},
        )

        assert response.status_code == 400
        assert "minimum" in response.json()["detail"].lower()

    def test_create_space_subscription_no_auth(self):
        """POST without auth header returns 401."""
        response = self.client.post(
            "/api/v1/webhooks/spaces/my-space/subscriptions",
            json={
                "webhook_url": "https://example.com/hook",
                "secret": "s",
                "event_types": ["*"],
            },
        )

        assert response.status_code == 401

    @patch("gbserver.api.webhooks._get_storage")
    def test_list_space_subscriptions(self, mock_get_storage):
        """GET returns list of active space-wide subscriptions."""
        mock_storage = MagicMock()
        sub = StoredWebhookSubscription(
            space_name="my-space",
            webhook_url="https://dashboard.example.com/hooks",
            secret="s",
            event_types=["*"],
            created_by="dashboard-service",
        )
        mock_storage.get_active_for_space.return_value = [sub]
        mock_get_storage.return_value = mock_storage

        response = self.client.get(
            "/api/v1/webhooks/spaces/my-space/subscriptions",
            headers={"X-Forwarded-User": "user"},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["subscriptions"]) == 1
        assert data["subscriptions"][0]["build_filter"] is None
        assert data["subscriptions"][0]["space_name"] == "my-space"
        assert data["subscriptions"][0]["webhook_url"] == (
            "https://dashboard.example.com/hooks"
        )
        # Secret must NEVER be returned
        assert "secret" not in data["subscriptions"][0]

    @patch("gbserver.api.webhooks._get_storage")
    def test_list_space_subscriptions_empty(self, mock_get_storage):
        """GET returns empty list when no space-wide subscriptions exist."""
        mock_storage = MagicMock()
        mock_storage.get_active_for_space.return_value = []
        mock_get_storage.return_value = mock_storage

        response = self.client.get(
            "/api/v1/webhooks/spaces/my-space/subscriptions",
            headers={"X-Forwarded-User": "user"},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["subscriptions"]) == 0

    def test_list_space_subscriptions_no_auth(self):
        """GET without auth header returns 401."""
        response = self.client.get(
            "/api/v1/webhooks/spaces/my-space/subscriptions"
        )

        assert response.status_code == 401
