#!/usr/bin/env python3

# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the event subscribe endpoint."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Request, status
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from gbserver.api.event_subscribe import SubscribeResponse, event_subscribe_router
from gbserver.storage.stored_build import StoredBuild
from gbserver.types.auth import User


def _make_fake_user(login: str = "testuser") -> User:
    """Create a minimal User object for testing."""
    return User(
        login=login,
        id=1,
        url="",
        html_url="",
        name=login,
        email=f"{login}@test.com",
        auth_provider="apikey",
    )


class _FakeAuthMiddleware(BaseHTTPMiddleware):
    """Middleware that injects a fake user if Authorization header present."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer ") and len(auth) > 7:
            request.state.data = {"user": _make_fake_user()}
        else:
            request.state.data = {}
        return await call_next(request)


class _NoAuthMiddleware(BaseHTTPMiddleware):
    """Middleware that never sets a user (simulates missing auth)."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request.state.data = {}
        return await call_next(request)


def _make_app(middleware_cls=_FakeAuthMiddleware) -> FastAPI:
    """Build a minimal FastAPI app with the event_subscribe_router."""
    app = FastAPI()
    app.include_router(event_subscribe_router, prefix="/api/v1")
    app.add_middleware(middleware_cls)
    return app


def _make_stored_build(build_id: str = "test-build-123") -> StoredBuild:
    """Create a minimal StoredBuild for testing."""
    return StoredBuild(
        uuid=build_id,
        name="test-build",
        space_name="test-space",
        source_uri="",
        username="testuser",
        build_archive="",
        status="submitted",
    )


class TestEventSubscribeEndpoint:
    """Tests for POST /api/v1/builds/{build_id}/events/subscribe."""

    @patch.dict(
        os.environ, {"RABBITMQ_HOST": "rmq.example.com", "RABBITMQ_PORT": "5672"}
    )
    @patch("gbserver.api.event_subscribe.provision_subscription")
    @patch("gbserver.api.event_subscribe.get_admin_storage")
    def test_successful_subscription(self, mock_get_storage, mock_provision):
        """Valid auth + existing build returns scoped credentials."""
        build_id = "abc-123-def"
        mock_build = _make_stored_build(build_id)

        # Mock storage to return a build
        mock_storage = MagicMock()
        mock_storage.build_storage.get_by_uuid.return_value = mock_build
        mock_get_storage.return_value = mock_storage

        # Mock provision_subscription
        mock_provision.return_value = {
            "delivery_type": "rabbitmq",
            "host": "rmq.example.com",
            "port": 5672,
            "username": "tmp-build-abc-123d-xyzabc",
            "password": "secret-password-12345678",
            "exchange": "build-events",
            "routing_key": f"build.{build_id}.#",
            "queue": f"events.{build_id}.xyzabc",
            "expires_at": 1748869200,
        }

        app = _make_app()
        client = TestClient(app)
        response = client.post(
            f"/api/v1/builds/{build_id}/events/subscribe",
            headers={"Authorization": "Bearer valid-token"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["delivery_type"] == "rabbitmq"
        assert data["host"] == "rmq.example.com"
        assert data["port"] == 5672
        assert data["username"] == "tmp-build-abc-123d-xyzabc"
        assert data["password"] == "secret-password-12345678"
        assert data["exchange"] == "build-events"
        assert data["routing_key"] == f"build.{build_id}.#"
        assert data["queue"] == f"events.{build_id}.xyzabc"
        assert data["expires_at"] == 1748869200

        mock_provision.assert_called_once_with(build_id)

    def test_missing_auth_returns_401(self):
        """Request without Authorization header returns 401."""
        app = _make_app(middleware_cls=_NoAuthMiddleware)
        client = TestClient(app)
        response = client.post(
            "/api/v1/builds/some-build-id/events/subscribe",
        )

        assert response.status_code == 401
        assert "Authorization required" in response.json()["detail"]

    @patch("gbserver.api.event_subscribe.get_admin_storage")
    def test_build_not_found_returns_404(self, mock_get_storage):
        """Request for non-existent build returns 404."""
        mock_storage = MagicMock()
        mock_storage.build_storage.get_by_uuid.return_value = None
        mock_get_storage.return_value = mock_storage

        app = _make_app()
        client = TestClient(app)
        response = client.post(
            "/api/v1/builds/nonexistent-build/events/subscribe",
            headers={"Authorization": "Bearer valid-token"},
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
