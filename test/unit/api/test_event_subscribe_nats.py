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

"""Unit tests for the NATS subscribe endpoint in standalone mode."""

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from gbserver.api.event_subscribe import event_subscribe_router
from gbserver.storage.stored_build import StoredBuild
from gbserver.types.auth import User


def _make_fake_user(login: str = "testuser") -> User:
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
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request.state.data = {"user": _make_fake_user()}
        return await call_next(request)


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(event_subscribe_router, prefix="/api/v1")
    app.add_middleware(_FakeAuthMiddleware)
    return app


def _make_stored_build(build_id: str = "test-build-123") -> StoredBuild:
    return StoredBuild(
        uuid=build_id,
        name="test-build",
        space_name="test-space",
        source_uri="",
        username="testuser",
        build_archive="",
        status="submitted",
    )


class TestNATSSubscribeEndpoint:
    """Tests for NATS subscribe response in standalone mode."""

    @patch.dict(os.environ, {"GB_ENVIRONMENT": "STANDALONE"}, clear=False)
    @patch("gbserver.messaging.subscription_service.HAS_NATS", True)
    @patch("gbserver.api.event_subscribe.get_admin_storage")
    def test_returns_nats_delivery_type(self, mock_get_storage):
        """In standalone, returns delivery_type='nats' with url and subject."""
        build_id = "abc-123-def"
        mock_storage = MagicMock()
        mock_storage.build_storage.get_by_uuid.return_value = _make_stored_build(
            build_id
        )
        mock_get_storage.return_value = mock_storage

        os.environ.pop("RABBITMQ_HOST", None)

        app = _make_app()
        client = TestClient(app)
        response = client.post(
            f"/api/v1/builds/{build_id}/events/subscribe",
            headers={"Authorization": "Bearer valid-token"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["delivery_type"] == "nats"
        assert data["url"] == "nats://localhost:4222"
        assert data["subject"] == f"gbserver.build.{build_id}.>"
        assert data["username"] is None
        assert data["password"] is None
        assert data["exchange"] is None
        assert data["routing_key"] is None
        assert data["queue"] is None
        assert data["host"] == "localhost"
        assert data["port"] == 4222
