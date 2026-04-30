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

import os
from unittest.mock import patch

import pytest
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from gbserver.api.auth import AuthMiddleware


def _make_app() -> FastAPI:
    """Build a minimal FastAPI app with AuthMiddleware and a /test endpoint."""
    app = FastAPI()
    app.add_middleware(AuthMiddleware)

    @app.get("/test")
    async def test_endpoint(request: Request):
        user = request.state.data["user"]
        return JSONResponse(content={"login": user.login})

    @app.get("/docs")
    async def docs_endpoint():
        return JSONResponse(content={"docs": True})

    @app.get("/openapi.json")
    async def openapi_endpoint():
        return JSONResponse(content={"openapi": "3.0.0"})

    return app


class TestAuthMiddlewareApiKeyMode:
    """Tests for the apikey auth mode in AuthMiddleware."""

    def test_valid_api_key_authenticates(self):
        """Valid API key should authenticate and return the default user login."""
        env = {
            "GBSERVER_AUTH_MODE": "apikey",
            "GBSERVER_API_KEY": "test-key-123",
        }
        with patch.dict(os.environ, env, clear=False):
            app = _make_app()
            client = TestClient(app)
            response = client.get("/test", headers={"Authorization": "Bearer test-key-123"})
        assert response.status_code == 200
        assert response.json()["login"] == "standalone"

    def test_wrong_api_key_returns_401(self):
        """Wrong API key should return 401 Unauthorized."""
        env = {
            "GBSERVER_AUTH_MODE": "apikey",
            "GBSERVER_API_KEY": "test-key-123",
        }
        with patch.dict(os.environ, env, clear=False):
            app = _make_app()
            client = TestClient(app)
            response = client.get("/test", headers={"Authorization": "Bearer wrong-key"})
        assert response.status_code == 401

    def test_custom_api_user(self):
        """Custom GBSERVER_API_USER should be used as the login."""
        env = {
            "GBSERVER_AUTH_MODE": "apikey",
            "GBSERVER_API_KEY": "test-key-123",
            "GBSERVER_API_USER": "myuser",
        }
        with patch.dict(os.environ, env, clear=False):
            app = _make_app()
            client = TestClient(app)
            response = client.get("/test", headers={"Authorization": "Bearer test-key-123"})
        assert response.status_code == 200
        assert response.json()["login"] == "myuser"

    def test_no_api_key_allows_localhost(self):
        """When no API key is set, localhost requests should be allowed."""
        env = {
            "GBSERVER_AUTH_MODE": "apikey",
            "GBSERVER_API_KEY": "",
        }
        with patch.dict(os.environ, env, clear=False):
            app = _make_app()
            client = TestClient(app)
            # TestClient sends from "testclient" which is in the localhost allow list
            response = client.get("/test")
        assert response.status_code == 200
        assert response.json()["login"] == "standalone"

    def test_is_localhost_rejects_non_localhost_ip(self):
        """_is_localhost returns False for non-localhost IPs."""
        from gbserver.api.auth import _is_localhost

        class FakeClient:
            host = "192.168.1.100"

        class FakeRequest:
            client = FakeClient()

        assert _is_localhost(FakeRequest()) is False

    def test_is_localhost_returns_false_when_client_is_none(self):
        """_is_localhost returns False when request.client is None."""
        from gbserver.api.auth import _is_localhost

        class FakeRequest:
            client = None

        assert _is_localhost(FakeRequest()) is False

    def test_docs_endpoint_always_allowed(self):
        """The /docs endpoint should not require authentication."""
        env = {
            "GBSERVER_AUTH_MODE": "apikey",
            "GBSERVER_API_KEY": "test-key-123",
        }
        with patch.dict(os.environ, env, clear=False):
            app = _make_app()
            client = TestClient(app)
            response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_json_always_allowed(self):
        """The /openapi.json endpoint should not require authentication."""
        env = {
            "GBSERVER_AUTH_MODE": "apikey",
            "GBSERVER_API_KEY": "test-key-123",
        }
        with patch.dict(os.environ, env, clear=False):
            app = _make_app()
            client = TestClient(app)
            response = client.get("/openapi.json")
        assert response.status_code == 200

    def test_missing_auth_header_with_api_key_set_returns_401(self):
        """Missing auth header should return 401 when API key is required."""
        env = {
            "GBSERVER_AUTH_MODE": "apikey",
            "GBSERVER_API_KEY": "test-key-123",
        }
        with patch.dict(os.environ, env, clear=False):
            app = _make_app()
            client = TestClient(app)
            response = client.get("/test")
        assert response.status_code == 401

    def test_localhost_without_auth_header_returns_401_when_api_key_set(self):
        """Localhost requests must still provide the API key when one is configured."""
        env = {
            "GBSERVER_AUTH_MODE": "apikey",
            "GBSERVER_API_KEY": "test-key-123",
        }
        with patch.dict(os.environ, env, clear=False):
            app = _make_app()
            # TestClient sends from "testclient" (in localhost allow list)
            client = TestClient(app)
            response = client.get("/test")  # no Authorization header
        assert response.status_code == 401
