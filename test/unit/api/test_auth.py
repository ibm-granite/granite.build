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

from gbserver.api.auth import AuthMiddleware, _is_public_path


def _make_app() -> FastAPI:
    """Build a minimal FastAPI app with AuthMiddleware and a /test endpoint.

    AuthMiddleware authenticates everything except an explicit allow-list
    (see _is_public_path in gbserver.api.auth) — /test deliberately isn't on
    that list, and doesn't need to live under /api/ to be protected: auth is
    the default everywhere now, not just under /api/.
    """
    app = FastAPI()
    app.add_middleware(AuthMiddleware)

    @app.get("/api/test")
    async def test_endpoint(request: Request):
        user = request.state.data["user"]
        return JSONResponse(content={"login": user.login})

    @app.get("/docs")
    async def docs_endpoint():
        return JSONResponse(content={"docs": True})

    @app.get("/openapi.json")
    async def openapi_endpoint():
        return JSONResponse(content={"openapi": "3.0.0"})

    @app.get("/redoc")
    async def redoc_endpoint():
        return JSONResponse(content={"redoc": True})

    @app.get("/docs/oauth2-redirect")
    async def oauth2_redirect_endpoint():
        return JSONResponse(content={"oauth2_redirect": True})

    @app.get("/")
    async def root_endpoint():
        return JSONResponse(content={"root": True})

    @app.get("/dashboard")
    async def frontend_page_endpoint():
        return JSONResponse(content={"page": True})

    @app.post("/dashboard")
    async def frontend_page_post_endpoint(request: Request):
        user = request.state.data["user"]
        return JSONResponse(content={"login": user.login})

    @app.get("/some/unregistered/path")
    async def unregistered_endpoint(request: Request):
        user = request.state.data["user"]
        return JSONResponse(content={"login": user.login})

    # Simulate the doc endpoints a mounted sub-app owns (see
    # openapi_security.enable_api) — e.g. builds_api mounted at /api/v1/builds.
    @app.get("/api/v1/builds/docs")
    async def subapp_docs_endpoint():
        return JSONResponse(content={"docs": True})

    @app.get("/api/v1/builds/openapi.json")
    async def subapp_openapi_endpoint():
        return JSONResponse(content={"openapi": "3.0.0"})

    @app.get("/api/v1/builds/redoc")
    async def subapp_redoc_endpoint():
        return JSONResponse(content={"redoc": True})

    @app.get("/api/v1/builds/docs/oauth2-redirect")
    async def subapp_oauth2_redirect_endpoint():
        return JSONResponse(content={"oauth2_redirect": True})

    # A regular sub-app data endpoint that must stay protected.
    @app.get("/api/v1/builds")
    async def subapp_data_endpoint(request: Request):
        user = request.state.data["user"]
        return JSONResponse(content={"login": user.login})

    # A data route whose trailing segment is a user-controlled path parameter,
    # mirroring GET /api/v1/secrets/user_secrets/{secret_name}. If the docs
    # exemption matched by suffix, a resource literally named "redoc"/"docs"/
    # "openapi.json" would slip past auth — this endpoint must stay protected.
    @app.get("/api/v1/secrets/user_secrets/{secret_name}")
    async def subapp_secret_endpoint(request: Request, secret_name: str):
        user = request.state.data["user"]
        return JSONResponse(content={"login": user.login, "secret": secret_name})

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
            response = client.get(
                "/api/test", headers={"Authorization": "Bearer test-key-123"}
            )
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
            response = client.get(
                "/api/test", headers={"Authorization": "Bearer wrong-key"}
            )
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
            response = client.get(
                "/api/test", headers={"Authorization": "Bearer test-key-123"}
            )
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
            response = client.get("/api/test")
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

    def test_redoc_always_allowed(self):
        """The /redoc endpoint should not require authentication.

        Missing from the allow-list before this fix — the original
        secure-by-default code only exempted /docs and /openapi.json."""
        env = {
            "GBSERVER_AUTH_MODE": "apikey",
            "GBSERVER_API_KEY": "test-key-123",
        }
        with patch.dict(os.environ, env, clear=False):
            app = _make_app()
            client = TestClient(app)
            response = client.get("/redoc")
        assert response.status_code == 200

    def test_docs_oauth2_redirect_always_allowed(self):
        """The /docs/oauth2-redirect endpoint should not require authentication."""
        env = {
            "GBSERVER_AUTH_MODE": "apikey",
            "GBSERVER_API_KEY": "test-key-123",
        }
        with patch.dict(os.environ, env, clear=False):
            app = _make_app()
            client = TestClient(app)
            response = client.get("/docs/oauth2-redirect")
        assert response.status_code == 200

    def test_root_path_always_allowed(self):
        """The root path (/) should not require authentication."""
        env = {
            "GBSERVER_AUTH_MODE": "apikey",
            "GBSERVER_API_KEY": "test-key-123",
        }
        with patch.dict(os.environ, env, clear=False):
            app = _make_app()
            client = TestClient(app)
            response = client.get("/")
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
            response = client.get("/api/test")
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
            response = client.get("/api/test")  # no Authorization header
        assert response.status_code == 401

    def test_analytics_path_requires_api_key(self):
        """/api/analytics/* must be authenticated like any other /api/ path —
        it must NOT be bypassed. These routes are included directly into
        root_api (see gbserver/api/root_api.py), so they get no special
        treatment from AuthMiddleware."""
        env = {
            "GBSERVER_AUTH_MODE": "apikey",
            "GBSERVER_API_KEY": "test-key-123",
        }
        with patch.dict(os.environ, env, clear=False):
            app = _make_app()
            client = TestClient(app)
            response = client.get("/api/analytics/builds/failure-trends/history")
        assert response.status_code == 401

    def test_known_frontend_path_bypasses_auth_even_with_api_key_set(self):
        """Known frontend page paths (/dashboard and friends) are public in
        every auth mode — the client needs to load the page before it has a
        token, and the API key requirement must not apply to them. This is
        an explicit allow-list entry, not "everything outside /api/" — see
        test_unlisted_non_api_path_requires_auth below for the contrast."""
        env = {
            "GBSERVER_AUTH_MODE": "apikey",
            "GBSERVER_API_KEY": "test-key-123",
        }
        with patch.dict(os.environ, env, clear=False):
            app = _make_app()
            client = TestClient(app)
            response = client.get("/dashboard")  # no Authorization header
        assert response.status_code == 200
        assert response.json() == {"page": True}

    def test_unlisted_non_api_path_requires_auth(self):
        """A non-/api/ path that ISN'T on the allow-list must still require
        auth — this is the regression test for the deny-by-default fix.
        Under the old "not /api/" bypass, this would have been public."""
        env = {
            "GBSERVER_AUTH_MODE": "apikey",
            "GBSERVER_API_KEY": "test-key-123",
        }
        with patch.dict(os.environ, env, clear=False):
            app = _make_app()
            client = TestClient(app)
            response = client.get("/some/unregistered/path")
        assert response.status_code == 401

    @pytest.mark.parametrize(
        "path",
        [
            "/api/v1/builds/docs",
            "/api/v1/builds/openapi.json",
            "/api/v1/builds/redoc",
            "/api/v1/builds/docs/oauth2-redirect",
        ],
    )
    def test_subapp_docs_paths_always_allowed(self, path):
        """A mounted sub-app's own Swagger/OpenAPI doc endpoints (e.g.
        /api/v1/builds/docs) must be public in every auth mode, just like the
        top-level /docs. This is the regression fix — only the top-level docs
        were exempted before, so sub-path docs started demanding a token."""
        env = {
            "GBSERVER_AUTH_MODE": "apikey",
            "GBSERVER_API_KEY": "test-key-123",
        }
        with patch.dict(os.environ, env, clear=False):
            app = _make_app()
            client = TestClient(app)
            response = client.get(path)  # no Authorization header
        assert response.status_code == 200

    def test_subapp_non_docs_api_path_requires_auth(self):
        """The docs exemption must not leak to real data routes —
        /api/v1/builds (not a docs path) still requires a token."""
        env = {
            "GBSERVER_AUTH_MODE": "apikey",
            "GBSERVER_API_KEY": "test-key-123",
        }
        with patch.dict(os.environ, env, clear=False):
            app = _make_app()
            client = TestClient(app)
            response = client.get("/api/v1/builds")  # no Authorization header
        assert response.status_code == 401

    @pytest.mark.parametrize("name", ["docs", "redoc", "openapi.json"])
    def test_data_route_ending_in_doc_name_requires_auth(self, name):
        """A resource whose name equals a doc endpoint (e.g. a secret named
        "redoc") must NOT bypass auth. The docs exemption is anchored to a
        single mount segment before the doc name — /api/v1/secrets/user_secrets/
        redoc has an extra segment, so it stays protected. Regression guard for
        the suffix-match auth bypass."""
        env = {
            "GBSERVER_AUTH_MODE": "apikey",
            "GBSERVER_API_KEY": "test-key-123",
        }
        with patch.dict(os.environ, env, clear=False):
            app = _make_app()
            client = TestClient(app)
            # no Authorization header
            response = client.get(f"/api/v1/secrets/user_secrets/{name}")
        assert response.status_code == 401

    def test_post_to_public_frontend_path_requires_auth(self):
        """A non-GET/HEAD request to an otherwise-public path must still
        require auth — the allow-list is GET/HEAD-only by design, so a
        mutating endpoint accidentally registered under a public-looking
        prefix (e.g. POST /dashboard/...) doesn't silently become public."""
        env = {
            "GBSERVER_AUTH_MODE": "apikey",
            "GBSERVER_API_KEY": "test-key-123",
        }
        with patch.dict(os.environ, env, clear=False):
            app = _make_app()
            client = TestClient(app)
            response = client.post("/dashboard")
        assert response.status_code == 401


def test_autotunex_proxy_prefix_is_public():
    assert _is_public_path("/api/autotunex") is True
    assert _is_public_path("/api/autotunex/job/by_build_id/abc") is True


def test_other_api_paths_still_require_auth():
    assert _is_public_path("/api/v1/builds") is False
    assert _is_public_path("/api/autotunexxx") is False


def test_autotunex_proxy_bypasses_auth_for_non_get_methods():
    """The AutoTuneX proxy exemption must be method-agnostic, so gbserver does
    not gate POST (or any other verb) before forwarding to the proxy.

    This asserts the exemption is wired as designed; it is NOT a claim that the
    request ends up authenticated. AutoTuneX defaults to
    auth_providers=["disabled"], so the 200 below is an unauthenticated write
    reaching the proxy — safe only in the localhost-only standalone deployment
    documented on auth._PUBLIC_PATH_PREFIXES. Regression guard for the
    GET/HEAD-only bypass in dispatch: without the `_is_autotunex_proxy`
    carve-out, this POST would 401 despite /api/autotunex being in
    _PUBLIC_PATH_PREFIXES. Control:
    POST /api/v1/thing (a non-public route) must still 401, proving auth is
    otherwise enforced and it's specifically the autotunex prefix that's
    exempt."""
    app = FastAPI()
    app.add_middleware(AuthMiddleware)

    @app.post("/api/autotunex/{path:path}")
    async def autotunex_proxy_endpoint(path: str):
        return JSONResponse(content={"path": path})

    @app.post("/api/v1/thing")
    async def other_data_endpoint(request: Request):
        user = request.state.data["user"]
        return JSONResponse(content={"login": user.login})

    env = {
        "GBSERVER_AUTH_MODE": "apikey",
        "GBSERVER_API_KEY": "secret",
        # Fixed explicitly so this machine's local .env GBSERVER_API_USER
        # can't leak in and change the outcome.
        "GBSERVER_API_USER": "test-user",
    }
    with patch.dict(os.environ, env, clear=False):
        client = TestClient(app)
        # No Authorization header on either request.
        proxy_response = client.post("/api/autotunex/jobs")
        other_response = client.post("/api/v1/thing")

    assert proxy_response.status_code == 200
    assert other_response.status_code == 401
