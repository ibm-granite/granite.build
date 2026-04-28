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

import base64
import hashlib
import os
import time
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gbserver.api.auth_routes import (
    _AuthSession,
    _sessions,
    _verify_pkce_s256,
    auth_api,
)

pytestmark = pytest.mark.g4os


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ENV_DEFAULTS = {
    "GBSERVER_IBMID_CLIENT_ID": "test-client-id",
    "GBSERVER_IBMID_CLIENT_SECRET": "test-client-secret",
    "GBSERVER_IBMID_AUTHORIZE_URL": "https://login.ibm.com/v1.0/endpoint/default/authorize",
    "GBSERVER_IBMID_TOKEN_URL": "https://login.ibm.com/v1.0/endpoint/default/token",
    "GBSERVER_IBMID_USERINFO_URL": "https://login.ibm.com/v1.0/endpoint/default/userinfo",
    "GBSERVER_IBMID_CALLBACK_URL": "https://gbserver.example.com/api/v1/auth/callback",
}


def _make_verifier_and_challenge():
    verifier = "test-code-verifier-that-is-at-least-43-characters-long-xxxxxxxxx"
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


@pytest.fixture(autouse=True)
def _clear_sessions():
    """Ensure sessions are clean before and after each test."""
    _sessions.clear()
    yield
    _sessions.clear()


# ---------------------------------------------------------------------------
# PKCE verification
# ---------------------------------------------------------------------------


class TestPKCEVerification:
    def test_verify_pkce_s256_correct(self):
        verifier, challenge = _make_verifier_and_challenge()
        assert _verify_pkce_s256(verifier, challenge) is True

    def test_verify_pkce_s256_wrong_verifier(self):
        _, challenge = _make_verifier_and_challenge()
        assert _verify_pkce_s256("wrong-verifier", challenge) is False

    def test_verify_pkce_s256_wrong_challenge(self):
        verifier, _ = _make_verifier_and_challenge()
        assert _verify_pkce_s256(verifier, "wrong-challenge") is False


# ---------------------------------------------------------------------------
# /authorize endpoint
# ---------------------------------------------------------------------------


class TestAuthorizeEndpoint:
    def test_authorize_redirects_to_ibmid(self):
        client = TestClient(auth_api, follow_redirects=False)
        _, challenge = _make_verifier_and_challenge()

        with patch.dict(os.environ, ENV_DEFAULTS, clear=False):
            resp = client.get(
                "/authorize",
                params={
                    "code_challenge": challenge,
                    "code_challenge_method": "S256",
                    "state": "test-state-1",
                },
            )

        assert resp.status_code == 302
        location = resp.headers["location"]
        parsed = urlparse(location)
        assert parsed.hostname == "login.ibm.com"
        qs = parse_qs(parsed.query)
        assert qs["response_type"] == ["code"]
        assert qs["client_id"] == ["test-client-id"]
        assert qs["redirect_uri"] == [
            "https://gbserver.example.com/api/v1/auth/callback"
        ]
        assert qs["scope"] == ["openid profile email"]
        assert qs["state"] == ["test-state-1"]

    def test_authorize_creates_pending_session(self):
        client = TestClient(auth_api, follow_redirects=False)
        _, challenge = _make_verifier_and_challenge()

        with patch.dict(os.environ, ENV_DEFAULTS, clear=False):
            client.get(
                "/authorize",
                params={
                    "code_challenge": challenge,
                    "code_challenge_method": "S256",
                    "state": "test-state-2",
                },
            )

        assert "test-state-2" in _sessions
        assert _sessions["test-state-2"].status == "pending"
        assert _sessions["test-state-2"].code_challenge == challenge

    def test_authorize_rejects_non_s256_method(self):
        client = TestClient(auth_api)
        _, challenge = _make_verifier_and_challenge()

        with patch.dict(os.environ, ENV_DEFAULTS, clear=False):
            resp = client.get(
                "/authorize",
                params={
                    "code_challenge": challenge,
                    "code_challenge_method": "plain",
                    "state": "test-state-3",
                },
            )

        assert resp.status_code == 400
        assert "S256" in resp.json()["detail"]

    def test_authorize_rejects_duplicate_state(self):
        client = TestClient(auth_api, follow_redirects=False)
        _, challenge = _make_verifier_and_challenge()

        with patch.dict(os.environ, ENV_DEFAULTS, clear=False):
            client.get(
                "/authorize",
                params={
                    "code_challenge": challenge,
                    "code_challenge_method": "S256",
                    "state": "dup-state",
                },
            )
            resp = client.get(
                "/authorize",
                params={
                    "code_challenge": challenge,
                    "code_challenge_method": "S256",
                    "state": "dup-state",
                },
            )

        assert resp.status_code == 409

    def test_authorize_fails_without_client_id(self):
        client = TestClient(auth_api)
        _, challenge = _make_verifier_and_challenge()
        env = {**ENV_DEFAULTS, "GBSERVER_IBMID_CLIENT_ID": ""}

        with patch.dict(os.environ, env, clear=False):
            resp = client.get(
                "/authorize",
                params={
                    "code_challenge": challenge,
                    "code_challenge_method": "S256",
                    "state": "test-state-4",
                },
            )

        assert resp.status_code == 500
        assert "CLIENT_ID" in resp.json()["detail"]

    def test_authorize_fails_without_callback_url(self):
        client = TestClient(auth_api)
        _, challenge = _make_verifier_and_challenge()
        env = {**ENV_DEFAULTS, "GBSERVER_IBMID_CALLBACK_URL": ""}

        with patch.dict(os.environ, env, clear=False):
            resp = client.get(
                "/authorize",
                params={
                    "code_challenge": challenge,
                    "code_challenge_method": "S256",
                    "state": "test-state-5",
                },
            )

        assert resp.status_code == 500
        assert "CALLBACK_URL" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# /callback endpoint
# ---------------------------------------------------------------------------


class TestCallbackEndpoint:
    def _setup_pending_session(self, state="cb-state"):
        _, challenge = _make_verifier_and_challenge()
        _sessions[state] = _AuthSession(code_challenge=challenge)
        return challenge

    def test_callback_exchanges_code_and_completes_session(self):
        self._setup_pending_session("cb-state-1")
        client = TestClient(auth_api)

        token_resp = MagicMock()
        token_resp.status_code = 200
        token_resp.json.return_value = {
            "access_token": "at_test",
            "id_token": "idt_test",
            "refresh_token": "rt_test",
            "expires_in": 3600,
        }

        userinfo_resp = MagicMock()
        userinfo_resp.status_code = 200
        userinfo_resp.json.return_value = {
            "sub": "IBMid-12345",
            "name": "Test User",
            "email": "test@ibm.com",
            "preferred_username": "testuser",
        }
        userinfo_resp.raise_for_status = MagicMock()

        with patch.dict(os.environ, ENV_DEFAULTS, clear=False):
            with patch(
                "gbserver.api.auth_routes.requests.post", return_value=token_resp
            ):
                with patch(
                    "gbserver.api.auth_routes.requests.get", return_value=userinfo_resp
                ):
                    resp = client.get(
                        "/callback",
                        params={"state": "cb-state-1", "code": "auth-code-123"},
                    )

        assert resp.status_code == 200
        assert "IBMid login successful." in resp.text

        session = _sessions["cb-state-1"]
        assert session.status == "complete"
        assert session.access_token == "at_test"
        assert session.user_info["email"] == "test@ibm.com"

    def test_callback_unknown_state(self):
        client = TestClient(auth_api)

        with patch.dict(os.environ, ENV_DEFAULTS, clear=False):
            resp = client.get(
                "/callback",
                params={"state": "unknown-state", "code": "some-code"},
            )

        assert resp.status_code == 200
        assert "not found or expired" in resp.text

    def test_callback_with_error_param(self):
        self._setup_pending_session("cb-state-err")
        client = TestClient(auth_api)

        with patch.dict(os.environ, ENV_DEFAULTS, clear=False):
            resp = client.get(
                "/callback",
                params={
                    "state": "cb-state-err",
                    "error": "access_denied",
                    "error_description": "User denied access",
                },
            )

        assert resp.status_code == 200
        assert "Authentication failed" in resp.text
        assert _sessions["cb-state-err"].status == "error"

    def test_callback_token_exchange_failure(self):
        self._setup_pending_session("cb-state-fail")
        client = TestClient(auth_api)

        token_resp = MagicMock()
        token_resp.status_code = 400
        token_resp.json.return_value = {
            "error": "invalid_grant",
            "error_description": "Code expired",
        }
        token_resp.text = "Code expired"

        with patch.dict(os.environ, ENV_DEFAULTS, clear=False):
            with patch(
                "gbserver.api.auth_routes.requests.post", return_value=token_resp
            ):
                resp = client.get(
                    "/callback",
                    params={"state": "cb-state-fail", "code": "expired-code"},
                )

        assert resp.status_code == 200
        assert "failed" in resp.text.lower()
        assert _sessions["cb-state-fail"].status == "error"

    def test_callback_no_code_sets_error(self):
        self._setup_pending_session("cb-state-nocode")
        client = TestClient(auth_api)

        with patch.dict(os.environ, ENV_DEFAULTS, clear=False):
            resp = client.get("/callback", params={"state": "cb-state-nocode"})

        assert resp.status_code == 200
        assert _sessions["cb-state-nocode"].status == "error"


# ---------------------------------------------------------------------------
# /status endpoint
# ---------------------------------------------------------------------------


class TestStatusEndpoint:
    def test_status_pending(self):
        _, challenge = _make_verifier_and_challenge()
        _sessions["st-pending"] = _AuthSession(code_challenge=challenge)

        client = TestClient(auth_api)
        resp = client.get("/status", params={"state": "st-pending"})

        assert resp.status_code == 200
        assert resp.json()["status"] == "pending"

    def test_status_complete_with_valid_verifier(self):
        verifier, challenge = _make_verifier_and_challenge()
        session = _AuthSession(code_challenge=challenge)
        session.status = "complete"
        session.access_token = "at_test"
        session.id_token = "idt_test"
        session.refresh_token = "rt_test"
        session.expires_in = 3600
        session.user_info = {"sub": "IBMid-12345", "email": "test@ibm.com"}
        _sessions["st-complete"] = session

        client = TestClient(auth_api)
        resp = client.get(
            "/status",
            params={"state": "st-complete", "code_verifier": verifier},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "complete"
        assert data["access_token"] == "at_test"
        assert data["user_info"]["email"] == "test@ibm.com"

    def test_status_complete_deletes_session(self):
        verifier, challenge = _make_verifier_and_challenge()
        session = _AuthSession(code_challenge=challenge)
        session.status = "complete"
        session.access_token = "at"
        _sessions["st-delete"] = session

        client = TestClient(auth_api)
        client.get(
            "/status",
            params={"state": "st-delete", "code_verifier": verifier},
        )

        assert "st-delete" not in _sessions

    def test_status_complete_with_invalid_verifier(self):
        _, challenge = _make_verifier_and_challenge()
        session = _AuthSession(code_challenge=challenge)
        session.status = "complete"
        session.access_token = "at"
        _sessions["st-badpkce"] = session

        client = TestClient(auth_api)
        resp = client.get(
            "/status",
            params={"state": "st-badpkce", "code_verifier": "wrong-verifier"},
        )

        assert resp.status_code == 401
        assert "PKCE" in resp.json()["detail"]

    def test_status_complete_without_verifier(self):
        _, challenge = _make_verifier_and_challenge()
        session = _AuthSession(code_challenge=challenge)
        session.status = "complete"
        session.access_token = "at"
        _sessions["st-noverifier"] = session

        client = TestClient(auth_api)
        resp = client.get("/status", params={"state": "st-noverifier"})

        assert resp.status_code == 400
        assert "code_verifier" in resp.json()["detail"]

    def test_status_error(self):
        _, challenge = _make_verifier_and_challenge()
        session = _AuthSession(code_challenge=challenge)
        session.status = "error"
        session.error_detail = "access_denied"
        _sessions["st-error"] = session

        client = TestClient(auth_api)
        resp = client.get("/status", params={"state": "st-error"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"
        assert data["error"] == "access_denied"
        # Error session should be deleted
        assert "st-error" not in _sessions

    def test_status_not_found(self):
        client = TestClient(auth_api)
        resp = client.get("/status", params={"state": "nonexistent"})

        assert resp.status_code == 404

    def test_status_expired_session_cleaned_up(self):
        _, challenge = _make_verifier_and_challenge()
        session = _AuthSession(code_challenge=challenge)
        session.created_at = time.time() - 600  # 10 minutes ago
        _sessions["st-expired"] = session

        client = TestClient(auth_api)
        resp = client.get("/status", params={"state": "st-expired"})

        assert resp.status_code == 404
        assert "st-expired" not in _sessions


# ---------------------------------------------------------------------------
# Middleware exclusion
# ---------------------------------------------------------------------------


class TestAuthMiddlewareExclusion:
    def _make_app(self):
        from gbserver.api.auth import AuthMiddleware

        app = FastAPI()
        app.add_middleware(AuthMiddleware)
        app.mount("/api/v1/auth", auth_api)

        @app.get("/api/v1/protected")
        def protected():
            return {"ok": True}

        return app

    def test_auth_routes_accessible_without_bearer_token(self):
        app = self._make_app()
        client = TestClient(app, follow_redirects=False)
        _, challenge = _make_verifier_and_challenge()

        with patch.dict(
            os.environ, {**ENV_DEFAULTS, "GBSERVER_AUTH_MODE": "ibmid"}, clear=False
        ):
            resp = client.get(
                "/api/v1/auth/authorize",
                params={
                    "code_challenge": challenge,
                    "code_challenge_method": "S256",
                    "state": "mw-test-state",
                },
            )

        # Should get a redirect (302), not a 401
        assert resp.status_code == 302

    def test_protected_route_still_requires_auth(self):
        app = self._make_app()
        client = TestClient(app)

        with patch.dict(os.environ, {"GBSERVER_AUTH_MODE": "ibmid"}, clear=False):
            resp = client.get("/api/v1/protected")

        assert resp.status_code == 401
