#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""Tests for the stateless IBMid flow: /oidc/authorize, /oidc/remote-callback,
/oidc/token."""

import base64
import json
import os
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import gbserver.api.auth_routes as auth_routes
from gbserver.api.auth_routes import auth_api, encode_remote_code

ENV = {
    "GBSERVER_IBMID_CLIENT_ID": "test-client-id",
    "GBSERVER_IBMID_CLIENT_SECRET": "test-client-secret",
    "GBSERVER_IBMID_AUTHORIZE_URL": "https://login.ibm.com/v1.0/endpoint/default/authorize",
    "GBSERVER_IBMID_TOKEN_URL": "https://login.ibm.com/v1.0/endpoint/default/token",
    "GBSERVER_IBMID_USERINFO_URL": "https://login.ibm.com/v1.0/endpoint/default/userinfo",
    "GBSERVER_IBMID_CALLBACK_URL": "https://gbserver.example.com/api/v1/auth/callback",
}
REMOTE_CALLBACK = "https://gbserver.example.com/api/v1/auth/oidc/remote-callback"
LOOPBACK = "http://127.0.0.1:48731/callback"


@pytest.fixture(autouse=True)
def _env():
    with patch.dict(os.environ, ENV, clear=False):
        os.environ.pop("GBSERVER_IBMID_REMOTE_CALLBACK_URL", None)
        yield


@pytest.fixture(autouse=True)
def _no_legacy_state():
    """The stateless flow must never touch the legacy in-memory sessions."""
    auth_routes._sessions.clear()
    yield
    assert auth_routes._sessions == {}


def _client():
    return TestClient(auth_api, follow_redirects=False)


def _authorize(**overrides):
    params = {
        "code_challenge": "challenge-abc",
        "code_challenge_method": "S256",
        "state": "state-1",
        "redirect_uri": LOOPBACK,
    }
    params.update(overrides)
    params = {k: v for k, v in params.items() if v is not None}
    return _client().get("/oidc/authorize", params=params)


def _ibmid_token_ok():
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "access_token": "at_test",
        "id_token": "idt_test",
        "refresh_token": "rt_test",
        "expires_in": 3600,
    }
    return resp


def _ibmid_userinfo_ok():
    resp = MagicMock()
    resp.json.return_value = {"sub": "IBMid-1", "email": "test@ibm.com"}
    return resp


# ---------------------------------------------------------------------------
# /oidc/authorize
# ---------------------------------------------------------------------------


class TestOIDCAuthorize:
    def test_redirects_to_ibmid_with_pkce_passed_through(self):
        resp = _authorize()

        assert resp.status_code == 302
        loc = urlparse(resp.headers["location"])
        assert loc.hostname == "login.ibm.com"
        qs = parse_qs(loc.query)
        assert qs["client_id"] == ["test-client-id"]
        assert qs["redirect_uri"] == [LOOPBACK]
        assert qs["state"] == ["state-1"]
        assert qs["code_challenge"] == ["challenge-abc"]
        assert qs["code_challenge_method"] == ["S256"]
        assert qs["response_type"] == ["code"]
        assert "client_secret" not in qs

    def test_omitted_redirect_uri_uses_remote_callback(self):
        resp = _authorize(redirect_uri=None)

        assert resp.status_code == 302
        qs = parse_qs(urlparse(resp.headers["location"]).query)
        assert qs["redirect_uri"] == [REMOTE_CALLBACK]

    def test_explicit_remote_callback_env_wins(self):
        with patch.dict(
            os.environ,
            {"GBSERVER_IBMID_REMOTE_CALLBACK_URL": "https://other.example.com/rc"},
        ):
            resp = _authorize(redirect_uri=None)

        qs = parse_qs(urlparse(resp.headers["location"]).query)
        assert qs["redirect_uri"] == ["https://other.example.com/rc"]

    def test_remote_without_any_callback_config_is_a_config_error(self):
        with patch.dict(os.environ, {"GBSERVER_IBMID_CALLBACK_URL": ""}):
            resp = _authorize(redirect_uri=None)
        assert resp.status_code == 500
        assert "not configured" in resp.json()["detail"]

    @pytest.mark.parametrize(
        "redirect_uri",
        [
            "https://evil.example.com/callback",
            "http://127.0.0.1.evil.example.com:48731/callback",
            "http://127.0.0.1:48731/other",
            "http://127.0.0.1:48731/callback?x=1",
            "https://127.0.0.1:48731/callback",
            "http://localhost:48731/callback",
            # The legacy callback: its handler expects an in-memory session.
            "https://gbserver.example.com/api/v1/auth/callback",
        ],
    )
    def test_rejects_disallowed_redirect_uri(self, redirect_uri):
        resp = _authorize(redirect_uri=redirect_uri)
        assert resp.status_code == 400
        assert "redirect_uri" in resp.json()["detail"]

    def test_rejects_non_s256(self):
        resp = _authorize(code_challenge_method="plain")
        assert resp.status_code == 400

    def test_requires_code_challenge(self):
        resp = _authorize(code_challenge=None)
        assert resp.status_code == 422

    def test_fails_without_client_id(self):
        with patch.dict(os.environ, {"GBSERVER_IBMID_CLIENT_ID": ""}):
            resp = _authorize()
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# /oidc/remote-callback
# ---------------------------------------------------------------------------


class TestOIDCRemoteCallback:
    def test_shows_code_bound_to_state(self):
        resp = _client().get(
            "/oidc/remote-callback", params={"state": "state-1", "code": "auth-code"}
        )

        assert resp.status_code == 200
        expected = encode_remote_code("state-1", "auth-code")
        assert expected in resp.text
        assert "Never share it" in resp.text
        raw = base64.urlsafe_b64decode(expected + "=" * (-len(expected) % 4))
        assert json.loads(raw) == {"s": "state-1", "c": "auth-code"}

    def test_ibmid_error_is_shown_escaped(self):
        resp = _client().get(
            "/oidc/remote-callback",
            params={
                "state": "state-1",
                "error": "access_denied",
                "error_description": "<script>x</script>",
            },
        )

        assert resp.status_code == 200
        assert "Authentication failed" in resp.text
        assert "<script>" not in resp.text

    def test_missing_code(self):
        resp = _client().get("/oidc/remote-callback", params={"state": "state-1"})
        assert "No authorization code" in resp.text

    def test_never_calls_ibmid(self):
        with patch("gbserver.api.auth_routes.requests.post") as post:
            _client().get("/oidc/remote-callback", params={"state": "s", "code": "c"})
        post.assert_not_called()


# ---------------------------------------------------------------------------
# /oidc/token
# ---------------------------------------------------------------------------


class TestOIDCToken:
    def _post(self, **body):
        payload = {"code": "auth-code", "code_verifier": "verifier-xyz"}
        payload.update(body)
        payload = {k: v for k, v in payload.items() if v is not None}
        return _client().post("/oidc/token", json=payload)

    def test_redeems_with_secret_and_verifier(self):
        with (
            patch(
                "gbserver.api.auth_routes.requests.post", return_value=_ibmid_token_ok()
            ) as post,
            patch(
                "gbserver.api.auth_routes.requests.get",
                return_value=_ibmid_userinfo_ok(),
            ) as get,
        ):
            resp = self._post(redirect_uri=LOOPBACK)

        assert resp.status_code == 200
        assert resp.json() == {
            "access_token": "at_test",
            "id_token": "idt_test",
            "refresh_token": "rt_test",
            "expires_in": 3600,
            "user_info": {"sub": "IBMid-1", "email": "test@ibm.com"},
        }
        assert post.call_args.args[0] == ENV["GBSERVER_IBMID_TOKEN_URL"]
        sent = post.call_args.kwargs["data"]
        assert sent == {
            "grant_type": "authorization_code",
            "code": "auth-code",
            "redirect_uri": LOOPBACK,
            "code_verifier": "verifier-xyz",
            "client_id": "test-client-id",
            "client_secret": "test-client-secret",
        }
        assert get.call_args.kwargs["headers"] == {"Authorization": "Bearer at_test"}

    def test_omitted_redirect_uri_uses_remote_callback(self):
        with (
            patch(
                "gbserver.api.auth_routes.requests.post", return_value=_ibmid_token_ok()
            ) as post,
            patch(
                "gbserver.api.auth_routes.requests.get",
                return_value=_ibmid_userinfo_ok(),
            ),
        ):
            resp = self._post()

        assert resp.status_code == 200
        assert post.call_args.kwargs["data"]["redirect_uri"] == REMOTE_CALLBACK

    def test_rejects_disallowed_redirect_uri_without_calling_ibmid(self):
        with patch("gbserver.api.auth_routes.requests.post") as post:
            resp = self._post(redirect_uri="https://evil.example.com/callback")

        assert resp.status_code == 400
        post.assert_not_called()

    def test_ibmid_rejection_maps_to_400(self):
        bad = MagicMock()
        bad.status_code = 400
        bad.json.return_value = {
            "error": "invalid_grant",
            "error_description": "PKCE verification failed",
        }
        with patch("gbserver.api.auth_routes.requests.post", return_value=bad):
            resp = self._post(redirect_uri=LOOPBACK)

        assert resp.status_code == 400
        assert "PKCE verification failed" in resp.json()["detail"]

    def test_ibmid_server_error_maps_to_502(self):
        bad = MagicMock()
        bad.status_code = 503
        bad.json.side_effect = ValueError("not json")
        with patch("gbserver.api.auth_routes.requests.post", return_value=bad):
            resp = self._post(redirect_uri=LOOPBACK)
        assert resp.status_code == 502

    def test_ibmid_unreachable_maps_to_502(self):
        with patch(
            "gbserver.api.auth_routes.requests.post",
            side_effect=ConnectionError("down"),
        ):
            resp = self._post(redirect_uri=LOOPBACK)
        assert resp.status_code == 502
        assert "down" not in resp.text

    def test_userinfo_failure_still_returns_tokens(self):
        ui = MagicMock()
        ui.raise_for_status.side_effect = RuntimeError("boom")
        with (
            patch(
                "gbserver.api.auth_routes.requests.post", return_value=_ibmid_token_ok()
            ),
            patch("gbserver.api.auth_routes.requests.get", return_value=ui),
        ):
            resp = self._post(redirect_uri=LOOPBACK)

        assert resp.status_code == 200
        assert resp.json()["user_info"] == {}

    @pytest.mark.parametrize(
        "unset", ["GBSERVER_IBMID_CLIENT_ID", "GBSERVER_IBMID_CLIENT_SECRET"]
    )
    def test_missing_client_credentials_is_a_config_error(self, unset):
        with (
            patch.dict(os.environ, {unset: ""}),
            patch("gbserver.api.auth_routes.requests.post") as post,
        ):
            resp = self._post(redirect_uri=LOOPBACK)
        assert resp.status_code == 500
        post.assert_not_called()

    def test_remote_without_callback_config_is_a_config_error(self):
        with (
            patch.dict(os.environ, {"GBSERVER_IBMID_CALLBACK_URL": ""}),
            patch("gbserver.api.auth_routes.requests.post") as post,
        ):
            resp = self._post()
        assert resp.status_code == 500
        post.assert_not_called()

    def test_token_response_without_access_token_is_an_error(self):
        empty = MagicMock()
        empty.status_code = 200
        empty.json.return_value = {"id_token": "idt"}
        with patch("gbserver.api.auth_routes.requests.post", return_value=empty):
            resp = self._post(redirect_uri=LOOPBACK)
        assert resp.status_code == 502

    def test_requires_verifier(self):
        resp = _client().post(
            "/oidc/token", json={"code": "c", "redirect_uri": LOOPBACK}
        )
        assert resp.status_code == 422

    def test_secret_never_in_response_or_logs(self, caplog):
        bad = MagicMock()
        bad.status_code = 400
        bad.json.return_value = {"error": "invalid_grant"}
        with patch("gbserver.api.auth_routes.requests.post", return_value=bad):
            resp = self._post(redirect_uri=LOOPBACK)

        for text in (resp.text, caplog.text):
            assert "test-client-secret" not in text
            assert "verifier-xyz" not in text
            assert "auth-code" not in text


# ---------------------------------------------------------------------------
# Stateless across workers
# ---------------------------------------------------------------------------


def test_steps_on_separate_app_instances():
    """Each step stands alone: authorize on one app instance and redeem on
    another (the multi-worker case the legacy flow breaks on)."""
    worker_a = FastAPI()
    worker_a.mount("/auth", auth_api)
    worker_b = FastAPI()
    worker_b.mount("/auth", auth_api)

    resp = TestClient(worker_a, follow_redirects=False).get(
        "/auth/oidc/authorize",
        params={
            "code_challenge": "c",
            "code_challenge_method": "S256",
            "state": "s",
            "redirect_uri": LOOPBACK,
        },
    )
    assert resp.status_code == 302

    with (
        patch("gbserver.api.auth_routes.requests.post", return_value=_ibmid_token_ok()),
        patch(
            "gbserver.api.auth_routes.requests.get", return_value=_ibmid_userinfo_ok()
        ),
    ):
        resp = TestClient(worker_b).post(
            "/auth/oidc/token",
            json={"code": "x", "code_verifier": "v", "redirect_uri": LOOPBACK},
        )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Auth middleware exemption
# ---------------------------------------------------------------------------


class TestMiddlewareExemption:
    def _app(self):
        from gbserver.api.auth import AuthMiddleware

        app = FastAPI()
        app.add_middleware(AuthMiddleware)
        app.mount("/api/v1/auth", auth_api)

        @app.post("/api/v1/builds/something")
        def protected_post():
            return {"ok": True}

        return app

    def test_token_post_needs_no_bearer(self):
        client = TestClient(self._app())
        with (
            patch.dict(os.environ, {"GBSERVER_AUTH_MODE": "ibmid"}),
            patch(
                "gbserver.api.auth_routes.requests.post", return_value=_ibmid_token_ok()
            ),
            patch(
                "gbserver.api.auth_routes.requests.get",
                return_value=_ibmid_userinfo_ok(),
            ),
        ):
            resp = client.post(
                "/api/v1/auth/oidc/token",
                json={"code": "x", "code_verifier": "v", "redirect_uri": LOOPBACK},
            )
        assert resp.status_code == 200

    @pytest.mark.parametrize(
        "path",
        [
            "/api/v1/builds/something",
            # Only the exact token path is exempt, not the whole auth prefix.
            "/api/v1/auth/oidc/authorize",
            "/api/v1/auth/oidc/token/extra",
        ],
    )
    def test_other_posts_still_need_auth(self, path):
        client = TestClient(self._app())
        with patch.dict(os.environ, {"GBSERVER_AUTH_MODE": "ibmid"}):
            resp = client.post(path, json={})
        assert resp.status_code == 401
