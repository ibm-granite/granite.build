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
import json
import os
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from gbserver.api.auth import AuthMiddleware
from gbserver.api.auth_providers import (
    GitHubAuthProvider,
    IBMidAuthProvider,
    _is_jwt_shaped,
    _peek_jwt_issuer,
    build_provider_list,
)
from gbserver.types.auth import User

pytestmark = pytest.mark.g4os


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rsa_keypair():
    """Generate an RSA private/public key pair for test JWT signing."""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    return private_key


def _make_test_jwt(private_key, claims: dict, headers: dict = None) -> str:
    """Create a signed JWT with the given claims."""
    return jwt.encode(claims, private_key, algorithm="RS256", headers=headers)


def _make_app() -> FastAPI:
    """Build a minimal FastAPI app with AuthMiddleware and a /test endpoint."""
    app = FastAPI()
    app.add_middleware(AuthMiddleware)

    @app.get("/test")
    async def test_endpoint(request: Request):
        user = request.state.data["user"]
        return JSONResponse(
            content={
                "login": user.login,
                "email": user.email,
                "auth_provider": user.auth_provider,
            }
        )

    @app.get("/docs")
    async def docs_endpoint():
        return JSONResponse(content={"docs": True})

    return app


# ---------------------------------------------------------------------------
# Token format detection tests
# ---------------------------------------------------------------------------


class TestTokenFormatDetection:
    """Test _is_jwt_shaped and _peek_jwt_issuer helpers."""

    def test_opaque_github_token_not_jwt(self):
        assert _is_jwt_shaped("ghp_abc123def456") is False

    def test_random_string_not_jwt(self):
        assert _is_jwt_shaped("some-random-token-value") is False

    def test_empty_string_not_jwt(self):
        assert _is_jwt_shaped("") is False

    def test_valid_jwt_structure_detected(self):
        # Create a minimal JWT-like structure
        header = base64.urlsafe_b64encode(json.dumps({"alg": "RS256"}).encode()).rstrip(b"=").decode()
        payload = base64.urlsafe_b64encode(json.dumps({"sub": "test"}).encode()).rstrip(b"=").decode()
        sig = base64.urlsafe_b64encode(b"fakesig").rstrip(b"=").decode()
        token = f"{header}.{payload}.{sig}"
        assert _is_jwt_shaped(token) is True

    def test_two_dots_but_invalid_base64_not_jwt(self):
        assert _is_jwt_shaped("not.valid!base64.token") is False

    def test_peek_issuer_from_jwt_payload(self):
        header = base64.urlsafe_b64encode(json.dumps({"alg": "RS256"}).encode()).rstrip(b"=").decode()
        payload = base64.urlsafe_b64encode(
            json.dumps({"iss": "https://login.ibm.com/oidc/endpoint/default"}).encode()
        ).rstrip(b"=").decode()
        sig = base64.urlsafe_b64encode(b"fakesig").rstrip(b"=").decode()
        token = f"{header}.{payload}.{sig}"
        assert _peek_jwt_issuer(token) == "https://login.ibm.com/oidc/endpoint/default"

    def test_peek_issuer_returns_none_for_opaque(self):
        assert _peek_jwt_issuer("ghp_abc123") is None


# ---------------------------------------------------------------------------
# GitHubAuthProvider tests
# ---------------------------------------------------------------------------


class TestGitHubAuthProvider:
    def test_identify_opaque_token(self):
        provider = GitHubAuthProvider()
        assert provider.identify_token("ghp_abc123def456") is True

    def test_identify_rejects_jwt(self):
        header = base64.urlsafe_b64encode(json.dumps({"alg": "RS256"}).encode()).rstrip(b"=").decode()
        payload = base64.urlsafe_b64encode(json.dumps({"sub": "test"}).encode()).rstrip(b"=").decode()
        sig = base64.urlsafe_b64encode(b"fakesig").rstrip(b"=").decode()
        token = f"{header}.{payload}.{sig}"
        provider = GitHubAuthProvider()
        assert provider.identify_token(token) is False

    def test_provider_name(self):
        provider = GitHubAuthProvider()
        assert provider.provider_name == "github"

    def test_validate_success(self):
        provider = GitHubAuthProvider(gh_domain="github.ibm.com")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "login": "testuser",
            "id": 12345,
            "url": "https://api.github.ibm.com/users/testuser",
            "html_url": "https://github.ibm.com/testuser",
            "name": "Test User",
            "email": "test@ibm.com",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("gbserver.api.auth_providers.requests") as mock_requests:
            mock_requests.get.return_value = mock_response
            # Need to use the correct import path
            user, error = provider.validate_token("ghp_validtoken")

        assert user is not None
        assert user.login == "testuser"
        assert user.email == "test@ibm.com"
        assert user.auth_provider == "github"
        assert error == ""

    def test_validate_failure(self):
        provider = GitHubAuthProvider(gh_domain="github.ibm.com")
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("401 Unauthorized")

        with patch("gbserver.api.auth_providers.requests") as mock_requests:
            mock_requests.get.return_value = mock_response
            user, error = provider.validate_token("ghp_invalidtoken")

        assert user is None
        assert "401" in error or "Unauthorized" in error


# ---------------------------------------------------------------------------
# IBMidAuthProvider tests
# ---------------------------------------------------------------------------


class TestIBMidAuthProvider:
    @pytest.fixture
    def rsa_keypair(self):
        return _make_rsa_keypair()

    @pytest.fixture
    def ibmid_issuer(self):
        return "https://login.ibm.com/oidc/endpoint/default"

    def _make_provider(self, rsa_keypair, issuer):
        """Create an IBMidAuthProvider with mocked JWKS."""
        provider = IBMidAuthProvider(
            issuer=issuer,
            jwks_uri="https://login.ibm.com/oidc/endpoint/default/jwks",
            client_id="test-client-id",
        )
        return provider

    def test_provider_name(self, rsa_keypair, ibmid_issuer):
        provider = self._make_provider(rsa_keypair, ibmid_issuer)
        assert provider.provider_name == "ibmid"

    def test_identify_valid_ibmid_jwt(self, rsa_keypair, ibmid_issuer):
        provider = self._make_provider(rsa_keypair, ibmid_issuer)
        token = _make_test_jwt(
            rsa_keypair,
            {"iss": ibmid_issuer, "sub": "testuser", "exp": int(time.time()) + 3600},
        )
        assert provider.identify_token(token) is True

    def test_identify_rejects_wrong_issuer(self, rsa_keypair, ibmid_issuer):
        provider = self._make_provider(rsa_keypair, ibmid_issuer)
        token = _make_test_jwt(
            rsa_keypair,
            {"iss": "https://other-issuer.com", "sub": "testuser", "exp": int(time.time()) + 3600},
        )
        assert provider.identify_token(token) is False

    def test_identify_rejects_opaque_token(self, rsa_keypair, ibmid_issuer):
        provider = self._make_provider(rsa_keypair, ibmid_issuer)
        assert provider.identify_token("ghp_abc123def456") is False

    def test_validate_valid_jwt(self, rsa_keypair, ibmid_issuer):
        provider = self._make_provider(rsa_keypair, ibmid_issuer)

        now = int(time.time())
        token = _make_test_jwt(
            rsa_keypair,
            {
                "iss": ibmid_issuer,
                "sub": "IBMid-12345",
                "email": "test@ibm.com",
                "name": "Test User",
                "preferred_username": "testuser@ibm.com",
                "aud": "test-client-id",
                "exp": now + 3600,
                "iat": now,
            },
        )

        # Mock the JWKS client to return our test key
        public_key = rsa_keypair.public_key()
        mock_signing_key = MagicMock()
        mock_signing_key.key = public_key

        with patch.object(provider._jwk_client, "get_signing_key_from_jwt", return_value=mock_signing_key):
            user, error = provider.validate_token(token)

        assert user is not None
        assert user.login == "testuser@ibm.com"
        assert user.email == "test@ibm.com"
        assert user.name == "Test User"
        assert user.auth_provider == "ibmid"
        assert error == ""

    def test_validate_expired_jwt(self, rsa_keypair, ibmid_issuer):
        provider = self._make_provider(rsa_keypair, ibmid_issuer)

        token = _make_test_jwt(
            rsa_keypair,
            {
                "iss": ibmid_issuer,
                "sub": "IBMid-12345",
                "email": "test@ibm.com",
                "aud": "test-client-id",
                "exp": int(time.time()) - 3600,  # expired
                "iat": int(time.time()) - 7200,
            },
        )

        public_key = rsa_keypair.public_key()
        mock_signing_key = MagicMock()
        mock_signing_key.key = public_key

        with patch.object(provider._jwk_client, "get_signing_key_from_jwt", return_value=mock_signing_key):
            user, error = provider.validate_token(token)

        assert user is None
        assert "expired" in error.lower()

    def test_validate_wrong_signature(self, rsa_keypair, ibmid_issuer):
        """Token signed with a different key should fail validation."""
        provider = self._make_provider(rsa_keypair, ibmid_issuer)

        # Sign with a different key
        other_key = _make_rsa_keypair()
        token = _make_test_jwt(
            other_key,
            {
                "iss": ibmid_issuer,
                "sub": "IBMid-12345",
                "email": "test@ibm.com",
                "aud": "test-client-id",
                "exp": int(time.time()) + 3600,
                "iat": int(time.time()),
            },
        )

        # Return the *correct* public key (different from the signing key)
        public_key = rsa_keypair.public_key()
        mock_signing_key = MagicMock()
        mock_signing_key.key = public_key

        with patch.object(provider._jwk_client, "get_signing_key_from_jwt", return_value=mock_signing_key):
            user, error = provider.validate_token(token)

        assert user is None
        assert "validation failed" in error.lower() or "signature" in error.lower()


# ---------------------------------------------------------------------------
# build_provider_list tests
# ---------------------------------------------------------------------------


class TestBuildProviderList:
    def test_github_mode(self):
        providers = build_provider_list("github")
        assert len(providers) == 1
        assert providers[0].provider_name == "github"

    @patch.dict(os.environ, {"GBSERVER_IBMID_CLIENT_ID": "test-id"})
    def test_ibmid_mode(self):
        providers = build_provider_list("ibmid")
        assert len(providers) == 1
        assert providers[0].provider_name == "ibmid"

    @patch.dict(os.environ, {"GBSERVER_IBMID_CLIENT_ID": "test-id"})
    def test_multi_mode(self):
        providers = build_provider_list("multi")
        assert len(providers) == 2
        # IBMid should come first (JWT providers checked first)
        assert providers[0].provider_name == "ibmid"
        assert providers[1].provider_name == "github"

    def test_unknown_mode_falls_back_to_github(self):
        providers = build_provider_list("unknown_mode")
        assert len(providers) == 1
        assert providers[0].provider_name == "github"


# ---------------------------------------------------------------------------
# User model auth_provider field tests
# ---------------------------------------------------------------------------


class TestUserAuthProvider:
    def test_default_auth_provider_is_github(self):
        user = User(
            login="test", id=1, url="", html_url="", name="Test", email="t@t.com"
        )
        assert user.auth_provider == "github"

    def test_auth_provider_can_be_set(self):
        user = User(
            login="test",
            id=1,
            url="",
            html_url="",
            name="Test",
            email="t@t.com",
            auth_provider="ibmid",
        )
        assert user.auth_provider == "ibmid"

    def test_github_api_response_without_auth_provider_defaults(self):
        """Simulate a GitHub API response that doesn't include auth_provider."""
        data = {
            "login": "testuser",
            "id": 12345,
            "url": "https://api.github.ibm.com/users/testuser",
            "html_url": "https://github.ibm.com/testuser",
            "name": "Test User",
            "email": "test@ibm.com",
        }
        user = User.model_validate(data)
        assert user.auth_provider == "github"


# ---------------------------------------------------------------------------
# AuthMiddleware integration tests (multi-provider)
# ---------------------------------------------------------------------------


class TestAuthMiddlewareMultiProvider:
    def test_github_mode_with_github_token(self):
        """Default github mode should validate opaque tokens via GitHub API."""
        env = {"GBSERVER_AUTH_MODE": "github"}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "login": "testuser",
            "id": 12345,
            "url": "https://api.github.ibm.com/users/testuser",
            "html_url": "https://github.ibm.com/testuser",
            "name": "Test User",
            "email": "test@ibm.com",
        }
        mock_response.raise_for_status = MagicMock()

        with patch.dict(os.environ, env, clear=False):
            app = _make_app()
            client = TestClient(app)
            with patch("gbserver.api.auth_providers.requests") as mock_requests:
                mock_requests.get.return_value = mock_response
                response = client.get(
                    "/test",
                    headers={"Authorization": "Bearer ghp_validtoken123"},
                )

        assert response.status_code == 200
        data = response.json()
        assert data["login"] == "testuser"
        assert data["auth_provider"] == "github"

    def test_missing_auth_header_returns_401(self):
        """Missing Authorization header should return 401."""
        env = {"GBSERVER_AUTH_MODE": "github"}
        with patch.dict(os.environ, env, clear=False):
            app = _make_app()
            client = TestClient(app)
            response = client.get("/test")
        assert response.status_code == 401

    def test_docs_allowed_without_auth(self):
        """Docs endpoint should work without authentication in any mode."""
        env = {"GBSERVER_AUTH_MODE": "multi"}
        with patch.dict(os.environ, env, clear=False):
            app = _make_app()
            client = TestClient(app)
            response = client.get("/docs")
        assert response.status_code == 200
