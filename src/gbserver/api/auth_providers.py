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

"""Pluggable authentication provider abstraction.

Each provider knows how to *identify* a Bearer token as its own and
*validate* it, returning a :class:`~gbserver.types.auth.User` on success.
"""

import base64
import hashlib
import json
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import jwt
import requests
from jwt import PyJWKClient

from gbcommon.types.constants import (
    DEFAULT_GH_DOMAIN,
    get_gh_api_base,
    is_public_github,
)
from gbserver.types.auth import User
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class AuthProvider(ABC):
    """Interface for an authentication provider."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Short identifier for this provider (e.g. ``"github"``, ``"ibmid"``)."""

    @abstractmethod
    def identify_token(self, token: str) -> bool:
        """Return ``True`` if *token* looks like it belongs to this provider.

        The check should be fast and heuristic — full validation happens in
        :meth:`validate_token`.
        """

    @abstractmethod
    def validate_token(self, token: str) -> Tuple[Optional[User], str]:
        """Validate *token* and return (User, "") on success.

        On failure return (None, error_message).
        """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_jwt_shaped(token: str) -> bool:
    """Return ``True`` if *token* has the three-segment Base64url structure of a JWT."""
    parts = token.split(".")
    if len(parts) != 3:
        return False
    for part in parts[:2]:
        try:
            # Add padding and decode
            padded = part + "=" * (-len(part) % 4)
            base64.urlsafe_b64decode(padded)
        except Exception:
            return False
    return True


def _peek_jwt_issuer(token: str) -> Optional[str]:
    """Decode the JWT payload *without* signature verification and return the ``iss`` claim."""
    try:
        parts = token.split(".")
        padded = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded))
        return payload.get("iss")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# GitHub Enterprise provider
# ---------------------------------------------------------------------------


class GitHubAuthProvider(AuthProvider):
    """Validates opaque GitHub Personal Access Tokens / OAuth tokens."""

    def __init__(self, gh_domain: Optional[str] = None):
        self._gh_domain = gh_domain or DEFAULT_GH_DOMAIN

    @property
    def provider_name(self) -> str:
        return "github"

    def identify_token(self, token: str) -> bool:
        # GitHub tokens are opaque (never JWT-shaped).
        return not _is_jwt_shaped(token)

    def validate_token(self, token: str) -> Tuple[Optional[User], str]:
        api_base = get_gh_api_base(self._gh_domain)
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        try:
            response = requests.get(
                f"{api_base}/user",
                headers=headers,
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            user = User.model_validate(data)
            user.auth_provider = "github"

            # Public GitHub may not include email if set to private.
            if not user.email and is_public_github(self._gh_domain):
                try:
                    emails_resp = requests.get(
                        f"{api_base}/user/emails", headers=headers, timeout=10
                    )
                    emails_resp.raise_for_status()
                    for entry in emails_resp.json():
                        if entry.get("primary") and entry.get("verified"):
                            user.email = entry["email"]
                            break
                except Exception as email_err:
                    logger.warning(
                        "Failed to fetch /user/emails for %s: %s",
                        user.login,
                        email_err,
                    )

            if not user.email:
                logger.warning(
                    "GitHub /user returned no email for user %s; "
                    "space-access checks may fail",
                    user.login,
                )
            return (user, "")
        except Exception as e:
            return (None, f"{e}")


# ---------------------------------------------------------------------------
# IBMid OIDC provider
# ---------------------------------------------------------------------------


class IBMidAuthProvider(AuthProvider):
    """Validates IBMid JWT access tokens using OIDC discovery / JWKS."""

    def __init__(
        self,
        issuer: str = "https://login.ibm.com/oidc/endpoint/default",
        jwks_uri: str = "https://login.ibm.com/oidc/endpoint/default/jwks",
        client_id: str = "",
    ):
        self._issuer = issuer
        self._client_id = client_id
        self._jwk_client = PyJWKClient(jwks_uri, cache_keys=True)

    @property
    def provider_name(self) -> str:
        return "ibmid"

    def identify_token(self, token: str) -> bool:
        if not _is_jwt_shaped(token):
            return False
        issuer = _peek_jwt_issuer(token)
        return issuer == self._issuer

    def validate_token(self, token: str) -> Tuple[Optional[User], str]:
        try:
            signing_key = self._jwk_client.get_signing_key_from_jwt(token)

            decode_options = {
                "verify_exp": True,
                "verify_aud": bool(self._client_id),
            }
            kwargs = {}
            if self._client_id:
                kwargs["audience"] = self._client_id

            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                issuer=self._issuer,
                options=decode_options,  # type: ignore[arg-type]
                **kwargs,  # type: ignore[arg-type]
            )

            sub = payload.get("sub", "")
            email = payload.get("email", "")
            name = payload.get("name", sub)
            login = payload.get("preferred_username", email or sub)

            # Produce a stable integer id from the ``sub`` claim.
            id_int = int(hashlib.sha256(sub.encode()).hexdigest()[:15], 16)

            user = User(
                login=login,
                id=id_int,
                url="",
                html_url="",
                name=name,
                email=email,
                auth_provider="ibmid",
            )
            return (user, "")
        except jwt.ExpiredSignatureError:
            return (None, "IBMid token has expired")
        except jwt.InvalidTokenError as e:
            return (None, f"IBMid token validation failed: {e}")
        except Exception as e:
            return (None, f"IBMid authentication error: {e}")


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------


def build_provider_list(auth_mode: str) -> List[AuthProvider]:
    """Build the list of active providers based on *auth_mode*.

    Provider order matters: JWT-based providers are checked first so that
    ``identify_token`` can distinguish token formats before falling through
    to the opaque-token provider (GitHub).
    """
    from gbserver.types.constants import (
        GBSERVER_IBMID_CLIENT_ID,
        GBSERVER_IBMID_ISSUER,
        GBSERVER_IBMID_JWKS_URI,
    )

    def _make_ibmid():
        return IBMidAuthProvider(
            issuer=GBSERVER_IBMID_ISSUER,
            jwks_uri=GBSERVER_IBMID_JWKS_URI,
            client_id=GBSERVER_IBMID_CLIENT_ID,
        )

    if auth_mode == "github":
        return [GitHubAuthProvider()]
    if auth_mode == "ibmid":
        return [_make_ibmid()]
    if auth_mode == "multi":
        return [_make_ibmid(), GitHubAuthProvider()]

    # Unknown mode – default to GitHub for backward compatibility
    logger.warning("Unknown GBSERVER_AUTH_MODE '%s', falling back to github", auth_mode)
    return [GitHubAuthProvider()]
