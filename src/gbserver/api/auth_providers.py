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


def resolve_github_email(user: User, domain: str, headers: dict) -> None:
    """Fetch the primary verified email from /user/emails if missing.

    Public GitHub may omit the email when the user has set it to private.
    This mutates *user* in place, setting ``user.email`` if found.
    """
    if user.email or not is_public_github(domain):
        return
    api_base = get_gh_api_base(domain)
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

            resolve_github_email(user, self._gh_domain, headers)

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
#
# ``provider_types`` maps a provider name to its class. Both the in-tree
# built-ins and any entry-point plugins (group ``gbserver.auth_providers``) are
# filed through the shared ``PluginRegistrar`` under the ``keys_by_name`` rule,
# exactly like the other name-keyed subsystems. The registry drives the *class
# lookup*; ordering and per-provider construction stay in ``build_provider_list``
# (see below), because an ``auth_mode`` selects an ordered *set* of providers and
# some (IBMid) need constructor arguments.
# ---------------------------------------------------------------------------

# Provider name -> class. Populated by ``_load_auth_providers`` (in-tree scan +
# entry-point plugin pass), keyed under both lowercased and verbatim names.
provider_types: dict = {}

# The built-in providers, registered by their public name. Kept as an explicit
# list rather than a directory scan: both live in this module.
_BUILTIN_PROVIDERS = [
    ("github", GitHubAuthProvider),
    ("ibmid", IBMidAuthProvider),
]

# auth_mode -> ordered provider names. Order matters: JWT-based providers are
# listed first so ``identify_token`` can claim JWT-shaped tokens before the
# opaque-token provider (GitHub) sees them. Ordering lives here, never in
# ``provider_types`` iteration order, so registration order carries no auth
# semantics.
#
# NOTE: this map is intentionally hardcoded to the built-in modes for now. A
# plugin AuthProvider is discovered into ``provider_types`` but is not yet
# selectable, because no ``auth_mode`` references it. Making plugin providers
# reachable (e.g. letting GBSERVER_AUTH_MODE name providers directly) is a
# deliberate follow-up; the registry lookup below is already provider-name
# driven, so that extension needs no change here.
_AUTH_MODES = {
    "github": ["github"],
    "ibmid": ["ibmid"],
    "multi": ["ibmid", "github"],
}


def _load_auth_providers(force: bool = False) -> None:
    """(Re)build ``provider_types`` from the built-ins and any plugins.

    ``build_provider_list`` is on the per-request auth path, so this is a
    **no-op once the registry is populated** — the built-ins and the installed
    plugin set do not change under a running server. Pass ``force=True`` to
    rebuild anyway (tests that reload modules). When it does (re)build it goes
    through the shared reset-and-rebuild contract, so the registry is reload-safe
    and a plugin can only *add* a provider, never shadow a built-in (core-wins).
    """
    if provider_types and not force:
        return
    from gbcommon.plugins import (
        GROUP_AUTH_PROVIDERS,
        PluginRegistrar,
        keys_by_name,
        rebuild_registry,
    )

    registrar = PluginRegistrar(provider_types, "Auth provider", keys_by_name)

    def populate() -> None:
        for name, cls in _BUILTIN_PROVIDERS:
            registrar.add(cls, name)
        registrar.discover(GROUP_AUTH_PROVIDERS, AuthProvider)

    rebuild_registry(provider_types, populate)


def _make_provider(name: str) -> Optional[AuthProvider]:
    """Instantiate the registered provider *name*, supplying its constructor args.

    Construction is kept out of the registry (which holds classes): IBMid needs
    its issuer/JWKS/client-id from configuration, and other providers may need
    their own arguments. Plugin providers with no special arguments construct
    with defaults.

    Returns ``None`` (with a warning) if *name* is not in the registry, so a
    misconfigured mode degrades gracefully rather than raising mid-request.
    """
    cls = provider_types.get(name)
    if cls is None:
        logger.warning("Auth provider '%s' is not registered; skipping", name)
        return None
    # Construction can fail (e.g. IBMid's PyJWKClient with a misconfigured JWKS
    # URI); degrade gracefully rather than raise on the request path, matching
    # the None-on-missing contract above.
    try:
        if name == "ibmid":
            from gbserver.types.constants import (
                GBSERVER_IBMID_CLIENT_ID,
                GBSERVER_IBMID_ISSUER,
                GBSERVER_IBMID_JWKS_URI,
            )

            return cls(
                issuer=GBSERVER_IBMID_ISSUER,
                jwks_uri=GBSERVER_IBMID_JWKS_URI,
                client_id=GBSERVER_IBMID_CLIENT_ID,
            )
        # Other providers construct with no args. A provider needing constructor
        # args has no arg source here yet (only the built-in modes are selectable).
        return cls()
    except Exception as e:
        logger.warning("Could not construct auth provider '%s': %s", name, e)
        return None


def build_provider_list(auth_mode: str) -> List[AuthProvider]:
    """Build the ordered list of active providers for *auth_mode*.

    Provider order matters: JWT-based providers are checked first so that
    ``identify_token`` can distinguish token formats before falling through
    to the opaque-token provider (GitHub). The ordering comes from
    :data:`_AUTH_MODES`; the class for each name comes from the registry.
    """
    _load_auth_providers()

    names = _AUTH_MODES.get(auth_mode)
    if names is None:
        # Unknown mode – default to GitHub for backward compatibility.
        logger.warning(
            "Unknown GBSERVER_AUTH_MODE '%s', falling back to github", auth_mode
        )
        names = ["github"]

    providers = [p for p in (_make_provider(name) for name in names) if p is not None]
    if not providers:
        # Every configured name was unregistered; fall back so auth still works.
        logger.warning(
            "No auth providers could be built for mode '%s'; falling back to github",
            auth_mode,
        )
        providers = [GitHubAuthProvider()]
    return providers
