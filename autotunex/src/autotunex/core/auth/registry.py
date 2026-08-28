# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Builds the configured :class:`Authenticator` from settings.

Also builds the standalone ID-token verifier the backend-for-frontend uses in
``/auth/callback`` (:func:`build_id_token_verifier`). One function per
concern, expanded by each auth phase: the session phase adds the last branch,
wiring a session-cookie verifier into the same :class:`RoutingAuthenticator`
the API-key and OIDC phases already share.
"""

from __future__ import annotations

from autotunex.core.auth.api_key import ApiKeyVerifier
from autotunex.core.auth.disabled import DisabledAuthenticator
from autotunex.core.auth.oidc import JwksSigningKeyResolver, OidcBearerVerifier
from autotunex.core.auth.protocols import Authenticator
from autotunex.core.auth.routing import RoutingAuthenticator
from autotunex.core.auth.session import SessionCookieVerifier
from autotunex.core.config import Settings


def build_authenticator(settings: Settings) -> Authenticator:
    """Return the ``Authenticator`` matching ``settings.auth_providers``."""
    if settings.auth_providers == ["disabled"]:
        return DisabledAuthenticator(settings)

    api_key_verifier = (
        ApiKeyVerifier(settings.api_keys) if "api_key" in settings.auth_providers else None
    )
    bearer_verifier = (
        _build_bearer_verifier(settings) if "oidc" in settings.auth_providers else None
    )
    session_verifier = (
        _build_session_verifier(settings) if "session" in settings.auth_providers else None
    )
    return RoutingAuthenticator(
        bearer_verifier=bearer_verifier,
        api_key_verifier=api_key_verifier,
        session_verifier=session_verifier,
    )


def build_id_token_verifier(settings: Settings) -> OidcBearerVerifier | None:
    """Build the verifier ``/auth/callback`` uses to check a W3ID ID token.

    ``"session"`` and ``"oidc"`` are independent providers, so this cannot
    reuse whatever ``build_authenticator`` wired for bearer tokens — either
    can be enabled without the other. Returns ``None`` when ``"session"`` is
    not enabled; nothing calls this dependency in that case.
    """
    if "session" not in settings.auth_providers:
        return None
    return _build_bearer_verifier(settings)


def _build_bearer_verifier(settings: Settings) -> OidcBearerVerifier:
    """Construct an OIDC verifier, once, from validated settings.

    Shared by two callers that each hand it a different token: the ``"oidc"``
    branch above, verifying a caller-presented bearer token, and
    :func:`build_id_token_verifier`, verifying the ID token
    ``/auth/callback`` receives from the IdP. Both check the same issuer,
    audience, and algorithm allowlist — only which token is passed in at
    ``verify`` time differs — so the construction logic is not duplicated.

    ``Settings``'s rule-2 and rule-4 validators already guarantee these three
    are set, non-empty, and non-whitespace whenever ``"oidc"`` or ``"session"``
    is enabled; the checks here exist only so mypy can narrow the type, not
    because this should ever trigger. They test falsy-or-whitespace rather
    than ``is None`` to state the same rule the validators do — an empty or
    whitespace-only value counts as unset.
    """
    if (
        not settings.oidc_issuer
        or not settings.oidc_issuer.strip()
        or not settings.oidc_jwks_uri
        or not settings.oidc_jwks_uri.strip()
        or not settings.oidc_audience
        or not settings.oidc_audience.strip()
    ):
        raise RuntimeError(
            "oidc enabled with missing settings — the startup validator should have caught this"
        )
    return OidcBearerVerifier(
        issuer=settings.oidc_issuer,
        audience=settings.oidc_audience,
        algorithms=settings.oidc_algorithms,
        email_claims=settings.oidc_email_claims,
        leeway_seconds=settings.oidc_leeway_seconds,
        key_resolver=JwksSigningKeyResolver(settings.oidc_jwks_uri),
    )


def _build_session_verifier(settings: Settings) -> SessionCookieVerifier:
    """Construct the session-cookie verifier, once, from validated settings.

    ``Settings``'s rule-4 validator already guarantees ``session_secret`` is
    set, non-empty, and non-whitespace whenever ``"session"`` is enabled; the
    check here exists only so mypy can narrow the type, not because this
    should ever trigger. It tests falsy-or-whitespace after unwrapping the
    ``SecretStr`` rather than ``is None`` alone, to state the same rule the
    validator does — a ``SecretStr`` is truthy no matter what it wraps, and
    an empty or whitespace-only secret counts as unset.
    """
    secret = settings.session_secret.get_secret_value() if settings.session_secret else None
    if not secret or not secret.strip():
        raise RuntimeError(
            "session enabled with missing settings — the startup validator should have caught this"
        )
    return SessionCookieVerifier(secret)
