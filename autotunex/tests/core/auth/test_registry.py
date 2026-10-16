"""The authenticator registry: one seam, built once from settings."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import pytest

from autotunex.core.auth.disabled import DisabledAuthenticator
from autotunex.core.auth.oidc import OidcBearerVerifier
from autotunex.core.auth.registry import build_authenticator, build_id_token_verifier
from autotunex.core.auth.routing import RoutingAuthenticator
from autotunex.core.auth.session import mint_session_token
from autotunex.core.config import Settings
from autotunex.core.exceptions import InvalidCredentialsError


def test_disabled_providers_builds_the_disabled_authenticator() -> None:
    authenticator = build_authenticator(
        Settings(_env_file=None, environment="test", auth_providers=["disabled"])
    )

    assert isinstance(authenticator, DisabledAuthenticator)


def test_api_key_provider_builds_a_routing_authenticator_with_the_verifier_wired() -> None:
    digest = "a" * 64
    settings = Settings(
        _env_file=None,
        environment="test",
        auth_providers=["api_key"],
        api_keys={digest: "svc@example.com"},
    )

    authenticator = build_authenticator(settings)

    assert isinstance(authenticator, RoutingAuthenticator)


async def test_the_wired_api_key_verifier_actually_authenticates() -> None:
    raw_key = "a-real-looking-key"
    digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    settings = Settings(
        _env_file=None,
        environment="test",
        auth_providers=["api_key"],
        api_keys={digest: "svc@example.com"},
    )
    authenticator = build_authenticator(settings)

    principal = await authenticator.authenticate(bearer=None, api_key=raw_key, session=None)

    assert principal.email == "svc@example.com"


async def test_a_bearer_token_is_rejected_when_only_the_api_key_provider_is_enabled() -> None:
    """Routed-to-an-unregistered-verifier is the invalid-credential 401, byte for byte.

    Naming which schemes are configured tells an attacker what to stop trying.
    """
    digest = "a" * 64
    settings = Settings(
        _env_file=None,
        environment="test",
        auth_providers=["api_key"],
        api_keys={digest: "svc@example.com"},
    )
    authenticator = build_authenticator(settings)

    with pytest.raises(InvalidCredentialsError):
        await authenticator.authenticate(bearer="anything", api_key=None, session=None)


async def test_a_valid_api_key_presented_as_a_bearer_token_is_rejected() -> None:
    """Verifier misrouting is caught: API keys in bearer slot must fail unconditionally.

    This pins the core invariant: each credential kind has its own transport.
    A valid credential presented on the wrong slot must be rejected, which
    requires that bearer_verifier is None, not ApiKeyVerifier. If someone
    mis-wires the registry to pass the same verifier to both slots, this
    test fails because the credential verifies successfully.
    """
    raw_key = "a-real-looking-key"
    digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    settings = Settings(
        _env_file=None,
        environment="test",
        auth_providers=["api_key"],
        api_keys={digest: "svc@example.com"},
    )
    authenticator = build_authenticator(settings)

    with pytest.raises(InvalidCredentialsError):
        await authenticator.authenticate(bearer=raw_key, api_key=None, session=None)


_OIDC_SETTINGS_KWARGS: dict[str, Any] = {
    # `Any`, not `object`: this dict is spread as `Settings(**kwargs)`, and
    # `BaseSettings.__init__` has dozens of precisely-typed `_cli_*` /
    # `_env_*` keyword parameters. Against `dict[str, object]`, mypy strict
    # must reject the spread because `object` cannot satisfy any of them; only
    # `Any` lets the unpack through while keeping every value here concrete.
    "_env_file": None,
    "environment": "test",
    "auth_providers": ["oidc"],
    "oidc_issuer": "https://idp.example.com/oauth2",
    "oidc_jwks_uri": "https://unused.invalid/jwks",
    "oidc_audience": "test-client-id",
}
"""Shared kwargs for the oidc-only tests below.

``oidc_jwks_uri`` deliberately points at an RFC 2606 ``.invalid`` host rather
than the real W3ID one: construction is lazy (``PyJWKClient`` never fetches
until a token actually needs verifying), so nothing here should ever resolve
it — but under the misrouting mutation the discriminating test below probes
for, a wrongly-wired verifier *would* attempt a real fetch, and a reachable
internal host would hang instead of failing cleanly offline.
"""


def test_oidc_provider_builds_a_routing_authenticator_with_a_bearer_verifier_wired() -> None:
    authenticator = build_authenticator(Settings(**_OIDC_SETTINGS_KWARGS))

    assert isinstance(authenticator, RoutingAuthenticator)


async def test_a_bearer_token_reaches_the_oidc_verifier_when_the_oidc_provider_is_enabled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The companion the isinstance check above cannot be: proof the bearer slot is wired.

    ``RoutingAuthenticator.bearer_verifier`` defaults to ``None`` whether or not
    ``build_authenticator`` wires anything for ``"oidc"`` — so
    ``isinstance(authenticator, RoutingAuthenticator)`` passes even if the
    ``_build_bearer_verifier(...)`` line were deleted and ``bearer_verifier``
    hardcoded to ``None``. Presenting a bearer token distinguishes the two
    states by which logger reports the rejection: unwired, the routing layer's
    "no verifier is registered" message fires (``core.auth.routing``); wired,
    ``"anything"`` reaches ``OidcBearerVerifier.verify``, fails to decode, and
    the OIDC logger (``core.auth.oidc``) reports it instead. No network is
    involved either way — a malformed token fails PyJWT's own decode before
    ``JwksSigningKeyResolver`` would ever reach the JWKS endpoint.
    """
    caplog.set_level(logging.WARNING)
    authenticator = build_authenticator(Settings(**_OIDC_SETTINGS_KWARGS))

    with pytest.raises(InvalidCredentialsError):
        await authenticator.authenticate(bearer="anything", api_key=None, session=None)

    assert any(
        record.name.startswith("autotunex") and "OIDC bearer token rejected" in record.getMessage()
        for record in caplog.records
    )
    assert not any(
        "no verifier is registered for that credential kind" in record.getMessage()
        for record in caplog.records
    )


_BOTH_PROVIDERS_RAW_KEY = "a-real-looking-key"


def _both_providers_settings() -> Settings:
    """Settings enabling ``["api_key", "oidc"]`` together — README's production shape."""
    digest = hashlib.sha256(_BOTH_PROVIDERS_RAW_KEY.encode("utf-8")).hexdigest()
    return Settings(
        **{
            **_OIDC_SETTINGS_KWARGS,
            "auth_providers": ["api_key", "oidc"],
            "api_keys": {digest: "svc@example.com"},
        }
    )


async def test_both_verifiers_are_wired_when_api_key_and_oidc_are_enabled_together(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The combination ``README.md``'s Production configuration blesses, and nothing covered.

    Every other test here passes a single-element ``auth_providers``, so a
    refactor of ``build_authenticator`` into an ``if/elif`` chain — the obvious
    "tidy up" of two independent ``in`` checks — would silently drop whichever
    branch lost, 401 that entire credential kind in production, and break no
    test. This asserts both slots are not merely populated but *reachable*.

    The API key half is proven positively, by authenticating. The bearer half
    cannot be, without a real signing key, so it uses the discriminating-log
    technique the rest of this file relies on: an unwired bearer slot is
    reported by ``RoutingAuthenticator._verify`` (``core.auth.routing``,
    "no verifier is registered"), whereas a wired one reaches
    ``OidcBearerVerifier.verify`` and is reported by ``core.auth.oidc``. Only
    the latter proves the verifier was constructed and installed. No network:
    ``"anything"`` fails PyJWT's own decode before any JWKS fetch.
    """
    caplog.set_level(logging.WARNING)
    authenticator = build_authenticator(_both_providers_settings())

    principal = await authenticator.authenticate(
        bearer=None, api_key=_BOTH_PROVIDERS_RAW_KEY, session=None
    )
    with pytest.raises(InvalidCredentialsError):
        await authenticator.authenticate(bearer="anything", api_key=None, session=None)

    assert principal.email == "svc@example.com"
    assert any(
        record.name.startswith("autotunex") and "OIDC bearer token rejected" in record.getMessage()
        for record in caplog.records
    )
    assert not any(
        "no verifier is registered for that credential kind" in record.getMessage()
        for record in caplog.records
    )


async def test_a_session_cookie_is_still_unregistered_when_both_providers_are_enabled() -> None:
    """Enabling two providers must not accidentally populate the third slot.

    ``RoutingAuthenticator`` takes three verifiers; ``_both_providers_settings``
    enables only ``api_key`` and ``oidc``, so a cookie must still be rejected.
    This pins that "both" means exactly api_key and bearer, not "everything".
    """
    authenticator = build_authenticator(_both_providers_settings())

    with pytest.raises(InvalidCredentialsError):
        await authenticator.authenticate(bearer=None, api_key=None, session="a-cookie")


async def test_an_api_key_is_rejected_when_only_the_oidc_provider_is_enabled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The mirror of the API-key phase's test: unregistered means invalid, not "not configured".

    A bare ``pytest.raises(InvalidCredentialsError)`` cannot catch a
    mis-wiring bug here: if ``build_authenticator`` wired the OIDC verifier
    into the ``api_key`` slot instead of (or as well as) the ``bearer`` slot,
    ``OidcBearerVerifier.verify`` would be handed ``"anything"``, fail to
    decode it, and raise the very same ``InvalidCredentialsError`` — so the
    test would pass either way. Asserting on *which* logger produced the
    rejection distinguishes "correctly unregistered" (logged by
    ``RoutingAuthenticator._verify``, in ``core.auth.routing``) from "wrongly
    wired into the api_key slot" (logged by ``OidcBearerVerifier.verify``, in
    ``core.auth.oidc``).
    """
    caplog.set_level(logging.WARNING)
    authenticator = build_authenticator(Settings(**_OIDC_SETTINGS_KWARGS))

    with pytest.raises(InvalidCredentialsError):
        await authenticator.authenticate(bearer=None, api_key="anything", session=None)

    assert any(
        record.name.startswith("autotunex")
        and "no verifier is registered for that credential kind" in record.getMessage()
        for record in caplog.records
    )
    assert not any("OIDC bearer token rejected" in record.getMessage() for record in caplog.records)


_SESSION_SECRET = "test-registry-session-secret-at-least-32-characters-long"
"""32+ characters: `session_secret` enforces this floor (see `core/config.py`).

Shared by `_SESSION_SETTINGS_KWARGS` below and
`test_the_wired_session_verifier_actually_authenticates`'s own
`mint_session_token` call, deliberately the same constant rather than two
literals that would need to agree by inspection — the settings build the
verifier around one secret and the test mints a token with the other, so
letting those drift apart would silently start rejecting the very token this
test exists to authenticate.
"""

_SESSION_SETTINGS_KWARGS: dict[str, Any] = {
    # See the comment on `_OIDC_SETTINGS_KWARGS` above: `Any`, not `object`,
    # because this dict is spread as `Settings(**kwargs)`.
    "_env_file": None,
    "environment": "test",
    "auth_providers": ["session"],
    "oidc_issuer": "https://idp.invalid/oauth2",
    "oidc_jwks_uri": "https://idp.invalid/jwks",
    "oidc_audience": "my-client-id",
    "oidc_client_id": "my-client-id",
    "oidc_client_secret": "shh",
    "oidc_authorization_endpoint": "https://idp.invalid/authorize",
    "oidc_token_endpoint": "https://idp.invalid/token",
    "public_base_url": "https://autotunex.invalid",
    "session_secret": _SESSION_SECRET,
}
"""Shared kwargs for the session tests below.

Every host is an RFC 2606 ``.invalid`` one, not a real IdP — see
``_OIDC_SETTINGS_KWARGS``'s docstring for the mechanism this guards against.
It matters even more here: ``build_id_token_verifier`` builds a
``JwksSigningKeyResolver`` around ``oidc_jwks_uri`` the same way
``_build_bearer_verifier`` does, so a mutation that made construction eager,
or a future refactor that did, would attempt a real outbound request against
whatever host is configured here rather than failing fast offline.
"""


def test_session_provider_builds_a_routing_authenticator_with_a_session_verifier_wired() -> None:
    authenticator = build_authenticator(Settings(**_SESSION_SETTINGS_KWARGS))

    assert isinstance(authenticator, RoutingAuthenticator)


async def test_the_wired_session_verifier_actually_authenticates() -> None:
    """The companion the isinstance check above cannot be, proven positively.

    Unlike the bearer-token tests above, this does not need the
    discriminating-log technique: a session token this process itself minted
    can be verified outright, with no signing key or network involved, so
    authenticating successfully is direct proof the session slot is wired —
    not merely populated with *something*.
    """
    settings = Settings(**_SESSION_SETTINGS_KWARGS)
    authenticator = build_authenticator(settings)
    token = mint_session_token(email="dev@example.com", secret=_SESSION_SECRET, ttl_hours=8)

    principal = await authenticator.authenticate(bearer=None, api_key=None, session=token)

    assert principal.email == "dev@example.com"


def test_build_id_token_verifier_is_none_when_session_is_not_enabled() -> None:
    assert build_id_token_verifier(Settings(_env_file=None, environment="test")) is None


def test_build_id_token_verifier_returns_an_oidc_bearer_verifier_when_session_is_enabled() -> None:
    settings = Settings(**_SESSION_SETTINGS_KWARGS)

    verifier = build_id_token_verifier(settings)

    assert isinstance(verifier, OidcBearerVerifier)


def _oidc_and_session_settings() -> Settings:
    """Settings enabling ``["oidc", "session"]`` together, sharing one ``oidc_audience``.

    README.md's OIDC subsection blesses this combination explicitly: the
    audience for the *ID* token session mints its verifier around is always
    the client id, the same setting the bearer-token path already requires —
    simpler than the access-token procedure oidc-alone documents, since ID
    tokens are unambiguously JWTs with ``aud`` equal to the client id.
    ``_SESSION_SETTINGS_KWARGS`` already sets ``oidc_issuer``,
    ``oidc_jwks_uri``, and ``oidc_audience``, so it satisfies rule 2 for
    ``"oidc"`` without needing to add anything.
    """
    return Settings(**{**_SESSION_SETTINGS_KWARGS, "auth_providers": ["oidc", "session"]})


async def test_both_verifiers_are_wired_when_oidc_and_session_are_enabled_together(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The combination README.md's OIDC subsection blesses, not yet covered above.

    Mirrors ``test_both_verifiers_are_wired_when_api_key_and_oidc_are_enabled_together``:
    the session half is proven positively, by authenticating a token this
    process itself minted; the bearer half uses the discriminating-log
    technique the rest of this file relies on, since proving it positively
    would need a real signing key. No network either way: ``"anything"``
    fails PyJWT's own decode before any JWKS fetch.
    """
    caplog.set_level(logging.WARNING)
    authenticator = build_authenticator(_oidc_and_session_settings())
    token = mint_session_token(email="dev@example.com", secret=_SESSION_SECRET, ttl_hours=8)

    principal = await authenticator.authenticate(bearer=None, api_key=None, session=token)
    with pytest.raises(InvalidCredentialsError):
        await authenticator.authenticate(bearer="anything", api_key=None, session=None)

    assert principal.email == "dev@example.com"
    assert any(
        record.name.startswith("autotunex") and "OIDC bearer token rejected" in record.getMessage()
        for record in caplog.records
    )
    assert not any(
        "no verifier is registered for that credential kind" in record.getMessage()
        for record in caplog.records
    )


def _api_key_and_session_settings() -> Settings:
    """Settings enabling ``["api_key", "session"]`` together."""
    digest = hashlib.sha256(_BOTH_PROVIDERS_RAW_KEY.encode("utf-8")).hexdigest()
    return Settings(
        **{
            **_SESSION_SETTINGS_KWARGS,
            "auth_providers": ["api_key", "session"],
            "api_keys": {digest: "svc@example.com"},
        }
    )


async def test_both_verifiers_are_wired_when_api_key_and_session_are_enabled_together() -> None:
    """The remaining two-provider combination not yet covered above.

    Unlike the oidc combinations elsewhere in this file, neither half here
    needs the discriminating-log technique: an api key can be authenticated
    outright, and so can a session token this process itself minted, so both
    slots are proven wired by actually authenticating through them.
    """
    authenticator = build_authenticator(_api_key_and_session_settings())
    token = mint_session_token(email="dev@example.com", secret=_SESSION_SECRET, ttl_hours=8)

    api_key_principal = await authenticator.authenticate(
        bearer=None, api_key=_BOTH_PROVIDERS_RAW_KEY, session=None
    )
    session_principal = await authenticator.authenticate(bearer=None, api_key=None, session=token)

    assert api_key_principal.email == "svc@example.com"
    assert session_principal.email == "dev@example.com"
