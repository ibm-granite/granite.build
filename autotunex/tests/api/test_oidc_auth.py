"""OIDC bearer authentication, proven through the real ASGI stack.

Builds the app with a real ``OidcBearerVerifier`` but a stubbed key resolver —
still no network, since the resolver is swapped on ``app.state`` after
``create_app`` rather than routed through a real JWKS endpoint.

Fixture shape: overriding the conftest ``settings`` fixture is enough to make
``conftest``'s ``app`` fixture build an OIDC-configured app, and a separate
``signing_key`` fixture holds the private key the tests mint tokens with. That
keeps the key out of ``app.state`` (where it does not belong) and off
``client._transport`` (which is httpx-private).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from http import HTTPStatus

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from httpx import AsyncClient

from autotunex.core.auth.oidc import OidcBearerVerifier
from autotunex.core.auth.routing import RoutingAuthenticator
from autotunex.core.config import Settings
from autotunex.db.tables import JobTable, UserTable
from tests.conftest import API, make_settings

ISSUER = "https://idp.example.com/oauth2"
AUDIENCE = "test-client-id"


class _FakeResolver:
    """Resolves every token to one fixed key. No network, no JWKS endpoint."""

    def __init__(self, key: object) -> None:
        self._key = key

    async def resolve_signing_key(self, token: str) -> object:
        return self._key


@pytest.fixture(scope="module")
def signing_key() -> rsa.RSAPrivateKey:
    """One keypair for the module — 2048-bit generation is slow enough to matter."""
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def settings() -> Settings:
    """Overrides the conftest default so ``app`` is built with OIDC enabled.

    Built through ``make_settings`` rather than a raw ``Settings(...)``: that
    factory is the one place ``_env_file=None`` and every auth field is pinned
    together, so a developer's own ``.env`` (which might carry real OIDC
    values) cannot leak into this test.
    """
    return make_settings(
        auth_providers=["oidc"],
        oidc_issuer=ISSUER,
        # RFC 2606 `.invalid`, which cannot resolve, rather than
        # `example.com`, which does. Nothing here should ever fetch it —
        # `stub_the_key_resolver` replaces the whole authenticator — but if
        # that fixture regressed, a resolvable host turns a bug into six tests
        # attempting real outbound HTTPS and hanging offline, instead of
        # failing immediately. Same ruling as `tests/core/auth/test_registry.py`.
        oidc_jwks_uri="https://unused.invalid/jwks",
        oidc_audience=AUDIENCE,
    )


@pytest.fixture(autouse=True)
def stub_the_key_resolver(app: FastAPI, signing_key: rsa.RSAPrivateKey) -> None:
    """Swap in a verifier whose key resolver never touches the network.

    ``app.state.authenticator`` is built in ``create_app`` and is not a
    dependency, so this replaces it outright rather than overriding anything.
    ``autouse`` because every test in this file needs it and forgetting it would
    surface as a JWKS fetch attempt against the unresolvable ``.invalid`` host
    the ``settings`` fixture configures — which fails fast rather than hanging.
    """
    app.state.authenticator = RoutingAuthenticator(
        bearer_verifier=OidcBearerVerifier(
            issuer=ISSUER,
            audience=AUDIENCE,
            algorithms=["RS256"],
            email_claims=["email", "emailAddress"],
            leeway_seconds=30,
            key_resolver=_FakeResolver(signing_key.public_key()),
        )
    )


def _token(private_key: rsa.RSAPrivateKey, **overrides: object) -> str:
    now = datetime.now(UTC)
    claims: dict[str, object] = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "email": "tester@example.com",
        "exp": int((now + timedelta(minutes=5)).timestamp()),
    }
    claims.update(overrides)
    return jwt.encode(claims, private_key, algorithm="RS256")


async def test_a_valid_bearer_token_authenticates(
    client: AsyncClient, signing_key: rsa.RSAPrivateKey, job: JobTable, user: UserTable
) -> None:
    token = _token(signing_key, email=user.email)

    response = await client.get(f"{API}/jobs", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == HTTPStatus.OK
    assert response.json()["total"] == 1


async def test_a_valid_token_for_an_unprovisioned_email_sees_an_empty_page(
    client: AsyncClient, signing_key: rsa.RSAPrivateKey, job: JobTable
) -> None:
    """Authentication succeeds; identity resolves to nothing. Spec decision 5."""
    token = _token(signing_key, email="ghost@example.com")

    response = await client.get(f"{API}/jobs", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == HTTPStatus.OK
    assert response.json()["total"] == 0


async def test_an_email_differing_only_in_case_still_resolves(
    client: AsyncClient, signing_key: rsa.RSAPrivateKey, job: JobTable, user: UserTable
) -> None:
    """W3ID may return a differently-cased address than the ``users`` row holds.

    ``docs/schema-review.md`` C7: MySQL folds case here, SQLite and Postgres do
    not — so without the lowered comparison this passes in production and fails
    in tests, or the reverse. This is the test that pins it end to end.
    """
    token = _token(signing_key, email=user.email.upper())

    response = await client.get(f"{API}/jobs", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == HTTPStatus.OK
    assert response.json()["total"] == 1


async def test_an_expired_bearer_token_is_rejected_with_the_expired_challenge(
    client: AsyncClient, signing_key: rsa.RSAPrivateKey
) -> None:
    expired = int((datetime.now(UTC) - timedelta(minutes=1)).timestamp())
    token = _token(signing_key, exp=expired)

    response = await client.get(f"{API}/jobs", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == HTTPStatus.UNAUTHORIZED
    assert "expired" in response.headers["www-authenticate"]


async def test_a_bearer_token_for_a_different_audience_is_rejected(
    client: AsyncClient, signing_key: rsa.RSAPrivateKey
) -> None:
    token = _token(signing_key, aud="someone-elses-client")

    response = await client.get(f"{API}/jobs", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == HTTPStatus.UNAUTHORIZED


async def test_no_fragment_of_a_rejected_token_is_logged(
    client: AsyncClient, signing_key: rsa.RSAPrivateKey, caplog: pytest.LogCaptureFixture
) -> None:
    """Spec §5: before the signature checks out, every claim is attacker-controlled.

    Absence of the token is necessary but not sufficient evidence — a test
    that only checks absence would pass equally well if nothing had been
    logged at all, which is why the assertion below also requires that our own
    WARNING actually fired. A bare ``assert caplog.records`` would not
    distinguish that from noise: httpx emits its own INFO record per request
    regardless of what this module logs.
    """
    token = _token(signing_key, iss="https://not-the-configured-issuer.example.com")

    with caplog.at_level(logging.DEBUG):
        response = await client.get(f"{API}/jobs", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == HTTPStatus.UNAUTHORIZED
    assert token not in caplog.text
    assert token.split(".")[1] not in caplog.text
    assert any(
        record.name.startswith("autotunex") and record.levelno == logging.WARNING
        for record in caplog.records
    )
