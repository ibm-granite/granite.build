"""The backend-for-frontend login flow.

No real network anywhere: the token endpoint is an ``httpx.MockTransport`` on the
client stored at ``app.state.http_client``, and the ID token is verified against
an in-process RSA keypair through a stubbed ``SigningKeyResolver`` — the same two
techniques already proven in the OIDC-bearer phase's own tests
(``tests/api/test_oidc_auth.py``).

Several things about the fixtures are load-bearing. Two of the claims below were
corrected during review after being checked directly against the installed
``httpx`` (0.28.1): a claim of "verified against the installed version" is only
as good as the check that produced it, and this file has already been wrong once
about what one of these libraries actually does — see the ``get_list`` bullet.

* ``base_url`` is **https**. Cookies here are set ``Secure``, and
  ``http.cookiejar`` refuses to return a ``Secure`` cookie on a plain-http
  request — it stores it, so ``response.cookies`` looks right, and then never
  sends it again. Over http, ``/auth/callback`` never sees ``oauth_flow``: the
  happy-path test fails, and the mismatched-``state`` test passes while asserting
  nothing, which is the worse of the two outcomes.
* ``app.state.http_client`` is set by ``stub_the_outbound_collaborators``, not by
  ``lifespan``. ``ASGITransport`` does not run lifespan, and FastAPI resolves
  every dependency *before* entering the handler — so without this, even a
  request the handler would reject on its first line 500s on an
  ``AttributeError`` instead of 401ing.
* Every host below is an RFC 2606 ``.invalid`` one, never the real
  ``idp.example.com``. Nothing here should ever resolve one of these
  hosts — every outbound call is stubbed — but a mutation probe that removed a
  stub would otherwise dial a real, reachable host and hang instead of failing
  fast offline.
* ``response.headers["set-cookie"]`` does **not** surface only the first
  ``Set-Cookie`` header — it joins every matching value into one string with
  ``", "``. ``/auth/callback`` emits two: the ``oauth_flow`` deletion, then
  ``session``. Indexing therefore returns something like
  ``'oauth_flow=""; Max-Age=0; Secure, session=TOKEN; HttpOnly; Secure'``, and an
  attribute assertion meant for the session cookie (``"secure" in set_cookie``,
  say) can be satisfied by the *deletion* header's attributes instead, silently
  passing regardless of what the session cookie actually carries. httpx's own
  cookie jar (``response.cookies``) is not subject to this — it does see every
  header — but that does not help an attribute-level assertion, which is why
  every test below selects the cookie it means by name from
  ``response.headers.get_list("set-cookie")`` rather than indexing.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable, Iterator
from datetime import UTC, datetime, timedelta
from http import HTTPStatus
from urllib.parse import parse_qs, urlparse

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from autotunex.core.auth.oidc import OidcBearerVerifier
from autotunex.core.config import Settings, get_settings
from autotunex.models.auth import Principal
from tests.conftest import API, make_settings

ISSUER = "https://idp.invalid/oauth2"
AUDIENCE = "my-client-id"
AUTHORIZATION_ENDPOINT = "https://idp.invalid/oauth2/authorize"
TOKEN_ENDPOINT = "https://idp.invalid/oauth2/token"
PUBLIC_BASE_URL = "https://autotunex.invalid"
# Long and distinctive on purpose: the leak tests below scan every response
# body/header for this exact substring. A short secret like the previous
# `"shh"` gives a three-character base64url run roughly a 1-in-1000 chance of
# turning up somewhere in the session JWT by coincidence — a rare false
# *alarm*, and a weak signal either way.
CLIENT_SECRET = "CLIENT-SECRET-MUST-NOT-LEAK-c0ffee"
SESSION_SECRET = "test-auth-routes-session-secret-at-least-32-chars"


class _FakeResolver:
    def __init__(self, key: object) -> None:
        self._key = key

    async def resolve_signing_key(self, token: str) -> object:
        return self._key


@pytest.fixture(scope="module")
def signing_key() -> rsa.RSAPrivateKey:
    """One keypair per module; 2048-bit generation is slow enough to matter."""
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def settings() -> Settings:
    """Overrides the conftest default so ``app`` is built with the BFF enabled.

    Built through ``make_settings`` (see ruling 5 of this task's brief
    amendments), not a raw ``Settings(...)``: that factory is the single place
    ``_env_file=None`` and every field is pinned together, so a developer's own
    ``.env`` cannot leak a real OIDC endpoint or a short session secret into
    this file.
    """
    return make_settings(
        auth_providers=["session"],
        oidc_issuer=ISSUER,
        oidc_jwks_uri="https://unused.invalid/jwks",
        oidc_audience=AUDIENCE,
        oidc_client_id=AUDIENCE,
        oidc_client_secret=CLIENT_SECRET,
        oidc_authorization_endpoint=AUTHORIZATION_ENDPOINT,
        oidc_token_endpoint=TOKEN_ENDPOINT,
        oidc_end_session_endpoint="https://idp.invalid/oauth2/logout",
        public_base_url=PUBLIC_BASE_URL,
        session_secret=SESSION_SECRET,
    )


def _id_token(private_key: rsa.RSAPrivateKey, **overrides: object) -> str:
    now = datetime.now(UTC)
    claims: dict[str, object] = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "email": "dev@example.com",
        "exp": int((now + timedelta(minutes=5)).timestamp()),
    }
    claims.update(overrides)
    return jwt.encode(claims, private_key, algorithm="RS256")


def _token_endpoint(body: dict[str, object], status: int = 200) -> AsyncClient:
    """An ``AsyncClient`` whose every request returns ``body``."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body)

    return AsyncClient(transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def stub_the_outbound_collaborators(app: FastAPI, signing_key: rsa.RSAPrivateKey) -> Iterator[None]:
    """Give the app an ID-token verifier and an HTTP client that reach no network.

    ``autouse`` because *every* request to ``/auth/callback`` resolves
    ``get_http_client`` before the handler runs, so a test that forgets this gets
    a 500 rather than the 401 it is asserting. A test that needs a different
    token-endpoint response replaces ``app.state.http_client`` itself.
    """
    app.state.id_token_verifier = OidcBearerVerifier(
        issuer=ISSUER,
        audience=AUDIENCE,
        algorithms=["RS256"],
        email_claims=["email", "emailAddress"],
        leeway_seconds=30,
        key_resolver=_FakeResolver(signing_key.public_key()),
    )
    app.state.http_client = _token_endpoint({"id_token": _id_token(signing_key)})

    yield

    # Not a close, just clearing the slot: the MockTransport-backed client
    # opened above holds no socket, so there is nothing to release, and this
    # fixture never awaits `.aclose()` on it (a fixture teardown after `yield`
    # runs synchronously here, so it could not `await` regardless). Confirmed
    # to leak nothing: `pytest -W error::ResourceWarning` passes this file.
    app.state.http_client = None


@pytest.fixture
async def bff_client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    """An https client, because the cookies under test are ``Secure``.

    ``conftest``'s ``client`` fixture uses ``http://testserver``, which is fine
    for the job endpoints but silently drops every cookie in this file.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="https://testserver") as client:
        yield client


async def _login(client: AsyncClient) -> str:
    """Complete ``/auth/login`` and return the ``state`` it generated."""
    response = await client.get("/auth/login", follow_redirects=False)
    return parse_qs(urlparse(response.headers["location"]).query)["state"][0]


def _flow_cookie(
    *,
    secret: str = SESSION_SECRET,
    state: str = "expected-state",
    omit_state: bool = False,
    omit_verifier: bool = False,
    omit_exp: bool = False,
    exp: int | None = None,
) -> str:
    """Build an ``oauth_flow`` JWT directly, bypassing ``/login``.

    Gives full control over the payload — an expired or absent ``exp``, the
    wrong signing secret, or a missing ``state``/``verifier`` claim — none of
    which ``/login`` itself can be made to produce, since it always mints a
    well-formed, freshly-signed cookie carrying all three. Set the return
    value on a client's cookie jar directly
    (``client.cookies.set("oauth_flow", ...)``) rather than going through
    ``/login``.
    """
    now = datetime.now(UTC)
    claims: dict[str, object] = {}
    if not omit_state:
        claims["state"] = state
    if not omit_exp:
        claims["exp"] = exp if exp is not None else int((now + timedelta(minutes=5)).timestamp())
    if not omit_verifier:
        claims["verifier"] = "expected-verifier"
    return jwt.encode(claims, secret, algorithm="HS256")


def _set_cookie_for(response: httpx.Response, name: str) -> str:
    """Return the ``Set-Cookie`` header for ``name``, lower-cased.

    ``/auth/callback`` emits two ``Set-Cookie`` headers — the ``oauth_flow``
    deletion first, then ``session``. Plain ``response.headers[...]`` indexing
    joins both into one comma-separated string rather than surfacing only the
    first, so an attribute assertion aimed at the session cookie can be
    satisfied by the deletion header's attributes instead. Selecting by name
    from ``get_list`` is what actually inspects the session cookie's own
    attributes rather than the flow cookie's deletion (see ruling 8).
    """
    for value in response.headers.get_list("set-cookie"):
        if value.split("=", 1)[0].strip() == name:
            return value.lower()
    raise AssertionError(f"no Set-Cookie header found for {name!r}")


# --- /auth/login ------------------------------------------------------------


async def test_login_redirects_to_the_authorization_endpoint_with_state_and_pkce(
    bff_client: AsyncClient,
) -> None:
    response = await bff_client.get("/auth/login", follow_redirects=False)

    assert response.status_code == HTTPStatus.FOUND
    location = urlparse(response.headers["location"])
    query = parse_qs(location.query)
    assert location.geturl().startswith(AUTHORIZATION_ENDPOINT)
    assert query["code_challenge_method"] == ["S256"]
    assert query["redirect_uri"] == [f"{PUBLIC_BASE_URL}/auth/callback"]
    assert "state" in query
    assert "code_challenge" in query
    assert "oauth_flow" in response.cookies


async def test_login_ignores_forwarded_headers_when_building_redirect_uri(
    bff_client: AsyncClient,
) -> None:
    """``redirect_uri`` comes from ``public_base_url`` only — spec §7.

    Building it from ``x-forwarded-host`` / ``x-forwarded-proto`` instead — an
    attacker-controlled pair — would redirect the authorization code to a host
    of their choosing.
    """
    response = await bff_client.get(
        "/auth/login",
        follow_redirects=False,
        headers={"X-Forwarded-Host": "evil.example.com", "X-Forwarded-Proto": "http"},
    )

    query = parse_qs(urlparse(response.headers["location"]).query)

    assert query["redirect_uri"] == [f"{PUBLIC_BASE_URL}/auth/callback"]
    assert "evil.example.com" not in response.headers["location"]


async def test_the_flow_cookie_is_hidden_from_javascript_but_not_encrypted(
    bff_client: AsyncClient,
) -> None:
    """States the protection that actually holds, not one that does not.

    The PKCE verifier genuinely *is* base64-readable inside this signed (not
    encrypted) JWT — anyone who can read the cookie can read the verifier. That
    is fine: ``httponly`` keeps it out of reach of any script running on the
    page, and PKCE's whole design binds the authorization code to whoever holds
    this cookie, not to whoever cannot decode a JWT.
    """
    response = await bff_client.get("/auth/login", follow_redirects=False)

    set_cookie = response.headers["set-cookie"].lower()

    assert "httponly" in set_cookie
    assert "secure" in set_cookie


async def test_login_returns_unauthorized_when_the_session_provider_is_not_enabled(
    app: FastAPI, caplog: pytest.LogCaptureFixture
) -> None:
    """The provider gate, not a bare ``session_secret is None`` check (ruling 1).

    ``auth_providers=["oidc"]`` here carries none of the BFF settings at all,
    which is the ordinary shape of an OIDC-only deployment. Only the
    ``get_settings`` override is replaced — the ``get_session`` override
    ``conftest``'s ``app`` fixture already installed is left alone, since
    clearing it is unrelated to what this test asserts and would silently
    change behaviour for any later use of ``app`` in the same test.

    Also pins that this rejection is logged: this gate used to fail silently,
    which is exactly the kind of misconfiguration (a deployment that meant to
    run ``"session"`` but left it off ``AUTOTUNEX_AUTH_PROVIDERS``) an operator
    most needs a trace for.
    """
    caplog.set_level(logging.WARNING)
    app.dependency_overrides[get_settings] = lambda: make_settings(
        auth_providers=["oidc"],
        oidc_issuer=ISSUER,
        oidc_jwks_uri="https://unused.invalid/jwks",
        oidc_audience=AUDIENCE,
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="https://testserver") as client:
        response = await client.get("/auth/login", follow_redirects=False)

    assert response.status_code == HTTPStatus.UNAUTHORIZED
    warnings = [
        record
        for record in caplog.records
        if record.name.startswith("autotunex") and record.levelno == logging.WARNING
    ]
    assert any("session" in record.getMessage() for record in warnings)


async def test_login_never_leaks_a_broken_redirect_when_only_session_secret_is_set(
    app: FastAPI,
) -> None:
    """The exact scenario ruling 1 names: a stray ``session_secret`` with "session" disabled.

    Checking only ``session_secret is None`` (the brief's original guard) lets
    this configuration fall through, since ``session_secret`` genuinely is set
    here — building a redirect to the literal URL
    ``None?response_type=code&client_id=None&...`` because
    ``oidc_authorization_endpoint``, ``oidc_client_id`` and ``public_base_url``
    are all still ``None``. Gating on ``"session" in auth_providers`` first
    closes that: no field is read until the provider check passes, so no
    response — success or failure — can ever contain the four-letter string
    ``"None"``.
    """
    app.dependency_overrides[get_settings] = lambda: make_settings(
        auth_providers=["oidc"],
        oidc_issuer=ISSUER,
        oidc_jwks_uri="https://unused.invalid/jwks",
        oidc_audience=AUDIENCE,
        session_secret=SESSION_SECRET,
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="https://testserver") as client:
        response = await client.get("/auth/login", follow_redirects=False)

    assert response.status_code == HTTPStatus.UNAUTHORIZED
    for value in response.headers.values():
        assert "None" not in value


# --- /auth/callback ----------------------------------------------------------


async def test_callback_with_a_mismatched_state_is_rejected(
    bff_client: AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    """Genuinely exercised only because the client speaks https and so has the cookie.

    Also pins that the rejection is logged, and that neither the real
    (flow-cookie) ``state`` nor the caller-supplied one reaches the log text —
    an operator needs to know a mismatch happened, not what either value was.
    """
    caplog.set_level(logging.WARNING)
    real_state = await _login(bff_client)

    response = await bff_client.get(
        "/auth/callback", params={"code": "irrelevant", "state": "wrong-state"}
    )

    assert response.status_code == HTTPStatus.UNAUTHORIZED
    warnings = [
        record
        for record in caplog.records
        if record.name.startswith("autotunex") and record.levelno == logging.WARNING
    ]
    assert any("state" in record.getMessage() for record in warnings)
    for record in warnings:
        message = record.getMessage()
        assert "wrong-state" not in message
        assert real_state not in message


async def test_callback_with_a_non_ascii_state_is_rejected_not_a_500(
    bff_client: AsyncClient,
) -> None:
    """The critical fix: ``secrets.compare_digest`` on ``str`` rejects non-ASCII input.

    A live flow cookie from ``/login`` matters here — without it, this request
    would 401 on the cookie-absence check several lines earlier, having never
    reached the comparison this test exists to exercise. ``state`` is a raw
    query parameter that Starlette percent-decodes as UTF-8, so an
    unauthenticated caller fully controls its content; comparing it against the
    cookie's ``state`` claim as ``str`` (not bytes) raises ``TypeError`` on any
    non-ASCII character, which would otherwise escape this handler as a 500.
    """
    await _login(bff_client)

    response = await bff_client.get("/auth/callback", params={"code": "irrelevant", "state": "ü"})

    assert response.status_code == HTTPStatus.UNAUTHORIZED


async def test_callback_with_no_flow_cookie_at_all_is_rejected(
    bff_client: AsyncClient,
) -> None:
    response = await bff_client.get(
        "/auth/callback", params={"code": "irrelevant", "state": "anything"}
    )

    assert response.status_code == HTTPStatus.UNAUTHORIZED


async def test_callback_with_missing_code_and_an_error_from_the_idp_is_rejected(
    bff_client: AsyncClient,
) -> None:
    """The "Cancel" path: W3ID redirects back with ``error=...`` and no ``code``.

    ``code``/``state`` are declared optional on the handler itself precisely so
    this request fails through the same opaque 401 every other rejection in
    this module does, rather than FastAPI's request-validation layer answering
    with a 422 before the handler runs — a second, distinguishable "rejected"
    shape a prober could otherwise use. The IdP's ``error`` value is not read
    by the handler and is not asserted on here either: only its presence
    (simulating a real cancellation redirect) and the resulting status code
    matter.
    """
    state = await _login(bff_client)

    response = await bff_client.get(
        "/auth/callback", params={"error": "access_denied", "state": state}
    )

    assert response.status_code == HTTPStatus.UNAUTHORIZED


async def test_callback_with_an_expired_flow_cookie_is_rejected(
    bff_client: AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    """Pins PyJWT's own default ``verify_exp``, not ruling 4's ``options=`` hardening.

    ``jwt.decode`` rejects an expired token under its *default* options — this
    holds even with ``options={}``, independent of the ``require`` list this
    module also passes. What this test actually depends on is that default
    plus this handler's ``except jwt.ExpiredSignatureError: ... raise
    InvalidCredentialsError() from None`` mapping the resulting exception to a
    401 rather than letting it escape unhandled. Confirmed by a mutation
    probe: dropping ``"exp"`` from ``require`` leaves this test passing for
    exactly this reason. Set directly rather than via ``/login`` — see
    :func:`_flow_cookie`.

    Also pins the level: an expired flow cookie logs at INFO, not WARNING,
    since it is routine (the user took a while, or returned to a stale tab)
    and self-correcting (clicking "log in" again mints a fresh one) — unlike
    every other rejection in this module.
    """
    caplog.set_level(logging.INFO)
    expired = _flow_cookie(exp=int((datetime.now(UTC) - timedelta(minutes=1)).timestamp()))
    bff_client.cookies.set("oauth_flow", expired)

    response = await bff_client.get(
        "/auth/callback", params={"code": "auth-code", "state": "expected-state"}
    )

    assert response.status_code == HTTPStatus.UNAUTHORIZED
    autotunex_records = [record for record in caplog.records if record.name.startswith("autotunex")]
    assert any(
        record.levelno == logging.INFO and "expired" in record.getMessage()
        for record in autotunex_records
    )
    assert not any(record.levelno == logging.WARNING for record in autotunex_records)


async def test_callback_with_a_flow_cookie_signed_by_a_different_secret_is_rejected(
    bff_client: AsyncClient,
) -> None:
    forged = _flow_cookie(secret="a-completely-different-secret-at-least-32-characters")
    bff_client.cookies.set("oauth_flow", forged)

    response = await bff_client.get(
        "/auth/callback", params={"code": "auth-code", "state": "expected-state"}
    )

    assert response.status_code == HTTPStatus.UNAUTHORIZED


async def test_callback_with_a_flow_cookie_missing_the_exp_claim_is_rejected(
    bff_client: AsyncClient,
) -> None:
    """Makes ``require: ["exp", ...]`` load-bearing — no later guard checks ``exp`` at all.

    Without ``require``, a token that simply omits ``exp`` decodes cleanly
    under ``jwt.decode``'s defaults (``verify_exp`` only checks an ``exp``
    that is present; it does not demand one exist), and nothing downstream —
    unlike ``state`` and ``verifier`` — has an ``isinstance``/truthiness guard
    to catch a missing claim. A mutation probe dropping ``"exp"`` from
    ``require`` confirms this: the request does not error, it *succeeds* — a
    302 with a freshly minted session cookie, since ``state`` and ``verifier``
    are both present and valid. That would be a flow cookie that never
    expires, gated only by knowledge of the signing secret.
    """
    no_exp = _flow_cookie(omit_exp=True)
    bff_client.cookies.set("oauth_flow", no_exp)

    response = await bff_client.get(
        "/auth/callback", params={"code": "auth-code", "state": "expected-state"}
    )

    assert response.status_code == HTTPStatus.UNAUTHORIZED


async def test_callback_with_a_flow_cookie_missing_the_state_claim_is_rejected(
    bff_client: AsyncClient,
) -> None:
    """Cheap once :func:`_flow_cookie` supports ``omit_exp`` — the ``state`` analogue.

    Unlike the ``exp`` case above, this one does *not* depend on ``require``:
    ``flow_claims.get("state")`` returns ``None`` for an absent claim, and the
    ``isinstance(flow_state, str)`` guard a few lines below the decode call
    catches that regardless of whether ``"state"`` is in ``require``. A
    mutation probe dropping ``"state"`` from ``require`` confirms this test
    keeps passing — the ``isinstance`` guard alone is sufficient here, unlike
    for ``exp``, which has no such guard.
    """
    no_state = _flow_cookie(omit_state=True)
    bff_client.cookies.set("oauth_flow", no_state)

    response = await bff_client.get(
        "/auth/callback", params={"code": "auth-code", "state": "expected-state"}
    )

    assert response.status_code == HTTPStatus.UNAUTHORIZED


async def test_callback_with_a_flow_cookie_missing_the_verifier_claim_is_rejected(
    bff_client: AsyncClient,
) -> None:
    """Pins the *disjunction* of ``require: ["verifier"]`` and the ``isinstance`` guard below.

    Not either alone: with ``"verifier"`` dropped from ``require``, decode
    still succeeds (the claim is merely absent, not malformed),
    ``flow_claims.get("verifier")`` returns ``None``, and
    ``isinstance(flow_verifier, str)`` catches that anyway — a mutation probe
    confirms this test keeps passing with ``require`` alone removed. It
    genuinely does depend on *one* of the two guards existing, just not on
    ``require`` specifically, unlike the ``exp`` case above.
    """
    malformed = _flow_cookie(omit_verifier=True)
    bff_client.cookies.set("oauth_flow", malformed)

    response = await bff_client.get(
        "/auth/callback", params={"code": "auth-code", "state": "expected-state"}
    )

    assert response.status_code == HTTPStatus.UNAUTHORIZED


async def test_callback_with_a_directly_set_valid_flow_cookie_completes_the_flow(
    bff_client: AsyncClient,
) -> None:
    """Positive control for :func:`_flow_cookie`: proves delivery, not just rejection.

    A 401 is also exactly what an *undelivered* cookie produces, so the four
    negative ``_flow_cookie`` tests above cannot, on their own, distinguish
    "the guard rejected this payload" from "the cookie never reached the
    handler" — a future change that silently stopped delivering the cookie
    would leave all four green for the wrong reason. This test uses the same
    delivery mechanism (``bff_client.cookies.set(...)``, bypassing ``/login``)
    with a valid, well-formed cookie whose ``state`` claim matches the query
    parameter, and expects the ordinary happy-path outcome — a 302 with a
    minted session cookie — which only happens if the cookie actually arrived
    and passed every guard in ``/callback``.
    """
    valid = _flow_cookie(state="expected-state")
    bff_client.cookies.set("oauth_flow", valid)

    response = await bff_client.get(
        "/auth/callback",
        params={"code": "auth-code", "state": "expected-state"},
        follow_redirects=False,
    )

    assert response.status_code == HTTPStatus.FOUND
    assert "session" in response.cookies


async def test_callback_returns_unauthorized_when_the_session_provider_is_not_enabled(
    app: FastAPI, bff_client: AsyncClient
) -> None:
    """``/callback``'s own provider gate, distinct from and untested by ``/login``'s.

    ``id_token_verifier is None`` is the only thing that gates ``/callback`` on
    ``"session"`` being enabled — nothing upstream of it re-checks
    ``settings.auth_providers`` (see :func:`_require_callback_settings`'s
    docstring). A valid, live flow cookie is present so this can only be the
    provider gate rejecting the request, not the cookie-absence check.
    """
    state = await _login(bff_client)
    app.state.id_token_verifier = None

    response = await bff_client.get("/auth/callback", params={"code": "auth-code", "state": state})

    assert response.status_code == HTTPStatus.UNAUTHORIZED


async def test_callback_completes_the_flow_and_sets_a_session_cookie(
    bff_client: AsyncClient,
) -> None:
    state = await _login(bff_client)

    response = await bff_client.get(
        "/auth/callback", params={"code": "auth-code", "state": state}, follow_redirects=False
    )

    assert response.status_code == HTTPStatus.FOUND
    assert "session" in response.cookies
    set_cookie = _set_cookie_for(response, "session")
    assert "httponly" in set_cookie
    assert "secure" in set_cookie
    assert "samesite=lax" in set_cookie


async def test_callback_clears_the_flow_cookie_before_setting_the_session_cookie(
    bff_client: AsyncClient,
) -> None:
    """Pins ruling 8's observation: two ``Set-Cookie`` headers, deletion first.

    Written as its own test, separate from the happy-path assertions above, so
    a future change to header order is caught by name rather than by
    coincidentally still passing.
    """
    state = await _login(bff_client)

    response = await bff_client.get(
        "/auth/callback", params={"code": "auth-code", "state": state}, follow_redirects=False
    )

    set_cookie_headers = response.headers.get_list("set-cookie")
    assert len(set_cookie_headers) == 2
    names = [value.split("=", 1)[0].strip() for value in set_cookie_headers]
    assert names == ["oauth_flow", "session"]
    # Starlette's `delete_cookie` expires immediately: `max-age=0` and an
    # `expires` date in the past, distinguishing "deleted" from "set".
    assert "max-age=0" in set_cookie_headers[0].lower()


async def test_a_failed_token_exchange_is_a_401(app: FastAPI, bff_client: AsyncClient) -> None:
    app.state.http_client = _token_endpoint({"error": "invalid_grant"}, status=400)
    state = await _login(bff_client)

    response = await bff_client.get("/auth/callback", params={"code": "bad", "state": state})

    assert response.status_code == HTTPStatus.UNAUTHORIZED


async def test_a_token_response_with_no_id_token_is_a_401(
    app: FastAPI, bff_client: AsyncClient
) -> None:
    app.state.http_client = _token_endpoint({"access_token": "opaque-thing"})
    state = await _login(bff_client)

    response = await bff_client.get("/auth/callback", params={"code": "code", "state": state})

    assert response.status_code == HTTPStatus.UNAUTHORIZED


async def test_callback_rejects_a_whitespace_only_email_claim(
    app: FastAPI, bff_client: AsyncClient, signing_key: rsa.RSAPrivateKey
) -> None:
    """The ``.strip()`` guard: ``OidcBearerVerifier`` treats any truthy string as present.

    Without ``.strip()``, a whitespace-only claim (``"   "``) passes
    ``principal.email is None`` and would mint a session cookie for an
    identity ``SessionCookieVerifier`` strips and rejects on the very next
    request — a login loop, not a bypass, but still an identity this
    codebase's own session verifier would refuse.
    """
    app.state.http_client = _token_endpoint({"id_token": _id_token(signing_key, email="   ")})
    state = await _login(bff_client)

    response = await bff_client.get("/auth/callback", params={"code": "auth-code", "state": state})

    assert response.status_code == HTTPStatus.UNAUTHORIZED


async def test_callback_rejects_an_id_token_for_the_wrong_audience(
    app: FastAPI, bff_client: AsyncClient, signing_key: rsa.RSAPrivateKey
) -> None:
    app.state.http_client = _token_endpoint(
        {"id_token": _id_token(signing_key, aud="someone-elses-client")}
    )
    state = await _login(bff_client)

    response = await bff_client.get("/auth/callback", params={"code": "auth-code", "state": state})

    assert response.status_code == HTTPStatus.UNAUTHORIZED


async def test_a_connect_error_from_the_token_endpoint_is_a_401_not_a_500(
    app: FastAPI, bff_client: AsyncClient
) -> None:
    """The brief's own defect: an unreachable token endpoint must not 500.

    ``httpx.MockTransport`` raising rather than returning a response is exactly
    what ``httpx.AsyncClient.post`` sees for a real connection failure — no
    real network is touched.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    app.state.http_client = AsyncClient(transport=httpx.MockTransport(handler))
    state = await _login(bff_client)

    response = await bff_client.get("/auth/callback", params={"code": "auth-code", "state": state})

    assert response.status_code == HTTPStatus.UNAUTHORIZED


async def test_a_non_json_token_response_is_a_401_not_a_500(
    app: FastAPI, bff_client: AsyncClient
) -> None:
    """An HTML error page from a misconfigured proxy must not 500 either."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>not json</html>")

    app.state.http_client = AsyncClient(transport=httpx.MockTransport(handler))
    state = await _login(bff_client)

    response = await bff_client.get("/auth/callback", params={"code": "auth-code", "state": state})

    assert response.status_code == HTTPStatus.UNAUTHORIZED


async def test_the_client_secret_never_reaches_the_browser(
    bff_client: AsyncClient,
) -> None:
    login = await bff_client.get("/auth/login", follow_redirects=False)
    state = parse_qs(urlparse(login.headers["location"]).query)["state"][0]

    callback = await bff_client.get(
        "/auth/callback", params={"code": "auth-code", "state": state}, follow_redirects=False
    )

    for response in (login, callback):
        assert CLIENT_SECRET not in response.headers.get("set-cookie", "")
        assert CLIENT_SECRET not in response.headers.get("location", "")
        assert CLIENT_SECRET not in response.text


async def test_the_session_secret_never_reaches_the_browser(
    bff_client: AsyncClient,
) -> None:
    login = await bff_client.get("/auth/login", follow_redirects=False)
    state = parse_qs(urlparse(login.headers["location"]).query)["state"][0]

    callback = await bff_client.get(
        "/auth/callback", params={"code": "auth-code", "state": state}, follow_redirects=False
    )

    for response in (login, callback):
        assert SESSION_SECRET not in response.headers.get("set-cookie", "")
        assert SESSION_SECRET not in response.headers.get("location", "")
        assert SESSION_SECRET not in response.text


# --- /auth/me -----------------------------------------------------------


async def test_me_requires_authentication(bff_client: AsyncClient) -> None:
    response = await bff_client.get("/auth/me")

    assert response.status_code == HTTPStatus.UNAUTHORIZED


async def test_me_reports_the_authenticated_principal(
    bff_client: AsyncClient, as_principal: Callable[[Principal], None]
) -> None:
    as_principal(Principal(email="dev@example.com", provider="session", is_admin=False))

    response = await bff_client.get("/auth/me")

    assert response.status_code == HTTPStatus.OK
    assert response.json()["email"] == "dev@example.com"


async def test_a_real_session_cookie_authenticates_the_rest_of_the_api(
    bff_client: AsyncClient,
) -> None:
    """The end-to-end point of this phase: log in, then use the API as that user.

    Everything before this proves a piece; this proves the cookie ``/callback``
    mints is one ``SessionCookieVerifier`` accepts, with no override anywhere in
    the chain — ``get_principal`` is not replaced in this test body, unlike
    every other test in this file that touches ``/auth/me``. (``get_session``
    *is* overridden, by the shared ``app`` fixture in ``tests/conftest.py``, for
    every test in this module including this one — that override is what lets
    the job query below run at all; it is unrelated to the principal chain this
    test actually exercises.)

    The job endpoint is asserted to return an empty page, not a 401 and not
    data. ``dev@example.com`` (the ID token's email, minted by ``_id_token``) is not
    a row in the ``users`` table this app's ``get_session`` override binds to,
    so stage two of principal resolution (``get_principal``, see
    ``api/deps.py``) resolves ``user_id=None, is_admin=False`` — the documented
    "authenticated but unprovisioned" case, which ``JobService`` answers with a
    200 and an empty page rather than a 401. That is the correct expectation
    here, not a weaker stand-in for one with rows.
    """
    state = await _login(bff_client)
    await bff_client.get(
        "/auth/callback", params={"code": "auth-code", "state": state}, follow_redirects=False
    )

    me_response = await bff_client.get("/auth/me")
    jobs_response = await bff_client.get(f"{API}/jobs")

    assert me_response.status_code == HTTPStatus.OK
    assert me_response.json()["email"] == "dev@example.com"
    assert jobs_response.status_code == HTTPStatus.OK
    assert jobs_response.json()["items"] == []
    assert jobs_response.json()["total"] == 0


# --- /auth/logout ------------------------------------------------------------


async def test_logout_requires_authentication(bff_client: AsyncClient) -> None:
    """An unauthenticated caller must not learn ``end_session_endpoint`` at all.

    Otherwise ``/logout`` becomes an oracle for whether an IdP is configured,
    with no credential required to ask it.
    """
    response = await bff_client.post("/auth/logout")

    assert response.status_code == HTTPStatus.UNAUTHORIZED


async def test_logout_clears_the_session_cookie_and_returns_the_end_session_endpoint(
    bff_client: AsyncClient, as_principal: Callable[[Principal], None]
) -> None:
    as_principal(Principal(email="dev@example.com", provider="session", is_admin=False))

    response = await bff_client.post("/auth/logout")

    assert response.status_code == HTTPStatus.OK
    assert response.json()["end_session_endpoint"] == "https://idp.invalid/oauth2/logout"
    set_cookie = _set_cookie_for(response, "session")
    assert "max-age=0" in set_cookie or "expires=thu, 01 jan 1970" in set_cookie
    assert "httponly" in set_cookie
    assert "secure" in set_cookie
    assert "samesite=lax" in set_cookie
