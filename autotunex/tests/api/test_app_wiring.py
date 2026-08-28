"""The app stores its authenticator on ``app.state`` rather than rebuilding it per request."""

from __future__ import annotations

import logging
from typing import cast

import httpx
import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from autotunex.core.auth.disabled import DisabledAuthenticator
from autotunex.core.auth.oidc import OidcBearerVerifier
from autotunex.main import create_app
from autotunex.services.reconcile.loop import ReconcileLoop
from tests.conftest import make_settings


def _has_cors_middleware(app: FastAPI) -> bool:
    """Whether ``CORSMiddleware`` is in ``app.user_middleware``.

    ``Middleware.cls`` is typed as Starlette's ``_MiddlewareFactory[P]`` — a
    callable protocol, not a ``type[...]`` — so mypy strict's
    ``comparison-overlap`` check rejects a bare ``middleware.cls is
    CORSMiddleware`` as comparing two types it cannot statically prove overlap.
    ``cast(object, ...)`` states the runtime fact plainly: at runtime ``cls`` is
    the exact class object handed to ``add_middleware``, and comparing that to
    ``CORSMiddleware`` by identity is exactly what "is this middleware attached"
    means.
    """
    return any(cast(object, middleware.cls) is CORSMiddleware for middleware in app.user_middleware)


def test_create_app_attaches_an_authenticator_to_app_state() -> None:
    app = create_app(make_settings())

    assert isinstance(app.state.authenticator, DisabledAuthenticator)


def test_create_app_attaches_an_id_token_verifier_slot_to_app_state() -> None:
    app = create_app(make_settings())

    assert app.state.id_token_verifier is None  # "session" is not enabled by default


def test_create_app_builds_a_real_id_token_verifier_when_session_is_enabled() -> None:
    """The companion the ``None`` check above cannot be.

    A stub that hardcoded ``app.state.id_token_verifier = None`` in ``create_app``
    would still satisfy the test above; this settings shape (mirroring
    ``tests/core/auth/test_registry.py``'s ``_SESSION_SETTINGS_KWARGS``) actually
    enables ``"session"``, so ``build_id_token_verifier`` takes its other branch and
    this asserts on what it returns rather than what it defaults to.
    """
    app = create_app(
        make_settings(
            auth_providers=["session"],
            oidc_issuer="https://idp.invalid/oauth2",
            oidc_jwks_uri="https://idp.invalid/jwks",
            oidc_audience="my-client-id",
            oidc_client_id="my-client-id",
            oidc_client_secret="shh",
            oidc_authorization_endpoint="https://idp.invalid/authorize",
            oidc_token_endpoint="https://idp.invalid/token",
            public_base_url="https://autotunex.invalid",
            session_secret="test-app-wiring-session-secret-at-least-32-chars",
        )
    )

    assert isinstance(app.state.id_token_verifier, OidcBearerVerifier)


def test_create_app_warns_when_auth_is_disabled_in_production(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A security-relevant default gets a loud signal, not a silent no-op.

    ``make_settings`` has no ``environment``/``allow_insecure_no_auth`` parameters
    — Task 1 added those to ``Settings`` itself, not to this shared test factory —
    so the prod+disabled shape is built by copying the disabled-provider defaults
    rather than adding parameters the factory doesn't have. ``model_copy`` skips
    ``_validate_auth``, which is fine here: that validator (and the opt-in flag it
    checks) is already covered directly in ``tests/core/test_config.py``.

    ``create_app`` also calls ``configure_logging``, which runs
    ``logging.basicConfig(force=True)`` — that tears down every handler already
    on the root logger, including the one pytest's own ``caplog`` fixture
    installs before this test body runs, so the warning below would otherwise
    reach stderr but never ``caplog.records``. Stubbing it out here is what
    makes the warning observable; it is not otherwise relevant to what this
    test asserts.
    """
    settings = make_settings(auth_providers=["disabled"]).model_copy(
        update={"environment": "prod", "allow_insecure_no_auth": True}
    )
    monkeypatch.setattr("autotunex.main.configure_logging", lambda level: None)

    with caplog.at_level(logging.WARNING):
        create_app(settings)

    assert any("authentication is disabled" in r.message.lower() for r in caplog.records)


def test_create_app_attaches_cors_middleware_when_an_allowlist_is_configured() -> None:
    """A security-relevant default, not a wiring detail — this is what the guard protects.

    ``create_app`` attaches ``CORSMiddleware(..., allow_credentials=True)`` (see
    ``main.py``). Credentialed CORS with no explicit origin allowlist would let a
    browser send the session cookie cross-origin to any site that asks, so the
    ``if settings.cors_allow_origins:`` guard exists specifically to keep that
    middleware off unless an allowlist was actually configured. ``app.user_middleware``
    is Starlette's own supported introspection list, not a private attribute.
    """
    app = create_app(make_settings(cors_allow_origins=["https://ui.invalid"]))

    assert _has_cors_middleware(app)


async def test_cors_preflight_allows_the_content_encoding_request_header() -> None:
    """A client-gzipped dataset upload sends a header CORS does not safelist.

    The UX compresses a compressible dataset in the browser and sends
    ``Content-Encoding: gzip``. That header is not on the CORS safelist, so a
    cross-origin deployment (``cors_allow_origins`` plus
    ``session_cookie_same_site=none``) only ever sends the upload if the
    preflight allows it — otherwise the browser drops the request itself and the
    UI can report nothing more useful than "network error". Asserting on the
    preflight rather than on ``allow_headers`` keeps this about the behaviour a
    browser actually observes.
    """
    app = create_app(make_settings(cors_allow_origins=["https://ui.invalid"]))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.options(
            "/api/v1/datasets/some-id/upload",
            headers={
                "Origin": "https://ui.invalid",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type,content-encoding",
            },
        )

    assert response.status_code == 200
    assert "content-encoding" in response.headers["access-control-allow-headers"].lower()


def test_create_app_does_not_attach_cors_middleware_with_the_default_empty_allowlist() -> None:
    app = create_app(make_settings())

    assert not _has_cors_middleware(app)


async def test_root_redirects_to_the_autotune() -> None:
    """The bare service root sends a browser to the interactive API docs.

    ``httpx.AsyncClient`` does not follow redirects by default, so this observes
    the redirect itself rather than the docs page it points at.
    """
    app = create_app(make_settings())

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/")

    assert response.status_code == 307
    assert response.headers["location"] == "/autotune"


async def test_the_app_opens_and_closes_an_http_client_across_its_lifespan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drives ``lifespan`` directly via Starlette's own router.

    ``httpx.ASGITransport`` does not run lifespan on its own, and driving it this
    way needs no extra dependency.

    Passing ``make_settings()`` to ``create_app`` does *not*, by itself, keep this
    test off the developer's real database. ``lifespan`` calls the module-level
    ``get_settings`` that ``main.py`` imports with ``from autotunex.core.config
    import ... get_settings`` — so the name it looks up at call time is
    ``autotunex.main.get_settings``, the same ``lru_cache``d process singleton
    every other module shares, not the ``settings`` argument handed to
    ``create_app``. ``get_engine()`` is a second singleton derived from that same
    call. Left unpatched, with the default ``auto_create_schema=True``, this test
    would run ``create_schema(get_engine())`` against whatever database the
    developer's own ``.env`` points at — a real file, or a real MySQL server.
    Monkeypatching ``autotunex.main.get_settings`` to return ``make_settings()``
    (which sets ``auto_create_schema=False``) is what actually neutralises that;
    the ``settings`` passed to ``create_app`` below only configures the app being
    built, not what ``lifespan`` reads.

    **Both** singletons have to be patched, and patching only the first is a trap
    this test fell into. ``auto_create_schema=False`` skips the ``create_schema``
    call on the way *in*, but ``lifespan`` ends with an unconditional
    ``await get_engine().dispose()`` on the way *out* — and ``get_engine`` is its
    own ``lru_cache``d singleton reading the *real* ``get_settings()``, not the
    patched name. So with a developer ``.env`` naming a driver that is not
    installed in this interpreter (``mysql+asyncmy`` without the ``mysql`` extra
    is the common case), merely constructing that engine raises
    ``ModuleNotFoundError`` and the test fails for a reason that has nothing to do
    with the HTTP client it is asserting on.
    """
    monkeypatch.setattr("autotunex.main.get_settings", lambda: make_settings())
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    monkeypatch.setattr("autotunex.main.get_engine", lambda: engine)
    app = create_app(make_settings())

    async with app.router.lifespan_context(app):
        assert isinstance(app.state.http_client, httpx.AsyncClient)
        assert not app.state.http_client.is_closed

    assert app.state.http_client.is_closed


async def test_lifespan_starts_and_stops_the_reconcile_loop_for_llmb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llmb = make_settings().model_copy(
        update={"job_backend": "llmb", "gb_server_url": "https://gbserver.example"}
    )
    monkeypatch.setattr("autotunex.main.get_settings", lambda: llmb)
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    monkeypatch.setattr("autotunex.main.get_engine", lambda: engine)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr("autotunex.main.get_session_factory", lambda: factory)
    app = create_app(llmb)

    async with app.router.lifespan_context(app):
        assert isinstance(app.state.reconcile_loop, ReconcileLoop)
        assert not app.state.reconcile_http_client.is_closed

    assert app.state.reconcile_http_client.is_closed


async def test_lifespan_does_not_start_the_loop_for_the_none_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("autotunex.main.get_settings", lambda: make_settings())
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    monkeypatch.setattr("autotunex.main.get_engine", lambda: engine)
    app = create_app(make_settings())

    async with app.router.lifespan_context(app):
        assert app.state.reconcile_loop is None
