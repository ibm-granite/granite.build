# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Application factory and ASGI entrypoint.

Run locally with::

    uvicorn autotunex.main:app --reload
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from autotunex import __version__
from autotunex.api.deps import get_principal
from autotunex.api.errors import register_exception_handlers
from autotunex.api.frontend import mount_frontend
from autotunex.api.mcp import mount_mcp
from autotunex.api.routers import (
    app_config,
    auth,
    chat,
    configurations,
    dataset_intelligence,
    datasets,
    health,
    jobs,
    reward_functions,
    users,
)
from autotunex.core.auth.registry import build_authenticator, build_id_token_verifier
from autotunex.core.config import Settings, get_settings
from autotunex.core.logging import configure_logging, get_logger
from autotunex.db.session import create_schema, get_engine, get_session_factory
from autotunex.services.reconcile.loop import ReconcileLoop
from autotunex.services.reconcile.registry import get_build_status_reader

logger = get_logger(__name__)

DESCRIPTION = """
Automated fine-tuning and hyperparameter optimization for large language models.

A **job** is one optimization run. It searches a hyperparameter space by running
**trials**, each testing one concrete parameter assignment, and reports the
metrics its objective is scored on.

Jobs are created either by submitting them through this API with `POST /jobs`
or by the tuning pipeline writing to the database directly; `GET /jobs` and
`GET /jobs/{id}` report what exists, and `DELETE /jobs/{id}` removes a job.
"""


def _load_local_env() -> None:
    """Populate ``os.environ`` from ``.env`` for non-Settings variables.

    pydantic-settings reads ``.env`` only into ``AUTOTUNEX_``-prefixed Settings
    fields, so the raw credential vars the ``llmb`` CLI needs — ``GB_TOKEN`` (which
    authenticates the CLI) and ``HF_TOKEN`` (the HuggingFace push destination) —
    never reach ``os.environ`` from ``.env`` alone, yet the storage backends read
    them from ``os.environ``. Load them here with ``setdefault`` so a real exported
    variable always wins and the token *values* still never enter ``Settings``.

    This runs from the lifespan hook (uvicorn startup). It is a hard no-op under
    pytest: a developer's real ``.env`` must never leak ``AUTOTUNEX_*`` or
    credential vars into ``os.environ`` for the test session — and some tests drive
    ``lifespan`` directly (see ``tests/api/test_app_wiring.py``), so the usual
    "ASGITransport skips lifespan" guarantee is not enough on its own.
    """
    if "PYTEST_CURRENT_TEST" in os.environ:
        return
    try:
        from dotenv import dotenv_values
    except ImportError:  # python-dotenv absent → nothing to load.
        return
    for key, value in dotenv_values(".env").items():
        if value is not None:
            os.environ.setdefault(key, value)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Prepare and release process-wide resources.

    Also owns the reconcile loop's lifecycle: for the ``llmb`` job backend, it
    starts the loop as a background task on startup and, on shutdown, cancels
    that task and closes its dedicated HTTP client. The ``none`` backend has no
    launcher and nothing to reconcile, so no loop is started.
    """
    _load_local_env()
    settings = get_settings()
    if settings.auto_create_schema:
        await create_schema(get_engine())
        logger.info("Database schema ensured at %s", settings.database_url)
    # Explicit rather than relying on httpx's default (5s): this client makes the
    # OIDC token-endpoint call from /auth/callback, where a hang holds an entire
    # login flow open rather than failing fast, so the number is stated here
    # instead of inherited invisibly from the library default.
    app.state.http_client = httpx.AsyncClient(timeout=httpx.Timeout(10.0))
    # The reconcile loop advances jobs.status from the cluster (off the read
    # path). It exists only for the llmb backend; the none backend has no
    # launcher and nothing to reconcile.
    app.state.reconcile_loop = None
    app.state.reconcile_http_client = None
    reconcile_task: asyncio.Task[None] | None = None
    if settings.job_backend == "llmb":
        # A dedicated client with its own explicit timeout, following the
        # http_client convention above; the sweep tolerates a slow gbserver by
        # leaving jobs untouched and retrying next interval.
        app.state.reconcile_http_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        reader = get_build_status_reader(settings, app.state.reconcile_http_client)
        loop = ReconcileLoop(
            session_factory=get_session_factory(),
            reader=reader,
            interval_seconds=settings.job_reconcile_interval_seconds,
            concurrency=settings.job_reconcile_concurrency,
        )
        app.state.reconcile_loop = loop
        reconcile_task = asyncio.create_task(loop.run())
    yield
    if reconcile_task is not None:
        reconcile_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reconcile_task
    if app.state.reconcile_http_client is not None:
        await app.state.reconcile_http_client.aclose()
    await app.state.http_client.aclose()
    await get_engine().dispose()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a configured application instance.

    Args:
        settings: Override configuration. Defaults to the environment-derived
            singleton.
    """
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=DESCRIPTION,
        debug=settings.debug,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    app.state.authenticator = build_authenticator(settings)
    logger.info("Auth providers active: %s", settings.auth_providers)
    if (
        settings.environment == "prod"
        and settings.auth_providers == ["disabled"]
        and settings.allow_insecure_no_auth
    ):
        logger.warning(
            "Authentication is DISABLED in production (allow_insecure_no_auth). "
            "Every caller acts as the standalone system owner and, with the default "
            "standalone_role=admin, can read all jobs, configurations, and datasets "
            "in this database — including any owned by other users."
        )
    app.state.id_token_verifier = build_id_token_verifier(settings)
    # `/auth/login`'s provider gate reads settings via `get_settings()`
    # (through `SettingsDep`); `/auth/callback`'s reads `app.state.id_token_verifier`,
    # built from the `settings` argument this function received. In production
    # both are the same object — `get_settings()` is the module-level singleton
    # and nothing here overrides it — so the two routes agree. They could only
    # diverge in a test that builds the app with an explicit `settings` argument
    # while a *different* `Settings` instance is cached behind `get_settings()`
    # (or vice versa), and even then both directions fail closed: `/login`
    # against a `settings.auth_providers` that omits `"session"` 401s before
    # touching `app.state` at all, and `/callback` against an
    # `id_token_verifier` built from settings that disabled `"session"` is
    # `None`, which is that route's own provider gate. Neither divergence
    # leaves a caller authenticated when it shouldn't be.
    if settings.cors_allow_origins:
        # `settings.cors_allow_origins` is validated at construction time to never
        # contain "*" (see core/config.py) — that guarantee is what makes
        # `allow_credentials=True` safe here. It is mandatory, not incidental: the
        # browser will not send the session cookie cross-origin without it, which
        # is the entire point of a separately-hosted UI talking to this BFF.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allow_origins,
            allow_credentials=True,
            # Mirror the mutating verbs the API actually exposes, so a
            # separately-hosted browser UI survives the CORS preflight for every
            # endpoint — not just GET/POST. PUT replaces a configuration, DELETE
            # removes jobs/configurations/datasets, and PATCH changes a user's role.
            allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE"],
            # `Content-Encoding` is NOT a CORS-safelisted request header, and the
            # dataset upload path sets it (`Content-Encoding: gzip`) whenever the
            # browser compresses the file client-side. Omitting it here fails the
            # preflight for exactly the large-upload case that compression exists
            # to help, and the browser never sends the request at all — which the
            # UI can only report as a bare network error.
            allow_headers=["Authorization", "X-API-Key", "Content-Type", "Content-Encoding"],
        )

    register_exception_handlers(app)

    @app.get("/", include_in_schema=False)
    async def _root_redirect() -> RedirectResponse:
        """Send the bare service root to the SPA at ``/autotune``.

        The SPA, when configured, mounts at ``frontend_base_path`` (``/autotune``
        by default), so ``/`` is otherwise unhandled and 404s. A temporary
        redirect (not permanent) keeps ``/`` free to become a real landing page
        later without a stale 308 cached in browsers.
        """
        return RedirectResponse(url="/autotune")

    app.include_router(health.router)
    app.include_router(app_config.router, prefix=settings.api_prefix)
    # No `dependencies=[Depends(get_principal)]` here, deliberately: you cannot
    # be logged in while logging in. `/login` and `/callback` are mounted
    # unconditionally (see `auth.py`'s module docstring for why 404-vs-401 must
    # never distinguish "not configured" from "rejected"); `/me` and `/logout`
    # protect themselves through their own `PrincipalDep` parameters instead of
    # a router-level attachment.
    app.include_router(auth.router)
    app.include_router(
        jobs.router, prefix=settings.api_prefix, dependencies=[Depends(get_principal)]
    )
    app.include_router(
        configurations.router, prefix=settings.api_prefix, dependencies=[Depends(get_principal)]
    )
    app.include_router(
        datasets.router, prefix=settings.api_prefix, dependencies=[Depends(get_principal)]
    )
    app.include_router(
        dataset_intelligence.router,
        prefix=settings.api_prefix,
        dependencies=[Depends(get_principal)],
    )
    app.include_router(
        users.router, prefix=settings.api_prefix, dependencies=[Depends(get_principal)]
    )
    app.include_router(
        chat.router, prefix=settings.api_prefix, dependencies=[Depends(get_principal)]
    )
    app.include_router(
        reward_functions.router,
        prefix=settings.api_prefix,
        dependencies=[Depends(get_principal)],
    )

    # Mounted before the SPA below so a broad frontend mount can never shadow
    # /mcp; mount_mcp itself is a no-op unless settings.enable_mcp is set, and
    # lazily imports fastmcp only then (see api/mcp.py's module docstring).
    mount_mcp(app, settings)
    # Optionally serve the built SPA (src/ux/build) from this service. Mounted
    # last so it never shadows /api, /auth, /health, or the OpenAPI routes.
    mount_frontend(app, settings)
    return app


app = create_app()
