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


import asyncio
import os

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from gbserver.api import (  # noqa: F401  registers routes on builds_api
    build_files as _build_files,
)
from gbserver.api import event_subscribe as _event_subscribe  # noqa: F401
from gbserver.api.artifacts import artifacts_api
from gbserver.api.auth import AuthMiddleware
from gbserver.api.auth_routes import auth_api
from gbserver.api.autotunex_proxy import aclose_client as _autotunex_aclose_client
from gbserver.api.autotunex_proxy import router as autotunex_router
from gbserver.api.builds import builds_api
from gbserver.api.environment_files import files_api
from gbserver.api.lineage import lineage_api
from gbserver.api.logs import logs_api
from gbserver.api.node_health import node_health_api
from gbserver.api.openapi_security import enable_api
from gbserver.api.secrets import secrets_api
from gbserver.api.spaces import spaces_api
from gbserver.types.constants import (
    API_BASE_PATH,
    GBSERVER_EVENT_PUBLISHING_ENABLED,
    GBSERVER_GIT_COMMIT,
    GBSERVER_REST_SERVER_WORKERS,
    analytics_backend_enabled,
    gbserver_ui_dir,
)
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


def get_app() -> FastAPI:
    app = FastAPI()
    if True:
        # perform an empty CLI call here just to let the work process go through the same init code in cli.py
        from gbserver.cli import gbserver

        gbserver(["rest-server-worker"], standalone_mode=False)
    return app


root_api = get_app()

root_api.add_middleware(AuthMiddleware)  # type: ignore[arg-type]


@root_api.get(API_BASE_PATH)
def read_root():
    return {
        "message": "Welcome to the REST API!",
        "git_commit": GBSERVER_GIT_COMMIT,
    }


# Mount each sub-app and advertise the Bearer-token scheme on it, so its /docs
# page shows an Authorize button that sends the header on "Try it out". This is
# purely a docs/UI convenience — enforcement stays in AuthMiddleware.
#
# auth_api is mounted with advertise_auth=False: AuthMiddleware exempts the
# entire /api/v1/auth/* login flow from authentication, so a lock icon there
# would wrongly claim a token is required before the user has one.
enable_api(root_api, f"{API_BASE_PATH}/auth", auth_api, advertise_auth=False)
enable_api(root_api, f"{API_BASE_PATH}/artifacts", artifacts_api)
enable_api(root_api, f"{API_BASE_PATH}/builds", builds_api)
enable_api(root_api, f"{API_BASE_PATH}/files", files_api)
enable_api(root_api, f"{API_BASE_PATH}/lineage", lineage_api)
enable_api(root_api, f"{API_BASE_PATH}/logs", logs_api)
enable_api(root_api, f"{API_BASE_PATH}/node-health", node_health_api)
enable_api(root_api, f"{API_BASE_PATH}/secrets", secrets_api)
enable_api(root_api, f"{API_BASE_PATH}/spaces", spaces_api)

# root_api is the top-level app (never mounted), so advertise the scheme on it
# directly — no mount, just the Swagger auth wiring.
enable_api(root_api)

# ── Frontend static file serving ──────────────────────────────────────────────

_UI_DIR = gbserver_ui_dir()

# ── Analytics (optional gb_ui_backend extra) ───────────────────────────────────
# Routers are include_router()'d (not mounted as a sub-app) so they run in-process
# behind AuthMiddleware and share root_api's startup hook (init_analytics, below).
# See analytics_backend_enabled() for the install + enable decision.
_ANALYTICS_ENABLED = analytics_backend_enabled()
if _ANALYTICS_ENABLED:
    from gb_ui_backend.api import ai as _gb_ai
    from gb_ui_backend.api import analytics as _gb_analytics
    from gb_ui_backend.api import builds as _gb_builds
    from gb_ui_backend.api import chat as _gb_chat
    from gb_ui_backend.api import data_processing as _gb_data_processing
    from gb_ui_backend.api import plans as _gb_plans

    for _router in (
        _gb_analytics.router,
        _gb_ai.router,
        _gb_builds.router,
        _gb_data_processing.router,
        _gb_plans.router,
        _gb_chat.router,
    ):
        root_api.include_router(_router, prefix="/api/analytics")

    logger.info("Analytics enabled — mounted /api/analytics routers")

    if GBSERVER_REST_SERVER_WORKERS > 1:
        # Each worker gets its own analytics DB engine / AI-daemon state.
        logger.warning(
            "GBSERVER_REST_SERVER_WORKERS=%d with analytics enabled — "
            "analytics assumes a single worker.",
            GBSERVER_REST_SERVER_WORKERS,
        )
else:
    logger.info("Analytics not enabled — /api/analytics not mounted")


# AutoTuneX reverse proxy — must be registered before the "/" static mount so it
# wins over the SPA catch-all. See api/autotunex_proxy.py.
root_api.include_router(autotunex_router)


def _is_rsc_request(request: Request) -> bool:
    return request.headers.get("rsc") == "1" or "_rsc" in request.query_params


@root_api.middleware("http")
async def _serve_rsc_payload(request: Request, call_next):
    """Serve the pre-rendered RSC flight payload for Next.js App Router
    client-side navigations, instead of letting StaticFiles return the page's
    HTML document for every request regardless of headers.

    `output: 'export'` ships a `<route>/index.txt` flight-stream payload
    alongside every route's `index.html`. The client router fetches it at
    `<route>.txt` (no trailing slash before `.txt`, e.g. `/dashboard/builds.txt`
    for the route `/dashboard/builds/`) — a *different* path than the payload's
    on-disk location — identified by the `rsc: 1` header or `_rsc` query param.
    Without this, every single navigation — `<Link>`, `router.push`, all of it
    — 404s fetching that payload, fails to parse a response as flight data,
    and silently falls back to a full hard reload, which is why the
    client-only theme attribute on <html> keeps resetting between page
    transitions.
    """
    path = request.url.path
    if (
        _is_rsc_request(request)
        and not path.startswith("/api/")
        and path.endswith(".txt")
    ):
        rel = path[: -len(".txt")].strip("/")
        txt_path = os.path.realpath(
            os.path.join(_UI_DIR, rel, "index.txt")
            if rel
            else os.path.join(_UI_DIR, "index.txt")
        )
        ui_dir_real = os.path.realpath(_UI_DIR)
        if (
            txt_path == ui_dir_real or txt_path.startswith(ui_dir_real + os.sep)
        ) and os.path.isfile(txt_path):
            return FileResponse(txt_path, media_type="text/x-component")
    return await call_next(request)


if os.path.isdir(_UI_DIR):
    # html=True makes StaticFiles serve index.html for directory paths
    # (e.g. /builds/ → builds/index.html). Unknown paths still return 404,
    # which the exception handler below catches for SPA client-side routes.
    root_api.mount("/", StaticFiles(directory=_UI_DIR, html=True), name="frontend")
    logger.info("Serving frontend from %s", _UI_DIR)
else:
    logger.info("No frontend UI directory found at %s — running API-only", _UI_DIR)


@root_api.exception_handler(404)
async def _spa_fallback(
    request: Request, exc: Exception
) -> FileResponse | JSONResponse:
    """Serve a clean SPA shell for unknown non-API paths so the client-side router can handle them.

    Uses dashboard/index.html rather than the root index.html because the root
    page has a server-side redirect baked into its RSC payload (NEXT_REDIRECT →
    /dashboard) that fires before the client router can navigate to the real path.

    RSC data requests (Next.js App Router prefetches/navigations identified by the
    `rsc: 1` header or `_rsc` query parameter) are intentionally NOT intercepted.
    Returning HTML for an RSC request causes the router to cache the wrong page
    data, resulting in the wrong page being rendered on navigation. A 404 here
    tells the router to render the route fresh from its client-side bundle.
    """
    if not request.url.path.startswith("/api/"):
        if _is_rsc_request(request):
            # RSC data requests must not be intercepted — returning HTML instead of
            # RSC data confuses the App Router and causes wrong-page renders.
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        shell = os.path.join(_UI_DIR, "dashboard", "index.html")
        if os.path.isfile(shell):
            return FileResponse(shell)
        index = os.path.join(_UI_DIR, "index.html")
        if os.path.isfile(index):
            return FileResponse(index)
    return JSONResponse({"detail": "Not Found"}, status_code=404)


@root_api.on_event("startup")
async def _start_background_tasks():
    """Launch background tasks that run for the lifetime of the server."""
    if _ANALYTICS_ENABLED:
        from gb_ui_backend.main import init_analytics

        # Analytics is optional and must never take down the core REST API — a
        # misconfigured or unwritable analytics DB should degrade gracefully
        # (mirrors the try/except around GbserverSource in gb_ui_backend.main).
        try:
            await init_analytics()
        except Exception:
            logger.exception(
                "Analytics init failed — continuing with analytics degraded"
            )

    if GBSERVER_EVENT_PUBLISHING_ENABLED:
        if os.getenv("RABBITMQ_HOST"):
            from gbserver.messaging.credential_cleanup import start_cleanup_loop

            logger.info("Event publishing enabled — starting credential cleanup task")
            asyncio.create_task(start_cleanup_loop())
        else:
            logger.info(
                "Event publishing enabled (NATS mode) — no credential cleanup needed"
            )


@root_api.on_event("shutdown")
async def _close_autotunex_client():
    """Close the AutoTuneX proxy's shared httpx client."""
    await _autotunex_aclose_client()
