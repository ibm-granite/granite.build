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
from gbserver.api.builds import builds_api
from gbserver.api.frontend_routes import frontend_router
from gbserver.api.lineage import lineage_api
from gbserver.api.logs import logs_api
from gbserver.api.node_health import node_health_api
from gbserver.api.secrets import secrets_api
from gbserver.api.spaces import spaces_api
from gbserver.types.constants import (
    API_BASE_PATH,
    GBSERVER_EVENT_PUBLISHING_ENABLED,
    GBSERVER_GIT_COMMIT,
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


root_api.include_router(frontend_router)
root_api.mount(f"{API_BASE_PATH}/auth", auth_api)
root_api.mount(f"{API_BASE_PATH}/artifacts", artifacts_api)
root_api.mount(f"{API_BASE_PATH}/builds", builds_api)
root_api.mount(f"{API_BASE_PATH}/lineage", lineage_api)
root_api.mount(f"{API_BASE_PATH}/logs", logs_api)
root_api.mount(f"{API_BASE_PATH}/node-health", node_health_api)
root_api.mount(f"{API_BASE_PATH}/secrets", secrets_api)
root_api.mount(f"{API_BASE_PATH}/spaces", spaces_api)

# ── Frontend static file serving ──────────────────────────────────────────────

# Default: static/ui/ sibling to this package directory (populated by make build-frontend).
# Override with GBSERVER_UI_DIR for non-standard layouts.
_UI_DIR = os.environ.get(
    "GBSERVER_UI_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "ui"),
)

if os.path.isdir(_UI_DIR):
    # html=True makes StaticFiles serve index.html for directory paths
    # (e.g. /builds/ → builds/index.html). Unknown paths still return 404,
    # which the exception handler below catches for SPA client-side routes.
    root_api.mount("/", StaticFiles(directory=_UI_DIR, html=True), name="frontend")
    logger.info("Serving frontend from %s", _UI_DIR)
else:
    logger.info("No frontend UI directory found at %s — running API-only", _UI_DIR)


@root_api.exception_handler(404)
async def _spa_fallback(request: Request, exc: Exception) -> FileResponse | JSONResponse:
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
        if request.headers.get("rsc") == "1" or "_rsc" in (request.url.query or ""):
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
    if GBSERVER_EVENT_PUBLISHING_ENABLED:
        if os.getenv("RABBITMQ_HOST"):
            from gbserver.messaging.credential_cleanup import start_cleanup_loop

            logger.info("Event publishing enabled — starting credential cleanup task")
            asyncio.create_task(start_cleanup_loop())
        else:
            logger.info(
                "Event publishing enabled (NATS mode) — no credential cleanup needed"
            )
