# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Serve the built SvelteKit SPA from this service.

The UI is built with ``@sveltejs/adapter-static`` in fallback mode
(``fallback: 'index.html'``), which produces a single-page app: real files under
``_app/`` plus an ``index.html`` that owns all client-side routing. Plain
``StaticFiles`` 404s on a client route like ``/autotune/jobs/42`` because no such
file exists on disk; :class:`SpaStaticFiles` serves ``index.html`` for those
instead so a deep link or a browser refresh loads the app rather than a 404.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from starlette.exceptions import HTTPException
from starlette.responses import Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

from autotunex.core.config import Settings
from autotunex.core.logging import get_logger

logger = get_logger(__name__)


class SpaStaticFiles(StaticFiles):
    """``StaticFiles`` that falls back to ``index.html`` on a missing path.

    Real assets (``_app/...``, images, ``favicon.png``) resolve normally; any
    other path — a client-side route — returns the SPA shell so the front-end
    router can take over. A genuinely missing ``index.html`` still 404s.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:
        """Serve *path*, or the SPA shell when *path* is a missing client route."""
        try:
            return await super().get_response(path, scope)
        except HTTPException as exc:
            if exc.status_code == 404:
                return await super().get_response("index.html", scope)
            raise


def mount_frontend(app: FastAPI, settings: Settings) -> None:
    """Mount the built SPA at ``settings.frontend_base_path`` if configured.

    A no-op when ``frontend_dir`` is unset or does not point at an existing
    directory, so the service still runs API-only in that case.
    """
    frontend_dir = settings.frontend_dir
    if frontend_dir is None:
        return
    if not Path(frontend_dir).is_dir():
        logger.warning("frontend_dir %s does not exist; serving API only", frontend_dir)
        return

    app.mount(
        settings.frontend_base_path,
        SpaStaticFiles(directory=frontend_dir, html=True),
        name="frontend",
    )
    logger.info("Serving frontend from %s at %s", frontend_dir, settings.frontend_base_path)
