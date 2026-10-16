"""gb-ui analytics service — FastAPI application.

Normally mounted directly into gbserver's own app (see
gbserver/api/root_api.py, which calls init_analytics() from its own startup
hook and include_router()s these routers under /api/analytics). This module
also remains independently runnable (`python -m gb_ui_backend` / the
`gb-ui-backend` console script) for local development and testing outside
gbserver.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load .env into os.environ so unprefixed vars are available to code that
# reads os.environ directly. override=False means real environment variables win.
_env_file = os.path.join(os.path.dirname(__file__), "../../../.env")
load_dotenv(_env_file, override=False)

from gb_ui_backend.api import ai, analytics, builds, chat, data_processing, plans
from gb_ui_backend.config import Config, get_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

config = get_config()


async def init_analytics(config: Config | None = None) -> None:
    """Create analytics tables and connect to gbserver's own DB if configured.

    Called both by this module's own `lifespan()` (standalone execution) and
    directly by gbserver's startup hook when these routers are mounted
    in-process (see gbserver/api/root_api.py).
    """
    if config is None:
        config = get_config()

    logger.info(
        "analytics service starting — db=%s ai=%s gbserver_db=%s",
        config.db_enabled,
        config.ai_enabled,
        bool(config.gbserver_db_url),
    )

    # Auto-create analytics tables (idempotent; required for SQLite which has no migrations)
    if config.db_enabled:
        from gb_ui_backend.services.db_schema import Base, _get_engine

        async with _get_engine().begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    # Initialize gbserver source (standalone analytics / AI data)
    if config.gbserver_db_url:
        try:
            from gb_ui_backend.services.gbserver_source import init_gbserver_source

            await init_gbserver_source(
                config.gbserver_db_url, schema=config.gbserver_db_schema
            )
            logger.info(
                "GbserverSource initialized from %s (schema=%s)",
                config.gbserver_db_url.split("///")[-1].split("@")[-1],
                config.gbserver_db_schema,
            )
        except Exception as e:
            logger.error("Failed to initialize GbserverSource: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_analytics(config)
    yield
    logger.info("analytics service stopped")


app = FastAPI(
    title="gb-ui analytics service",
    description="Optional analytics backend for gb-ui. Provides build history charts, failure trend analysis, and AI-powered build analysis.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analytics.router, prefix="/api/analytics")
app.include_router(ai.router, prefix="/api/analytics")
app.include_router(builds.router, prefix="/api/analytics")
app.include_router(data_processing.router, prefix="/api/analytics")
app.include_router(plans.router, prefix="/api/analytics")
app.include_router(chat.router, prefix="/api/analytics")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "features": {
            "database": config.db_enabled,
            "ai_analysis": config.ai_enabled,
        },
    }
