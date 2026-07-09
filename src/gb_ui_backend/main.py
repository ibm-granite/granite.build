"""gb-ui analytics sidecar — FastAPI application."""

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

from gb_ui_backend.api import ai, analytics, builds, data_processing, plans
from gb_ui_backend.config import get_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

config = get_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "gb-ui backend starting — db=%s ai=%s gbserver_db=%s",
        config.db_enabled,
        config.ai_enabled,
        bool(config.gbserver_db_url),
    )

    # Auto-create sidecar tables (idempotent; required for SQLite which has no migrations)
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

    yield

    logger.info("gb-ui backend stopped")


app = FastAPI(
    title="gb-ui analytics sidecar",
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

app.include_router(analytics.router)
app.include_router(ai.router)
app.include_router(builds.router)
app.include_router(data_processing.router)
app.include_router(plans.router)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "features": {
            "database": config.db_enabled,
            "ai_analysis": config.ai_enabled,
        },
    }
