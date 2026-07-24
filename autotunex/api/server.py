# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
from contextlib import asynccontextmanager

import dependencies
import models as api
import uvicorn
from auth import auth_router
from fastapi import (
    APIRouter,
    FastAPI,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from routes import (
    chat_routes,
    config_routes,
    dataset_routes,
    dmf_routes,
    gb_routes,
    health_checker,
    jobs_routes,
    reward_routes,
    user_routes,
    utils_routes,
)
from services import gb_service
from starlette.exceptions import HTTPException as StarletteHTTPException
from utils import str_to_bool

logger = logging.getLogger("server")


# Run the check at startup
@asynccontextmanager
async def startup_event(app: FastAPI):
    async def startup():
        from services.db_service import init_pool

        await init_pool()
        database = dependencies.get_database()
        job = dependencies.get_job_service(database)
        await database.test_db_connection_and_structure()
        job.start_monitoring()
        await job.terminate_jobs(
            api.JobStatus.TERMINATED
        )  # Status correction for old job with still RUNNING status (NOT IDEAL SOLUTION)

        async def download_cleanup_loop():
            db = dependencies.get_database()
            job_svc = dependencies.get_job_service(db)
            while True:
                await asyncio.sleep(1800)  # 30 minutes
                try:
                    await job_svc.cleanup_expired_downloads(max_age_minutes=60)
                except Exception as e:
                    logger.warning(f"Download cleanup failed: {e}")

        app.state.download_cleanup_task = asyncio.create_task(download_cleanup_loop())

        async def chunk_staging_cleanup_loop():
            # Remove staging dirs from chunked uploads the client never finished,
            # so abandoned partial uploads don't accumulate on disk.
            db = dependencies.get_database()
            dataset_svc = dependencies.get_dataset_service(db)
            while True:
                try:
                    removed = await dataset_svc.cleanup_stale_chunk_uploads(
                        max_age_minutes=360
                    )
                    if removed:
                        logger.info(
                            f"Removed {removed} stale chunk-upload staging dir(s)"
                        )
                except Exception as e:
                    logger.warning(f"Chunk staging cleanup failed: {e}")
                await asyncio.sleep(3600)  # hourly

        app.state.chunk_staging_cleanup_task = asyncio.create_task(
            chunk_staging_cleanup_loop()
        )

    async def shutdown():
        from services.db_service import shutdown_pool

        database = dependencies.get_database()
        job = dependencies.get_job_service(database)
        # await job.terminate_jobs(api.JobStatus.TERMINATED)
        job.stop_monitoring()
        if hasattr(app.state, "download_cleanup_task"):
            app.state.download_cleanup_task.cancel()
        if hasattr(app.state, "chunk_staging_cleanup_task"):
            app.state.chunk_staging_cleanup_task.cancel()
        logger.info("All jobs are terminated")
        await shutdown_pool()

    await startup()
    yield
    await shutdown()


app = FastAPI(
    title="AutoTune Server",
    description="""
# AutoTuneX API

**AutoTuneX** is an automated fine-tuning platform for foundation models with integrated hyperparameter optimization.

## Features

- **Automated Fine-Tuning**: Fine-tune foundation models with automated HPO
- **Multiple Tuning Methods**: LORA, PREFIX_TUNING, P_TUNING, and more
- **Experiment Tracking**: Monitor trials, metrics, and model performance
- **Dataset Management**: Upload and manage training datasets
- **Configuration Management**: Save and reuse hyperparameter configurations
- **Model Registry**: Publish models to Data Model Factory (DMF)
- **Enterprise Auth**: OAuth 2.0 authentication with IBM w3id

## Quick Start

1. **Authenticate**: Use `/api/login` to start OAuth flow
2. **Upload Dataset**: Create dataset with `/api/dataset` and upload files
3. **Create Configuration**: Define HPO search space with `/api/config`
4. **Start Tuning**: Launch job with `/api/job`
5. **Monitor Progress**: Check status and trials
6. **Download Results**: Get trained models and metrics

## API Organization

- **Tunings**: Core fine-tuning job management
- **Configurations**: Hyperparameter search space definitions
- **Data sets**: Training/validation data management
- **DMF**: Model registry integration
- **Auth**: Authentication and authorization
- **User**: User management and metadata
    """,
    version="0.1.0",
    terms_of_service="IBM Research",
    contact={
        "name": "IBM Research",
        "url": "https://research.ibm.com/",
        "email": "daniel.karl@ibm.com",
    },
    license_info={"name": "IBM Research", "url": "https://research.ibm.com/"},
    servers=[
        {
            "url": os.getenv("AUTOTUNE_SERVER_URL", "http://localhost:8000"),
            "description": "AutoTune Server URL",
        }
    ],
    openapi_url="/fmtune/openapi.json",
    redoc_url="/fmtune/docs",
    docs_url="/fmtune/try",
    lifespan=startup_event,
    openapi_tags=[
        {
            "name": "Tunings",
            "description": "Fine-tuning job management - create, monitor, and manage tuning experiments",
        },
        {
            "name": "Configurations",
            "description": "Hyperparameter configuration management - define and save HPO search spaces",
        },
        {
            "name": "Data sets",
            "description": "Dataset management - upload and manage training/validation data",
        },
        {
            "name": "DMF",
            "description": "Data Model Factory integration - publish and manage models in IBM's model registry",
        },
        {
            "name": "Auth",
            "description": "Authentication and authorization - OAuth 2.0 login and token management",
        },
        {
            "name": "User",
            "description": "User management - user profiles, metadata, and administration",
        },
        {
            "name": "Tasks",
            "description": "Task management - individual task tracking within jobs",
        },
        {"name": "health", "description": "Health check endpoints"},
        {"name": "gb", "description": "Granite Build integration"},
        {
            "name": "Chat",
            "description": "AI assistant for natural language interaction with AutoTuneX",
        },
        {"name": "Utils", "description": "Utility endpoints for internal use"},
    ],
)

prefix_router = APIRouter(prefix="/fmtune")
prefix_router.include_router(gb_routes.router, tags=["gb"])
prefix_router.include_router(health_checker.router, tags=["health"])
prefix_router.include_router(utils_routes.router, tags=["Utils"])
prefix_router.include_router(auth_router, tags=["Auth"])
prefix_router.include_router(chat_routes.router)
prefix_router.include_router(user_routes.router)
prefix_router.include_router(config_routes.router)
prefix_router.include_router(dmf_routes.router)
prefix_router.include_router(reward_routes.router)
prefix_router.include_router(dataset_routes.router)
prefix_router.include_router(jobs_routes.router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "Location",
        "Upload-Offset",
        "Upload-Length",
        "Tus-Resumable",
        "Tus-Version",
        "Tus-Extension",
        "Tus-Max-Size",
        "Upload-Expires",
        "Upload-Metadata",
    ],
)


# Resumable dataset uploads (tus protocol). The router carries its own
# 'datasets/tus' prefix, so mounting under '/api' yields /fmtune/api/datasets/tus.
# Gated on tuspyserver being importable so a minimal environment without it still
# boots (mirrors the ENABLE_MCP gating below).
try:
    from services.datasets.tus_app import create_dataset_tus_router

    prefix_router.include_router(create_dataset_tus_router(), prefix="/api")
except Exception as _tus_exc:  # pragma: no cover - import/env guard
    logger.warning("tus upload router not mounted: %s", _tus_exc)

app.include_router(prefix_router)

# Mount MCP server at /mcp
enable_mcp = str_to_bool(os.getenv("ENABLE_MCP", "false"))
if enable_mcp:
    from mcp_server import mcp as mcp_instance

    mcp_app = mcp_instance.http_app(path="/", transport="sse")
    app.mount("/mcp", mcp_app)

# ── Serve the SvelteKit SPA (single-container mode) ───────────────
# Built with base path /autotune (see ux/svelte.config.js). Gated on the
# build dir existing so local `python server.py` (no build) is a no-op and
# the Vite dev server + proxy keep working unchanged.
UX_BUILD_DIR = os.getenv("UX_BUILD_DIR", "/app/api/ux_build")
if os.path.isdir(UX_BUILD_DIR):

    class SPAStaticFiles(StaticFiles):
        """StaticFiles that falls back to index.html on 404 so SvelteKit
        client-side routes (e.g. /autotune/tunings) survive deep links and
        hard refreshes instead of returning 404."""

        async def get_response(self, path, scope):
            # Starlette signals a missing file either by returning a 404
            # response or by raising HTTPException(404) (version-dependent);
            # handle both so SPA deep links fall back to index.html.
            try:
                response = await super().get_response(path, scope)
            except StarletteHTTPException as exc:
                if exc.status_code == 404:
                    return await super().get_response("index.html", scope)
                raise
            if response.status_code == 404:
                return await super().get_response("index.html", scope)
            return response

    app.mount(
        "/autotune",
        SPAStaticFiles(directory=UX_BUILD_DIR, html=True),
        name="ux",
    )

    @app.get("/")
    async def root_redirect():
        return RedirectResponse(url="/autotune")


if __name__ == "__main__":
    gb_service.GBService().login_gb()
    dev_mode = str_to_bool(os.getenv("DEV_MODE", "false"))
    if dev_mode:
        print("Running in development mode")
    uvicorn.run(
        "server:app", host="0.0.0.0", port=8000, reload=os.getenv("DEV_MODE", False)
    )
