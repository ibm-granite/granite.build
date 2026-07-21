"""Server lifespan: build-cache + telemetry init/teardown.

gbmcp is standalone-only (it ships bundled in granite.build and runs as a local
stdio process the MCP client launches). The source contains only standalone-usable tools,
so there is no runtime tool pruning — this lifespan just initializes the build
cache and telemetry DB (both best-effort, degrading gracefully when their
Postgres backends aren't configured) and tears them down on shutdown.
"""

from contextlib import asynccontextmanager

from fastmcp.utilities.logging import get_logger

from gbmcp.services.build_cache.build_cache import close_build_cache, init_build_cache
from gbmcp.services.telemetry.telemetry_db import close_telemetry_db, init_telemetry_db

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(server):
    # Build cache backs partial build-id resolution for the build_* tools; it
    # degrades gracefully when no Postgres is configured (the standalone case).
    logger.info("Initializing build cache...")
    try:
        await init_build_cache()
        logger.info("Build cache ready.")
    except Exception as e:
        logger.warning(f"Build cache unavailable (postgres not configured?): {e}")

    try:
        await init_telemetry_db()
        logger.info("Telemetry DB ready.")
    except ValueError as e:
        logger.info(f"Telemetry DB not configured: {e}")
    except Exception as e:
        logger.warning(f"Telemetry DB unavailable: {e}")

    try:
        yield
    finally:
        await close_build_cache()
        logger.info("Build cache closed.")
        await close_telemetry_db()
