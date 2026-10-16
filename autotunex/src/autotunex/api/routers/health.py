# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from autotunex import __version__
from autotunex.api.deps import SessionDep, SettingsDep
from autotunex.core.exceptions import DatabaseUnavailableError
from autotunex.models.common import HealthResponse, ReadinessResponse

router = APIRouter(tags=["meta"])


@router.get("/health", summary="Liveness probe")
async def health(settings: SettingsDep) -> HealthResponse:
    """Report that the service is up.

    Does not touch the database — this is a liveness probe, not a readiness one.
    """
    return HealthResponse(status="ok", service=settings.app_name, version=__version__)


@router.get("/health/live", summary="Liveness probe (alias)")
async def health_live(settings: SettingsDep) -> HealthResponse:
    """Alias of ``GET /health`` giving orchestrators an explicit live/ready split.

    Like ``/health``, it does not touch the database.
    """
    return HealthResponse(status="ok", service=settings.app_name, version=__version__)


@router.get("/health/ready", summary="Readiness probe")
async def health_ready(session: SessionDep) -> ReadinessResponse:
    """Report readiness to serve traffic, gated on database reachability.

    Runs a trivial ``SELECT 1``. A ``SQLAlchemyError`` — a dead pool connection,
    an unreachable host — is turned into a 503 (:class:`DatabaseUnavailableError`)
    rather than the generic 500 the global handler would emit for an uncaught
    error, so an orchestrator can keep traffic away until the database is back.
    Catching ``SQLAlchemyError`` specifically (not a broad ``Exception``) is the
    point: any other failure still surfaces as a 500.
    """
    try:
        await session.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        raise DatabaseUnavailableError() from exc
    return ReadinessResponse()
