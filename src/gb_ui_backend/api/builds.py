"""Build-level analytics endpoints (logs, etc.)."""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gb_ui_backend.config import Config, get_config
from gb_ui_backend.services.db_schema import GbdK8sResource

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/builds")


class BuildLogsResponse(BaseModel):
    lines: list[str]
    total: int


@router.get("/{build_id}/logs", response_model=BuildLogsResponse)
async def get_build_logs(
    build_id: str,
    container: str = "main",
    limit: int = Query(default=500, le=10000),
    offset: int = Query(default=0, ge=0),
    config: Config = Depends(get_config),
) -> BuildLogsResponse:
    """Fetch build logs from cloud logs service or K8s pod logs fallback."""

    # Try cloud logs first if configured
    if config.cloud_logs_url and config.cloud_logs_api_key:
        try:
            from gb_ui_backend.services.cloud_logs import get_cloud_logs_client

            client = get_cloud_logs_client(
                config.cloud_logs_url, config.cloud_logs_api_key
            )
            response = await client.query_logs(
                build_id=build_id,
                container_name=container if container == "sidecar" else None,
                page_size=limit + offset,
            )
            exclude = "sidecar" if container == "main" else None
            all_lines: list[str] = client.parse_logs(
                response, exclude_container=exclude
            )
            total = len(all_lines)
            page = (
                all_lines[offset : offset + limit]
                if offset
                else (all_lines[-limit:] if len(all_lines) > limit else all_lines)
            )
            return BuildLogsResponse(lines=page, total=total)
        except ImportError:
            pass
        except Exception as e:
            logger.warning("Cloud logs fetch for %s failed: %s", build_id, e)

    # K8s pod logs fallback from the analytics-service DB
    if not config.database_url:
        return BuildLogsResponse(lines=[], total=0)

    try:
        build_uuid = UUID(build_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid build ID")

    pods: list[GbdK8sResource] = []
    try:
        from gb_ui_backend.services.db_schema import _get_session_factory

        async with _get_session_factory()() as db:
            result = await db.execute(
                select(GbdK8sResource)
                .where(GbdK8sResource.build_id == build_uuid)
                .where(GbdK8sResource.kind == "Pod")
            )
            pods = list(result.scalars().all())
    except Exception as e:
        logger.warning("DB query for pod logs failed for %s: %s", build_id, e)
        return BuildLogsResponse(lines=[], total=0)

    all_lines = []
    for pod in pods:
        extra = pod.extra or {}
        container_log: str = (extra.get("logs") or {}).get(container, "")
        if container_log:
            if len(pods) > 1:
                all_lines.append(f"=== {pod.name} ===")
            all_lines.extend(container_log.split("\n"))

    total = len(all_lines)
    page = (
        all_lines[offset : offset + limit]
        if offset
        else (all_lines[-limit:] if len(all_lines) > limit else all_lines)
    )
    return BuildLogsResponse(lines=page, total=total)
