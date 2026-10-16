"""AI analysis endpoints."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid as uuid_mod
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from gb_ui_backend.config import Config, get_config
from gb_ui_backend.services.db_schema import GbdMeta, get_db
from gb_ui_backend.services.request_identity import resolve_identity

logger = logging.getLogger(__name__)
router = APIRouter()

_trigger_task: Optional[asyncio.Task] = None
_trigger_analyzing = False

_ANALYZE_LOGS_RATE_LIMIT_WINDOW_SECONDS = 60
_ANALYZE_LOGS_RATE_LIMIT_MAX_CALLS = 5
_analyze_logs_call_times: dict[str, list[float]] = {}


def _rate_limit_analyze_logs(request: Request) -> None:
    """Minimal per-identity sliding-window rate limit for analyze_logs.

    This endpoint makes a billable LLM call per request — bound abuse by
    identity. Routes mounted in gbserver see the trusted user gbserver's
    AuthMiddleware already resolved (`request.state.data["user"]`); the
    X-User-Email header and client IP are fallbacks for running this app
    standalone, outside gbserver, where there's no AuthMiddleware at all.
    """
    identity = resolve_identity(
        request, fallback=request.client.host if request.client else "unknown"
    )
    now = time.monotonic()
    recent = [
        t
        for t in _analyze_logs_call_times.get(identity, [])
        if now - t < _ANALYZE_LOGS_RATE_LIMIT_WINDOW_SECONDS
    ]
    if len(recent) >= _ANALYZE_LOGS_RATE_LIMIT_MAX_CALLS:
        raise HTTPException(429, "Rate limit exceeded — try again later")
    recent.append(now)
    _analyze_logs_call_times[identity] = recent


class AIAnalysisOut(BaseModel):
    update_id: str
    build_id: str
    source: str
    analysis_type: Optional[str] = None
    summary: str
    root_cause: str
    suggested_action: str
    issues: list[dict]
    confidence: float
    model_name: Optional[str] = None
    error_category_1: Optional[str] = None
    error_category_2: Optional[str] = None
    kb_recommendation: Optional[str] = None
    parent_uid: Optional[str] = None
    created_at: str
    feedback_rating: Optional[int] = None
    feedback_helpful: Optional[bool] = None
    corrected_root_cause: Optional[str] = None
    feedback_comment: Optional[str] = None
    upvotes: int = 0
    downvotes: int = 0


class AnalyzeLogsIn(BaseModel):
    log_content: str
    build_name: str = ""
    status: str = "running"


class FeedbackIn(BaseModel):
    update_id: str
    rating: Optional[int] = None
    helpful: Optional[bool] = None
    corrected_root_cause: Optional[str] = None
    comment: Optional[str] = None


@router.get("/ai/status")
async def get_ai_daemon_status(config: Config = Depends(get_config)):
    return {
        "running": _trigger_task is not None and not _trigger_task.done(),
        "analyzing": _trigger_analyzing,
        "llm_configured": bool(config.llm_base_url and config.llm_api_key),
    }


class RunAnalysisIn(BaseModel):
    mode: str = "auto"
    categories: Optional[List[str]] = None
    days_back: int = 90


@router.post("/ai/run")
async def run_analysis(body: RunAnalysisIn, config: Config = Depends(get_config)):
    global _trigger_task, _trigger_analyzing

    if _trigger_analyzing:
        raise HTTPException(409, "Analysis already running")
    if not config.llm_base_url or not config.llm_api_key:
        raise HTTPException(
            503, "LLM not configured — set GB_UI_LLM_BASE_URL and GB_UI_LLM_API_KEY"
        )
    if not config.db_enabled:
        raise HTTPException(503, "Database not configured — set GB_UI_DATABASE_URL")
    if body.mode == "custom" and not body.categories:
        raise HTTPException(422, "categories required for custom mode")

    async def do_run():
        global _trigger_analyzing
        _trigger_analyzing = True
        try:
            engine = create_async_engine(
                config.database_url, echo=False, pool_size=3, max_overflow=5
            )
            session_factory = async_sessionmaker(engine, expire_on_commit=False)
            if body.mode == "auto":
                from gb_ui_backend.services.ai_daemon import create_ai_daemon

                daemon = await create_ai_daemon(
                    database_url=config.database_url,
                    llm_base_url=config.llm_base_url,
                    llm_api_key=config.llm_api_key,
                    llm_models=config.llm_models_list,
                    llm_timeout=config.llm_timeout,
                    gbserver_db_url=config.gbserver_db_url,
                    gbserver_db_schema=config.gbserver_db_schema,
                )
                await daemon._backfill_categories()
                await daemon._process_gbserver_batch()
            else:
                from gb_ui_backend.services.ai_daemon import run_custom_categorization

                await run_custom_categorization(
                    session_factory=session_factory,
                    llm_base_url=config.llm_base_url,
                    llm_api_key=config.llm_api_key,
                    llm_models=config.llm_models_list,
                    categories=body.categories,
                    days_back=body.days_back,
                    llm_timeout=config.llm_timeout,
                )
            await engine.dispose()
        except Exception as e:
            logger.error("Triggered analysis failed: %s", e, exc_info=True)
        finally:
            _trigger_analyzing = False

    _trigger_task = asyncio.create_task(do_run())
    return {"started": True, "mode": body.mode}


@router.get("/builds/{build_id}/ai-analysis", response_model=list[AIAnalysisOut])
async def get_ai_analysis(
    build_id: str,
    db: AsyncSession = Depends(get_db),
    config: Config = Depends(get_config),
):
    if not config.db_enabled:
        raise HTTPException(503, "Database not configured")

    try:
        build_uuid = UUID(build_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid build ID")

    stmt = (
        select(GbdMeta)
        .where(GbdMeta.build_id == build_uuid)
        .order_by(GbdMeta.created_at.desc())
        .limit(10)
    )
    result = await db.execute(stmt)
    metas = result.scalars().all()

    return [
        AIAnalysisOut(
            update_id=str(m.update_id),
            build_id=str(m.build_id),
            source=m.source or "system",
            analysis_type=m.analysis_type,
            summary=m.summary or "",
            root_cause=m.root_cause or "",
            suggested_action=m.suggested_action or "",
            issues=m.issues or [],
            confidence=m.confidence or 0.0,
            model_name=m.model_name,
            error_category_1=m.error_category_1,
            error_category_2=m.error_category_2,
            kb_recommendation=m.kb_recommendation,
            parent_uid=str(m.parent_uid) if m.parent_uid else None,
            created_at=m.created_at.isoformat(),
            feedback_rating=m.feedback_rating,
            feedback_helpful=m.feedback_helpful,
            corrected_root_cause=m.corrected_root_cause,
            feedback_comment=m.feedback_comment,
            upvotes=m.upvotes or 0,
            downvotes=m.downvotes or 0,
        )
        for m in metas
    ]


@router.post("/builds/{build_id}/analyze-logs", response_model=AIAnalysisOut)
async def analyze_logs(
    build_id: str,
    body: AnalyzeLogsIn,
    config: Config = Depends(get_config),
    _rate_limit: None = Depends(_rate_limit_analyze_logs),
):
    if not config.llm_base_url:
        raise HTTPException(503, "AI analysis not configured")

    from gb_ui_backend.services.ai_daemon import _analyze_build_async

    # Truncate to ~50k tokens to keep the request reasonable
    log_content = body.log_content[:200_000]

    result = await _analyze_build_async(
        build_data={
            "build_id": build_id,
            "build_name": body.build_name or build_id[:8],
            "status": body.status,
            "pod_logs": log_content,
        },
        llm_base_url=config.llm_base_url,
        llm_api_key=config.llm_api_key,
        llm_models=config.llm_models_list,
        llm_timeout=config.llm_timeout,
    )

    if result.error:
        raise HTTPException(500, f"Analysis failed: {result.error}")

    return AIAnalysisOut(
        update_id=str(uuid_mod.uuid4()),
        build_id=build_id,
        source="llm_phase1",
        analysis_type=result.analysis_type,
        summary=result.summary or "",
        root_cause=result.root_cause or "",
        suggested_action=result.suggested_action or "",
        issues=result.issues or [],
        confidence=result.confidence or 0.0,
        model_name=result.model_name,
        error_category_1=result.error_category_1,
        error_category_2=result.error_category_2,
        created_at=result.created_at,
        upvotes=0,
        downvotes=0,
    )


@router.post("/builds/{build_id}/ai-feedback")
async def submit_feedback(
    build_id: str,
    body: FeedbackIn,
    db: AsyncSession = Depends(get_db),
    config: Config = Depends(get_config),
):
    if not config.db_enabled:
        raise HTTPException(503, "Database not configured")

    stmt = select(GbdMeta).where(GbdMeta.update_id == body.update_id)
    result = await db.execute(stmt)
    meta = result.scalar_one_or_none()
    if not meta:
        raise HTTPException(404, "Analysis record not found")

    if body.rating is not None:
        meta.feedback_rating = body.rating
    if body.helpful is not None:
        meta.feedback_helpful = body.helpful
        if body.helpful:
            meta.upvotes = (meta.upvotes or 0) + 1
        else:
            meta.downvotes = (meta.downvotes or 0) + 1
    if body.corrected_root_cause:
        meta.corrected_root_cause = body.corrected_root_cause
    if body.comment:
        meta.feedback_comment = body.comment

    await db.commit()
    return {"success": True}
