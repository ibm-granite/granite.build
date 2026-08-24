"""Analytics API endpoints — build status chart and failure trends.

Data source priority:
  1. analytics-service PostgreSQL (gbd_* tables, populated by K8s sync daemon)
  2. gbserver SQLite/PostgreSQL (gb_* tables, read directly via GbserverSource)

This means standalone users get charts and trends from gbserver's own data
without needing a separate database.
"""

from __future__ import annotations

import logging
import time
import uuid as uuid_module
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy import case, delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from gb_ui_backend.config import Config, get_config
from gb_ui_backend.services.db_schema import GbdBuild, GbdMeta, get_db, get_optional_db
from gb_ui_backend.services.gbserver_source import get_gbserver_source
from gb_ui_backend.services.request_identity import resolve_identity

TREND_SENTINEL_BUILD_ID = uuid_module.UUID("00000000-0000-0000-0000-000000000000")

logger = logging.getLogger(__name__)
router = APIRouter()


class BuildStatusPoint(BaseModel):
    """Mirrors gbserver's Status enum (src/gbserver/types/status.py)."""

    date: str
    running: int = 0
    success: int = 0
    failed: int = 0
    invalid: int = 0
    pending: int = 0
    submitted: int = 0
    retry_pending: int = 0
    cancel_requested: int = 0
    cancelled: int = 0
    running_test: int = 0
    success_test: int = 0
    failed_test: int = 0
    invalid_test: int = 0
    pending_test: int = 0
    submitted_test: int = 0
    retry_pending_test: int = 0
    cancel_requested_test: int = 0
    cancelled_test: int = 0


class FailureTrendRequest(BaseModel):
    days_back: int = 30
    date_from: Optional[str] = None  # YYYY-MM-DD; overrides days_back when both are set
    source: str = (
        "llm_phase1"  # "llm_phase1" (auto) or "llm_custom" (user-triggered custom)
    )
    date_to: Optional[str] = None  # YYYY-MM-DD inclusive
    categories: Optional[list[str]] = None
    exclude_tests: bool = False


class CategorizedBuild(BaseModel):
    build_id: str
    name: str
    username: str
    space_name: str
    created_at: str
    category: str
    confidence: float
    summary: Optional[str] = None


class FailureTrendResponse(BaseModel):
    labels: list[str]
    categories: list[str]
    series: dict[str, list[int]]
    builds_by_category: dict[str, list[CategorizedBuild]]
    total_analyzed: int
    analysis_time_ms: float


@router.get("/builds/status-chart", response_model=list[BuildStatusPoint])
async def get_build_status_chart(
    days_back: int = Query(default=30, ge=1, le=365),
    exclude_tests: bool = Query(default=False),
    db: Optional[AsyncSession] = Depends(get_optional_db),
    config: Config = Depends(get_config),
):
    # ── Path 1: analytics-service database ──────────────────────────────────────────────
    if db is not None:
        since = datetime.now(timezone.utc) - timedelta(days=days_back)
        is_test_expr = case(
            (GbdBuild.username.like("test%"), True),
            else_=False,
        )
        stmt = (
            select(
                func.date(GbdBuild.created_at).label("date"),
                GbdBuild.status,
                is_test_expr.label("is_test"),
                func.count().label("count"),
            )
            .where(GbdBuild.created_at >= since)
            .group_by(func.date(GbdBuild.created_at), GbdBuild.status, is_test_expr)
            .order_by(func.date(GbdBuild.created_at))
        )
        result = await db.execute(stmt)
        rows = result.all()
        if rows:
            pivot: dict[str, dict[str, dict[bool, int]]] = {}
            for row in rows:
                d = str(row.date)
                s = (row.status or "").lower()
                it = bool(row.is_test)
                pivot.setdefault(d, {}).setdefault(s, {})[it] = row.count
            statuses = [
                "running",
                "success",
                "failed",
                "invalid",
                "pending",
                "submitted",
                "retry_pending",
                "cancel_requested",
                "cancelled",
            ]
            return [
                BuildStatusPoint(
                    date=d,
                    **{s: pivot[d].get(s, {}).get(False, 0) for s in statuses},
                    **{
                        f"{s}_test": (
                            0 if exclude_tests else pivot[d].get(s, {}).get(True, 0)
                        )
                        for s in statuses
                    },
                )
                for d in sorted(pivot)
            ]
        # Analytics-service DB is empty (AI analysis hasn't run yet) — fall through to Path 2.

    # ── Path 2: gbserver source (standalone) ──────────────────────────────────
    source = get_gbserver_source()
    if source:
        points = await source.get_status_chart(
            days_back=days_back, exclude_tests=exclude_tests
        )
        return [BuildStatusPoint(**p) for p in points]

    raise HTTPException(503, "No data source configured")


@router.post("/builds/failure-trends", response_model=FailureTrendResponse)
async def get_failure_trends(
    req: FailureTrendRequest,
    db: Optional[AsyncSession] = Depends(get_optional_db),
    config: Config = Depends(get_config),
):
    t0 = time.monotonic()

    # Resolve date bounds — explicit date_from/date_to take precedence over days_back
    if req.date_from:
        try:
            since = datetime.strptime(req.date_from, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            since = datetime.now(timezone.utc) - timedelta(days=req.days_back)
    else:
        since = datetime.now(timezone.utc) - timedelta(days=req.days_back)

    until: Optional[datetime] = None
    if req.date_to:
        try:
            # Add one day so the upper bound is inclusive
            until = datetime.strptime(req.date_to, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            ) + timedelta(days=1)
        except ValueError:
            pass

    # ── Path 1: analytics-service database ──────────────────────────────────────────────
    if db is not None:
        from gb_ui_backend.services.db_schema import GbdMeta

        meta_source = (
            req.source if req.source in ("llm_phase1", "llm_custom") else "llm_phase1"
        )
        # Subquery: only the most recent analysis per build for the requested source
        latest_meta = (
            select(
                GbdMeta.build_id,
                func.max(GbdMeta.created_at).label("latest"),
            )
            .where(GbdMeta.source == meta_source)
            .group_by(GbdMeta.build_id)
            .subquery()
        )
        meta_alias = GbdMeta.__table__.alias("m")
        stmt = (
            select(GbdBuild, GbdMeta)
            .outerjoin(
                latest_meta,
                GbdBuild.id == latest_meta.c.build_id,
            )
            .outerjoin(
                GbdMeta,
                (GbdBuild.id == GbdMeta.build_id)
                & (GbdMeta.source == meta_source)
                & (GbdMeta.created_at == latest_meta.c.latest),
            )
            .where(GbdBuild.status == "failed", GbdBuild.created_at >= since)
            .order_by(GbdBuild.created_at)
        )
        if until:
            stmt = stmt.where(GbdBuild.created_at < until)
        if req.exclude_tests:
            stmt = stmt.where(~GbdBuild.username.like("test%"))
        result = await db.execute(stmt)
        rows = result.all()
        if rows:
            return _pivot_failure_trends(
                [
                    {
                        "build_id": str(b.id),
                        "name": b.name,
                        "username": b.username,
                        "space_name": b.space_name,
                        "created_at": b.created_at.isoformat() if b.created_at else "",
                        "category": (m.error_category_1 if m else None)
                        or "Uncategorized",
                        "confidence": m.confidence if m else 0.0,
                        "summary": m.summary if m else None,
                    }
                    for b, m in rows
                ],
                req.categories,
                t0,
            )
        # Analytics-service DB is empty (AI analysis hasn't run yet) — fall through to Path 2.

    # ── Path 2: gbserver source (standalone) ──────────────────────────────────
    source = get_gbserver_source()
    if source:
        builds = await source.get_failed_builds(
            days_back=req.days_back,
            date_from=req.date_from,
            date_to=req.date_to,
            exclude_tests=req.exclude_tests,
        )
        # No AI categories without the analytics-service DB — everything is "Uncategorized"
        rows_simple = [
            {
                "build_id": b["uuid"],
                "name": b["name"],
                "username": b["username"],
                "space_name": b["space_name"],
                "created_at": (
                    b["updated_time"].isoformat()
                    if hasattr(b.get("updated_time"), "isoformat")
                    else str(b.get("updated_time", ""))
                ),
                "category": "Uncategorized",
                "confidence": 0.0,
                "summary": None,
            }
            for b in builds
        ]
        return _pivot_failure_trends(rows_simple, req.categories, t0)

    raise HTTPException(503, "No data source configured")


def _pivot_failure_trends(
    rows: list[dict],
    filter_categories: Optional[list[str]],
    t0: float,
) -> FailureTrendResponse:
    DEFAULT_CAT = "Uncategorized"
    date_cat_counts: dict[str, dict[str, int]] = {}
    cat_builds: dict[str, list[CategorizedBuild]] = {}
    all_cats: set[str] = set()

    for row in rows:
        raw_dt = row["created_at"]
        if hasattr(raw_dt, "strftime"):
            date_str = raw_dt.strftime("%Y-%m-%d")
        else:
            date_str = str(raw_dt)[:10]

        cat = row.get("category") or DEFAULT_CAT
        if filter_categories and cat not in filter_categories:
            cat = DEFAULT_CAT
        all_cats.add(cat)
        date_cat_counts.setdefault(date_str, {})[cat] = (
            date_cat_counts.get(date_str, {}).get(cat, 0) + 1
        )
        cat_builds.setdefault(cat, []).append(
            CategorizedBuild(
                build_id=row["build_id"],
                name=row["name"],
                username=row["username"],
                space_name=row["space_name"],
                created_at=str(raw_dt)[:19],
                category=cat,
                confidence=row.get("confidence") or 0.0,
                summary=row.get("summary"),
            )
        )

    labels = sorted(date_cat_counts)
    categories = sorted(all_cats)
    return FailureTrendResponse(
        labels=labels,
        categories=categories,
        series={
            cat: [date_cat_counts.get(l, {}).get(cat, 0) for l in labels]
            for cat in categories
        },
        builds_by_category=cat_builds,
        total_analyzed=len(rows),
        analysis_time_ms=(time.monotonic() - t0) * 1000,
    )


# ── Saved trend analyses ──────────────────────────────────────────────────────


class SaveTrendRequest(BaseModel):
    data: dict
    title: Optional[str] = None
    is_public: bool = False


def get_current_author(request: Request) -> str:
    """Derive the caller's identity — see resolve_identity(). Used as a
    FastAPI dependency to scope/authorize saved-analysis ownership, so this
    must never be trusted from a client-supplied field."""
    return resolve_identity(request)


class TrendHistoryItem(BaseModel):
    update_id: str
    title: Optional[str] = None
    summary: str
    date_range_start: str
    date_range_end: str
    category_count: int
    total_builds: int
    is_public: bool
    author: str
    created_at: str


class TrendHistoryResponse(BaseModel):
    items: list[TrendHistoryItem]
    total_count: int


@router.post("/builds/failure-trends/save")
async def save_trend_analysis(
    body: SaveTrendRequest,
    author: str = Depends(get_current_author),
    db: AsyncSession = Depends(get_db),
    config: Config = Depends(get_config),
):
    if not config.db_enabled:
        raise HTTPException(503, "Database not configured")

    labels = body.data.get("labels", [])
    categories = body.data.get("categories", [])
    total = body.data.get("total_analyzed", 0)
    date_start = labels[0] if labels else ""
    date_end = labels[-1] if labels else ""
    summary = (
        f"{date_start} to {date_end}: {total} builds, {len(categories)} categories"
    )

    stmt = pg_insert(GbdMeta).values(
        update_id=uuid_module.uuid4(),
        build_id=TREND_SENTINEL_BUILD_ID,
        source="trend_analysis",
        analysis_type="trend_analysis",
        prompt_version="v1",
        summary=summary,
        raw_response=body.data,
        feedback_author=author,
        extras={
            "is_public": body.is_public,
            "title": body.title,
            "date_range_start": date_start,
            "date_range_end": date_end,
            "category_count": len(categories),
            "total_builds": total,
        },
        created_at=datetime.now(timezone.utc),
    )
    result = await db.execute(stmt.returning(GbdMeta.update_id))
    await db.commit()
    row = result.fetchone()
    return {"success": True, "update_id": str(row[0])}


@router.get("/builds/failure-trends/history", response_model=TrendHistoryResponse)
async def get_trend_history(
    tab: str = Query(default="mine"),
    author: str = Depends(get_current_author),
    db: AsyncSession = Depends(get_db),
    config: Config = Depends(get_config),
):
    if not config.db_enabled:
        return TrendHistoryResponse(items=[], total_count=0)

    filters = [
        GbdMeta.source == "trend_analysis",
        GbdMeta.build_id == TREND_SENTINEL_BUILD_ID,
    ]
    if tab == "mine":
        filters.append(GbdMeta.feedback_author == author)
    else:
        # .astext is PostgreSQL JSONB-specific and doesn't exist on the generic
        # JSON comparator (extras is declared as plain JSON to stay portable
        # across the analytics service's SQLite/Postgres backends) — as_boolean() is the
        # dialect-agnostic equivalent.
        filters.append(GbdMeta.extras["is_public"].as_boolean() == True)  # noqa: E712

    total = (
        await db.execute(select(func.count(GbdMeta.id)).where(*filters))
    ).scalar() or 0
    rows = (
        (
            await db.execute(
                select(GbdMeta)
                .where(*filters)
                .order_by(GbdMeta.created_at.desc())
                .limit(50)
            )
        )
        .scalars()
        .all()
    )

    items = []
    for r in rows:
        ex = r.extras or {}
        items.append(
            TrendHistoryItem(
                update_id=str(r.update_id),
                title=ex.get("title"),
                summary=r.summary or "",
                date_range_start=ex.get("date_range_start", ""),
                date_range_end=ex.get("date_range_end", ""),
                category_count=ex.get("category_count", 0),
                total_builds=ex.get("total_builds", 0),
                is_public=ex.get("is_public", False),
                author=r.feedback_author or "unknown",
                created_at=r.created_at.isoformat() if r.created_at else "",
            )
        )
    return TrendHistoryResponse(items=items, total_count=total)


@router.get("/builds/failure-trends/{update_id}")
async def get_saved_trend(
    update_id: str,
    db: AsyncSession = Depends(get_db),
    config: Config = Depends(get_config),
):
    if not config.db_enabled:
        raise HTTPException(503, "Database not configured")
    try:
        uid = uuid_module.UUID(update_id)
    except ValueError:
        raise HTTPException(400, "Invalid update_id")
    row = (
        await db.execute(
            select(GbdMeta).where(
                GbdMeta.update_id == uid, GbdMeta.source == "trend_analysis"
            )
        )
    ).scalar_one_or_none()
    if not row:
        raise HTTPException(404, "Not found")
    return {
        "update_id": str(row.update_id),
        "data": row.raw_response,
        "title": (row.extras or {}).get("title"),
    }


@router.patch("/builds/failure-trends/{update_id}/visibility")
async def toggle_trend_visibility(
    update_id: str,
    is_public: bool = Query(),
    author: str = Depends(get_current_author),
    db: AsyncSession = Depends(get_db),
    config: Config = Depends(get_config),
):
    if not config.db_enabled:
        raise HTTPException(503, "Database not configured")
    try:
        uid = uuid_module.UUID(update_id)
    except ValueError:
        raise HTTPException(400, "Invalid update_id")
    row = (
        await db.execute(
            select(GbdMeta).where(
                GbdMeta.update_id == uid, GbdMeta.source == "trend_analysis"
            )
        )
    ).scalar_one_or_none()
    if not row:
        raise HTTPException(404, "Not found")
    if row.feedback_author != author:
        raise HTTPException(403, "Not your analysis")
    ex = dict(row.extras or {})
    ex["is_public"] = is_public
    row.extras = ex
    await db.commit()
    return {"success": True}


@router.delete("/builds/failure-trends/{update_id}")
async def delete_saved_trend(
    update_id: str,
    author: str = Depends(get_current_author),
    db: AsyncSession = Depends(get_db),
    config: Config = Depends(get_config),
):
    if not config.db_enabled:
        raise HTTPException(503, "Database not configured")
    try:
        uid = uuid_module.UUID(update_id)
    except ValueError:
        raise HTTPException(400, "Invalid update_id")
    row = (
        await db.execute(
            select(GbdMeta).where(
                GbdMeta.update_id == uid, GbdMeta.source == "trend_analysis"
            )
        )
    ).scalar_one_or_none()
    if not row:
        raise HTTPException(404, "Not found")
    if row.feedback_author != author:
        raise HTTPException(403, "Not your analysis")
    await db.execute(delete(GbdMeta).where(GbdMeta.update_id == uid))
    await db.commit()
    return {"success": True}
