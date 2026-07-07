"""
AI analysis daemon — analyzes builds with an LLM and writes results to gbd_meta.

Ported from gb_dashboard/services/ai_daemon.py.
RITS-specific client replaced with gb_ui_backend.services.llm_client (OpenAI-compatible).
Multi-phase analysis (Phase 1 failure/health, Phase 2 knowledge base search) retained.
Progress analysis (Phase 3) and consensus refinement (Phase 5) are simplified/omitted
in this initial port — the core failure/health analysis is the primary value.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from gb_ui_backend.config import get_config
from gb_ui_backend.services.ai_data_collectors import (
    CompositeDataCollector,
    get_build_with_data,
    query_knowledge_base,
)
from gb_ui_backend.services.ai_prompts import (
    PROMPT_VERSION,
    BuildContext,
    KnowledgeBaseEntry,
    determine_analysis_type,
    format_phase2_prompt,
    format_user_prompt,
    get_system_prompt,
)
from gb_ui_backend.services.db_schema import GbdBuild, GbdMeta

logger = logging.getLogger(__name__)

_daemon_instance: Optional["AIDaemon"] = None


def get_daemon() -> Optional["AIDaemon"]:
    return _daemon_instance


_ISSUE_TYPE_TO_CATEGORY: Dict[str, str] = {
    "oom": "OOM",
    "timeout": "Timeout",
    "network": "Network",
    "gpu": "GPU",
    "storage": "Storage",
    "config": "Configuration",
    "configuration": "Configuration",
    "infrastructure": "Infrastructure",
    "code": "Code Error",
}

_CATEGORY_KEYWORDS: List[tuple] = [
    ("OOM",           ["out of memory", "cuda out of memory", "oom", "memory exhausted", "outofmemoryerror"]),
    ("GPU",           ["cuda", "nccl", "gpu"]),
    ("Timeout",       ["timeout", "timed out", "deadline exceeded"]),
    ("Network",       ["network", "connection refused", "connection reset", "dns", "socket"]),
    ("Storage",       ["no space left", "disk full", "storage", "filesystem"]),
    ("Configuration", ["misconfigured", "invalid config", "environment variable", "missing env"]),
    ("Code Error",    ["traceback", "exception", "assertionerror", "importerror", "syntaxerror"]),
]


def _infer_category(parsed: Dict[str, Any]) -> Optional[str]:
    """Infer error_category_1 from response fields when the model omits it."""
    for issue in (parsed.get("issues") or []):
        cat = _ISSUE_TYPE_TO_CATEGORY.get((issue.get("type") or "").lower())
        if cat:
            return cat
    text = " ".join([
        (parsed.get("summary") or "").lower(),
        (parsed.get("root_cause") or "").lower(),
    ])
    for cat, keywords in _CATEGORY_KEYWORDS:
        if any(kw in text for kw in keywords):
            return cat
    return "Other"


# Statuses that should be analyzed
ANALYZE_STATUSES = {"failed", "error", "deleted", "running", "pending", "suspended"}
# Statuses that get re-analyzed periodically (build is still active)
ACTIVE_STATUSES = {"running", "pending", "suspended"}


@dataclass
class AnalysisResult:
    build_id: str
    build_name: str
    analysis_type: str
    model_name: str
    prompt_version: str
    created_at: str
    build_status_at_analysis: str
    summary: Optional[str]
    root_cause: Optional[str]
    suggested_action: Optional[str]
    issues: Optional[List[Dict[str, Any]]]
    error_messages: Optional[List[str]]
    error_category_1: Optional[str]
    error_category_2: Optional[str]
    confidence: Optional[float]
    raw_response: Dict[str, Any]
    tokens_prompt: int
    tokens_completion: int
    latency_ms: int
    error: Optional[str] = None


async def _analyze_build_async(
    build_data: Dict[str, Any],
    llm_base_url: str,
    llm_api_key: str,
    llm_models: List[str],
    llm_timeout: int = 60,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> AnalysisResult:
    """Async LLM analysis — runs directly in the event loop via asyncio.gather."""
    from gb_ui_backend.services.llm_client import LLMClient

    build_id = build_data["build_id"]
    build_name = build_data["build_name"]
    status = build_data["status"]
    short_id = build_id[:8]

    try:
        analysis_type = determine_analysis_type(status)
        context = BuildContext(
            build_id=build_id, build_name=build_name, status=status,
            failure_reason=build_data.get("failure_reason"),
            failure_message=build_data.get("failure_message"),
            total_cpu=build_data.get("total_cpu"),
            total_memory=build_data.get("total_memory"),
            total_gpu=build_data.get("total_gpu", 0),
            created_at=build_data.get("created_at"),
            finished_at=build_data.get("finished_at"),
            duration_str=build_data.get("duration_str"),
            k8s_resources=build_data.get("k8s_resources", []),
            events=build_data.get("events", []),
            pod_logs=build_data.get("pod_logs"),
            step_logs=build_data.get("step_logs", {}),
            clusterqueue_capacity_gpu=build_data.get("clusterqueue_capacity_gpu", 0),
            clusterqueue_usage_gpu=build_data.get("clusterqueue_usage_gpu", 0),
            clusterqueue_pending_workloads=build_data.get("clusterqueue_pending_workloads", 0),
            gbserver_events=build_data.get("gbserver_events", []),
            gbserver_status_msgs=build_data.get("gbserver_status_msgs", []),
        )
        system_prompt = get_system_prompt(analysis_type)
        user_prompt = format_user_prompt(context)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        logger.info("LLM call: %s (%s) — %d chars", short_id, status,
                    len(system_prompt) + len(user_prompt))

        client = LLMClient(
            base_url=llm_base_url,
            api_key=llm_api_key,
            models=llm_models,
            timeout=llm_timeout,
        )

        t0 = time.monotonic()
        raw_response = await client.chat_completion(
            messages=messages, temperature=temperature, max_tokens=max_tokens)
        latency_ms = int((time.monotonic() - t0) * 1000)

        content = raw_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        model_used = raw_response.get("model", llm_models[0] if llm_models else "unknown")
        usage = raw_response.get("usage", {})
        tokens_prompt = usage.get("prompt_tokens", 0)
        tokens_completion = usage.get("completion_tokens", 0)

        try:
            clean = content.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            parsed = json.loads(clean.strip())
        except json.JSONDecodeError:
            parsed = {
                "summary": content[:500],
                "root_cause": None,
                "suggested_action": None,
                "issues": [],
                "confidence": 0.5,
            }

        logger.info("LLM response: %s — %dms, %d tokens — %s",
                    short_id, latency_ms, tokens_completion, parsed.get("summary", "")[:80])

        error_category_1 = parsed.get("error_category_1")
        if not error_category_1 and analysis_type == "failure":
            error_category_1 = _infer_category(parsed)

        return AnalysisResult(
            build_id=build_id, build_name=build_name, analysis_type=analysis_type,
            model_name=model_used, prompt_version=PROMPT_VERSION,
            created_at=datetime.now(timezone.utc).isoformat(),
            build_status_at_analysis=status,
            summary=parsed.get("summary"),
            root_cause=parsed.get("root_cause"),
            suggested_action=parsed.get("suggested_action"),
            issues=parsed.get("issues"),
            error_messages=parsed.get("error_messages"),
            error_category_1=error_category_1,
            error_category_2=parsed.get("error_category_2"),
            confidence=parsed.get("confidence"),
            raw_response=raw_response,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            latency_ms=latency_ms,
        )

    except Exception as e:
        logger.error("Error analyzing %s: %s", short_id, e)
        return AnalysisResult(
            build_id=build_id, build_name=build_name, analysis_type="error",
            model_name="", prompt_version=PROMPT_VERSION,
            created_at=datetime.now(timezone.utc).isoformat(),
            build_status_at_analysis=status,
            summary=None, root_cause=None, suggested_action=None,
            issues=None, error_messages=None, error_category_1=None, error_category_2=None,
            confidence=None, raw_response={},
            tokens_prompt=0, tokens_completion=0, latency_ms=0,
            error=str(e),
        )


class AIDaemon:
    """Analyzes builds with an LLM and stores results in gbd_meta."""

    def __init__(
        self,
        session_factory: async_sessionmaker,
        data_collector: CompositeDataCollector,
        llm_base_url: str,
        llm_api_key: str,
        llm_models: List[str],
        llm_timeout: int = 60,
        poll_interval: int = 30,
        reanalyze_active_interval: int = 300,
    ):
        self.session_factory = session_factory
        self.data_collector = data_collector
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.llm_models = llm_models
        self.llm_timeout = llm_timeout
        self.poll_interval = poll_interval
        self.reanalyze_active_interval = reanalyze_active_interval
        self._running = False
        self.is_analyzing = False
        self._last_analyzed_active: Dict[str, datetime] = {}

    async def _backfill_categories(self) -> None:
        """Infer error_category_1 for existing rows where the model omitted it.

        Runs once at startup. Reads the already-stored raw_response and applies
        _infer_category so no LLM calls are needed.
        """
        from sqlalchemy import update
        async with self.session_factory() as session:
            result = await session.execute(
                select(GbdMeta.id, GbdMeta.raw_response)
                .where(
                    GbdMeta.analysis_type == "failure",
                    GbdMeta.error_category_1.is_(None),
                    GbdMeta.raw_response.isnot(None),
                )
            )
            rows = result.all()

        if not rows:
            return

        logger.info("Backfilling error_category_1 for %d existing analyses", len(rows))
        updated = 0
        async with self.session_factory() as session:
            for row_id, raw_response in rows:
                try:
                    content = raw_response.get("choices", [{}])[0].get("message", {}).get("content", "")
                    clean = content.strip()
                    if clean.startswith("```"):
                        clean = clean.split("```")[1]
                        if clean.startswith("json"):
                            clean = clean[4:]
                    parsed = json.loads(clean.strip())
                    cat = parsed.get("error_category_1") or _infer_category(parsed)
                    if cat:
                        await session.execute(
                            update(GbdMeta)
                            .where(GbdMeta.id == row_id)
                            .values(error_category_1=cat)
                        )
                        updated += 1
                except Exception:
                    pass
            await session.commit()

        logger.info("Backfilled error_category_1 for %d rows", updated)

    async def run(self) -> None:
        self._running = True
        logger.info("AI daemon starting (models=%s)", self.llm_models)
        await self._backfill_categories()
        while self._running:
            self.is_analyzing = True
            try:
                await self._process_batch()
            except Exception as e:
                logger.error("AI daemon error: %s", e, exc_info=True)
            try:
                await self._process_gbserver_batch()
            except Exception as e:
                logger.error("AI daemon (gbserver) error: %s", e, exc_info=True)
            self.is_analyzing = False
            await asyncio.sleep(self.poll_interval)

    def stop(self) -> None:
        self._running = False

    async def _find_gbserver_builds_needing_analysis(
        self,
        session: AsyncSession,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find failed builds in gbserver that don't yet have a failure analysis."""
        from gb_ui_backend.services.gbserver_source import get_gbserver_source
        source = get_gbserver_source()
        if not source:
            return []

        analyzed_stmt = (
            select(GbdMeta.build_id)
            .where(GbdMeta.analysis_type == "failure")
            .distinct()
        )
        result = await session.execute(analyzed_stmt)
        analyzed_ids = {str(row[0]) for row in result.fetchall()}

        failed = await source.get_failed_builds(days_back=90, limit=50)
        return [b for b in failed if b["uuid"] not in analyzed_ids][:limit]

    async def _upsert_gbserver_build(
        self,
        session: AsyncSession,
        raw: Dict[str, Any],
    ) -> GbdBuild:
        """Insert (or refresh) a minimal GbdBuild row from gbserver data.

        Creates the row so the failure-trends query (which starts from gbd_builds)
        can join against the gbd_meta analysis result.
        """
        build_id = uuid.UUID(raw["uuid"])
        now = datetime.now(timezone.utc)
        stmt = (
            insert(GbdBuild)
            .values(
                id=build_id,
                name=raw["name"],
                space_name=raw.get("space_name") or "",
                username=raw.get("username") or "",
                status=raw.get("status") or "failed",
                created_at=raw.get("created_time") or now,
                updated_at=raw.get("updated_time") or now,
                finished_at=raw.get("updated_time"),
            )
            .on_conflict_do_update(
                index_elements=["id"],
                set_={"status": raw.get("status") or "failed", "updated_at": now},
            )
        )
        await session.execute(stmt)
        result = await session.execute(select(GbdBuild).where(GbdBuild.id == build_id))
        return result.scalar_one()

    async def _process_gbserver_batch(self) -> None:
        """Analyze failed builds discovered directly from gbserver's database.

        Runs alongside _process_batch so the AI daemon works even when the K8s
        sync daemon is not running (e.g. no GB_UI_KUBECONFIG configured).
        """
        async with self.session_factory() as session:
            builds_raw = await self._find_gbserver_builds_needing_analysis(session)
        if not builds_raw:
            return

        logger.info("AI daemon: analyzing %d gbserver-only builds", len(builds_raw))

        build_objects: List[GbdBuild] = []
        async with self.session_factory() as session:
            for raw in builds_raw:
                try:
                    obj = await self._upsert_gbserver_build(session, raw)
                    build_objects.append(obj)
                except Exception as e:
                    logger.warning("Failed to upsert gbserver build %s: %s", raw.get("name"), e)
            await session.commit()

        if not build_objects:
            return

        build_data_list: List[Dict[str, Any]] = []
        async with self.session_factory() as session:
            for build in build_objects:
                try:
                    context = await self.data_collector.collect_build_context(session, build)
                    build_data_list.append(context.to_dict())
                except Exception as e:
                    logger.warning("Failed to collect context for %s: %s", build.name, e)

        if not build_data_list:
            return

        results = await asyncio.gather(
            *[
                _analyze_build_async(
                    bd,
                    self.llm_base_url,
                    self.llm_api_key,
                    self.llm_models,
                    self.llm_timeout,
                    0.1,
                    4096,
                )
                for bd in build_data_list
            ],
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, Exception):
                logger.error("Analysis error (gbserver): %s", result)
                continue
            if result.error:
                logger.warning("Analysis error for %s: %s", result.build_id[:8], result.error)
                continue
            try:
                async with self.session_factory() as session:
                    await self._save_result(session, result)
                    await session.commit()
            except Exception as e:
                logger.error("Failed to save gbserver analysis for %s: %s", result.build_id[:8], e)

    async def _process_batch(self) -> None:
        now = datetime.now(timezone.utc)

        async with self.session_factory() as session:
            # Exclude terminal builds already analyzed in this session or in the DB.
            # Active builds are always included so the in-memory interval check applies.
            already_analyzed_subq = select(GbdMeta.build_id).where(
                GbdMeta.source == "llm_phase1",
                GbdMeta.error_category_1.isnot(None),
            )
            stmt = (select(GbdBuild)
                    .where(
                        GbdBuild.status.in_(list(ANALYZE_STATUSES)),
                        or_(
                            GbdBuild.status.in_(list(ACTIVE_STATUSES)),
                            GbdBuild.id.not_in(already_analyzed_subq),
                        ),
                    )
                    .order_by(GbdBuild.updated_at.desc())
                    .limit(50))
            result = await session.execute(stmt)
            all_builds = result.scalars().all()

        builds_to_analyze = []
        for build in all_builds:
            build_id = str(build.id)
            is_terminal = build.status not in ACTIVE_STATUSES
            if not is_terminal:
                last = self._last_analyzed_active.get(build_id)
                if last and (now - last).total_seconds() < self.reanalyze_active_interval:
                    continue
            builds_to_analyze.append(build)

        if not builds_to_analyze:
            return

        logger.info("AI daemon: analyzing %d builds", len(builds_to_analyze))

        # Collect context for each build
        build_data_list = []
        async with self.session_factory() as session:
            for build in builds_to_analyze:
                try:
                    context = await self.data_collector.collect_build_context(session, build)
                    build_data_list.append(context.to_dict())
                except Exception as e:
                    logger.warning("Failed to collect context for %s: %s", build.name, e)

        if not build_data_list:
            return

        # Run LLM analysis concurrently — the client is async (httpx) so no
        # subprocess or thread pool is needed.
        results = await asyncio.gather(
            *[
                _analyze_build_async(
                    bd,
                    self.llm_base_url,
                    self.llm_api_key,
                    self.llm_models,
                    self.llm_timeout,
                    0.1,
                    4096,
                )
                for bd in build_data_list
            ],
            return_exceptions=True,
        )

        # Persist results — each in its own session so one failure doesn't
        # abort the transaction and block the rest of the batch.
        for result in results:
            if isinstance(result, Exception):
                logger.error("Analysis worker exception: %s", result)
                continue
            if result.error:
                logger.warning("Analysis error for %s: %s", result.build_id[:8], result.error)
                continue
            try:
                async with self.session_factory() as session:
                    await self._save_result(session, result)
                    await session.commit()
                is_terminal = result.build_status_at_analysis not in ACTIVE_STATUSES
                if not is_terminal:
                    self._last_analyzed_active[result.build_id] = now
            except Exception as e:
                logger.error("Failed to save analysis for %s: %s", result.build_id[:8], e)

        # Phase 2: knowledge base search for failed builds with results
        failed_results = [r for r in results
                          if not isinstance(r, Exception) and not r.error
                          and r.analysis_type == "failure" and r.root_cause]
        if failed_results:
            await self._run_phase2(failed_results)

    async def _save_result(self, session: AsyncSession, result: AnalysisResult) -> None:
        try:
            build_uuid = uuid.UUID(result.build_id)
        except ValueError:
            return
        stmt = insert(GbdMeta).values(
            update_id=uuid.uuid4(),
            build_id=build_uuid,
            source="llm_phase1",
            analysis_type=result.analysis_type,
            model_name=result.model_name,
            prompt_version=result.prompt_version,
            summary=result.summary,
            root_cause=result.root_cause,
            suggested_action=result.suggested_action,
            issues=result.issues or [],
            confidence=result.confidence,
            raw_response=result.raw_response,
            tokens_prompt=result.tokens_prompt,
            tokens_completion=result.tokens_completion,
            latency_ms=result.latency_ms,
            error_category_1=result.error_category_1,
            error_category_2=result.error_category_2,
            upvotes=0,
            downvotes=0,
            created_at=datetime.now(timezone.utc),
        )
        await session.execute(stmt)
        logger.info("Saved analysis for %s (%s): %s",
                    result.build_id[:8], result.analysis_type,
                    (result.summary or "")[:80])

    async def _run_phase2(self, phase1_results: List[AnalysisResult]) -> None:
        """Search knowledge base and store phase 2 recommendations."""
        for r in phase1_results:
            try:
                build_uuid = uuid.UUID(r.build_id)
            except ValueError:
                continue
            async with self.session_factory() as session:
                try:
                    kb = await query_knowledge_base(
                        session,
                        error_category_1=r.error_category_1,
                        error_category_2=r.error_category_2,
                        max_entries=20,
                        exclude_build_id=build_uuid,
                    )
                    if not kb:
                        continue
                    # Run phase 2 in-process (it's much lighter than phase 1)
                    phase2_prompt = format_phase2_prompt(
                        root_cause=r.root_cause or "",
                        error_messages=r.error_messages or [],
                        error_categories=[r.error_category_1, r.error_category_2],
                        knowledge_base=kb,
                    )
                    system = get_system_prompt("solution_search")
                    from gb_ui_backend.services.llm_client import LLMClient
                    client = LLMClient(
                        base_url=self.llm_base_url,
                        api_key=self.llm_api_key,
                        models=self.llm_models,
                        timeout=self.llm_timeout,
                    )
                    raw = await client.chat_completion(
                        messages=[{"role": "system", "content": system},
                                   {"role": "user", "content": phase2_prompt}],
                        temperature=0.1, max_tokens=1024,
                    )
                    content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
                    try:
                        parsed = json.loads(content.strip())
                    except json.JSONDecodeError:
                        continue
                    recommendation = parsed.get("recommendation")
                    if not recommendation:
                        continue

                    # Find the phase 1 meta record to link to
                    existing = await session.execute(
                        select(GbdMeta)
                        .where(GbdMeta.build_id == build_uuid, GbdMeta.source == "llm_phase1")
                        .order_by(GbdMeta.created_at.desc())
                        .limit(1)
                    )
                    parent = existing.scalar_one_or_none()
                    parent_uid = parent.update_id if parent else None

                    stmt = insert(GbdMeta).values(
                        update_id=uuid.uuid4(),
                        build_id=build_uuid,
                        source="llm_phase2",
                        analysis_type="solution_search",
                        model_name=raw.get("model", ""),
                        prompt_version=PROMPT_VERSION,
                        summary=recommendation,
                        root_cause=r.root_cause,
                        kb_search_query=parsed.get("search_query", ""),
                        kb_recommendation=recommendation,
                        parent_uid=parent_uid,
                        upvotes=0, downvotes=0,
                        created_at=datetime.now(timezone.utc),
                    )
                    await session.execute(stmt)
                    await session.commit()
                    logger.debug("Phase 2 saved for %s", r.build_id[:8])
                except Exception as e:
                    logger.warning("Phase 2 error for %s: %s", r.build_id[:8], e)


async def run_custom_categorization(
    session_factory: async_sessionmaker,
    llm_base_url: str,
    llm_api_key: str,
    llm_models: List[str],
    categories: List[str],
    days_back: int = 90,
    llm_timeout: int = 60,
) -> int:
    """Classify failed builds into user-provided categories.

    Uses each build's existing summary/root_cause from gbd_meta so no full
    context collection is needed — just a short LLM classification call per build.
    Results are stored with source='llm_custom', replacing any prior custom run.
    """
    from gb_ui_backend.services.llm_client import LLMClient
    from sqlalchemy import delete as sa_delete

    since = datetime.now(timezone.utc) - timedelta(days=days_back)
    categories_str = ", ".join(categories)

    # Latest auto-analysis per build (for summary/root_cause)
    latest_phase1 = (
        select(GbdMeta.build_id, func.max(GbdMeta.created_at).label("latest"))
        .where(GbdMeta.source == "llm_phase1")
        .group_by(GbdMeta.build_id)
        .subquery()
    )

    async with session_factory() as session:
        result = await session.execute(
            select(GbdBuild, GbdMeta.summary, GbdMeta.root_cause)
            .outerjoin(latest_phase1, GbdBuild.id == latest_phase1.c.build_id)
            .outerjoin(
                GbdMeta,
                (GbdBuild.id == GbdMeta.build_id)
                & (GbdMeta.source == "llm_phase1")
                & (GbdMeta.created_at == latest_phase1.c.latest),
            )
            .where(GbdBuild.status == "failed", GbdBuild.created_at >= since)
            .order_by(GbdBuild.created_at.desc())
        )
        rows = result.all()

    if not rows:
        return 0

    client = LLMClient(
        base_url=llm_base_url, api_key=llm_api_key,
        models=llm_models, timeout=llm_timeout,
    )

    async def classify_one(build: GbdBuild, summary: Optional[str], root_cause: Optional[str]) -> tuple:
        if not summary and not root_cause:
            return build.id, "Uncategorized"
        prompt = (
            f"Classify this build failure into exactly one of these categories: {categories_str}\n\n"
            f"Summary: {summary or ''}\nRoot cause: {root_cause or ''}\n\n"
            f'Respond with JSON only: {{"category": "chosen category"}}'
        )
        try:
            raw = await client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=50,
            )
            content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
            clean = content.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1].lstrip("json").strip()
            cat = json.loads(clean).get("category", "Uncategorized")
            return build.id, cat if cat in categories else "Uncategorized"
        except Exception:
            return build.id, "Uncategorized"

    results = await asyncio.gather(
        *[classify_one(b, s, r) for b, s, r in rows],
        return_exceptions=True,
    )

    build_ids = [b.id for b, _, _ in rows]
    now = datetime.now(timezone.utc)
    saved = 0
    async with session_factory() as session:
        await session.execute(
            sa_delete(GbdMeta).where(
                GbdMeta.source == "llm_custom",
                GbdMeta.build_id.in_(build_ids),
            )
        )
        for res in results:
            if isinstance(res, Exception):
                continue
            build_id, category = res
            await session.execute(
                insert(GbdMeta).values(
                    update_id=uuid.uuid4(),
                    build_id=build_id,
                    source="llm_custom",
                    analysis_type="failure",
                    error_category_1=category,
                    created_at=now,
                    upvotes=0,
                    downvotes=0,
                )
            )
            saved += 1
        await session.commit()

    logger.info("Custom categorization complete: %d builds classified into %s", saved, categories)
    return saved


async def create_ai_daemon(  # noqa: PLR0913
    database_url: str,
    llm_base_url: str,
    llm_api_key: str,
    llm_models: List[str],
    llm_timeout: int = 60,
    cloud_logs_api_url: str = "",
    cloud_logs_api_key: str = "",
    gbserver_db_url: str = "",
    gbserver_db_schema: str = "public",
    poll_interval: int = 30,
) -> AIDaemon:
    """Create and return a ready-to-run AIDaemon."""
    engine = create_async_engine(database_url, echo=False, pool_size=5, max_overflow=10)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    collector = CompositeDataCollector(
        cloud_logs_api_url=cloud_logs_api_url,
        cloud_logs_api_key=cloud_logs_api_key,
        gbserver_db_url=gbserver_db_url,
        gbserver_db_schema=gbserver_db_schema,
    )
    global _daemon_instance
    _daemon_instance = AIDaemon(
        session_factory=session_factory,
        data_collector=collector,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        llm_models=llm_models,
        llm_timeout=llm_timeout,
        poll_interval=poll_interval,
    )
    return _daemon_instance
