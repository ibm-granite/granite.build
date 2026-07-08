"""
Data collectors for AI analysis.

Ported from gb_dashboard/services/ai_data_collectors.py.
Cloud logs collection (e.g. IBM Cloud Logs) is optional — falls back to K8s pod logs stored in the DB.
GbserverDataCollector connects to gbserver's PostgreSQL schema if configured.
"""
from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import selectinload

from gb_ui_backend.services.db_schema import (
    GbdBuild,
    GbdClusterQueue,
    GbdEvent,
    GbdK8sResource,
    GbdMeta,
)
from gb_ui_backend.services.ai_prompts import BuildContext, KnowledgeBaseEntry

logger = logging.getLogger(__name__)


# ── Base ──────────────────────────────────────────────────────────────────────

class DataCollector(ABC):
    @abstractmethod
    async def collect(self, session: AsyncSession, build: GbdBuild, context: BuildContext) -> None:
        pass


# ── Core collectors ───────────────────────────────────────────────────────────

class BuildMetadataCollector(DataCollector):
    async def collect(self, session: AsyncSession, build: GbdBuild, context: BuildContext) -> None:
        context.build_id = str(build.id)
        context.build_name = build.name
        context.status = build.status
        context.failure_reason = build.failure_reason
        context.failure_message = build.failure_message
        context.total_cpu = build.total_cpu
        context.total_memory = build.total_memory
        context.total_gpu = build.total_gpu or 0
        if build.created_at:
            context.created_at = build.created_at.isoformat()
            end_time = build.finished_at or datetime.now(timezone.utc)
            duration = end_time - build.created_at
            h, rem = divmod(int(duration.total_seconds()), 3600)
            m, s = divmod(rem, 60)
            context.duration_str = f"{h}h {m}m" if h > 0 else (f"{m}m {s}s" if m > 0 else f"{s}s")
        if build.finished_at:
            context.finished_at = build.finished_at.isoformat()


class K8sResourceCollector(DataCollector):
    async def collect(self, session: AsyncSession, build: GbdBuild, context: BuildContext) -> None:
        stmt = (select(GbdK8sResource)
                .where(GbdK8sResource.build_id == build.id)
                .order_by(GbdK8sResource.kind, GbdK8sResource.name))
        result = await session.execute(stmt)
        context.k8s_resources = [
            {
                "kind": r.kind, "name": r.name, "namespace": r.namespace,
                "status": r.status, "build_status": r.build_status,
                "failure_reason": r.failure_reason, "failure_message": r.failure_message,
                "cpu": r.cpu, "memory": r.memory, "gpu": r.gpu,
                "deleted_at": r.deleted_at.isoformat() if r.deleted_at else None,
                "extra": r.extra,
            }
            for r in result.scalars().all()
        ]


class EventCollector(DataCollector):
    def __init__(self, max_events: int = 20):
        self.max_events = max_events

    async def collect(self, session: AsyncSession, build: GbdBuild, context: BuildContext) -> None:
        stmt = (select(GbdEvent)
                .where(GbdEvent.build_id == build.id)
                .order_by(GbdEvent.type.desc(), GbdEvent.last_timestamp.desc().nullslast())
                .limit(self.max_events))
        result = await session.execute(stmt)
        context.events = [
            {
                "type": e.type, "reason": e.reason, "message": e.message,
                "count": e.count, "object_kind": e.object_kind, "object_name": e.object_name,
                "last_timestamp": e.last_timestamp.isoformat() if e.last_timestamp else None,
            }
            for e in result.scalars().all()
        ]


class ClusterQueueCollector(DataCollector):
    async def collect(self, session: AsyncSession, build: GbdBuild, context: BuildContext) -> None:
        stmt = (select(GbdClusterQueue)
                .where(GbdClusterQueue.cluster_name == build.cluster_name,
                       GbdClusterQueue.namespace == build.space_name)
                .limit(1))
        result = await session.execute(stmt)
        cq = result.scalar_one_or_none()
        if cq:
            context.clusterqueue_capacity_gpu = cq.capacity_gpu or 0
            context.clusterqueue_usage_gpu = cq.usage_gpu or 0
            context.clusterqueue_pending_workloads = cq.pending_workloads or 0


class PodLogCollector(DataCollector):
    """Collects pod logs stored in K8s resource extra data (from sync daemon)."""
    def __init__(self, max_lines: int = 50):
        self.max_lines = max_lines

    async def collect(self, session: AsyncSession, build: GbdBuild, context: BuildContext) -> None:
        stmt = (select(GbdK8sResource)
                .where(GbdK8sResource.build_id == build.id, GbdK8sResource.kind == "Pod"))
        result = await session.execute(stmt)
        log_lines = []
        for pod in result.scalars().all():
            extra = pod.extra or {}
            logs = extra.get("logs", {})
            if isinstance(logs, dict):
                for container_name, container_logs in logs.items():
                    if container_logs:
                        if isinstance(container_logs, list):
                            lines = container_logs[-self.max_lines:]
                        else:
                            lines = str(container_logs).split("\n")[-self.max_lines:]
                        log_lines.append(f"=== {pod.name}/{container_name} ===")
                        log_lines.extend(lines)
            elif isinstance(logs, str) and logs:
                lines = logs.split("\n")[-self.max_lines:]
                log_lines.append(f"=== {pod.name} ===")
                log_lines.extend(lines)
        if log_lines:
            context.pod_logs = "\n".join(log_lines)


# ── Optional: gbserver data collector ────────────────────────────────────────

class GbserverClient:
    """Reads build events, targets, and steps from gbserver's PostgreSQL schema."""

    def __init__(self, db_url: str, schema: str = "public"):
        self.db_url = db_url
        self.schema = schema
        self._engine = None
        self._session_factory = None

    async def initialize(self) -> None:
        self._engine = create_async_engine(
            self.db_url, echo=False, pool_size=2, max_overflow=3, pool_pre_ping=True)
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

    async def close(self) -> None:
        if self._engine:
            await self._engine.dispose()

    async def get_build_data(self, build_id: str, max_events: int = 5000
                              ) -> Tuple[List[Dict], List[Dict]]:
        if not self._session_factory:
            return [], []
        async with self._session_factory() as session:
            result = await session.execute(
                text(f"""
                    SELECT 'event' as source_table, e.uuid as row_id, NULL as name,
                           e.type as type_or_status, e.json as json_data, e.created_time,
                           e.index, e.source, e.target_id, e.step_id
                    FROM {self.schema}.gb_events e WHERE e.build_id = :build_id
                    UNION ALL
                    SELECT 'target', t.uuid, t.name, t.status, t.json, NULL, NULL, NULL, NULL, NULL
                    FROM {self.schema}.gb_targets t WHERE t.build_id = :build_id
                    UNION ALL
                    SELECT 'step', s.uuid, NULL, s.status, s.json, NULL, NULL, NULL, NULL, NULL
                    FROM {self.schema}.gb_steps s WHERE s.build_id = :build_id
                """),
                {"build_id": build_id}
            )
            rows = result.fetchall()
            events, status_msgs = [], []
            for row in rows:
                try:
                    payload = json.loads(row.json_data) if row.json_data else {}
                except json.JSONDecodeError:
                    payload = {}
                if row.source_table == "event":
                    ev = payload.get("build_event", payload)
                    ep = ev.get("payload", {})
                    event: Dict[str, Any] = {
                        "index": row.index or 0, "type": row.type_or_status,
                        "source": row.source,
                        "created_time": row.created_time.isoformat() if row.created_time else None,
                    }
                    if row.type_or_status == "MESSAGE_EVENT":
                        event["level"] = ep.get("level", "INFO")
                        event["message"] = ep.get("msg", "")[:10240]
                    elif row.type_or_status == "STATUS_EVENT":
                        event["status"] = ep.get("status", "")
                        event["message"] = ep.get("msg", "")[:10240]
                    elif row.type_or_status == "WORKLOAD_STATUS_EVENT":
                        event["appwrapper_state"] = ep.get("state", "")
                        event["failed_pods"] = ep.get("failed_pods", {})
                    events.append(event)
                elif row.source_table in ("target", "step"):
                    sm = payload.get("status_msg", "")
                    if sm or row.type_or_status in ("FAILED", "ERROR", "INVALID"):
                        status_msgs.append({
                            "entity": row.source_table,
                            "name": row.name or payload.get("definition_uri", "unknown"),
                            "status": row.type_or_status,
                            "status_msg": sm[:10240] if sm else "",
                        })

            def event_priority(e: Dict) -> tuple:
                if e["type"] == "MESSAGE_EVENT" and e.get("level") == "ERROR":
                    return (0, -e["index"])
                if e["type"] == "STATUS_EVENT" and e.get("status") in ("FAILED", "ERROR"):
                    return (1, -e["index"])
                return (4, -e["index"])
            events.sort(key=event_priority)
            return events[:max_events], status_msgs


class GbserverDataCollector(DataCollector):
    """Collects build events and step status from gbserver's database.

    Uses GbserverSource (which supports both SQLite and PostgreSQL) when
    GB_UI_GBSERVER_DB_URL is set. This is the primary data source for AI
    analysis in standalone mode.
    """
    def __init__(self, db_url: str = "", schema: str = "public"):
        self.db_url = db_url
        self.schema = schema
        self._client: Optional[GbserverClient] = None
        self._initialized = False
        self._init_failed = False

    async def _ensure_initialized(self) -> bool:
        if self._initialized:
            return self._client is not None
        if self._init_failed or not self.db_url:
            self._init_failed = True
            return False
        # For SQLite URLs, use GbserverSource directly (it handles aiosqlite)
        if "sqlite" in self.db_url:
            self._initialized = True
            return True  # will use GbserverSource in collect()
        try:
            self._client = GbserverClient(self.db_url, self.schema)
            await self._client.initialize()
            self._initialized = True
            return True
        except Exception as e:
            logger.warning("Failed to init gbserver client: %s", e)
            self._init_failed = True
            return False

    async def collect(self, session: AsyncSession, build: GbdBuild, context: BuildContext) -> None:
        if not await self._ensure_initialized():
            return
        build_id = str(build.id)
        try:
            # SQLite path: use GbserverSource
            if "sqlite" in (self.db_url or ""):
                from gb_ui_backend.services.gbserver_source import GbserverSource, _resolve_db_url
                import os
                url = _resolve_db_url(self.db_url)
                source = GbserverSource(url)
                try:
                    events, status_msgs = await source.get_build_events(build_id)
                finally:
                    await source.close()
            else:
                events, status_msgs = await self._client.get_build_data(build_id)
            context.gbserver_events = events
            context.gbserver_status_msgs = status_msgs
        except Exception as e:
            logger.warning("Failed to collect gbserver data for %s: %s", build.name, e)


# ── Optional: cloud logs collector ────────────────────────────────────────────

class CloudLogsCollector(DataCollector):
    """Fetches logs from a cloud logging service (e.g. IBM Cloud Logs).

    Falls back to K8s pod logs stored in the database when unavailable.
    The cloud logs client is loaded lazily via gb_ui_backend.services.cloud_logs
    if that module exists; otherwise falls back immediately.
    """
    def __init__(self, api_url: str = "", api_key: str = "", max_lines: int = 100):
        self.api_url = api_url
        self.api_key = api_key
        self.max_lines = max_lines

    def _collect_k8s_fallback(self, context: BuildContext) -> Optional[str]:
        lines = []
        for res in context.k8s_resources:
            if res.get("kind") == "Pod":
                logs = (res.get("extra") or {}).get("logs", "")
                if logs and isinstance(logs, str):
                    lines.append(f"=== Pod: {res.get('name','unknown')} (K8s) ===")
                    lines.extend(logs.split("\n")[-self.max_lines:])
        return "\n".join(lines) if lines else None

    async def collect(self, session: AsyncSession, build: GbdBuild, context: BuildContext) -> None:
        if not self.api_url or not self.api_key:
            k8s_logs = self._collect_k8s_fallback(context)
            if k8s_logs:
                context.pod_logs = k8s_logs
            return
        try:
            from gb_ui_backend.services.cloud_logs import get_cloud_logs_client
            client = get_cloud_logs_client(self.api_url, self.api_key)
            build_id_str = str(build.id)
            # Collect step IDs from AppWrapper extra fields
            steps: List[Tuple[str, str]] = []
            seen: Set[str] = set()
            for res in context.k8s_resources:
                if res.get("kind") == "AppWrapper":
                    extra = res.get("extra") or {}
                    step_id = extra.get("build_step_id", "")
                    source_uri = extra.get("source_uri", "")
                    if step_id and step_id not in seen:
                        steps.append((step_id, source_uri or res.get("name", "main")))
                        seen.add(step_id)
            if steps:
                tasks = [self._fetch_step_logs(client, build_id_str, sid, sname)
                         for sid, sname in steps]
                results = await asyncio.gather(*tasks)
                context.step_logs = {name: content for name, content in results if content}
                if not context.step_logs:
                    k8s_logs = self._collect_k8s_fallback(context)
                    if k8s_logs:
                        context.pod_logs = k8s_logs
            else:
                response = await client.query_logs(build_id=build_id_str, page_size=self.max_lines * 2)
                log_lines = client.parse_logs(response)
                if log_lines:
                    context.pod_logs = "\n".join(log_lines[-self.max_lines:])
                else:
                    k8s_logs = self._collect_k8s_fallback(context)
                    if k8s_logs:
                        context.pod_logs = k8s_logs
        except ImportError:
            k8s_logs = self._collect_k8s_fallback(context)
            if k8s_logs:
                context.pod_logs = k8s_logs
        except Exception as e:
            logger.warning("Cloud logs collection failed for %s: %s", build.name, e)
            k8s_logs = self._collect_k8s_fallback(context)
            if k8s_logs:
                context.pod_logs = k8s_logs

    async def _fetch_step_logs(self, client: Any, build_id: str,
                                step_id: str, step_name: str) -> Tuple[str, str]:
        try:
            response = await client.query_logs(build_id=build_id, step_id=step_id,
                                               page_size=self.max_lines * 2)
            lines = client.parse_logs(response)
            if lines:
                return (step_name, "\n".join(lines[-self.max_lines:]))
        except Exception as e:
            logger.warning("Failed to fetch logs for step %s: %s", step_name, e)
        return (step_name, "")


# ── Composite collector ────────────────────────────────────────────────────────

class CompositeDataCollector:
    """Orchestrates all collectors to build a complete BuildContext."""

    def __init__(
        self,
        cloud_logs_api_url: str = "",
        cloud_logs_api_key: str = "",
        log_lines: int = 100,
        gbserver_db_url: str = "",
        gbserver_db_schema: str = "public",
        collectors: Optional[List[DataCollector]] = None,
    ):
        if collectors is not None:
            self.collectors = collectors
        else:
            self.collectors: List[DataCollector] = [
                BuildMetadataCollector(),
                K8sResourceCollector(),
                EventCollector(max_events=20),
                ClusterQueueCollector(),
            ]
            if gbserver_db_url:
                self.collectors.append(
                    GbserverDataCollector(db_url=gbserver_db_url, schema=gbserver_db_schema))
            if cloud_logs_api_url and cloud_logs_api_key:
                self.collectors.append(
                    CloudLogsCollector(api_url=cloud_logs_api_url, api_key=cloud_logs_api_key,
                                       max_lines=log_lines))
            else:
                self.collectors.append(PodLogCollector(max_lines=log_lines))

    async def collect_build_context(self, session: AsyncSession, build: GbdBuild) -> BuildContext:
        context = BuildContext(build_id=str(build.id), build_name=build.name, status=build.status)
        for collector in self.collectors:
            try:
                await collector.collect(session, build, context)
            except Exception as e:
                logger.warning("Collector %s failed for %s: %s",
                               collector.__class__.__name__, build.name, e)
        return context


async def get_build_with_data(session: AsyncSession, build_id: uuid.UUID) -> Optional[GbdBuild]:
    stmt = (select(GbdBuild)
            .options(selectinload(GbdBuild.k8s_resources), selectinload(GbdBuild.events))
            .where(GbdBuild.id == build_id))
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def query_knowledge_base(
    session: AsyncSession,
    error_category_1: Optional[str] = None,
    error_category_2: Optional[str] = None,
    max_entries: int = 20,
    exclude_build_id: Optional[uuid.UUID] = None,
) -> List[KnowledgeBaseEntry]:
    from sqlalchemy import desc, or_, and_, case, nullslast

    stmt = (select(GbdMeta)
            .where(or_(GbdMeta.human_solution.isnot(None),
                       GbdMeta.feedback_helpful == True,
                       GbdMeta.upvotes > 0)))
    if exclude_build_id:
        stmt = stmt.where(GbdMeta.build_id != exclude_build_id)

    case_conditions = []
    if error_category_1 and error_category_2:
        case_conditions.append((and_(
            GbdMeta.error_category_1 == error_category_1,
            GbdMeta.error_category_2 == error_category_2), 100))
    if error_category_1:
        case_conditions.append((GbdMeta.error_category_1 == error_category_1, 50))
    score = case(*case_conditions, else_=0) if case_conditions else 0

    stmt = stmt.order_by(
        desc(case((GbdMeta.human_solution.isnot(None), 1), else_=0)),
        desc(score),
        desc(GbdMeta.upvotes),
        nullslast(desc(GbdMeta.feedback_rating)),
        desc(GbdMeta.created_at),
    ).limit(max_entries)

    result = await session.execute(stmt)
    return [
        KnowledgeBaseEntry(
            meta_id=m.id, build_id=str(m.build_id), source=m.source or "llm_phase1",
            error_category_1=m.error_category_1, error_category_2=m.error_category_2,
            root_cause=m.root_cause, summary=m.summary, suggested_action=m.suggested_action,
            human_solution=m.human_solution, feedback_rating=m.feedback_rating,
            feedback_helpful=m.feedback_helpful, upvotes=m.upvotes or 0,
            created_at=m.created_at.isoformat() if m.created_at else None,
        )
        for m in result.scalars().all()
    ]
