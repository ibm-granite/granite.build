"""Infrastructure endpoints — leaderboard, queues, node pools."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from gb_ui_backend.config import Config, get_config
from gb_ui_backend.services.db_schema import (
    GbdBuild,
    GbdClusterQueue,
    GbdK8sResource,
    GbdNodeCapacity,
    get_db,
    get_optional_db,
)
from gb_ui_backend.services.gbserver_source import get_gbserver_source

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/analytics/infra")


def _parse_cpu_cores(s: Optional[str]) -> float:
    """Parse a Kubernetes CPU string ("500m", "2", "0.5") into fractional cores."""
    if not s:
        return 0.0
    s = s.strip()
    if s.endswith("m"):
        return float(s[:-1]) / 1000
    return float(s)


def _parse_memory_gib(s: Optional[str]) -> float:
    """Parse a Kubernetes memory string ("2Gi", "512Mi") into GiB."""
    if not s:
        return 0.0
    s = s.strip()
    for sfx, gib in [("Ti", 1024.0), ("Gi", 1.0), ("Mi", 1 / 1024), ("Ki", 1 / (1024 ** 2)),
                     ("T", 1000 ** 4 / 1024 ** 3), ("G", 1000 ** 3 / 1024 ** 3)]:
        if s.endswith(sfx):
            return float(s[:-len(sfx)]) * gib
    return float(s) / (1024 ** 3)


class LeaderboardEntry(BaseModel):
    username: str
    running_jobs: int = 0
    gpu_count: int = 0
    cpu_cores: float = 0.0
    memory_gib: float = 0.0
    total_builds: int = 0


class QueueCapacityOut(BaseModel):
    name: str
    cluster_name: str
    gpu_capacity: int
    gpu_used: int
    cpu_capacity_cores: float
    cpu_used_cores: float
    memory_capacity_gib: float
    memory_used_gib: float
    admitted_workloads: int
    pending_workloads: int
    reserving_workloads: int


class NodePoolOut(BaseModel):
    pool_name: str
    cluster_name: str
    node_count: int
    ready_nodes: int
    cpu_allocatable_cores: float
    cpu_requested_cores: float
    memory_allocatable_gib: float
    memory_requested_gib: float
    gpu_allocatable: int
    gpu_requested: int
    running_pods: int
    pending_pods: int
    autoscale_enabled: bool
    min_nodes: Optional[int] = None
    max_nodes: Optional[int] = None


@router.get("/leaderboard", response_model=list[LeaderboardEntry])
async def get_leaderboard(
    view: str = Query(default="running_jobs"),
    env_id: Optional[str] = Query(default=None),
    db: Optional[AsyncSession] = Depends(get_optional_db),
    config: Config = Depends(get_config),
):
    # When env_id is provided, always prefer the env-specific gbserver source.
    # When db is None (no GB_UI_DATABASE_URL), fall back to the default GbserverSource.
    if env_id or db is None:
        source = get_gbserver_source(env_id)
        if source:
            entries = await source.get_leaderboard(view=view)
            return [LeaderboardEntry(**e) for e in entries]
        if db is None:
            raise HTTPException(503, "No data source configured")

    if view == "running_jobs":
        stmt = (
            select(
                GbdBuild.username,
                func.count().label("running_jobs"),
                func.sum(GbdBuild.total_gpu).label("gpu_count"),
            )
            .where(GbdBuild.status == "running")
            .group_by(GbdBuild.username)
            .order_by(func.count().desc())
            .limit(10)
        )
    elif view == "total_builds":
        stmt = (
            select(GbdBuild.username, func.count().label("total_builds"))
            .group_by(GbdBuild.username)
            .order_by(func.count().desc())
            .limit(10)
        )
    else:
        # total_cpu / total_memory are stored as Kubernetes strings ("500m", "2Gi")
        # so we fetch raw values and aggregate in Python — all-time, no status filter.
        raw_stmt = select(GbdBuild.username, GbdBuild.total_cpu, GbdBuild.total_memory, GbdBuild.total_gpu)
        raw_result = await db.execute(raw_stmt)
        totals: dict[str, LeaderboardEntry] = {}
        for row in raw_result.all():
            e = totals.setdefault(row.username, LeaderboardEntry(username=row.username))
            e.cpu_cores += _parse_cpu_cores(row.total_cpu)
            e.memory_gib += _parse_memory_gib(row.total_memory)
            e.gpu_count += int(row.total_gpu or 0)
        if view == "cpu":
            return sorted(totals.values(), key=lambda e: -e.cpu_cores)[:10]
        if view == "memory":
            return sorted(totals.values(), key=lambda e: -e.memory_gib)[:10]
        return sorted(totals.values(), key=lambda e: -e.gpu_count)[:10]

    result = await db.execute(stmt)
    rows = result.all()

    entries = []
    for row in rows:
        e = LeaderboardEntry(username=row.username)
        if view == "running_jobs":
            e.running_jobs = row.running_jobs or 0
            e.gpu_count = int(row.gpu_count or 0)
        elif view == "total_builds":
            e.total_builds = row.total_builds or 0
        entries.append(e)
    return entries


@router.get("/queues", response_model=list[QueueCapacityOut])
async def get_queue_capacity(
    db: AsyncSession = Depends(get_db),
    config: Config = Depends(get_config),
):
    if not config.db_enabled:
        raise HTTPException(503, "Database not configured")

    result = await db.execute(select(GbdClusterQueue).order_by(GbdClusterQueue.name))
    queues = result.scalars().all()

    def milli_to_cores(milli: Optional[str]) -> float:
        if not milli:
            return 0.0
        if milli.endswith("m"):
            return float(milli[:-1]) / 1000
        return float(milli)

    def parse_memory_gib(mem: Optional[str]) -> float:
        if not mem:
            return 0.0
        if mem.endswith("Pi"):
            return float(mem[:-2]) * 1024 * 1024
        if mem.endswith("Ti"):
            return float(mem[:-2]) * 1024
        if mem.endswith("Gi"):
            return float(mem[:-2])
        if mem.endswith("Mi"):
            return float(mem[:-2]) / 1024
        if mem.endswith("Ki"):
            return float(mem[:-2]) / (1024 * 1024)
        return float(mem) / (1024 ** 3)

    return [
        QueueCapacityOut(
            name=q.name,
            cluster_name=q.cluster_name,
            gpu_capacity=q.capacity_gpu or 0,
            gpu_used=q.usage_gpu or 0,
            cpu_capacity_cores=milli_to_cores(q.capacity_cpu),
            cpu_used_cores=milli_to_cores(q.usage_cpu),
            memory_capacity_gib=parse_memory_gib(q.capacity_memory),
            memory_used_gib=parse_memory_gib(q.usage_memory),
            admitted_workloads=q.admitted_workloads or 0,
            pending_workloads=q.pending_workloads or 0,
            reserving_workloads=q.reserving_workloads or 0,
        )
        for q in queues
    ]


class K8sResourceOut(BaseModel):
    kind: str
    name: str
    namespace: Optional[str] = None
    cluster_name: Optional[str] = None
    status: Optional[str] = None
    build_status: Optional[str] = None
    failure_reason: Optional[str] = None
    failure_message: Optional[str] = None
    cpu: Optional[str] = None
    memory: Optional[str] = None
    gpu: Optional[int] = None
    storage: Optional[str] = None
    replicas: Optional[int] = None
    created_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None


@router.get("/nodes", response_model=list[NodePoolOut])
async def get_node_pools(
    db: AsyncSession = Depends(get_db),
    config: Config = Depends(get_config),
):
    if not config.db_enabled:
        raise HTTPException(503, "Database not configured")

    result = await db.execute(
        select(GbdNodeCapacity).order_by(GbdNodeCapacity.cluster_name, GbdNodeCapacity.pool_name)
    )
    pools = result.scalars().all()

    return [
        NodePoolOut(
            pool_name=p.pool_name,
            cluster_name=p.cluster_name,
            node_count=p.node_count or 0,
            ready_nodes=p.ready_nodes or 0,
            cpu_allocatable_cores=(p.allocatable_cpu_milli or 0) / 1000,
            cpu_requested_cores=(p.requested_cpu_milli or 0) / 1000,
            memory_allocatable_gib=(p.allocatable_memory_mb or 0) / 1024,
            memory_requested_gib=(p.requested_memory_mb or 0) / 1024,
            gpu_allocatable=p.allocatable_gpu or 0,
            gpu_requested=p.requested_gpu or 0,
            running_pods=p.running_pods or 0,
            pending_pods=p.pending_pods or 0,
            autoscale_enabled=p.autoscale_enabled or False,
            min_nodes=p.min_nodes,
            max_nodes=p.max_nodes,
        )
        for p in pools
    ]


class UserResourceDay(BaseModel):
    username: str
    date: str
    build_count: int = 0
    gpu_count: int = 0
    cpu_cores: float = 0.0
    memory_gib: float = 0.0


@router.get("/resource-usage", response_model=list[UserResourceDay])
async def get_resource_usage(
    days_back: int = Query(default=14),
    env_id: Optional[str] = Query(default=None),
    db: AsyncSession = Depends(get_db),
    config: Config = Depends(get_config),
):
    # When env_id is provided, pull from the env-specific gbserver source
    if env_id:
        source = get_gbserver_source(env_id)
        if source:
            builds = await source.list_builds(days_back=days_back)
            groups: dict[tuple[str, str], UserResourceDay] = {}
            for b in builds:
                date = str(b.get("created_time", ""))[:10]
                if not date:
                    continue
                key = (b.get("username", ""), date)
                if key not in groups:
                    groups[key] = UserResourceDay(username=b.get("username", ""), date=date)
                groups[key].build_count += 1
            return sorted(groups.values(), key=lambda x: (x.date, x.username))

    if not config.db_enabled:
        raise HTTPException(503, "Database not configured")

    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    stmt = (
        select(
            GbdBuild.username,
            func.date(GbdBuild.created_at).label("date"),
            GbdBuild.total_gpu,
            GbdBuild.total_cpu,
            GbdBuild.total_memory,
        )
        .where(GbdBuild.created_at >= cutoff)
        .order_by(func.date(GbdBuild.created_at), GbdBuild.username)
    )
    result = await db.execute(stmt)

    groups: dict[tuple[str, str], UserResourceDay] = {}
    for row in result.all():
        key = (row.username, str(row.date))
        if key not in groups:
            groups[key] = UserResourceDay(username=row.username, date=str(row.date))
        g = groups[key]
        g.build_count += 1
        g.gpu_count += int(row.total_gpu or 0)
        g.cpu_cores += _parse_cpu_cores(row.total_cpu)
        g.memory_gib += _parse_memory_gib(row.total_memory)

    return sorted(groups.values(), key=lambda x: (x.date, x.username))


class BuildResourcesOut(BaseModel):
    build_id: str
    cpu: Optional[str] = None
    memory: Optional[str] = None
    gpu: Optional[int] = None


@router.get("/builds/resources", response_model=list[BuildResourcesOut])
async def get_builds_resources(
    build_id: list[str] = Query(default=[]),
    db: AsyncSession = Depends(get_db),
    config: Config = Depends(get_config),
):
    """Return total resource allocations for a batch of build IDs from the sidecar DB."""
    if not config.db_enabled or not build_id:
        return []
    valid_uids: list[UUID] = []
    for bid in build_id:
        try:
            valid_uids.append(UUID(bid))
        except ValueError:
            pass
    if not valid_uids:
        return []
    result = await db.execute(
        select(GbdBuild.id, GbdBuild.total_cpu, GbdBuild.total_memory, GbdBuild.total_gpu)
        .where(GbdBuild.id.in_(valid_uids))
    )
    return [
        BuildResourcesOut(
            build_id=str(r.id),
            cpu=r.total_cpu,
            memory=r.total_memory,
            gpu=r.total_gpu,
        )
        for r in result.all()
    ]


@router.get("/builds/{build_id}/k8s-resources", response_model=list[K8sResourceOut])
async def get_build_k8s_resources(
    build_id: str,
    db: AsyncSession = Depends(get_db),
    config: Config = Depends(get_config),
):
    if not config.db_enabled:
        raise HTTPException(503, "Database not configured")
    try:
        uid = UUID(build_id)
    except ValueError:
        raise HTTPException(400, "Invalid build_id")
    result = await db.execute(
        select(GbdK8sResource)
        .where(GbdK8sResource.build_id == uid)
        .order_by(GbdK8sResource.kind, GbdK8sResource.name)
    )
    resources = result.scalars().all()
    return [
        K8sResourceOut(
            kind=r.kind,
            name=r.name,
            namespace=r.namespace,
            cluster_name=r.cluster_name,
            status=r.status,
            build_status=r.build_status,
            failure_reason=r.failure_reason,
            failure_message=r.failure_message,
            cpu=r.cpu,
            memory=r.memory,
            gpu=r.gpu,
            storage=r.storage,
            replicas=r.replicas,
            created_at=r.created_at,
            deleted_at=r.deleted_at,
        )
        for r in resources
    ]
