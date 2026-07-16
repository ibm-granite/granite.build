"""
SQLAlchemy models — ported from gb_dashboard's gbd_* tables.
IBM-specific fields (DMF, Tekton, Aspera) are omitted.
"""

from __future__ import annotations

import ssl
from datetime import datetime
from typing import AsyncGenerator, List, Optional
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from gb_ui_backend.config import get_config

_engine = None
_session_factory = None


def _build_connect_args(raw: dict) -> dict:
    """Translate the JSON-serializable connect args crossing the env-var boundary
    (see Config.database_connect_args_json) into what create_async_engine() expects.

    ssl.SSLContext isn't JSON-serializable, so gbserver's _configure_analytics_env
    only ever sends the cert file *path* under "sslrootcert_file" (mirroring the main
    SQL store's own TLS cert, translated for asyncpg) — built into a context here,
    right before the engine is created. Any other key is passed through unchanged.
    """
    connect_args = dict(raw)
    cert_file = connect_args.pop("sslrootcert_file", None)
    if cert_file:
        connect_args["ssl"] = ssl.create_default_context(cafile=cert_file)
    return connect_args


def _get_engine():
    global _engine
    if _engine is None:
        config = get_config()
        if not config.database_url:
            raise HTTPException(
                status_code=503,
                detail="Analytics database not configured. Set GB_UI_DATABASE_URL.",
            )
        is_sqlite = config.database_url.startswith("sqlite")
        connect_args = _build_connect_args(config.database_connect_args)
        _engine = create_async_engine(
            config.database_url,
            echo=False,
            # pool_pre_ping not supported by SQLite's StaticPool
            **({} if is_sqlite else {"pool_pre_ping": True}),
            **({"connect_args": connect_args} if connect_args else {}),
        )
    return _engine


def _get_session_factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(_get_engine(), expire_on_commit=False)
    return _session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with _get_session_factory()() as session:
        yield session


async def get_optional_db() -> AsyncGenerator[Optional[AsyncSession], None]:
    """Yields None when GB_UI_DATABASE_URL is not set; yields a session otherwise.

    Use this as the dependency for endpoints that have a GbserverSource fallback
    path so they can serve data even without an analytics database.
    """
    if not get_config().database_url:
        yield None
        return
    async with _get_session_factory()() as session:
        yield session


class Base(DeclarativeBase):
    pass


class GbdBuild(Base):
    __tablename__ = "gbd_builds"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    space_name: Mapped[str] = mapped_column(String(255))
    username: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(50))
    failure_reason: Mapped[Optional[str]] = mapped_column(String(255))
    failure_message: Mapped[Optional[str]] = mapped_column(Text)
    total_cpu: Mapped[Optional[str]] = mapped_column(String(50))
    total_memory: Mapped[Optional[str]] = mapped_column(String(50))
    total_gpu: Mapped[Optional[int]] = mapped_column(Integer)
    total_storage: Mapped[Optional[str]] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now()
    )
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    cluster_name: Mapped[Optional[str]] = mapped_column(String(255))
    source_uri: Mapped[Optional[str]] = mapped_column(Text)
    description: Mapped[Optional[str]] = mapped_column(Text)
    tags: Mapped[Optional[dict]] = mapped_column(JSON)

    __table_args__ = (
        Index("ix_gbd_builds_status", "status"),
        Index("ix_gbd_builds_username", "username"),
        Index("ix_gbd_builds_space_name", "space_name"),
        Index("ix_gbd_builds_created_at", "created_at"),
        Index("ix_gbd_builds_updated_at", "updated_at"),
        UniqueConstraint("name", "cluster_name", name="uq_gbd_builds_name_cluster"),
    )

    k8s_resources: Mapped[List["GbdK8sResource"]] = relationship(
        "GbdK8sResource", lazy="noload"
    )
    events: Mapped[List["GbdEvent"]] = relationship("GbdEvent", lazy="noload")


class GbdMeta(Base):
    """AI analysis results for a build."""

    __tablename__ = "gbd_meta"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    update_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), unique=True)
    build_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True))
    source: Mapped[Optional[str]] = mapped_column(String(50))
    analysis_type: Mapped[Optional[str]] = mapped_column(String(50))
    prompt_version: Mapped[Optional[str]] = mapped_column(String(20))
    model_name: Mapped[Optional[str]] = mapped_column(String(255))
    summary: Mapped[Optional[str]] = mapped_column(Text)
    root_cause: Mapped[Optional[str]] = mapped_column(Text)
    suggested_action: Mapped[Optional[str]] = mapped_column(Text)
    issues: Mapped[Optional[list]] = mapped_column(JSON)
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    raw_response: Mapped[Optional[dict]] = mapped_column(JSON)
    tokens_prompt: Mapped[Optional[int]] = mapped_column(Integer)
    tokens_completion: Mapped[Optional[int]] = mapped_column(Integer)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer)
    human_solution: Mapped[Optional[str]] = mapped_column(Text)
    feedback_rating: Mapped[Optional[int]] = mapped_column(Integer)
    feedback_helpful: Mapped[Optional[bool]] = mapped_column(Boolean)
    corrected_root_cause: Mapped[Optional[str]] = mapped_column(Text)
    feedback_comment: Mapped[Optional[str]] = mapped_column(Text)
    upvotes: Mapped[int] = mapped_column(Integer, default=0)
    downvotes: Mapped[int] = mapped_column(Integer, default=0)
    feedback_author: Mapped[Optional[str]] = mapped_column(String(255))
    extras: Mapped[Optional[dict]] = mapped_column(JSON)
    error_category_1: Mapped[Optional[str]] = mapped_column(String(255))
    error_category_2: Mapped[Optional[str]] = mapped_column(String(255))
    error_category_3: Mapped[Optional[str]] = mapped_column(String(255))
    error_category_4: Mapped[Optional[str]] = mapped_column(String(255))
    # Phase 2 knowledge base search
    kb_search_query: Mapped[Optional[str]] = mapped_column(Text)
    kb_recommendation: Mapped[Optional[str]] = mapped_column(Text)
    # Links phase 2/5 results to their parent phase 1 analysis
    parent_uid: Mapped[Optional[UUID]] = mapped_column(Uuid(as_uuid=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )

    __table_args__ = (
        Index("ix_gbd_meta_build_id", "build_id"),
        Index("ix_gbd_meta_source", "source"),
        Index("ix_gbd_meta_created_at", "created_at"),
        Index("ix_gbd_meta_error_cat_1", "error_category_1"),
        Index("ix_gbd_meta_parent_uid", "parent_uid"),
    )


class GbdClusterQueue(Base):
    """Kueue ClusterQueue capacity snapshot."""

    __tablename__ = "gbd_clusterqueues"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255))
    cluster_name: Mapped[str] = mapped_column(String(255))
    namespace: Mapped[Optional[str]] = mapped_column(String(255))
    capacity_cpu: Mapped[Optional[str]] = mapped_column(String(50))
    capacity_memory: Mapped[Optional[str]] = mapped_column(String(50))
    capacity_gpu: Mapped[Optional[int]] = mapped_column(Integer)
    usage_cpu: Mapped[Optional[str]] = mapped_column(String(50))
    usage_memory: Mapped[Optional[str]] = mapped_column(String(50))
    usage_gpu: Mapped[Optional[int]] = mapped_column(Integer)
    admitted_workloads: Mapped[Optional[int]] = mapped_column(Integer)
    pending_workloads: Mapped[Optional[int]] = mapped_column(Integer)
    reserving_workloads: Mapped[Optional[int]] = mapped_column(Integer)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("name", "cluster_name", name="uq_gbd_cq_name_cluster"),
    )


class GbdNodeCapacity(Base):
    """Node pool utilization snapshot."""

    __tablename__ = "gbd_node_capacity"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cluster_name: Mapped[str] = mapped_column(String(255))
    pool_name: Mapped[str] = mapped_column(String(255))
    node_count: Mapped[Optional[int]] = mapped_column(Integer)
    ready_nodes: Mapped[Optional[int]] = mapped_column(Integer)
    allocatable_cpu_milli: Mapped[Optional[int]] = mapped_column(BigInteger)
    allocatable_memory_mb: Mapped[Optional[int]] = mapped_column(BigInteger)
    allocatable_gpu: Mapped[Optional[int]] = mapped_column(Integer)
    requested_cpu_milli: Mapped[Optional[int]] = mapped_column(BigInteger)
    requested_memory_mb: Mapped[Optional[int]] = mapped_column(BigInteger)
    requested_gpu: Mapped[Optional[int]] = mapped_column(Integer)
    running_pods: Mapped[Optional[int]] = mapped_column(Integer)
    pending_pods: Mapped[Optional[int]] = mapped_column(Integer)
    autoscale_enabled: Mapped[Optional[bool]] = mapped_column(Boolean)
    min_nodes: Mapped[Optional[int]] = mapped_column(Integer)
    max_nodes: Mapped[Optional[int]] = mapped_column(Integer)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("cluster_name", "pool_name", name="uq_gbd_nc_cluster_pool"),
    )


class GbdK8sResource(Base):
    """AppWrapper, Pod, Workload, or HelmRelease associated with a build."""

    __tablename__ = "gbd_k8s_resources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    build_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("gbd_builds.id")
    )
    kind: Mapped[str] = mapped_column(String(50))
    name: Mapped[str] = mapped_column(String(255))
    namespace: Mapped[Optional[str]] = mapped_column(String(255))
    cluster_name: Mapped[Optional[str]] = mapped_column(String(255))
    uid: Mapped[Optional[str]] = mapped_column(String(255))
    status: Mapped[Optional[str]] = mapped_column(String(100))
    build_status: Mapped[Optional[str]] = mapped_column(String(50))
    failure_reason: Mapped[Optional[str]] = mapped_column(String(255))
    failure_message: Mapped[Optional[str]] = mapped_column(Text)
    cpu: Mapped[Optional[str]] = mapped_column(String(50))
    memory: Mapped[Optional[str]] = mapped_column(String(50))
    gpu: Mapped[Optional[int]] = mapped_column(Integer)
    storage: Mapped[Optional[str]] = mapped_column(String(50))
    replicas: Mapped[Optional[int]] = mapped_column(Integer, default=1)
    extra: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("ix_gbd_k8s_resources_build_id", "build_id"),
        Index("ix_gbd_k8s_resources_kind", "kind"),
        Index("ix_gbd_k8s_resources_status", "build_status"),
        UniqueConstraint(
            "kind",
            "name",
            "namespace",
            "cluster_name",
            name="uq_gbd_k8s_resources_kind_name_ns_cluster",
        ),
    )


class GbdEvent(Base):
    """Kubernetes event associated with a build."""

    __tablename__ = "gbd_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    build_id: Mapped[Optional[UUID]] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("gbd_builds.id")
    )
    event_uid: Mapped[Optional[str]] = mapped_column(String(255))
    cluster_name: Mapped[Optional[str]] = mapped_column(String(255))
    namespace: Mapped[Optional[str]] = mapped_column(String(255))
    type: Mapped[Optional[str]] = mapped_column(String(50))
    reason: Mapped[Optional[str]] = mapped_column(String(255))
    message: Mapped[Optional[str]] = mapped_column(Text)
    count: Mapped[Optional[int]] = mapped_column(Integer)
    object_kind: Mapped[Optional[str]] = mapped_column(String(100))
    object_name: Mapped[Optional[str]] = mapped_column(String(255))
    first_timestamp: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_timestamp: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    recorded_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("ix_gbd_events_build_id", "build_id"),
        UniqueConstraint("event_uid", "cluster_name", name="uq_gbd_events_uid_cluster"),
    )
