# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Async engine and session factory."""

from __future__ import annotations

import ssl
from functools import lru_cache
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from autotunex.core.config import Settings, get_settings
from autotunex.db.base import Base


def build_ssl_context(ca_path: str | None, mode: str | None = None) -> ssl.SSLContext | None:
    """Return a TLS context for the database according to ``mode``, or ``None``.

    Managed MySQL — IBM Cloud Databases for MySQL in particular — accepts *only*
    TLS connections and rejects plaintext auth outright (a bare ``asyncmy``
    connection fails with ``(1045, "Access denied ...")`` before it ever reaches
    the schema). There are two ways to satisfy that:

    - ``"verify"`` — the provider signs its server certificate with a private CA,
      so the stock trust store cannot verify it; the operator supplies that CA via
      ``database_ssl_ca`` and this builds a context that checks the server against
      it, with ``check_hostname`` on and ``verify_mode`` at ``CERT_REQUIRED``
      (verifying the cert but not the hostname would still let a holder of *any*
      cert that CA signed impersonate the server). IBM Cloud's CA marks its
      ``basicConstraints`` extension ``CA:TRUE`` but **not critical**, which
      RFC 5280 §4.2.1.9 requires; OpenSSL 3.x enables ``VERIFY_X509_STRICT`` in
      the default context and rejects the cert with "Basic Constraints of CA cert
      not marked critical". Clearing that one flag accepts the provider's CA while
      leaving chain and hostname verification fully intact.
    - ``"require"`` — encrypt without authenticating the server: no CA needed.
      This is the "connect without a certificate" path. It stops a passive
      eavesdropper but not an active man-in-the-middle; use it for dev, not prod.

    ``mode`` of ``None`` derives from ``ca_path`` (``"verify"`` when a CA is set,
    ``"disable"`` otherwise), preserving the historical behaviour where a lone CA
    path meant verified TLS. ``"disable"`` returns ``None`` so local
    SQLite/Postgres/MySQL send no TLS parameters at all.
    """
    effective = mode if mode is not None else ("verify" if ca_path else "disable")

    if effective == "disable":
        return None
    if effective == "require":
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context
    if effective == "verify":
        if ca_path is None:
            raise ValueError("database_ssl_mode='verify' requires database_ssl_ca to be set")
        context = ssl.create_default_context(cafile=ca_path)
        context.verify_flags &= ~ssl.VERIFY_X509_STRICT
        return context
    raise ValueError(f"unknown database_ssl_mode: {mode!r}")


def build_connect_args(
    database_url: str, ssl_context: ssl.SSLContext | None = None
) -> dict[str, Any]:
    """Return driver-specific connection arguments for ``database_url``.

    MySQL converts ``TIMESTAMP`` values into the connection's session timezone on
    read, so without pinning it every timestamp would come back shifted by
    whatever the server happens to be set to. This is what
    ``resources/autotunex_schema.sql``'s ``SET GLOBAL time_zone = '+00:00'`` was
    working around — done per-connection instead, it needs no
    ``SYSTEM_VARIABLES_ADMIN`` and cannot affect other databases on the server.

    Sent as ``init_command`` rather than a ``time_zone`` keyword: ``asyncmy`` has no
    such parameter and raises ``TypeError: connect() got an unexpected keyword
    argument 'time_zone'`` if given one. (PyMySQL and aiomysql do accept it, which
    is where the mistaken form comes from.) ``init_command`` runs once per new
    connection, before any query.

    ``ssl_context``, when given, is attached as ``asyncmy``'s ``ssl`` keyword so
    the connection negotiates TLS (see :func:`build_ssl_context`). It is applied
    only to MySQL URLs on purpose: ``asyncpg`` and ``aiosqlite`` take TLS through
    different parameters, so an ``ssl`` keyword would either be ignored or
    rejected there.

    SQLite and Postgres need nothing: ``UtcDateTime`` already normalizes both.
    """
    if database_url.startswith("mysql"):
        args: dict[str, Any] = {"init_command": "SET time_zone = '+00:00'"}
        if ssl_context is not None:
            args["ssl"] = ssl_context
        return args
    return {}


def build_pool_kwargs(database_url: str, settings: Settings) -> dict[str, Any]:
    """Return connection-pool arguments for :func:`create_async_engine`.

    ``pool_pre_ping`` liveness-checks a connection on checkout and is understood by
    every pool implementation, so it is always passed. The remaining knobs tune
    SQLAlchemy's ``QueuePool`` (the async ``AsyncAdaptedQueuePool``) and are passed
    **only** for server databases (MySQL/PostgreSQL): SQLite's async engine uses a
    non-queue pool (``StaticPool``/``NullPool``) that raises ``TypeError`` at engine
    construction when handed ``pool_size``, ``max_overflow``, ``pool_timeout``,
    ``pool_recycle`` or ``pool_use_lifo``. The dev default and the entire test suite
    run on SQLite, so gating those is what keeps SQLite's behaviour byte-for-byte what
    it was — pre-ping only.

    Why the server-only knobs matter against managed MySQL (IBM Cloud Databases),
    where every *new* connection pays a full TLS + ``caching_sha2_password`` handshake
    worth seconds:

    - ``pool_recycle`` refreshes a connection before the server's ``wait_timeout``
      (IBM Cloud defaults to 3600s) silently closes it, so a request is not handed a
      dead connection nor forced into a surprise full reconnect mid-request.
    - ``pool_size`` keeps that many connections *warm* and reused. It must cover the
      request handlers **and** the background reconcile loop
      (``job_reconcile_concurrency``, ``job_backend="llmb"``), which draws from this
      same pool — undersize it and a sweep pushes request handlers onto cold,
      expensive ``max_overflow`` connections.
    - ``pool_timeout`` bounds how long a request waits for a slot when the pool is
      saturated, failing loudly instead of hanging.
    - ``pool_use_lifo`` keeps a small set of connections hot under bursty traffic,
      cutting how often a request lands on a cold or server-closed one.
    """
    kwargs: dict[str, Any] = {"pool_pre_ping": settings.database_pool_pre_ping}
    if not database_url.startswith("sqlite"):
        kwargs["pool_size"] = settings.database_pool_size
        kwargs["max_overflow"] = settings.database_max_overflow
        kwargs["pool_timeout"] = settings.database_pool_timeout_seconds
        kwargs["pool_recycle"] = settings.database_pool_recycle_seconds
        kwargs["pool_use_lifo"] = settings.database_pool_use_lifo
    return kwargs


@lru_cache
def get_engine() -> AsyncEngine:
    """Return the process-wide async engine."""
    settings = get_settings()
    ssl_context = build_ssl_context(settings.database_ssl_ca, settings.database_ssl_mode)
    return create_async_engine(
        settings.database_url,
        echo=settings.database_echo,
        connect_args=build_connect_args(settings.database_url, ssl_context),
        **build_pool_kwargs(settings.database_url, settings),
    )


@lru_cache
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the process-wide session factory."""
    return async_sessionmaker(
        bind=get_engine(),
        expire_on_commit=False,
        autoflush=False,
    )


async def create_schema(engine: AsyncEngine) -> None:
    """Create every table that does not yet exist.

    Development and test convenience only — production schema changes go
    through Alembic (``make migrate``).
    """
    from autotunex.db import tables  # noqa: F401  (registers tables on Base.metadata)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
