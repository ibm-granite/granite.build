"""Telemetry database client — records MCP tool call events to PostgreSQL.

Uses the same GBD_DB_* environment variables as gbd_build_source.py.
Tables: gbmcp_tool_call_events, gbmcp_telemetry_sessions.
"""

import json
import os
from typing import Optional

from fastmcp.utilities.logging import get_logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

logger = get_logger(__name__)


class TelemetryDB:
    """Write-only telemetry client for recording tool call events."""

    def __init__(self, db_url: str):
        self.db_url = db_url
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker] = None

    async def initialize(self) -> None:
        """Create engine and ensure schema exists."""
        self._engine = create_async_engine(
            self.db_url,
            echo=False,
            pool_size=2,
            max_overflow=3,
            pool_pre_ping=True,
            pool_timeout=10,
        )
        self._session_factory = async_sessionmaker(
            self._engine,
            expire_on_commit=False,
        )
        await self._ensure_schema()
        logger.info("Telemetry DB initialized")

    async def close(self) -> None:
        """Dispose connection pool."""
        if self._engine:
            await self._engine.dispose()
            logger.info("Telemetry DB connection closed")

    async def _ensure_schema(self) -> None:
        """Create telemetry tables and indexes if they don't exist."""
        async with self._session_factory() as session:
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS gbmcp_tool_call_events (
                    id                        UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
                    created_at                TIMESTAMPTZ NOT NULL DEFAULT now(),
                    server_version            TEXT,

                    session_id                TEXT,
                    mcp_request_id            TEXT,
                    mcp_client_id             TEXT,

                    github_username           TEXT,
                    github_user_id            TEXT,
                    github_email              TEXT,
                    github_name               TEXT,

                    tool_name                 TEXT    NOT NULL,
                    tool_category             TEXT,
                    tool_module               TEXT,

                    started_at                TIMESTAMPTZ NOT NULL,
                    ended_at                  TIMESTAMPTZ,
                    duration_ms               INTEGER,

                    success                   BOOLEAN,
                    error_message             TEXT,
                    error_type                TEXT,

                    input_tokens              INTEGER,
                    output_tokens             INTEGER,
                    total_tokens              INTEGER,

                    argument_keys             TEXT[],
                    argument_input_lengths    INTEGER[],
                    argument_count            INTEGER,
                    result_length             INTEGER,
                    result_truncated          BOOLEAN DEFAULT FALSE,

                    call_sequence_in_session  INTEGER,
                    previous_tool_name        TEXT,

                    server_instance_id        TEXT,
                    gb_environment            TEXT,
                    tool_version              TEXT,
                    tool_git_commit           TEXT,
                    mcp_server_commit         TEXT,
                    mcp_image                 TEXT,

                    metadata                  JSONB DEFAULT '{}'::jsonb
                )
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_gbmcp_tce_created_at
                    ON gbmcp_tool_call_events (created_at DESC)
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_gbmcp_tce_session_id
                    ON gbmcp_tool_call_events (session_id)
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_gbmcp_tce_github_username
                    ON gbmcp_tool_call_events (github_username)
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_gbmcp_tce_tool_name
                    ON gbmcp_tool_call_events (tool_name)
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_gbmcp_tce_success
                    ON gbmcp_tool_call_events (success)
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_gbmcp_tce_gb_environment
                    ON gbmcp_tool_call_events (gb_environment)
            """))

            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS gbmcp_telemetry_sessions (
                    session_id              TEXT        PRIMARY KEY,
                    first_seen_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
                    last_seen_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
                    github_username         TEXT,
                    github_user_id          TEXT,
                    gb_environment          TEXT,
                    server_instance_id      TEXT,
                    total_calls             INTEGER     NOT NULL DEFAULT 0,
                    successful_calls        INTEGER     NOT NULL DEFAULT 0,
                    failed_calls            INTEGER     NOT NULL DEFAULT 0,
                    distinct_tools          TEXT[],
                    metadata                JSONB DEFAULT '{}'::jsonb
                )
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_gbmcp_ts_github_username
                    ON gbmcp_telemetry_sessions (github_username)
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_gbmcp_ts_last_seen_at
                    ON gbmcp_telemetry_sessions (last_seen_at DESC)
            """))
            await session.commit()

    async def insert_tool_call_event(self, event: dict) -> None:
        """Insert a single tool call event row with atomic trajectory computation.

        call_sequence_in_session and previous_tool_name are computed by the SQL
        using a per-session advisory lock to prevent race conditions under
        concurrent tool calls.
        """
        if not self._session_factory:
            return

        # Trajectory columns are computed by SQL; exclude from params
        params = {
            k: v
            for k, v in event.items()
            if k not in ("call_sequence_in_session", "previous_tool_name")
        }
        params["metadata"] = json.dumps(event.get("metadata") or {})

        async with self._session_factory() as session:
            await session.execute(
                text("""
                    INSERT INTO gbmcp_tool_call_events (
                        server_version,
                        session_id, mcp_request_id, mcp_client_id,
                        github_username, github_user_id, github_email, github_name,
                        tool_name, tool_category, tool_module,
                        started_at, ended_at, duration_ms,
                        success, error_message, error_type,
                        input_tokens, output_tokens, total_tokens,
                        argument_keys, argument_input_lengths, argument_count,
                        result_length, result_truncated,
                        call_sequence_in_session, previous_tool_name,
                        server_instance_id, gb_environment,
                        tool_version, tool_git_commit, mcp_server_commit, mcp_image,
                        metadata
                    )
                    SELECT
                        :server_version,
                        :session_id, :mcp_request_id, :mcp_client_id,
                        :github_username, :github_user_id, :github_email, :github_name,
                        :tool_name, :tool_category, :tool_module,
                        :started_at, :ended_at, :duration_ms,
                        :success, :error_message, :error_type,
                        :input_tokens, :output_tokens, :total_tokens,
                        :argument_keys, :argument_input_lengths, :argument_count,
                        :result_length, :result_truncated,
                        COALESCE(prev.last_seq, 0) + 1,
                        prev.last_tool,
                        :server_instance_id, :gb_environment,
                        :tool_version, :tool_git_commit, :mcp_server_commit, :mcp_image,
                        CAST(:metadata AS jsonb)
                    FROM (SELECT pg_advisory_xact_lock(hashtext(:session_id))) AS _lock
                    LEFT JOIN LATERAL (
                        SELECT call_sequence_in_session AS last_seq,
                               tool_name                AS last_tool
                        FROM gbmcp_tool_call_events
                        WHERE session_id = :session_id
                        ORDER BY call_sequence_in_session DESC NULLS LAST
                        LIMIT 1
                    ) prev ON TRUE
                """),
                params,
            )
            await session.commit()

    async def upsert_session(self, session_data: dict) -> None:
        """Upsert a session row, incrementing call counters."""
        if not self._session_factory:
            return

        async with self._session_factory() as session:
            await session.execute(
                text("""
                    INSERT INTO gbmcp_telemetry_sessions (
                        session_id, first_seen_at, last_seen_at,
                        github_username, github_user_id, gb_environment,
                        server_instance_id,
                        total_calls, successful_calls, failed_calls,
                        distinct_tools
                    ) VALUES (
                        :session_id, now(), now(),
                        :github_username, :github_user_id, :gb_environment,
                        :server_instance_id,
                        1,
                        CASE WHEN :success THEN 1 ELSE 0 END,
                        CASE WHEN :success THEN 0 ELSE 1 END,
                        ARRAY[:tool_name]::TEXT[]
                    )
                    ON CONFLICT (session_id) DO UPDATE SET
                        last_seen_at     = now(),
                        total_calls      = gbmcp_telemetry_sessions.total_calls + 1,
                        successful_calls = gbmcp_telemetry_sessions.successful_calls
                                          + CASE WHEN :success THEN 1 ELSE 0 END,
                        failed_calls     = gbmcp_telemetry_sessions.failed_calls
                                          + CASE WHEN :success THEN 0 ELSE 1 END,
                        distinct_tools   = CASE
                            WHEN :tool_name = ANY(COALESCE(gbmcp_telemetry_sessions.distinct_tools, ARRAY[]::TEXT[]))
                            THEN gbmcp_telemetry_sessions.distinct_tools
                            ELSE array_append(COALESCE(gbmcp_telemetry_sessions.distinct_tools, ARRAY[]::TEXT[]), :tool_name)
                        END
                """),
                session_data,
            )
            await session.commit()


# ---------------------------------------------------------------------------
# Singleton lifecycle
# ---------------------------------------------------------------------------

_telemetry_db: Optional[TelemetryDB] = None


def get_telemetry_db() -> Optional[TelemetryDB]:
    """Return the global TelemetryDB instance, or None if not initialised."""
    return _telemetry_db


def _build_telemetry_url() -> str:
    """Construct asyncpg DB URL from GBD_DB_* environment variables."""
    host = os.environ.get("GBD_DB_HOST", "")
    port = os.environ.get("GBD_DB_PORT", "")
    user = os.environ.get("GBD_DB_USER", "")
    passwd = os.environ.get("GBD_DB_PASSWD", "")
    name = os.environ.get("GBD_DB_NAME", "")
    missing = [
        k
        for k, v in [
            ("GBD_DB_HOST", host),
            ("GBD_DB_PORT", port),
            ("GBD_DB_USER", user),
            ("GBD_DB_PASSWD", passwd),
        ]
        if not v or v.startswith("<")
    ]
    if missing:
        raise ValueError(
            f"Missing or unconfigured GBD postgres env vars: {', '.join(missing)}"
        )
    return f"postgresql+asyncpg://{user}:{passwd}@{host}:{port}/{name}"


async def init_telemetry_db() -> TelemetryDB:
    """Initialise the global TelemetryDB from GBD_DB_* environment variables.

    Raises ValueError if required vars are missing or unconfigured.
    """
    global _telemetry_db
    db = TelemetryDB(db_url=_build_telemetry_url())
    await db.initialize()
    _telemetry_db = db
    return db


async def close_telemetry_db() -> None:
    """Close and clear the global TelemetryDB."""
    global _telemetry_db
    if _telemetry_db:
        await _telemetry_db.close()
        _telemetry_db = None
