"""
GbserverSource — reads build data directly from gbserver's database.

Works with both:
  - SQLite (standalone mode):  sqlite+aiosqlite:///~/.llmb/llmb-server.db
  - PostgreSQL (hosted mode):  postgresql+asyncpg://user:pass@host/db

This is the analytics data source for standalone mode. Instead of syncing
from K8s into separate gbd_* tables, we read gbserver's own tables directly:

  gb_builds   — build records (uuid, name, space_name, username, status, timestamps)
  gb_events   — build events (MESSAGE_EVENT, STATUS_EVENT, WORKLOAD_STATUS_EVENT)
  gb_targets  — target runs (status, status_msg, started_at, finished_at)
  gb_steps    — step runs (definition_uri, status, status_msg, started_at, finished_at)

gbserver stores richer data in JSON blobs; we extract what we need via raw SQL.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

logger = logging.getLogger(__name__)

# gbserver status values (uppercase in the DB)
TERMINAL_STATUSES = {"SUCCESS", "FAILED", "CANCELLED", "INVALID", "ERROR"}
ACTIVE_STATUSES = {"PENDING", "RUNNING", "SUBMITTED", "SUSPENDED"}


def _resolve_db_url(url: str) -> str:
    """Expand ~ in SQLite paths."""
    if url.startswith("sqlite+aiosqlite:///"):
        path = url[len("sqlite+aiosqlite:///"):]
        return "sqlite+aiosqlite:///" + os.path.expanduser(path)
    return url


def _default_sqlite_url() -> str:
    return f"sqlite+aiosqlite:///{os.path.expanduser('~')}/.llmb/llmb-server.db"


class GbserverSource:
    """
    Async read-only client for gbserver's database.
    Instantiate once and reuse — the engine is shared.
    """

    def __init__(self, db_url: str, schema: str = "public"):
        url = _resolve_db_url(db_url)
        if "sqlite" in url:
            connect_args: dict = {"check_same_thread": False}
        else:
            # asyncpg: set search_path so bare table names resolve to the right schema
            connect_args = {"server_settings": {"search_path": schema}}
        self._engine = create_async_engine(
            url,
            echo=False,
            connect_args=connect_args,
            pool_pre_ping=True,
        )
        self._sessions = async_sessionmaker(self._engine, expire_on_commit=False)
        self._is_sqlite = "sqlite" in url
        logger.info(
            "GbserverSource connected to %s (schema=%s)",
            url.split("///")[-1].split("@")[-1], schema,
        )

    async def close(self) -> None:
        await self._engine.dispose()

    # ── Builds ────────────────────────────────────────────────────────────────

    async def list_builds(
        self,
        days_back: int = 30,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        status: Optional[str] = None,
        space_name: Optional[str] = None,
        username: Optional[str] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Return builds from gbserver ordered by updated_time desc."""
        if date_from:
            try:
                since = datetime.strptime(date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                since = datetime.now(timezone.utc) - timedelta(days=days_back)
        else:
            since = datetime.now(timezone.utc) - timedelta(days=days_back)

        conditions = ["updated_time >= :since"]
        params: Dict[str, Any] = {"since": since, "limit": limit}

        if date_to:
            try:
                until = datetime.strptime(date_to, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)
                conditions.append("updated_time < :until")
                params["until"] = until
            except ValueError:
                pass

        if status:
            conditions.append("status = :status")
            params["status"] = status.upper()
        if space_name:
            conditions.append("space_name = :space_name")
            params["space_name"] = space_name
        if username:
            conditions.append("username = :username")
            params["username"] = username

        where = " AND ".join(conditions)
        sql = f"""
            SELECT
                uuid,
                name,
                space_name,
                username,
                status,
                created_time,
                updated_time
            FROM gb_builds
            WHERE {where}
            ORDER BY updated_time DESC
            LIMIT :limit
        """
        async with self._sessions() as session:
            result = await session.execute(text(sql), params)
            rows = result.fetchall()

        builds = []
        for row in rows:
            builds.append({
                "uuid": str(row[0]),
                "name": row[1],
                "space_name": row[2],
                "username": row[3],
                "status": (row[4] or "").lower(),
                "created_time": row[5],
                "updated_time": row[6],
            })
        return builds

    async def list_builds_for_dp_scan(
        self,
        days_back: int = 7,
        limit: int = 10000,
    ) -> tuple[List[Dict[str, Any]], str | None]:
        """Return (builds_with_yaml, warning_or_None) for data processing path scanning.

        Tries to read build_archive (base64-encoded ZIP) and extract YAML inline.
        Falls back to returning builds without YAML if the column doesn't exist,
        so the caller can report the gap rather than silently returning nothing.
        """
        import base64
        import io
        import zipfile

        since = datetime.now(timezone.utc) - timedelta(days=days_back)
        params = {"since": since, "limit": limit}

        # --- attempt 1: fetch build_archive column directly ---
        # Filter by created_time OR updated_time so we catch:
        #   - builds started in the window (not yet updated)
        #   - builds from earlier that are still active / recently updated
        sql_with_archive = text("""
            SELECT uuid, name, space_name, username, status, created_time, updated_time,
                   build_archive
            FROM gb_builds
            WHERE created_time >= :since OR updated_time >= :since
            ORDER BY CASE WHEN created_time > updated_time THEN created_time ELSE updated_time END DESC
            LIMIT :limit
        """)
        has_archive_column = True
        rows = []
        async with self._sessions() as session:
            try:
                result = await session.execute(sql_with_archive, params)
                rows = result.fetchall()
            except Exception as exc:
                err_str = str(exc).lower()
                if "build_archive" in err_str or "column" in err_str or "undefined" in err_str:
                    has_archive_column = False
                    logger.info("list_builds_for_dp_scan: build_archive column absent, fetching without it")
                else:
                    raise

        # --- fallback: build_archive missing — introspect table for alternatives ---
        if not has_archive_column:
            # Discover what columns actually exist.
            # SQLite uses PRAGMA table_info; PostgreSQL uses information_schema.
            async with self._sessions() as session:
                if self._is_sqlite:
                    col_result = await session.execute(text("PRAGMA table_info(gb_builds)"))
                    available_cols = [r[1] for r in col_result.fetchall()]  # column 1 = name
                else:
                    col_result = await session.execute(text("""
                        SELECT column_name FROM information_schema.columns
                        WHERE table_schema = current_schema() AND table_name = 'gb_builds'
                        ORDER BY ordinal_position
                    """))
                    available_cols = [r[0] for r in col_result.fetchall()]

            logger.info("gb_builds columns: %s", available_cols)

            # gbserver stores the full StoredBuild as JSON in the "json" column —
            # build_archive is inside that JSON blob, same as how steps use it.
            if "json" in available_cols:
                logger.info("list_builds_for_dp_scan: extracting build_archive from json column")
                sql_json = text("""
                    SELECT uuid, name, space_name, username, status, created_time, updated_time,
                           "json"
                    FROM gb_builds
                    WHERE created_time >= :since OR updated_time >= :since
                    ORDER BY CASE WHEN created_time > updated_time THEN created_time ELSE updated_time END DESC
                    LIMIT :limit
                """)
                async with self._sessions() as session:
                    result = await session.execute(sql_json, params)
                    rows = result.fetchall()

                builds = []
                for row in rows:
                    uuid_, name, space_name, username, status, created_time, updated_time, json_blob = row
                    yaml_content: Optional[str] = None
                    if json_blob:
                        try:
                            import json as _json
                            blob = _json.loads(json_blob) if isinstance(json_blob, str) else json_blob
                            archive = blob.get("build_archive") if isinstance(blob, dict) else None
                            if archive:
                                import base64, io, zipfile
                                zip_bytes = base64.b64decode(archive)
                                with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                                    names = zf.namelist()
                                    # Prefer build.yaml; fall back to first yaml found
                                    target = next(
                                        (n for n in names if n.lower() in ("build.yaml", "build.yml")),
                                        next((n for n in names if n.endswith((".yaml", ".yml"))), None),
                                    )
                                    if target:
                                        yaml_content = zf.read(target).decode("utf-8", errors="replace")
                        except Exception as exc:
                            logger.debug("json column decode error for build %s: %s", uuid_, exc)
                    builds.append({
                        "uuid": str(uuid_), "name": name, "space_name": space_name,
                        "username": username, "status": (status or "").lower(),
                        "created_time": created_time, "updated_time": updated_time,
                        "yaml_content": yaml_content,
                    })
                return builds, None

            # No usable column found — return metadata-only with a detailed warning
            sql_no_archive = text("""
                SELECT uuid, name, space_name, username, status, created_time, updated_time
                FROM gb_builds
                WHERE created_time >= :since OR updated_time >= :since
                ORDER BY CASE WHEN created_time > updated_time THEN created_time ELSE updated_time END DESC
                LIMIT :limit
            """)
            async with self._sessions() as session:
                result = await session.execute(sql_no_archive, params)
                raw_rows = result.fetchall()

            builds = [
                {
                    "uuid": str(r[0]), "name": r[1], "space_name": r[2],
                    "username": r[3], "status": (r[4] or "").lower(),
                    "created_time": r[5], "yaml_content": None,
                }
                for r in raw_rows
            ]
            cols_str = ", ".join(available_cols) if available_cols else "(could not read)"
            warning = (
                f"build_archive column not found in gb_builds — "
                f"found {len(builds)} builds but cannot read their YAMLs to detect DP patterns. "
                f"Available columns: {cols_str}"
            )
            return builds, warning

        # --- decode archives ---
        builds = []
        for row in rows:
            uuid_, name, space_name, username, status, created_time, updated_time, archive = row
            yaml_content: Optional[str] = None
            if archive:
                try:
                    raw = archive if isinstance(archive, (bytes, bytearray)) else archive.encode()
                    zip_bytes = base64.b64decode(raw)
                    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                        for fname in zf.namelist():
                            if fname.endswith((".yaml", ".yml")):
                                yaml_content = zf.read(fname).decode("utf-8", errors="replace")
                                break
                except Exception:
                    pass
            builds.append({
                "uuid": str(uuid_),
                "name": name,
                "space_name": space_name,
                "username": username,
                "status": (status or "").lower(),
                "created_time": created_time,
                "yaml_content": yaml_content,
            })
        return builds, None

    async def get_build(self, build_id: str) -> Optional[Dict[str, Any]]:
        sql = """
            SELECT uuid, name, space_name, username, status, created_time, updated_time
            FROM gb_builds WHERE uuid = :build_id
        """
        async with self._sessions() as session:
            result = await session.execute(text(sql), {"build_id": build_id})
            row = result.fetchone()
        if not row:
            return None
        return {
            "uuid": str(row[0]), "name": row[1], "space_name": row[2],
            "username": row[3], "status": (row[4] or "").lower(),
            "created_time": row[5], "updated_time": row[6],
        }

    # ── Build status chart ────────────────────────────────────────────────────

    async def get_status_chart(
        self,
        days_back: int = 30,
        exclude_tests: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return daily build status counts from gb_builds."""
        since = datetime.now(timezone.utc) - timedelta(days=days_back)

        # SQLite uses strftime; PostgreSQL uses date()
        if self._is_sqlite:
            date_expr = "strftime('%Y-%m-%d', updated_time)"
        else:
            date_expr = "date(updated_time)"

        sql = f"""
            SELECT
                {date_expr} AS day,
                lower(status) AS status,
                count(*) AS cnt
            FROM gb_builds
            WHERE updated_time >= :since
            {'AND username NOT LIKE :test_prefix' if exclude_tests else ''}
            GROUP BY day, lower(status)
            ORDER BY day
        """
        params: Dict[str, Any] = {"since": since}
        if exclude_tests:
            params["test_prefix"] = "test%"

        async with self._sessions() as session:
            result = await session.execute(text(sql), params)
            rows = result.fetchall()

        # Pivot: date → {status: count}
        pivot: Dict[str, Dict[str, int]] = {}
        for row in rows:
            day_str = str(row[0])
            if day_str not in pivot:
                pivot[day_str] = {}
            pivot[day_str][row[1]] = int(row[2])

        statuses = ["running", "success", "failed", "pending", "submitted", "suspended"]
        return [
            {
                "date": day,
                **{s: pivot[day].get(s, 0) for s in statuses},
            }
            for day in sorted(pivot)
        ]

    # ── Failure trends ────────────────────────────────────────────────────────

    async def get_failed_builds(
        self,
        days_back: int = 30,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        exclude_tests: bool = False,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Return failed builds for trend analysis."""
        return await self.list_builds(
            days_back=days_back,
            date_from=date_from,
            date_to=date_to,
            status="FAILED",
            limit=limit,
        )

    # ── Events / targets / steps (for AI analysis) ────────────────────────────

    async def get_build_events(
        self,
        build_id: str,
        max_events: int = 200,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Fetch build events, targets, and steps from gbserver.
        Returns (events, status_msgs) in the same format as GbserverDataCollector.
        """
        import json as _json

        sql = """
            SELECT 'event'  AS src, uuid, NULL     AS name,
                   type     AS type_status, json   AS json_data,
                   created_time, COALESCE(index_col, 0) AS idx,
                   source, target_id, step_id
            FROM gb_events WHERE build_id = :build_id

            UNION ALL

            SELECT 'target', uuid, name,
                   status, json,
                   NULL, NULL, NULL, NULL, NULL
            FROM gb_targets WHERE build_id = :build_id

            UNION ALL

            SELECT 'step',   uuid, NULL,
                   status,   json,
                   NULL, NULL, NULL, NULL, NULL
            FROM gb_steps WHERE build_id = :build_id
        """
        # gb_events uses "index" which is a reserved word in some DBs —
        # try the query, fall back to a simpler version if it fails
        try:
            async with self._sessions() as session:
                result = await session.execute(text(sql), {"build_id": build_id})
                rows = result.fetchall()
        except Exception:
            # Simpler fallback without index column
            sql_simple = """
                SELECT 'event' AS src, uuid, NULL AS name,
                       type AS type_status, json AS json_data,
                       created_time, 0 AS idx, source, target_id, step_id
                FROM gb_events WHERE build_id = :build_id
                UNION ALL
                SELECT 'target', uuid, name, status, json, NULL, NULL, NULL, NULL, NULL
                FROM gb_targets WHERE build_id = :build_id
                UNION ALL
                SELECT 'step', uuid, NULL, status, json, NULL, NULL, NULL, NULL, NULL
                FROM gb_steps WHERE build_id = :build_id
            """
            async with self._sessions() as session:
                result = await session.execute(text(sql_simple), {"build_id": build_id})
                rows = result.fetchall()

        events: List[Dict[str, Any]] = []
        status_msgs: List[Dict[str, Any]] = []

        for row in rows:
            src, _uid, name, type_status, json_data, created_time, idx, source, target_id, step_id = row
            try:
                payload = _json.loads(json_data) if json_data else {}
            except Exception:
                payload = {}

            if src == "event":
                ev_payload = payload.get("build_event", payload).get("payload", {})
                event: Dict[str, Any] = {
                    "index": idx or 0,
                    "type": type_status,
                    "source": source,
                    "created_time": created_time.isoformat() if hasattr(created_time, "isoformat") else str(created_time or ""),
                }
                if type_status == "MESSAGE_EVENT":
                    event["level"] = ev_payload.get("level", "INFO")
                    event["message"] = ev_payload.get("msg", "")[:10240]
                elif type_status == "STATUS_EVENT":
                    event["status"] = ev_payload.get("status", "")
                    event["message"] = ev_payload.get("msg", "")[:10240]
                elif type_status == "WORKLOAD_STATUS_EVENT":
                    event["appwrapper_state"] = ev_payload.get("state", "")
                    event["failed_pods"] = ev_payload.get("failed_pods", {})
                events.append(event)

            elif src in ("target", "step"):
                status_msg = payload.get("status_msg", "")
                definition_uri = payload.get("definition_uri", "")
                if status_msg or (type_status or "").upper() in ("FAILED", "ERROR", "INVALID"):
                    status_msgs.append({
                        "entity": src,
                        "name": name or definition_uri or "unknown",
                        "definition_uri": definition_uri,
                        "status": type_status,
                        "status_msg": status_msg[:10240] if status_msg else "",
                        "started_at": payload.get("started_at"),
                        "finished_at": payload.get("finished_at"),
                    })

        # Sort events by priority: errors first
        def _priority(e: Dict) -> tuple:
            if e["type"] == "MESSAGE_EVENT" and e.get("level") == "ERROR":
                return (0, -(e.get("index") or 0))
            if e["type"] == "STATUS_EVENT" and e.get("status", "").upper() in ("FAILED", "ERROR"):
                return (1, -(e.get("index") or 0))
            return (4, -(e.get("index") or 0))

        events.sort(key=_priority)
        return events[:max_events], status_msgs

    # ── Leaderboard ───────────────────────────────────────────────────────────

    async def get_leaderboard(
        self,
        view: str = "running_jobs",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Simple leaderboard from gb_builds — no GPU data in standalone."""
        if view == "running_jobs":
            sql = """
                SELECT username, count(*) AS cnt
                FROM gb_builds WHERE status = 'RUNNING'
                GROUP BY username ORDER BY cnt DESC LIMIT :limit
            """
        elif view == "total_builds":
            sql = """
                SELECT username, count(*) AS cnt
                FROM gb_builds GROUP BY username ORDER BY cnt DESC LIMIT :limit
            """
        else:
            # cpu/memory/gpu — not available in gbserver SQLite, fall back to total
            sql = """
                SELECT username, count(*) AS cnt
                FROM gb_builds GROUP BY username ORDER BY cnt DESC LIMIT :limit
            """

        async with self._sessions() as session:
            result = await session.execute(text(sql), {"limit": limit})
            rows = result.fetchall()

        return [
            {
                "username": row[0],
                "running_jobs": int(row[1]) if view == "running_jobs" else 0,
                "total_builds": int(row[1]) if view == "total_builds" else 0,
                "gpu_count": 0,
                "cpu_cores": 0.0,
                "memory_gib": 0.0,
            }
            for row in rows
        ]


_default_source: Optional[GbserverSource] = None
_sources_by_schema: Dict[str, GbserverSource] = {}
_env_to_schema: Dict[str, str] = {}  # uppercase env_id → schema


def get_gbserver_source(env_id: Optional[str] = None) -> Optional[GbserverSource]:
    """Return the GbserverSource for the given environment ID, or the default."""
    if env_id:
        schema = _env_to_schema.get(env_id.upper())
        if schema and schema in _sources_by_schema:
            return _sources_by_schema[schema]
    return _default_source


async def init_gbserver_sources_from_environments(
    db_url: str,
    default_schema: str = "public",
) -> None:
    """Initialize one GbserverSource per unique dbSchema found in ENVIRONMENTS_JSON.

    Falls back to a single source using default_schema when the env var is absent
    or unparseable. The default_schema source is always created.
    """
    import json as _json
    import os as _os

    global _default_source, _sources_by_schema, _env_to_schema

    async def _get_or_create(schema: str) -> GbserverSource:
        if schema not in _sources_by_schema:
            _sources_by_schema[schema] = GbserverSource(db_url, schema=schema)
        return _sources_by_schema[schema]

    # Always create the deployment-default source first
    _default_source = await _get_or_create(default_schema)

    # Parse ENVIRONMENTS_JSON and create one source per unique schema
    raw = _os.environ.get("ENVIRONMENTS_JSON", "")
    if raw:
        try:
            for env in _json.loads(raw):
                env_id = str(env.get("id", "")).upper()
                schema = env.get("dbSchema") or default_schema
                await _get_or_create(schema)
                _env_to_schema[env_id] = schema
        except Exception as exc:
            logger.warning("ENVIRONMENTS_JSON parse error in gbserver_source: %s", exc)

    logger.info(
        "GbserverSource: %d schema(s) initialised, %d env mappings",
        len(_sources_by_schema), len(_env_to_schema),
    )


# Kept for backward compat — callers that don't care about env switching.
async def init_gbserver_source(db_url: str, schema: str = "public") -> GbserverSource:
    await init_gbserver_sources_from_environments(db_url, default_schema=schema)
    return _default_source  # type: ignore[return-value]
