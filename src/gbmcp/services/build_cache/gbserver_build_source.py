"""GBServer database client for querying gb_builds table.

Copied from gb_dashboard.services.gbserver_build_source.
Read-only access to the gbserver PostgreSQL database.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Set

from fastmcp.utilities.logging import get_logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

logger = get_logger(__name__)

# Safety limit: never query more than this many days back
MAX_LOOKBACK_DAYS = 60


class GBServerBuildSource:
    """Query builds from gbserver database (gb_builds table).

    Read-only access to the gbserver database for fetching build YAML archives
    and other build metadata. Uses raw SQL since we don't own the schema.
    """

    def __init__(
        self,
        db_url: str,
        schema: str = "granite_dot_build_prod",
        pool_size: int = 2,
        max_overflow: int = 3,
    ):
        self.db_url = db_url
        self.schema = schema
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker] = None
        self._pool_size = pool_size
        self._max_overflow = max_overflow

    async def initialize(self) -> None:
        """Initialize database connection pool."""
        self._engine = create_async_engine(
            self.db_url,
            echo=False,
            pool_size=self._pool_size,
            max_overflow=self._max_overflow,
            pool_pre_ping=True,
            pool_timeout=10,
        )
        self._session_factory = async_sessionmaker(
            self._engine,
            expire_on_commit=False,
        )
        logger.info(f"GBServer database initialized (schema: {self.schema})")

    async def close(self) -> None:
        """Close database connection pool."""
        if self._engine:
            await self._engine.dispose()
            logger.info("GBServer database connection closed")

    def _get_time_bounds(self, days_ago: int) -> tuple[datetime, datetime]:
        now = datetime.now(timezone.utc)
        actual_days = min(days_ago, MAX_LOOKBACK_DAYS)
        start_time = now - timedelta(days=actual_days)
        return start_time, now

    async def get_build_uuids_in_timerange(
        self,
        days_ago: int = 1,
        space_names: Optional[list[str]] = None,
        username: Optional[str] = None,
    ) -> Set[str]:
        if not self._session_factory:
            return set()

        start_time, end_time = self._get_time_bounds(days_ago)

        async with self._session_factory() as session:
            try:
                query = f"""
                    SELECT uuid::text
                    FROM {self.schema}.gb_builds
                    WHERE created_time >= :start_time
                      AND created_time <= :end_time
                """
                params: dict = {"start_time": start_time, "end_time": end_time}

                if space_names:
                    if "public" in space_names:
                        query += " AND (space_name = ANY(:space_names) OR space_name IS NULL OR space_name = '')"
                    else:
                        query += " AND space_name = ANY(:space_names)"
                    params["space_names"] = space_names

                if username:
                    query += " AND username ILIKE :username"
                    params["username"] = f"%{username}%"

                result = await session.execute(text(query), params)
                return {str(row[0]) for row in result.fetchall()}
            except Exception as e:
                logger.warning(f"Failed to query gbserver builds: {e}")
                return set()

    async def get_builds_not_in_k8s(
        self,
        exclude_uuids: Set[str],
        days_ago: int = 1,
        space_names: Optional[list[str]] = None,
        username: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        if not self._session_factory:
            return []

        start_time, end_time = self._get_time_bounds(days_ago)

        async with self._session_factory() as session:
            try:
                query = f"""
                    SELECT uuid::text, name, space_name, username, status,
                           source_uri, created_time, updated_time
                    FROM {self.schema}.gb_builds
                    WHERE created_time >= :start_time
                      AND created_time <= :end_time
                """
                params: dict = {"start_time": start_time, "end_time": end_time}

                if exclude_uuids:
                    query += " AND uuid::text != ALL(:exclude_uuids)"
                    params["exclude_uuids"] = list(exclude_uuids)

                if space_names:
                    if "public" in space_names:
                        query += " AND (space_name = ANY(:space_names) OR space_name IS NULL OR space_name = '')"
                    else:
                        query += " AND space_name = ANY(:space_names)"
                    params["space_names"] = space_names

                if username:
                    query += " AND username ILIKE :username"
                    params["username"] = f"%{username}%"

                query += " ORDER BY created_time DESC LIMIT :limit"
                params["limit"] = limit

                result = await session.execute(text(query), params)
                return [
                    {
                        "uuid": row[0],
                        "name": row[1],
                        "space_name": row[2],
                        "username": row[3],
                        "status": row[4],
                        "source_uri": row[5],
                        "created_time": row[6],
                        "updated_time": row[7],
                    }
                    for row in result.fetchall()
                ]
            except Exception as e:
                logger.warning(f"Failed to query gbserver builds not in K8s: {e}")
                return []

    async def get_builds_by_uuids(self, uuids: Set[str]) -> dict[str, dict]:
        if not self._session_factory or not uuids:
            return {}

        async with self._session_factory() as session:
            try:
                query = f"""
                    SELECT uuid::text, name, space_name, username, status,
                           source_uri, created_time, updated_time
                    FROM {self.schema}.gb_builds
                    WHERE uuid::text = ANY(:uuids)
                """
                result = await session.execute(text(query), {"uuids": list(uuids)})
                return {
                    row[0]: {
                        "uuid": row[0],
                        "name": row[1],
                        "space_name": row[2],
                        "username": row[3],
                        "status": row[4],
                        "source_uri": row[5],
                        "created_time": row[6],
                        "updated_time": row[7],
                    }
                    for row in result.fetchall()
                }
            except Exception as e:
                logger.warning(f"Failed to fetch gbserver builds by UUIDs: {e}")
                return {}

    async def count_builds_in_timerange(
        self, days_ago: int = 1, username: Optional[str] = None
    ) -> int:
        if not self._session_factory:
            return 0

        start_time, end_time = self._get_time_bounds(days_ago)

        async with self._session_factory() as session:
            try:
                query = f"""
                    SELECT COUNT(*)
                    FROM {self.schema}.gb_builds
                    WHERE created_time >= :start_time
                      AND created_time <= :end_time
                """
                params: dict = {"start_time": start_time, "end_time": end_time}

                if username:
                    query += " AND username ILIKE :username"
                    params["username"] = f"%{username}%"

                result = await session.execute(text(query), params)
                row = result.fetchone()
                return row[0] if row else 0
            except Exception as e:
                logger.warning(f"Failed to count gbserver builds: {e}")
                return 0

    async def get_build_yaml(self, build_id: str) -> Optional[dict]:
        """Fetch build.yaml content for a single build.

        Extracts the raw build.yaml string from the base64+zipped build_archive
        stored in the gb_builds.json column.

        Returns:
            Dict with keys: yaml, build_name, space_name, username, targets,
            parameters_applied — or None if not found.
        """
        import base64
        import io
        import json
        import zipfile

        if not self._session_factory:
            return None

        async with self._session_factory() as session:
            try:
                result = await session.execute(
                    text(
                        f"SELECT name, json FROM {self.schema}.gb_builds WHERE uuid = :build_id"
                    ),
                    {"build_id": build_id},
                )
                row = result.fetchone()
                if not row:
                    return None

                try:
                    payload = json.loads(row[1]) if row[1] else {}
                except json.JSONDecodeError:
                    payload = {}

                build_archive = payload.get("build_archive")
                if not build_archive:
                    return {
                        "yaml": "",
                        "build_name": row[0] or "",
                        "space_name": payload.get("space_name", ""),
                        "username": payload.get("username", ""),
                        "targets": payload.get("targets", []),
                    }

                yaml_content = ""
                parameters_applied = ""
                try:
                    zip_bytes = base64.b64decode(build_archive)
                    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
                        names = zf.namelist()
                        if "build.yaml" in names:
                            yaml_content = zf.read("build.yaml").decode("utf-8")
                        if "parameters-applied.yaml" in names:
                            parameters_applied = zf.read(
                                "parameters-applied.yaml"
                            ).decode("utf-8")
                except Exception as e:
                    logger.warning(
                        f"Failed to extract build archive for {build_id}: {e}"
                    )

                return {
                    "yaml": yaml_content,
                    "build_name": row[0] or "",
                    "space_name": payload.get("space_name", ""),
                    "username": payload.get("username", ""),
                    "targets": payload.get("targets", []),
                    "parameters_applied": parameters_applied,
                }
            except Exception as e:
                logger.warning(f"Failed to fetch build yaml for {build_id}: {e}")
                return None

    async def search_builds(
        self,
        search: str,
        exclude_uuids: Optional[Set[str]] = None,
        days_ago: int = 60,
        limit: int = 100,
        space_names: Optional[list[str]] = None,
        username: Optional[str] = None,
    ) -> list[dict]:
        """Search builds by UUID, name, username, or space_name."""
        if not self._session_factory or not search:
            return []

        start_time, end_time = self._get_time_bounds(days_ago)
        search_pattern = f"%{search}%"

        async with self._session_factory() as session:
            try:
                query = f"""
                    SELECT uuid::text, name, space_name, username, status,
                           source_uri, created_time, updated_time
                    FROM {self.schema}.gb_builds
                    WHERE created_time >= :start_time
                      AND created_time <= :end_time
                      AND (
                          uuid::text ILIKE :search
                          OR name ILIKE :search
                          OR username ILIKE :search
                          OR space_name ILIKE :search
                      )
                """
                params: dict = {
                    "start_time": start_time,
                    "end_time": end_time,
                    "search": search_pattern,
                }

                if exclude_uuids:
                    query += " AND uuid::text != ALL(:exclude_uuids)"
                    params["exclude_uuids"] = list(exclude_uuids)

                if space_names:
                    if "public" in space_names:
                        query += " AND (space_name = ANY(:space_names) OR space_name IS NULL OR space_name = '')"
                    else:
                        query += " AND space_name = ANY(:space_names)"
                    params["space_names"] = space_names

                if username:
                    query += " AND username ILIKE :filter_username"
                    params["filter_username"] = username

                query += " ORDER BY created_time DESC LIMIT :limit"
                params["limit"] = limit

                result = await session.execute(text(query), params)
                return [
                    {
                        "uuid": row[0],
                        "name": row[1],
                        "space_name": row[2],
                        "username": row[3],
                        "status": row[4],
                        "source_uri": row[5],
                        "created_time": row[6],
                        "updated_time": row[7],
                    }
                    for row in result.fetchall()
                ]
            except Exception as e:
                logger.warning(f"Failed to search gbserver builds: {e}")
                return []
