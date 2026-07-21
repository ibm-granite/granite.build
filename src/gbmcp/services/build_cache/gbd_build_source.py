"""GB Dashboard database client for querying gbd_meta AI analysis table.

Read-only access to the gb_dashboard PostgreSQL database (gbd_meta table).
Mirrors the _fetch_ai_analysis logic from gb_dashboard.services.build_cache.
"""

import os
from typing import Any, Dict, List, Optional

from fastmcp.utilities.logging import get_logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

logger = get_logger(__name__)


class GBDashboardSource:
    """Query AI analysis from gb_dashboard's gbd_meta table.

    Read-only. Uses raw SQL to avoid importing gb_dashboard's ORM.
    """

    def __init__(
        self,
        db_url: str,
        pool_size: int = 2,
        max_overflow: int = 3,
    ):
        self.db_url = db_url
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
        logger.info("GB Dashboard database initialized")

    async def close(self) -> None:
        """Close database connection pool."""
        if self._engine:
            await self._engine.dispose()
            logger.info("GB Dashboard database connection closed")

    async def fetch_ai_analysis(
        self, build_uuids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch AI analysis from gbd_meta for the given build UUIDs.

        Returns {uuid: analysis_dict} matching gb_dashboard's ai_analysis_by_build
        structure. Only returns the most recent analysis per build (parent_uid IS NULL).
        """
        if not self._session_factory or not build_uuids:
            return {}

        try:
            async with self._session_factory() as session:
                result = await session.execute(
                    text("""
                        SELECT DISTINCT ON (build_id)
                            build_id::text,
                            update_id::text,
                            source,
                            summary,
                            root_cause,
                            model_name,
                            created_at,
                            suggested_action,
                            issues,
                            confidence,
                            extras,
                            corrected_root_cause,
                            feedback_comment,
                            feedback_helpful,
                            error_category_1,
                            error_category_2,
                            error_category_3,
                            error_category_4
                        FROM gbd_meta
                        WHERE build_id::text = ANY(:uuids)
                          AND parent_uid IS NULL
                        ORDER BY build_id, created_at DESC
                    """),
                    {"uuids": build_uuids},
                )
                rows = result.fetchall()

            ai_analysis: Dict[str, Dict[str, Any]] = {}
            for row in rows:
                (
                    build_id,
                    update_id,
                    source,
                    summary,
                    root_cause,
                    model_name,
                    created_at,
                    suggested_action,
                    issues,
                    confidence,
                    extras,
                    corrected_root_cause,
                    feedback_comment,
                    feedback_helpful,
                    error_category_1,
                    error_category_2,
                    error_category_3,
                    error_category_4,
                ) = row
                ai_analysis[build_id] = {
                    "update_id": update_id,
                    "source": source,
                    "summary": summary,
                    "root_cause": root_cause,
                    "model_name": model_name or "unknown",
                    "created_at": created_at,
                    "suggested_action": suggested_action,
                    "issues": issues,
                    "confidence": confidence,
                    "extras": extras or {},
                    "corrected_root_cause": corrected_root_cause,
                    "feedback_comment": feedback_comment,
                    "feedback_helpful": feedback_helpful,
                    "error_category_1": error_category_1,
                    "error_category_2": error_category_2,
                    "error_category_3": error_category_3,
                    "error_category_4": error_category_4,
                }

            logger.debug(
                f"Fetched AI analysis for {len(ai_analysis)}/{len(build_uuids)} builds"
            )
            return ai_analysis

        except Exception as e:
            logger.warning(f"Failed to fetch AI analysis from gbd_meta: {e}")
            return {}

    async def fetch_k8s_resources(self, build_uuid: str) -> List[Dict[str, Any]]:
        """Fetch K8s resources from gbd_k8s_resources for a single build UUID.

        Returns a list of resource dicts with keys: kind, name, namespace, cluster,
        status, build_status, failure_reason, failure_message, cpu, memory, gpu,
        storage, replicas, extra, created_at, deleted_at.
        """
        if not self._session_factory:
            return []

        try:
            async with self._session_factory() as session:
                result = await session.execute(
                    text("""
                        SELECT
                            kind,
                            name,
                            namespace,
                            cluster_name,
                            status,
                            build_status,
                            failure_reason,
                            failure_message,
                            cpu,
                            memory,
                            gpu,
                            storage,
                            replicas,
                            extra,
                            created_at,
                            deleted_at
                        FROM gbd_k8s_resources
                        WHERE build_id = :uuid
                        ORDER BY kind, name
                    """),
                    {"uuid": build_uuid},
                )
                rows = result.fetchall()

            resources = []
            for row in rows:
                (
                    kind,
                    name,
                    namespace,
                    cluster_name,
                    status,
                    build_status,
                    failure_reason,
                    failure_message,
                    cpu,
                    memory,
                    gpu,
                    storage,
                    replicas,
                    extra,
                    created_at,
                    deleted_at,
                ) = row
                resources.append(
                    {
                        "kind": kind,
                        "name": name,
                        "namespace": namespace,
                        "cluster": cluster_name,
                        "status": status,
                        "build_status": build_status,
                        "failure_reason": failure_reason or "",
                        "failure_message": failure_message or "",
                        "cpu": cpu or "",
                        "memory": memory or "",
                        "gpu": str(gpu) if gpu else "",
                        "storage": storage or "",
                        "replicas": replicas or 1,
                        "extra": extra or {},
                        "created_at": created_at,
                        "deleted_at": deleted_at,
                    }
                )

            return resources

        except Exception as e:
            logger.warning(
                f"Failed to fetch K8s resources for build {build_uuid[:8]}: {e}"
            )
            return []


def _build_gbd_url() -> str:
    """Construct asyncpg DB URL from environment variables."""
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


async def init_gbd_source() -> GBDashboardSource:
    """Initialize a GBDashboardSource from environment variables.

    Reads: GBD_DB_HOST, GBD_DB_PORT, GBD_DB_USER, GBD_DB_PASSWD, GBD_DB_NAME
    Raises ValueError if required vars are missing or unconfigured.
    """
    source = GBDashboardSource(db_url=_build_gbd_url())
    await source.initialize()
    return source
