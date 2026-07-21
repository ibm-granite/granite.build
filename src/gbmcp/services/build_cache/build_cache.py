"""Build cache service - in-memory cache of build data backed by gbserver PostgreSQL.

Copied and adapted from gb_dashboard.services.build_cache.
Removed: K8s dashboard DB, metrics, auto-cleanup (dashboard-only).
Kept: gbserver build list, gbserver YAML archive extraction, gb_dashboard AI analysis,
      background refresh.

Cache strategy (matches gb_dashboard):
- Startup: warms with 1 day of history, then a background loader progressively
  extends to cache_days over background_duration seconds.
- Periodic refresh: re-fetches only the last 1 day but carries forward all older
  builds already in cache (preserving the accumulated history indefinitely).
"""

import asyncio
import base64
import io
import json
import os
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from fastmcp.utilities.logging import get_logger
from sqlalchemy import text as sa_text

from gbmcp.services.build_cache.build_source import Build, BuildStatus, BuildStatusInfo
from gbmcp.services.build_cache.gbd_build_source import (
    GBDashboardSource,
    init_gbd_source,
)
from gbmcp.services.build_cache.gbserver_build_source import GBServerBuildSource

logger = get_logger(__name__)


@dataclass
class CachedBuildData:
    """In-memory snapshot of build data from gbserver + gb_dashboard."""

    builds: List[Build]
    total_count: int
    last_refresh: datetime
    yaml_by_build: Dict[str, str] = field(
        default_factory=dict
    )  # UUID → build.yaml content
    ai_analysis_by_build: Dict[str, Dict[str, Any]] = field(
        default_factory=dict
    )  # UUID → AI analysis dict (from gbd_meta)


class BuildCache:
    """In-memory cache of gbserver build data with background refresh.

    Initialised with a GBServerBuildSource and a refresh interval (seconds).
    Call start() to warm the cache and begin background refreshes.

    Cache depth strategy:
    - Warms with 1 day at startup to minimise startup latency.
    - A background task progressively extends history to cache_days over
      background_duration seconds (e.g. 60 days over 300 s).
    - Each periodic refresh only re-fetches the last 1 day, then carries
      forward all older builds already in memory — so history accumulates
      and is never dropped by a routine refresh cycle.
    """

    def __init__(
        self,
        gbserver_source: GBServerBuildSource,
        refresh_interval: int = 60,
        cache_days: int = 7,
        background_duration: int = 300,
        gbd_source: Optional[GBDashboardSource] = None,
    ):
        self._gbserver_source = gbserver_source
        self._gbd_source = gbd_source
        self._refresh_interval = refresh_interval
        self._cache_days = cache_days
        self._background_duration = background_duration
        self._cache: Optional[CachedBuildData] = None
        self._lock = asyncio.Lock()
        self._refresh_task: Optional[asyncio.Task] = None
        self._background_load_task: Optional[asyncio.Task] = None
        self._current_loaded_days: int = 1
        self._running = False

    async def start(self) -> None:
        """Warm cache with 1 day and start background refresh + extension loops."""
        logger.info("Starting build cache warmup (1 day)...")
        await self._refresh_cache(days=1)
        if self._cache:
            logger.info(f"Build cache warmed up with {len(self._cache.builds)} builds")
        self._running = True
        self._refresh_task = asyncio.create_task(self._refresh_loop())
        logger.info(
            f"Build cache background refresh started (interval: {self._refresh_interval}s)"
        )

        if self._cache_days > 1:
            self._background_load_task = asyncio.create_task(
                self._background_load_loop()
            )
            logger.info(
                f"Background cache loading started "
                f"(target: {self._cache_days} days over {self._background_duration}s)"
            )

    async def stop(self) -> None:
        """Stop background refresh and extension tasks."""
        self._running = False
        for task in (self._refresh_task, self._background_load_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._refresh_task = None
        self._background_load_task = None
        logger.info("Build cache stopped")

    async def _refresh_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self._refresh_interval)
                if not self._running:
                    break
                await self._refresh_cache(days=1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache refresh loop: {e}", exc_info=True)

    async def _background_load_loop(self) -> None:
        """Gradually extend cache history from 1 day up to cache_days."""
        target_days = self._cache_days
        duration_seconds = self._background_duration

        days_to_load = target_days - 1
        batches = min(30, days_to_load)
        days_per_batch = max(1, days_to_load // batches)
        interval = duration_seconds / batches

        logger.info(
            f"Background loading: {days_to_load} days in {batches} batches, "
            f"{days_per_batch} days every {interval:.1f}s"
        )

        current_days = 1
        while self._running and current_days < target_days:
            try:
                await asyncio.sleep(interval)
                if not self._running:
                    break
                next_days = min(current_days + days_per_batch, target_days)
                await self._extend_cache_to_days(next_days)
                current_days = next_days
                self._current_loaded_days = current_days
                logger.info(
                    f"Background cache: loaded {current_days}/{target_days} days, "
                    f"{len(self._cache.builds) if self._cache else 0} total builds"
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Background load batch failed: {e}")

        logger.info(
            f"Background cache loading complete: {current_days} days, "
            f"{len(self._cache.builds) if self._cache else 0} builds"
        )

    async def _refresh_cache(self, days: int = 1) -> None:
        """Fetch fresh data for the last `days` days and merge with preserved older builds.

        When days=1 (normal refresh cycle): re-fetches recent builds and carries
        forward all older builds already in cache so accumulated history is never lost.
        When days>1 (initial warmup): full replace with no older cache to carry forward.
        """
        try:
            builds_raw = await self._gbserver_source.get_builds_not_in_k8s(
                exclude_uuids=set(),
                days_ago=days,
                limit=10_000,
            )

            fresh_builds: List[Build] = [self._dict_to_build(b) for b in builds_raw]
            fresh_uuids: Set[str] = {b.uuid for b in fresh_builds}

            # Carry forward older builds not covered by this refresh window
            older_builds: List[Build] = []
            async with self._lock:
                if self._cache and self._current_loaded_days > 1:
                    for build in self._cache.builds:
                        if build.uuid not in fresh_uuids:
                            older_builds.append(build)

            all_builds = fresh_builds + older_builds
            all_builds.sort(
                key=lambda b: b.created_time
                or datetime.min.replace(tzinfo=timezone.utc),
                reverse=True,
            )

            fresh_yaml = await self._fetch_yaml_for_builds(fresh_builds)
            fresh_ai = await self._fetch_ai_analysis(fresh_builds)

            async with self._lock:
                old_yaml = self._cache.yaml_by_build if self._cache else {}
                old_ai = self._cache.ai_analysis_by_build if self._cache else {}
                all_uuids = {b.uuid for b in all_builds}
                preserved_yaml = {k: v for k, v in old_yaml.items() if k in all_uuids}
                preserved_ai = {k: v for k, v in old_ai.items() if k in all_uuids}
                self._cache = CachedBuildData(
                    builds=all_builds,
                    total_count=len(all_builds),
                    last_refresh=datetime.now(timezone.utc),
                    yaml_by_build={**preserved_yaml, **fresh_yaml},
                    ai_analysis_by_build={**preserved_ai, **fresh_ai},
                )

            logger.debug(
                f"Cache refreshed: {len(fresh_builds)} fresh + {len(older_builds)} preserved = {len(all_builds)} total"
            )
        except Exception as e:
            logger.error(f"Failed to refresh cache: {e}", exc_info=True)

    async def _extend_cache_to_days(self, days: int) -> None:
        """Fetch builds for the extended window and merge new ones into cache."""
        try:
            builds_raw = await self._gbserver_source.get_builds_not_in_k8s(
                exclude_uuids=set(),
                days_ago=days,
                limit=10_000,
            )

            async with self._lock:
                if not self._cache:
                    return
                existing_uuids = {b.uuid for b in self._cache.builds}
                new_builds = [
                    self._dict_to_build(b)
                    for b in builds_raw
                    if b["uuid"] not in existing_uuids
                ]

                if not new_builds:
                    logger.debug(f"No new builds found for {days}-day extension")
                    return

            new_yaml = await self._fetch_yaml_for_builds(new_builds)
            new_ai = await self._fetch_ai_analysis(new_builds)

            async with self._lock:
                if not self._cache:
                    return
                # Re-check for duplicates in case of concurrent refresh
                existing_uuids = {b.uuid for b in self._cache.builds}
                deduped = [b for b in new_builds if b.uuid not in existing_uuids]
                all_builds = self._cache.builds + deduped
                all_builds.sort(
                    key=lambda b: b.created_time
                    or datetime.min.replace(tzinfo=timezone.utc),
                    reverse=True,
                )
                self._cache = CachedBuildData(
                    builds=all_builds,
                    total_count=len(all_builds),
                    last_refresh=self._cache.last_refresh,
                    yaml_by_build={**self._cache.yaml_by_build, **new_yaml},
                    ai_analysis_by_build={**self._cache.ai_analysis_by_build, **new_ai},
                )
                logger.debug(
                    f"Extended cache: +{len(deduped)} builds, total {len(all_builds)}"
                )
        except Exception as e:
            logger.warning(f"Failed to extend cache to {days} days: {e}")

    async def _fetch_yaml_for_builds(self, builds: List[Build]) -> Dict[str, str]:
        """Batch-extract build.yaml from gbserver's base64+zip archives."""
        if not builds:
            return {}

        existing_yaml = self._cache.yaml_by_build if self._cache else {}
        uuids = [b.uuid for b in builds if b.uuid not in existing_yaml]
        if not uuids:
            return {}

        yaml_map: Dict[str, str] = {}
        try:
            async with self._gbserver_source._session_factory() as session:
                result = await session.execute(
                    sa_text(f"""
                        SELECT uuid::text, json
                        FROM {self._gbserver_source.schema}.gb_builds
                        WHERE uuid::text = ANY(:uuids)
                    """),
                    {"uuids": uuids},
                )
                rows = result.fetchall()

            def _extract(rows_data):
                extracted = {}
                for uuid_str, json_str in rows_data:
                    try:
                        payload = json.loads(json_str) if json_str else {}
                    except (json.JSONDecodeError, TypeError):
                        continue
                    build_archive = payload.get("build_archive")
                    if not build_archive:
                        continue
                    try:
                        zip_bytes = base64.b64decode(build_archive)
                        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
                            if "build.yaml" in zf.namelist():
                                extracted[uuid_str] = zf.read("build.yaml").decode(
                                    "utf-8"
                                )
                    except Exception:
                        continue
                return extracted

            loop = asyncio.get_event_loop()
            yaml_map = await loop.run_in_executor(None, _extract, rows)
            logger.debug(f"Fetched build.yaml for {len(yaml_map)}/{len(uuids)} builds")
        except Exception as e:
            logger.warning(f"Failed to fetch build YAML archives: {e}")

        return yaml_map

    async def _fetch_ai_analysis(
        self, builds: List[Build]
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch AI analysis from gb_dashboard's gbd_meta for the given builds."""
        if not self._gbd_source or not builds:
            return {}
        existing_ai = self._cache.ai_analysis_by_build if self._cache else {}
        uuids = [b.uuid for b in builds if b.uuid not in existing_ai]
        if not uuids:
            return {}
        return await self._gbd_source.fetch_ai_analysis(uuids)

    def _dict_to_build(self, gb: Dict[str, Any]) -> Build:
        status_map = {
            "submitted": BuildStatus.SUBMITTED,
            "pending": BuildStatus.PENDING,
            "running": BuildStatus.RUNNING,
            "success": BuildStatus.SUCCESS,
            "failed": BuildStatus.FAILED,
            "invalid": BuildStatus.INVALID,
            "cancelled": BuildStatus.CANCELLED,
            "cancel_requested": BuildStatus.CANCEL_REQUESTED,
            "suspended": BuildStatus.SUSPENDED,
        }
        build_status = status_map.get(
            (gb.get("status") or "").lower(), BuildStatus.PENDING
        )
        status_info = BuildStatusInfo(
            is_completed=build_status
            in (
                BuildStatus.SUCCESS,
                BuildStatus.FAILED,
                BuildStatus.CANCELLED,
                BuildStatus.INVALID,
            ),
            has_failures=build_status in (BuildStatus.FAILED, BuildStatus.INVALID),
        )
        return Build(
            uuid=gb["uuid"],
            name=gb.get("name") or "",
            space_name=gb.get("space_name") or "",
            username=gb.get("username") or "",
            status=build_status,
            source_uri=gb.get("source_uri") or "",
            created_time=gb.get("created_time"),
            updated_time=gb.get("updated_time"),
            data_sources={"gbserver"},
            status_info=status_info,
            gbserver_status=gb.get("status"),
            gbserver_name=gb.get("name"),
            gbserver_space_name=gb.get("space_name"),
        )

    @property
    def yaml_cache(self) -> Dict[str, str]:
        """UUID → build.yaml content map."""
        return self._cache.yaml_by_build if self._cache else {}

    @property
    def ai_analysis(self) -> Dict[str, Dict[str, Any]]:
        """UUID → AI analysis dict (from gbd_meta)."""
        return self._cache.ai_analysis_by_build if self._cache else {}

    @property
    def build_count(self) -> int:
        return len(self._cache.builds) if self._cache else 0

    @property
    def last_refresh(self) -> Optional[datetime]:
        return self._cache.last_refresh if self._cache else None


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_build_cache: Optional[BuildCache] = None


def get_build_cache() -> Optional[BuildCache]:
    """Return the global BuildCache instance, or None if not initialised."""
    return _build_cache


def _build_gbserver_url() -> str:
    """Construct asyncpg DB URL from environment variables."""
    host = os.environ.get("GBSERVER_DB_HOST", "")
    port = os.environ.get("GBSERVER_DB_PORT", "")
    user = os.environ.get("GBSERVER_DB_USER", "")
    passwd = os.environ.get("GBSERVER_DB_PASSWD", "")
    name = os.environ.get("GBSERVER_DB_NAME", "ibmclouddb")
    missing = [
        k
        for k, v in [
            ("GBSERVER_DB_HOST", host),
            ("GBSERVER_DB_PORT", port),
            ("GBSERVER_DB_USER", user),
            ("GBSERVER_DB_PASSWD", passwd),
        ]
        if not v or v.startswith("<")
    ]
    if missing:
        raise ValueError(
            f"Missing or unconfigured postgres env vars: {', '.join(missing)}"
        )
    return f"postgresql+asyncpg://{user}:{passwd}@{host}:{port}/{name}"


async def init_build_cache(
    refresh_interval: int = 60,
    cache_days: int = 60,
    background_duration: int = 300,
) -> BuildCache:
    """Initialise and start the global BuildCache.

    Reads connection config from environment variables:
      GBSERVER_DB_HOST, GBSERVER_DB_PORT, GBSERVER_DB_USER,
      GBSERVER_DB_PASSWD, GBSERVER_DB_NAME, GBSERVER_DB_SCHEMA

    Optional — gb_dashboard AI analysis (gbd_meta):
      GBD_DB_HOST, GBD_DB_PORT, GBD_DB_USER, GBD_DB_PASSWD, GBD_DB_NAME

    Optional tuning via environment variables:
      BUILD_CACHE_REFRESH_INTERVAL    — seconds between periodic refreshes (default 60)
      BUILD_CACHE_DAYS                — days of history to load (default 60)
      BUILD_CACHE_BACKGROUND_DURATION — seconds to spread background loading over (default 300)
    """
    global _build_cache
    schema = os.environ.get("GBSERVER_DB_SCHEMA", "granite_dot_build_prod")
    refresh_interval = int(
        os.environ.get("BUILD_CACHE_REFRESH_INTERVAL", str(refresh_interval))
    )
    cache_days = int(os.environ.get("BUILD_CACHE_DAYS", str(cache_days)))
    background_duration = int(
        os.environ.get("BUILD_CACHE_BACKGROUND_DURATION", str(background_duration))
    )
    source = GBServerBuildSource(db_url=_build_gbserver_url(), schema=schema)
    await source.initialize()

    gbd_source = None
    try:
        gbd_source = await init_gbd_source()
        logger.info("GB Dashboard AI analysis source initialized")
    except ValueError as e:
        logger.info(f"GB Dashboard AI analysis not configured (skipping): {e}")

    _build_cache = BuildCache(
        source,
        refresh_interval=refresh_interval,
        cache_days=cache_days,
        background_duration=background_duration,
        gbd_source=gbd_source,
    )
    await _build_cache.start()
    return _build_cache


async def close_build_cache() -> None:
    """Stop and close the global BuildCache."""
    global _build_cache
    if _build_cache:
        await _build_cache.stop()
        if _build_cache._gbserver_source:
            await _build_cache._gbserver_source.close()
        if _build_cache._gbd_source:
            await _build_cache._gbd_source.close()
        _build_cache = None
