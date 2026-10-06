#!/usr/bin/env python3

# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The lineage indexer: fills ``gb_lineage_index`` incrementally from one source.

The index exists to navigate and rebuild the graph from any artifact URI, so how
it is filled does not depend on builds. Both sources walk **jobs** in timestamp
order and share one checkpoint: the ``timestamp`` of the last job indexed, plus
the ``item_ids`` already indexed *at exactly that instant*. A scan reads from the
timestamp *inclusive*, so a job sharing the boundary instant is never lost, and
skips the ids listed, so a caught-up scan neither re-indexes the boundary job nor
rewrites the checkpoint. The id list only ever holds the jobs tied on one
instant: it is reset whenever the timestamp moves forward.

The sink is always the index (``DBLineageStore``). The source follows how the
server was started:

``admin_db`` (standalone)
    Successful ``gb_targets`` rows, ordered by ``finished_at``. A target run *is*
    a job there (``job_id = targetrun.uuid``). This is not ``LineageWatcher``:
    that one walks builds for ``lineage-watch`` and is left alone.

``lineage_store`` (every other deployment)
    The configured lineage store read back out. For W&B that is the project's
    runs, ordered by ``createdAt``. When the configured store *is* the index
    (``GBSERVER_LINEAGE_PROVIDER=db``) source and sink are the same table, so the
    indexer says so and does nothing; ``none`` has nothing to read.

No checkpoint means "from the beginning". ``--base-timestamp`` seeds one
(``from-latest``, ``all`` or an ISO-8601 timestamp), only when none exists yet.
A timestamp is the natural anchor: it is what the checkpoint already holds, it
means the same thing in both sources, and "index everything since this date" is
the question an operator actually has.
"""

import threading
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from gbserver.lineage.db_jobstats import DBLineageStore
from gbserver.lineage.jobstats import (
    LINEAGE_PROVIDER_DB,
    LINEAGE_PROVIDER_NONE,
    _resolve_lineage_provider,
)
from gbserver.lineage.lineage_reconciler import (
    _SCAN_PAGE_SIZE,
    _successful_targets_page,
    as_aware,
)
from gbserver.lineage.lineage_seeding import (
    SEED_ALL,
    SEED_FROM_LATEST,
    LineageSeedError,
)
from gbserver.storage.singleton_storage import SingletonAdminStorage, get_admin_storage
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

INDEXER_SOURCE_ADMIN_DB = "admin_db"
INDEXER_SOURCE_LINEAGE_STORE = "lineage_store"

# One key for both sources: the source is fixed by the startup mode, and the
# value has the same shape either way. Separate from every lineage-watch key.
INDEXER_CHECKPOINT_KEY = "lineage_index_checkpoint"
INDEXER_CHECKPOINT_VERSION = 1

# Attempts at one job before it is skipped. The scan stops at a failing job
# rather than stepping past it, so without a bound one unreadable job would pin
# the checkpoint forever.
_MAX_JOB_ATTEMPTS = 5

_WANDB_PAGE_SIZE = 200

# Config keys WandBLineageService.emit_event copies out of job_details; read back
# here into the same place. Duplicated rather than imported so this module does
# not import wandb_service (and so wandb) in admin_db mode.
_JOB_DETAIL_KEYS = (
    "job_id",
    "job_type",
    "category",
    "job_status",
    "job_started_at",
    "job_completed_at",
    "release_id",
    "owner",
    "job_output_stats",
)
_PASSTHROUGH_FACET_KEYS = ("job_input_params", "execution_stats")


def resolve_indexer_source() -> str:
    """Pick the source from how the server was started.

    Standalone reads ``gb_targets`` directly (``admin_db``); every other
    deployment reads the configured lineage store (``lineage_store``).
    Deliberately not configurable: the mode already says which one is right.
    """
    from gbcommon.types.gbenvconfig import is_standalone

    if is_standalone():
        return INDEXER_SOURCE_ADMIN_DB
    return INDEXER_SOURCE_LINEAGE_STORE


def create_indexer(
    source: str,
    monitoring_interval: float = 30.0,
    sink: Optional[DBLineageStore] = None,
):
    """Build the indexer loop for ``source``, or ``None`` when it has nothing to do.

    The returned object has ``start()``, ``stop()``, a ``stop_event`` to block on,
    and ``seed_if_absent()``.
    """
    sink = sink or DBLineageStore()
    if source == INDEXER_SOURCE_ADMIN_DB:
        return TargetLineageIndexer(monitoring_interval=monitoring_interval, sink=sink)

    provider = _resolve_lineage_provider()
    if provider == LINEAGE_PROVIDER_DB:
        logger.warning(
            "The configured lineage store is the lineage index itself "
            "(GBSERVER_LINEAGE_PROVIDER=db): lineage-watch already writes "
            "gb_lineage_index, so there is nothing to copy. The indexer is a no-op."
        )
        return None
    if provider == LINEAGE_PROVIDER_NONE:
        logger.info(
            "No lineage store is configured (GBSERVER_LINEAGE_PROVIDER=none); the "
            "indexer has nothing to read."
        )
        return None
    return WandBLineageIndexer(monitoring_interval=monitoring_interval, sink=sink)


def _parse_ts(value: str) -> datetime:
    """Parse a checkpoint timestamp; naive is read as local, like ``as_aware``."""
    return as_aware(datetime.fromisoformat(value.replace("Z", "+00:00")))


class JobLineageIndexer:
    """Walks one source's jobs in timestamp order into the index.

    Subclasses say how to list jobs from a timestamp and how to index one; the
    checkpoint, seeding, retry bound and lifecycle live here, so both sources
    advance the same way.

    **The checkpoint** is written after each job that changes it, so a crash
    mid-scan resumes at the job it was on. It is ``(timestamp, item_ids)``: jobs
    at the checkpoint instant whose id is listed are done and skipped; any other
    job at or after it is indexed. Two jobs on the same instant are therefore both
    indexed, and a crash between them resumes at the second. It never moves
    backwards, and a scan with nothing new writes nothing.

    **A pending job stops the scan.** A source may list a job whose lineage is
    not complete yet; advancing past it would skip whatever lands later.
    """

    _thread_name = "lineage-indexer"

    def __init__(
        self,
        monitoring_interval: float = 30.0,
        sink: Optional[DBLineageStore] = None,
    ) -> None:
        self.monitoring_interval = monitoring_interval
        self.stop_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None
        self._sink = sink or DBLineageStore()
        self._failed_attempts: Dict[str, int] = {}

    # -- Source (per subclass) -----------------------------------------------

    def _jobs_since(
        self, storage: SingletonAdminStorage, timestamp: Optional[str]
    ) -> Iterable[Any]:
        """Jobs at or after ``timestamp`` (all when ``None``), oldest first."""
        raise NotImplementedError

    def _timestamp(self, job: Any) -> str:
        """The instant a job is ordered by, as the checkpoint stores it."""
        raise NotImplementedError

    def _item_id(self, job: Any) -> str:
        """The source item's id, for logs and counting failed attempts."""
        raise NotImplementedError

    def _is_pending(self, job: Any) -> bool:
        return False

    def _index(self, storage: SingletonAdminStorage, job: Any) -> bool:
        """Index one job; return whether it was written (vs. nothing to write)."""
        raise NotImplementedError

    def _latest_job(self, storage: SingletonAdminStorage) -> Optional[Any]:
        raise NotImplementedError

    def _format_timestamp(self, instant: datetime) -> str:
        """An instant spelled as this source's jobs spell ``_timestamp``.

        A seeded checkpoint must read like one a scan wrote, so the same mark
        does not look different depending on who wrote it.
        """
        raise NotImplementedError

    # -- Checkpoint ----------------------------------------------------------

    @staticmethod
    def read_checkpoint(storage: SingletonAdminStorage) -> Optional[dict]:
        value = storage.kv_pair_storage.get_value(INDEXER_CHECKPOINT_KEY)
        if not value or not value.get("timestamp"):
            return None
        return value

    @staticmethod
    def _write_timestamp(
        storage: SingletonAdminStorage,
        timestamp: str,
        item_ids: Optional[List[str]] = None,
    ) -> None:
        storage.kv_pair_storage.set_value(
            INDEXER_CHECKPOINT_KEY,
            {
                "timestamp": timestamp,
                "item_ids": list(item_ids or []),
                "version": INDEXER_CHECKPOINT_VERSION,
            },
        )

    def seed_if_absent(self, storage: SingletonAdminStorage, spec: str) -> bool:
        """Place the checkpoint at a timestamp, only when there is none yet.

        ``spec`` is ``from-latest`` (the newest job's timestamp), ``all``, or an
        ISO-8601 timestamp; a timestamp without an offset is read as local time,
        like every other naive instant here. ``all`` writes nothing: no
        checkpoint already means "from the beginning". Seed-if-absent so the flag
        is safe to leave in a pod spec. Returns True if written.

        Scans read from the checkpoint inclusively, so a job at exactly the
        seeded instant is indexed.

        Raises:
            LineageSeedError: When ``from-latest`` finds no job, or ``spec`` is
                not a timestamp.
        """
        existing = self.read_checkpoint(storage)
        if existing is not None:
            logger.info(
                "Lineage index checkpoint %s already exists (%s); ignoring the "
                "requested seed (%s).",
                INDEXER_CHECKPOINT_KEY,
                existing,
                spec,
            )
            return False
        if spec == SEED_ALL:
            return False

        if spec == SEED_FROM_LATEST:
            job = self._latest_job(storage)
            if job is None:
                raise LineageSeedError(
                    "No job found in the source; nothing to anchor a checkpoint at."
                )
            timestamp = self._timestamp(job)
        else:
            try:
                instant = _parse_ts(spec)
            except ValueError as exc:
                raise LineageSeedError(
                    f"{spec!r} is not '{SEED_FROM_LATEST}', '{SEED_ALL}', or an "
                    "ISO-8601 timestamp (e.g. 2026-09-01T00:00:00+00:00)."
                ) from exc
            timestamp = self._format_timestamp(instant)
        self._write_timestamp(storage, timestamp)
        logger.info(
            "Seeded lineage index checkpoint %s at %s.",
            INDEXER_CHECKPOINT_KEY,
            timestamp,
        )
        return True

    # -- Scanning ------------------------------------------------------------

    def scan_once(self, storage: Optional[SingletonAdminStorage] = None) -> int:
        """Index every job since the checkpoint; return how many were written.

        A job that fails is retried on the next scan, up to ``_MAX_JOB_ATTEMPTS``
        times, and then skipped with an error log.
        """
        storage = storage or get_admin_storage()
        checkpoint = self.read_checkpoint(storage)
        mark: Optional[Tuple[datetime, List[str]]] = (
            (_parse_ts(checkpoint["timestamp"]), list(checkpoint.get("item_ids") or []))
            if checkpoint
            else None
        )
        indexed = 0
        for job in self._jobs_since(storage, checkpoint and checkpoint["timestamp"]):
            if self.stop_event.is_set():
                break
            item_id = self._item_id(job)
            job_ts = _parse_ts(self._timestamp(job))
            if mark is not None and (
                job_ts < mark[0] or (job_ts == mark[0] and item_id in mark[1])
            ):
                # Behind the mark, or on it and already done.
                continue
            if self._is_pending(job):
                logger.debug("Job %s still pending; stopping the scan", item_id)
                break
            try:
                if self._index(storage, job):
                    indexed += 1
            except Exception:
                attempts = self._failed_attempts.get(item_id, 0) + 1
                self._failed_attempts[item_id] = attempts
                if attempts < _MAX_JOB_ATTEMPTS:
                    logger.exception(
                        "Failed to index job %s (attempt %d/%d); retrying next scan.",
                        item_id,
                        attempts,
                        _MAX_JOB_ATTEMPTS,
                    )
                    break
                logger.error(
                    "Giving up on job %s after %d attempts; its lineage is NOT in "
                    "the index.",
                    item_id,
                    attempts,
                )
            self._failed_attempts.pop(item_id, None)
            if mark is not None and job_ts == mark[0]:
                ids = mark[1] + [item_id]
            else:
                ids = [item_id]
            self._write_timestamp(storage, self._timestamp(job), ids)
            mark = (job_ts, ids)
        return indexed

    # -- Lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if self.worker_thread is not None and self.worker_thread.is_alive():
            logger.error("lineage indexer thread is already running")
            return
        self.worker_thread = threading.Thread(
            target=self._run, name=self._thread_name, daemon=True
        )
        self.worker_thread.start()
        logger.info("%s started", type(self).__name__)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                count = self.scan_once()
                if count:
                    logger.info("Indexed %d lineage job(s)", count)
            except Exception:
                logger.exception("Lineage index scan failed; retrying next scan")
            self.stop_event.wait(self.monitoring_interval)

    def stop(self, timeout: float = 5.0) -> None:
        self.stop_event.set()
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=timeout)


class TargetLineageIndexer(JobLineageIndexer):
    """Standalone source: successful ``gb_targets`` rows by ``finished_at``.

    Every page is read, never stopping at the first row behind the checkpoint:
    SQLite orders ``finished_at`` as text and the column holds two spellings
    (``' '`` and ``'T'`` separators), so a newer row can sort below an older one
    (see ``select_builds_from_checkpoint``, which reads every page for the same
    reason).

    Builds play no part in the walk; a target's ``build_id`` is only used to load
    the target's own lineage. Targets with no ``finished_at`` are not finished
    and are not listed, so there is nothing pending to stop at. Targets with no
    artifacts at all have no edge to index and are passed over (the checkpoint
    still advances past them).
    """

    def _timestamp(self, job: StoredTargetRun) -> str:
        return as_aware(job.finished_at).isoformat()

    def _item_id(self, job: StoredTargetRun) -> str:
        return job.uuid

    def _jobs_since(
        self, storage: SingletonAdminStorage, timestamp: Optional[str]
    ) -> List[StoredTargetRun]:
        cutoff = _parse_ts(timestamp) if timestamp else None
        selected: List[StoredTargetRun] = []
        page_index = 0
        while True:
            page = _successful_targets_page(storage, page_index)
            if not page:
                break
            for target in page:
                # Unfinished rows are skipped, never a stop: NULLs can interleave.
                if target.finished_at is None:
                    continue
                if cutoff is not None and as_aware(target.finished_at) < cutoff:
                    continue
                selected.append(target)
            if len(page) < _SCAN_PAGE_SIZE:
                break
            page_index += 1
        # The DB sort can disagree with instant order (SQLite stores wall-clock
        # text), so order by the aware instant here; uuid only makes ties stable.
        selected.sort(key=lambda t: (as_aware(t.finished_at), t.uuid))
        return selected

    def _index(self, storage: SingletonAdminStorage, job: StoredTargetRun) -> bool:
        if not job.input_artifacts and not any(job.output_artifacts.values()):
            return False
        # Recorded only if *this* system wrote it: a copy another source imported
        # first still needs granite.build's view merged in. See recorded_by_self.
        if self._sink.row_storage.has_rows_for_job(
            job.uuid
        ) and self._sink.recorded_by_self([job.uuid]):
            return False
        self._sink.add_jobstats_for_build_target(
            storage, build_id=job.build_id, target_id=job.uuid
        )
        return True

    def _latest_job(self, storage: SingletonAdminStorage) -> Optional[StoredTargetRun]:
        page_index = 0
        while True:
            page = _successful_targets_page(storage, page_index)
            if not page:
                return None
            finished = [t for t in page if t.finished_at is not None]
            if finished:
                return max(finished, key=lambda t: as_aware(t.finished_at))
            page_index += 1

    def _format_timestamp(self, instant: datetime) -> str:
        # Same form as _timestamp: the aware isoformat, offset kept as given.
        return as_aware(instant).isoformat()


def _tags_of(run: Any) -> Dict[str, str]:
    tags: Dict[str, str] = {}
    for tag in run.tags or []:
        if "=" in tag:
            key, value = tag.split("=", 1)
            tags[key] = value
    return tags


def wandb_run_to_job(run: Any) -> Optional[dict]:
    """Rebuild the job entry a W&B lineage run was recorded from.

    The inverse of ``WandBLineageService.emit_event``, in the shape
    ``DBLineageStore.write_job`` takes -- the same shape the admin-DB path builds,
    so both sources decompose identically.

    The namespace comes from ``config.job_namespace``, NOT from the run's
    entity/project the way ``_get_run_lineage`` fills it: the index splits it to
    recover the space for authorization, and ``entity/project`` would prune every
    node as belonging to a space nobody is in.

    Returns ``None`` for a run with no ``job_id`` -- not a granite.build lineage run
    (anything else logged to the project), so there is nothing to index.
    """
    from gbserver.lineage.wandb_service import WandBLineageService

    config = dict(run.config or {})
    job_details = {k: config[k] for k in _JOB_DETAIL_KEYS if k in config}
    if not job_details.get("job_id"):
        return None

    facets: Dict[str, Any] = {
        k: config[k] for k in _PASSTHROUGH_FACET_KEYS if config.get(k) is not None
    }
    tags = _tags_of(run)
    if tags:
        facets["tags"] = tags
    if config.get("source_code_url"):
        facets["source_code"] = {
            "url": config["source_code_url"],
            "commit_hash": "",
            "path": "",
        }

    to_dataset = WandBLineageService._artifact_to_openlineage_dataset
    is_system = WandBLineageService._is_wandb_system_artifact
    sources = [to_dataset(a) for a in run.used_artifacts() if not is_system(a)]
    targets = [to_dataset(a) for a in run.logged_artifacts() if not is_system(a)]

    return {
        "job_name": config.get("job_name") or run.name,
        "release_id": job_details.get("release_id", ""),
        "job_details": job_details,
        "job": {"namespace": config.get("job_namespace", ""), "name": run.name},
        "run": {"runId": run.id, "facets": facets},
        "eventType": config.get("event_type", ""),
        "description": config.get("description", ""),
        "sources": sources,
        "targets": targets,
    }


class WandBLineageIndexer(JobLineageIndexer):
    """Lineage-store source: W&B lineage runs by ``createdAt``.

    **Run state is ignored.** A W&B run executes nothing: it is only the record
    ``emit_event`` writes for a job's lineage, so every run is indexed with the
    inputs and outputs it carries.

    **Several runs, one job.** The W&B sink writes one run per output artifact, all
    carrying the target's ``job_id``. Each is indexed on its own and adds its own
    rows under the shared job, which is why dedup here is by the sink's unique
    indexes and not by "does this job already have rows".
    """

    _thread_name = "lineage-indexer-wandb"

    def __init__(
        self,
        monitoring_interval: float = 30.0,
        sink: Optional[DBLineageStore] = None,
        api: Any = None,
    ) -> None:
        super().__init__(monitoring_interval=monitoring_interval, sink=sink)
        self._api = api

    def _wandb_api(self) -> Any:
        if self._api is None:
            import wandb

            from gbserver.types.constants import (
                GBSERVER_WANDB_API_KEY,
                GBSERVER_WANDB_BASE_URL,
            )

            wandb.login(key=GBSERVER_WANDB_API_KEY, host=GBSERVER_WANDB_BASE_URL)
            self._api = wandb.Api()
        return self._api

    @staticmethod
    def _project_path() -> str:
        from gbserver.types.constants import (
            GBSERVER_WANDB_ENTITY,
            GBSERVER_WANDB_PROJECT,
        )

        if GBSERVER_WANDB_ENTITY:
            return f"{GBSERVER_WANDB_ENTITY}/{GBSERVER_WANDB_PROJECT}"
        return GBSERVER_WANDB_PROJECT

    def _timestamp(self, job: Any) -> str:
        return job.created_at

    def _item_id(self, job: Any) -> str:
        # Per run: the runs of one job share its job_id but fail independently.
        return job.id

    def _jobs_since(
        self, storage: SingletonAdminStorage, timestamp: Optional[str]
    ) -> Iterable[Any]:
        filters = {"createdAt": {"$gte": timestamp}} if timestamp else {}
        return self._wandb_api().runs(
            self._project_path(),
            filters=filters,
            order="+created_at",
            per_page=_WANDB_PAGE_SIZE,
        )

    def _index(self, storage: SingletonAdminStorage, job: Any) -> bool:
        entry = wandb_run_to_job(job)
        if entry is None:
            return False
        tags = _tags_of(job)
        self._sink.write_job(
            entry,
            build_id=tags.get("build_id", ""),
            target_run_uuid=tags.get("target_id", ""),
        )
        return True

    def _first_run(self, **kwargs: Any) -> Optional[Any]:
        return next(iter(self._wandb_api().runs(self._project_path(), **kwargs)), None)

    def _latest_job(self, storage: SingletonAdminStorage) -> Optional[Any]:
        return self._first_run(order="-created_at", per_page=1)

    def _format_timestamp(self, instant: datetime) -> str:
        # W&B's createdAt is UTC with a trailing Z, and the $gte filter compares
        # against it; spell a seed the same way.
        return instant.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
