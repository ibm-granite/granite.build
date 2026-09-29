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

The sink is always the index (``DBLineageStore``). What varies is where new
lineage is read from, chosen by ``GBSERVER_LINEAGE_INDEXER_SOURCE``:

``admin_db`` (the standalone default)
    Reads ``gb_build``/``gb_targets`` directly. This is the ``LineageWatcher``
    reconciliation loop pointed at the index sink, under its own checkpoint keys
    so it never shares a mark with a ``lineage-watch`` recording elsewhere.

``lineage_store`` (the default everywhere else)
    Reads the configured lineage store back out. For W&B that is the project's
    runs, paged by ``createdAt`` from a checkpoint. When the configured store *is*
    the index (``GBSERVER_LINEAGE_PROVIDER=db``) source and sink are the same
    table, so there is nothing to copy: the indexer says so and does nothing,
    rather than re-writing every row onto itself. ``none`` has nothing to read and
    is a no-op for the same reason.

Both modes are idempotent at the sink -- jobs are unique by ``job_id`` and rows by
``(job_id, input, output)`` -- so the overlap each scan re-reads on purpose costs a
rejected insert, never a duplicate.
"""

import os
import threading
from typing import Any, Dict, List, Optional

from gbserver.lineage.db_jobstats import DBLineageStore
from gbserver.lineage.jobstats import (
    LINEAGE_PROVIDER_DB,
    LINEAGE_PROVIDER_NONE,
    _resolve_lineage_provider,
)
from gbserver.lineage.lineage_watcher import LineageWatcher
from gbserver.storage.singleton_storage import SingletonAdminStorage, get_admin_storage
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

INDEXER_SOURCE_ADMIN_DB = "admin_db"
INDEXER_SOURCE_LINEAGE_STORE = "lineage_store"
VALID_INDEXER_SOURCES = (INDEXER_SOURCE_ADMIN_DB, INDEXER_SOURCE_LINEAGE_STORE)

# Separate from the lineage-watch keys: in standard mode lineage-watch records to
# W&B from gb_build while this indexer may read gb_build too, and one shared mark
# would let either advance past lineage only the other had recorded.
INDEXER_CHECKPOINT_KEY = "lineage_index_latest_build_id"
INDEXER_DROPPED_KEY = "lineage_index_dropped_target_ids"
INDEXER_WANDB_CHECKPOINT_KEY = "lineage_index_wandb_created_at"
INDEXER_WANDB_CHECKPOINT_VERSION = 1

# Attempts at one W&B run before it is skipped. The scan stops at a failing run
# rather than stepping past it, so without a bound one unreadable run would pin
# the checkpoint forever.
_MAX_RUN_ATTEMPTS = 5

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


class UnknownIndexerSource(ValueError):
    """``GBSERVER_LINEAGE_INDEXER_SOURCE`` names no known source."""


def resolve_indexer_source(override: Optional[str] = None) -> str:
    """Resolve which source the indexer reads.

    ``override`` (the CLI flag) wins, then the environment variable, then the
    deployment default: ``admin_db`` standalone, ``lineage_store`` otherwise.
    Unknown values raise rather than fall back, for the same reason
    ``_resolve_lineage_provider`` does.
    """
    from gbcommon.types.gbenvconfig import is_standalone
    from gbserver.types.constants import ENV_VAR_PREFIX

    default = (
        INDEXER_SOURCE_ADMIN_DB if is_standalone() else INDEXER_SOURCE_LINEAGE_STORE
    )
    source = (
        override or os.getenv(ENV_VAR_PREFIX + "_LINEAGE_INDEXER_SOURCE") or default
    ).strip()
    if source not in VALID_INDEXER_SOURCES:
        raise UnknownIndexerSource(
            f"{ENV_VAR_PREFIX}_LINEAGE_INDEXER_SOURCE is {source!r}; expected one "
            f"of {', '.join(VALID_INDEXER_SOURCES)}"
        )
    return source


def create_indexer(
    source: str,
    monitoring_interval: float = 30.0,
    sink: Optional[DBLineageStore] = None,
):
    """Build the indexer loop for ``source``, or ``None`` when it has nothing to do.

    The returned object has the watcher's lifecycle: ``start()``, ``stop()`` and a
    ``stop_event`` to block on.
    """
    sink = sink or DBLineageStore()
    if source == INDEXER_SOURCE_ADMIN_DB:
        return LineageWatcher(
            monitoring_interval=monitoring_interval,
            store=sink,
            checkpoint_key=INDEXER_CHECKPOINT_KEY,
            dropped_key=INDEXER_DROPPED_KEY,
        )

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
            "indexer has nothing to read. Use --source admin_db to index gb_build."
        )
        return None
    return WandBLineageIndexer(monitoring_interval=monitoring_interval, sink=sink)


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
    if config.get("source_code_url"):
        facets["source_code"] = {
            "url": config["source_code_url"],
            "commit_hash": "",
            "path": "",
        }

    to_dataset = WandBLineageService._artifact_to_openlineage_dataset
    sources = [to_dataset(a) for a in run.used_artifacts()]
    targets = [to_dataset(a) for a in run.logged_artifacts()]

    return {
        "job_name": config.get("job_name") or run.name,
        "release_id": job_details.get("release_id", ""),
        "job_details": job_details,
        "job": {"namespace": config.get("job_namespace", ""), "name": run.name},
        "run": {"runId": run.id, "facets": facets},
        "sources": sources,
        "targets": targets,
    }


class WandBLineageIndexer:
    """Copies new W&B lineage runs into the index, in ``createdAt`` order.

    **The checkpoint** is the ``createdAt`` of the last run indexed, and a scan
    reads from it *inclusive*. Re-reading that one run each scan is the price of
    never losing a run that shares its timestamp; the unique indexes make it free.

    **A running run stops the scan.** ``emit_event`` logs artifacts before it
    finishes the run, so a running run may not have all its outputs yet. Advancing
    past it would skip whatever lands later; stopping keeps it in range until it
    finishes. Runs are finished within one event, so this holds for seconds.

    **Several runs, one job.** The W&B sink writes one run per output artifact, all
    carrying the target's ``job_id``. Each is indexed on its own; they add their
    own rows under the shared job, which is why dedup here is by the sink's unique
    indexes and not by "does this job already have rows".
    """

    def __init__(
        self,
        monitoring_interval: float = 30.0,
        sink: Optional[DBLineageStore] = None,
        api: Any = None,
    ) -> None:
        self.monitoring_interval = monitoring_interval
        self.stop_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None
        self._sink = sink or DBLineageStore()
        self._api = api
        self._failed_attempts: Dict[str, int] = {}

    # -- W&B access ----------------------------------------------------------

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

    def _runs_since(self, created_at: Optional[str]) -> List[Any]:
        filters = {"createdAt": {"$gte": created_at}} if created_at else {}
        return self._wandb_api().runs(
            self._project_path(),
            filters=filters,
            order="+created_at",
            per_page=_WANDB_PAGE_SIZE,
        )

    # -- Checkpoint ----------------------------------------------------------

    @staticmethod
    def _read_checkpoint(storage: SingletonAdminStorage) -> Optional[str]:
        value = storage.kv_pair_storage.get_value(INDEXER_WANDB_CHECKPOINT_KEY)
        return (value or {}).get("created_at") or None

    @staticmethod
    def _write_checkpoint(storage: SingletonAdminStorage, run: Any) -> None:
        storage.kv_pair_storage.set_value(
            INDEXER_WANDB_CHECKPOINT_KEY,
            {
                "created_at": run.created_at,
                "run_id": run.id,
                "version": INDEXER_WANDB_CHECKPOINT_VERSION,
            },
        )

    # -- Scanning ------------------------------------------------------------

    def scan_once(self, storage: Optional[SingletonAdminStorage] = None) -> int:
        """Index every run created since the checkpoint; return how many were indexed.

        The checkpoint is written after each run, so a crash mid-scan resumes at
        the run it was on. A run that fails is retried on the next scan, up to
        ``_MAX_RUN_ATTEMPTS`` times, and then skipped with an error log.
        """
        storage = storage or get_admin_storage()
        indexed = 0
        for run in self._runs_since(self._read_checkpoint(storage)):
            if self.stop_event.is_set():
                break
            if run.state == "running":
                logger.debug("W&B run %s still running; stopping the scan", run.id)
                break
            try:
                job = wandb_run_to_job(run)
                if job is not None:
                    tags = _tags_of(run)
                    self._sink.write_job(
                        job,
                        build_id=tags.get("build_id", ""),
                        target_run_uuid=tags.get("target_id", ""),
                    )
                    indexed += 1
            except Exception:
                attempts = self._failed_attempts.get(run.id, 0) + 1
                self._failed_attempts[run.id] = attempts
                if attempts < _MAX_RUN_ATTEMPTS:
                    logger.exception(
                        "Failed to index W&B run %s (attempt %d/%d); retrying next "
                        "scan.",
                        run.id,
                        attempts,
                        _MAX_RUN_ATTEMPTS,
                    )
                    break
                logger.error(
                    "Giving up on W&B run %s after %d attempts; its lineage is NOT "
                    "in the index.",
                    run.id,
                    attempts,
                )
            self._failed_attempts.pop(run.id, None)
            self._write_checkpoint(storage, run)
        return indexed

    # -- Lifecycle (same shape as LineageWatcher) ----------------------------

    def start(self) -> None:
        if self.worker_thread is not None and self.worker_thread.is_alive():
            logger.error("lineage indexer thread is already running")
            return
        self.worker_thread = threading.Thread(
            target=self._run, name="lineage-indexer", daemon=True
        )
        self.worker_thread.start()
        logger.info("WandBLineageIndexer started")

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                count = self.scan_once()
                if count:
                    logger.info("Indexed %d W&B lineage run(s)", count)
            except Exception:
                logger.exception("W&B lineage index scan failed; retrying next scan")
            self.stop_event.wait(self.monitoring_interval)

    def stop(self, timeout: float = 5.0) -> None:
        self.stop_event.set()
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=timeout)
