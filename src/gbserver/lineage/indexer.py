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
    runs, ordered by ``createdAt``.

``lineage_job`` (``GBSERVER_LINEAGE_PROVIDER=db``)
    ``gb_lineage_job`` records, ordered by ``recorded_at``.

**No checkpoint** means "from the beginning" in standalone, where the indexer
stays automatic -- except for W&B, whose project is shared with every other
deployment: there the first scan seeds ``from-latest`` instead of copying the
whole project into the local index. Everywhere else it means "index nothing yet": the indexer logs
once and re-reads the checkpoint every scan until ``gbserver lineage-index-init
--base-timestamp`` seeds one (``from-latest``, ``all`` -- an explicit
:data:`FROM_BEGINNING` mark -- or an ISO-8601 timestamp), as ``lineage-init``
does for the watcher. The command works in standalone too. A timestamp is the
natural anchor: it is what the checkpoint already holds, it
means the same thing in both sources, and "index everything since this date" is
the question an operator actually has.
"""

import os
import sys
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from gbcommon.types.gbenvconfig import is_standalone
from gbserver.lineage.db_jobstats import DBLineageStore
from gbserver.lineage.jobstats import (
    LINEAGE_PROVIDER_DB,
    LINEAGE_PROVIDER_NONE,
    LINEAGE_PROVIDER_WANDB,
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
from gbserver.lineage.row_indexing import LineageRowIndexer
from gbserver.storage.singleton_storage import SingletonAdminStorage, get_admin_storage
from gbserver.storage.storage import Pagination, QueryControl, SortOrder
from gbserver.storage.stored_lineage_job import StoredLineageJob
from gbserver.storage.stored_lineage_row import JobStore
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

INDEXER_SOURCE_ADMIN_DB = "admin_db"
INDEXER_SOURCE_LINEAGE_STORE = "lineage_store"
INDEXER_SOURCE_LINEAGE_JOB = "lineage_job"

# Checkpoint keys are per-source to avoid mixing their different sync semantics.
# The key is determined by which provider/source is being indexed from.
# These are separate from every lineage-watch key.
INDEXER_CHECKPOINT_PREFIX = "lineage_index_checkpoint"
INDEXER_CHECKPOINT_VERSION = 1

# The checkpoint ``all`` writes: "index from the beginning", said explicitly
# because outside standalone no checkpoint means "index nothing yet". Stored in
# each source's own timestamp form; a scan reading it does a full read.
FROM_BEGINNING = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _checkpoint_key_for_provider(provider: str) -> str:
    """Build the checkpoint key for a lineage provider.

    The suffix is the ``job_store`` value the provider's indexer writes, so a
    checkpoint names the same store as the rows it tracks.
    """
    if provider == LINEAGE_PROVIDER_NONE:
        return f"{INDEXER_CHECKPOINT_PREFIX}:{JobStore.TARGETS.value}"
    if provider == LINEAGE_PROVIDER_DB:
        return f"{INDEXER_CHECKPOINT_PREFIX}:{JobStore.LINEAGE_JOB.value}"
    if provider == LINEAGE_PROVIDER_WANDB:
        return f"{INDEXER_CHECKPOINT_PREFIX}:{JobStore.WANDB.value}"
    return INDEXER_CHECKPOINT_PREFIX  # fallback


# Attempts at one job before it is skipped. The scan stops at a failing job
# rather than stepping past it, so without a bound one unreadable job would pin
# the checkpoint forever. Only failures of the job itself count: a storage or
# network outage (see _is_transient_error) is waited out however long it lasts,
# since skipping a job over one would leave it out of the index for good.
_MAX_JOB_ATTEMPTS = 5

_WANDB_PAGE_SIZE = 200

# A W&B run still "running" is still receiving its artifacts and job_id (the sink
# logs them asynchronously until finish()), so the checkpoint waits on it -- but
# only this long, so a run whose writer died is eventually indexed as it is.
_WANDB_RUNNING_MAX_AGE_S = 6 * 3600
# How long one running lineage run may hold the checkpoint before it is warned
# about (once per run): past this its writer has likely died. Runs after it are
# still indexed (see _MAX_DONE_AHEAD); only the checkpoint waits.
_WANDB_PINNED_WARN_AFTER_S = 30 * 60
# Jobs indexed past a held (pending) job, remembered in the checkpoint so a later
# scan does not redo them while the checkpoint waits behind the held job. Past
# this many the scan stops instead, as it did before jobs could be held.
_MAX_DONE_AHEAD = 5000
# wandb_service.LINEAGE_RUN_JOB_TYPE, restated so this module does not import wandb.
_WANDB_LINEAGE_RUN_JOB_TYPE = "gb-lineage-event"


# How old a job's timestamp must be before the scan takes it. The timestamp is
# stamped before the row commits (finished_at when the event is created,
# recorded_at in Python), so rows can commit out of timestamp order; a checkpoint
# moved to the newest timestamp would skip one that commits later behind it.
# Only taking settled jobs keeps the checkpoint at most ``now - grace``. The
# default outlasts the writers' own retry budget (10 attempts, up to 30s apart, in
# sql_storage), since a write retried that long commits that far behind its
# stamp. A row
# that commits later than the grace behind its timestamp (a long event backlog,
# clock skew between pods) is still skipped until a full reindex; raise
# GBSERVER_LINEAGE_INDEX_SETTLE_GRACE_S where that is expected. It sizes the
# gb_targets source only: W&B needs none and gb_lineage_job a few seconds (see
# their settle_grace_s).
def _settle_grace_from_env(default: float = 360.0) -> float:
    raw = os.environ.get("GBSERVER_LINEAGE_INDEX_SETTLE_GRACE_S", "")
    try:
        return float(raw) if raw.strip() else default
    except ValueError:
        return default


_SETTLE_GRACE_S = _settle_grace_from_env()

# Checkpoint writes are batched: one per job turned a 300k-record source into 300k
# gb_kv updates per scan. Losing a batch on a crash only re-reads it, and the
# re-read is deduplicated by the sink.
_CHECKPOINT_EVERY = 500

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
    "submitted_by",
    "job_output_stats",
)
_PASSTHROUGH_FACET_KEYS = ("job_input_params", "execution_stats")


def resolve_indexer_source() -> str:
    """Pick the source from ``GBSERVER_LINEAGE_PROVIDER``, in every mode.

    The index is filled from wherever the sink records lineage: ``none`` records
    nowhere, so ``gb_targets`` (``admin_db``) is read; ``db`` records to
    ``gb_lineage_job``; ``wandb`` to W&B (``lineage_store``). Standalone only
    changes the provider's default (``none``), never this mapping.
    """
    provider = _resolve_lineage_provider()
    if provider == LINEAGE_PROVIDER_DB:
        return INDEXER_SOURCE_LINEAGE_JOB
    if provider == LINEAGE_PROVIDER_WANDB:
        return INDEXER_SOURCE_LINEAGE_STORE
    return INDEXER_SOURCE_ADMIN_DB


def create_indexer(
    source: str,
    monitoring_interval: float = 30.0,
    sink: Optional[DBLineageStore] = None,
    rows: Optional[LineageRowIndexer] = None,
):
    """Build the indexer loop for ``source``, or ``None`` when it has nothing to do.

    The returned object has ``start()``, ``stop()``, a ``stop_event`` to block on,
    and ``seed_if_absent()``.

    ``sink`` builds a target's job entries (the ``gb_targets`` source); ``rows``
    writes the index the indexer owns. See :class:`JobLineageIndexer`.
    """
    sink = sink or DBLineageStore()
    rows = rows or LineageRowIndexer()
    if source == INDEXER_SOURCE_ADMIN_DB:
        return TargetLineageIndexer(
            monitoring_interval=monitoring_interval, sink=sink, rows=rows
        )
    if source == INDEXER_SOURCE_LINEAGE_JOB:
        return LineageJobIndexer(
            monitoring_interval=monitoring_interval, sink=sink, rows=rows
        )
    if source == INDEXER_SOURCE_LINEAGE_STORE:
        return WandBLineageIndexer(
            monitoring_interval=monitoring_interval, sink=sink, rows=rows
        )
    logger.warning("Unknown lineage indexer source %r; not indexing.", source)
    return None


def _build_tags(storage: SingletonAdminStorage, build_id: str) -> List[str]:
    """The build's user tags, or none when it cannot be read.

    Tolerant on purpose: the tags label a row, so losing them must not cost the
    whole execution its place in the index.
    """
    try:
        build = storage.build_storage.get_by_uuid(build_id)
    except Exception:
        return []
    tags = getattr(build, "tags", None)
    return list(tags) if isinstance(tags, list) else []


def _parse_ts(value: str) -> datetime:
    """Parse a checkpoint timestamp; naive is read as local, like ``as_aware``."""
    return as_aware(datetime.fromisoformat(value.replace("Z", "+00:00")))


def _parse_utc_ts(value: str) -> datetime:
    """Parse a timestamp whose source is UTC; naive is read as UTC, not local."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


# Exception names (anywhere in the chain) of failures that are about reaching the
# store, not about the job: retrying them later can succeed, so they never count
# toward _MAX_JOB_ATTEMPTS. Matched by name so this module does not import
# sqlalchemy, and so a driver's subclass (psycopg's OperationalError) matches too.
_TRANSIENT_ERROR_NAMES = frozenset(
    {
        "OperationalError",
        "InterfaceError",
        "DisconnectionError",
        "TimeoutError",
        "ConnectionError",
        "CommError",
    }
)
_TRANSIENT_ERROR_MARKERS = (
    "database is locked",
    "could not connect",
    "connection refused",
    "connection reset",
    "server closed the connection",
    "timed out",
)
# Messages that mark an error as the job's or the schema's, even when its class
# (sqlalchemy's OperationalError covers both) or another marker reads as an
# outage: retrying cannot fix them, so they must count toward the give-up instead
# of pinning the checkpoint forever.
_DETERMINISTIC_ERROR_MARKERS = (
    "no such table",
    "no such column",
    "syntax error",
    "does not exist",
    "value too long",
    "statement timeout",
)
# After this many transient failures in a row on the same job the warning becomes
# an error: still not counted (an outage must skip nothing), but no longer quiet.
_TRANSIENT_ALERT_AFTER = 20


def _is_transient_error(exc: BaseException) -> bool:
    """Whether a failure is an outage of the store rather than a bad job.

    Unlike ``is_permanent_sink_failure``, unrecognized errors are *not* transient
    here: a job that keeps failing for its own reasons must eventually be skipped,
    or it would pin the checkpoint forever.
    """
    seen: set = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if any(m in str(current).lower() for m in _DETERMINISTIC_ERROR_MARKERS):
            return False
        current = current.__cause__ or current.__context__
    seen.clear()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (ConnectionError, TimeoutError)):
            return True
        if _TRANSIENT_ERROR_NAMES & {c.__name__ for c in type(current).__mro__}:
            return True
        if getattr(current, "connection_invalidated", False):
            return True
        message = str(current).lower()
        if any(marker in message for marker in _TRANSIENT_ERROR_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False


class JobLineageIndexer:
    """Walks one source's jobs in timestamp order into the index.

    Subclasses say how to list jobs from a timestamp and how to index one; the
    checkpoint, seeding, retry bound and lifecycle live here, so both sources
    advance the same way.

    **The checkpoint** is written every ``_CHECKPOINT_EVERY`` jobs and when the
    scan ends, so a crash mid-scan re-reads at most one batch (deduplicated by the
    sink). It is ``(timestamp, item_ids)``: jobs at the checkpoint instant whose id
    is listed are done and skipped; any other job at or after it is indexed. Two
    jobs on the same instant are therefore both indexed. A scan only moves it
    forward, and a scan with nothing new writes nothing; only an explicit
    ``seed_if_absent(force=True)`` moves it back. A scan re-reads the checkpoint
    before each write and stops if someone else changed it (a forced re-seed, or
    another indexer during a rollout), rather than overwrite it.

    **A pending job holds the checkpoint, not the scan.** A source may list a job
    whose lineage is not complete yet (a W&B run still ``running``); advancing
    past it would skip whatever lands later. The checkpoint stays behind it while
    the jobs after it are indexed and listed as ``done_ahead`` in the checkpoint,
    up to ``_MAX_DONE_AHEAD``. An unsettled job (younger than the settle grace)
    still stops the scan.

    **Two collaborators, two directions.** Jobs are *read* from whichever source
    this subclass is for -- ``gb_targets``, ``gb_lineage_job`` or the W&B API --
    and index rows are *written* through ``self._rows``, which owns
    ``gb_lineage_index``. The index belongs to the indexer, not to any lineage
    store: a store answers for its own source only. ``self._sink`` is the ``db``
    store, used only by the ``gb_targets`` source to build a target's job entries;
    it never writes, and no source copies jobs into ``gb_lineage_job``.
    """

    PROVIDER: Optional[str] = None

    _thread_name = "lineage-indexer"

    def __init__(
        self,
        monitoring_interval: float = 30.0,
        sink: Optional[DBLineageStore] = None,
        rows: Optional[LineageRowIndexer] = None,
    ) -> None:
        self.monitoring_interval = monitoring_interval
        self.stop_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None
        self._sink = sink or DBLineageStore()
        self._rows = rows or LineageRowIndexer()
        self._failed_attempts: Dict[str, int] = {}
        self._transient_streak: Dict[str, int] = {}
        # The checkpoint as this indexer last read or wrote it; a write first checks
        # the stored one still matches, so a concurrent re-seed is not overwritten.
        self._expected_checkpoint: Optional[dict] = None
        # Set when _advance_checkpoint finds the checkpoint changed; ends the scan.
        self._checkpoint_moved = False
        # Whether the "no checkpoint yet" notice was logged, so it is not repeated
        # every scan; reset once a checkpoint appears.
        self._missing_checkpoint_logged = False

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

    def _parse_job_ts(self, value: str) -> datetime:
        """Parse a timestamp in this source's form (``_timestamp``, the checkpoint).

        Naive is local by default, like ``as_aware``; a source whose timestamps are
        zoneless UTC overrides this, so the settle age and the mark comparisons
        agree with how the source wrote them.
        """
        return _parse_ts(value)

    # Instance-overridable so tests can turn the grace off.
    settle_grace_s: float = _SETTLE_GRACE_S

    def _is_settled(self, job: Any) -> bool:
        """Whether the job is old enough that nothing can still commit behind it."""
        age = datetime.now(timezone.utc) - self._parse_job_ts(self._timestamp(job))
        return age.total_seconds() >= self.settle_grace_s

    def _record_failure(self, item_id: str) -> bool:
        """Note a failed attempt at ``item_id``; return whether to give up on it.

        Called from an ``except`` block. A transient failure (the store or network
        is down) is not the job's fault: it is logged and retried next scan without
        counting, so an outage of any length skips nothing.
        """
        exc = sys.exc_info()[1]
        if exc is not None and _is_transient_error(exc):
            streak = self._transient_streak.get(item_id, 0) + 1
            self._transient_streak = {item_id: streak}
            log = logger.error if streak >= _TRANSIENT_ALERT_AFTER else logger.warning
            log(
                "Transient failure %d in a row indexing job %s; retrying next "
                "scan: %s",
                streak,
                item_id,
                exc,
            )
            return False
        self._transient_streak.pop(item_id, None)
        attempts = self._failed_attempts.get(item_id, 0) + 1
        self._failed_attempts[item_id] = attempts
        if attempts < _MAX_JOB_ATTEMPTS:
            logger.exception(
                "Failed to index job %s (attempt %d/%d); retrying next scan.",
                item_id,
                attempts,
                _MAX_JOB_ATTEMPTS,
            )
            return False
        logger.exception(
            "Giving up on job %s after %d attempts; its lineage is NOT in the index.",
            item_id,
            attempts,
        )
        self._failed_attempts.pop(item_id, None)
        return True

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

    def _get_checkpoint_key(self) -> str:
        """Get the checkpoint key for this indexer's provider.

        Keyed off the class, not ``GBSERVER_LINEAGE_PROVIDER``: an indexer reads
        exactly one source, so its checkpoint cannot depend on the environment.
        """
        return _checkpoint_key_for_provider(self.PROVIDER)

    def read_checkpoint(self, storage: SingletonAdminStorage) -> Optional[dict]:
        key = self._get_checkpoint_key()
        value = storage.kv_pair_storage.get_value(key)
        if not value or not value.get("timestamp"):
            return None
        return value

    def _advance_checkpoint(
        self,
        storage: SingletonAdminStorage,
        timestamp: str,
        item_ids: List[str],
        done_ahead: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Write a scan's progress unless the checkpoint changed under the scan.

        Returns False, writing nothing, when the stored checkpoint is no longer the
        one this scan started from or last wrote: a ``--force`` re-seed, or another
        indexer pod overlapping during a rollout. Overwriting it would undo the
        re-seed, so the scan stops and the next one starts from what is stored.
        The write is a compare-and-set against the value just read, so a re-seed
        landing between the check and the write is not overwritten either.
        """
        key = self._get_checkpoint_key()
        raw = storage.kv_pair_storage.get_value(key)
        current = raw if raw and raw.get("timestamp") else None
        if current == self._expected_checkpoint:
            value = self._checkpoint_value(timestamp, item_ids, done_ahead)
            if storage.kv_pair_storage.compare_and_set(key, raw, value):
                self._expected_checkpoint = value
                return True
            current = self.read_checkpoint(storage)
        logger.warning(
            "Lineage index checkpoint %s changed during the scan (%s -> %s); "
            "not overwriting it.",
            key,
            self._expected_checkpoint,
            current,
        )
        self._checkpoint_moved = True
        return False

    @staticmethod
    def _checkpoint_value(
        timestamp: str,
        item_ids: Optional[List[str]],
        done_ahead: Optional[Dict[str, str]] = None,
    ) -> dict:
        value = {
            "timestamp": timestamp,
            "item_ids": list(item_ids or []),
            "version": INDEXER_CHECKPOINT_VERSION,
        }
        if done_ahead:
            # Jobs past a held one, already indexed: ``{item_id: timestamp}``.
            value["done_ahead"] = dict(done_ahead)
        return value

    def _write_timestamp(
        self,
        storage: SingletonAdminStorage,
        timestamp: str,
        item_ids: Optional[List[str]] = None,
    ) -> None:
        storage.kv_pair_storage.set_value(
            self._get_checkpoint_key(), self._checkpoint_value(timestamp, item_ids)
        )

    def _is_from_beginning(self, checkpoint: dict) -> bool:
        """Whether ``checkpoint`` is the explicit "index everything" mark."""
        try:
            return self._parse_job_ts(checkpoint["timestamp"]) <= FROM_BEGINNING
        except (TypeError, ValueError):
            return False

    def _resolve_seed(
        self,
        storage: SingletonAdminStorage,
        spec: str,
        empty_from_beginning: bool = False,
    ) -> str:
        """The checkpoint timestamp ``spec`` names, in this source's form.

        Raises:
            LineageSeedError: When ``from-latest`` finds no job (unless
                ``empty_from_beginning``), or ``spec`` is not a timestamp.
        """
        if spec == SEED_ALL:
            return self._format_timestamp(FROM_BEGINNING)
        if spec == SEED_FROM_LATEST:
            job = self._latest_job(storage)
            if job is None and empty_from_beginning:
                # Nothing to skip yet: every job written from now on is indexed.
                return self._format_timestamp(FROM_BEGINNING)
            if job is None:
                raise LineageSeedError(
                    "No job found in the source; nothing to anchor a checkpoint at."
                )
            return self._timestamp(job)
        try:
            instant = _parse_ts(spec)
        except ValueError as exc:
            raise LineageSeedError(
                f"{spec!r} is not '{SEED_FROM_LATEST}', '{SEED_ALL}', or an "
                "ISO-8601 timestamp (e.g. 2026-09-01T00:00:00+00:00)."
            ) from exc
        return self._format_timestamp(instant)

    def seed_if_absent(
        self,
        storage: SingletonAdminStorage,
        spec: str,
        force: bool = False,
        empty_from_beginning: bool = False,
    ) -> bool:
        """Place the checkpoint at a timestamp, by default only when there is none.

        ``spec`` is ``from-latest`` (the newest job's timestamp), ``all``, or an
        ISO-8601 timestamp; a timestamp without an offset is read as local time,
        like every other naive instant here. ``all`` writes the explicit
        :data:`FROM_BEGINNING` mark: outside standalone a missing checkpoint means
        "index nothing yet", so "everything" has to be said. Returns True if
        written.

        ``force`` replaces an existing checkpoint. The new anchor is resolved
        first, so a seed that cannot be resolved raises and leaves the existing one
        in place. Moving it back re-indexes (deduplicated, not free); moving it
        forward skips jobs for good.

        ``empty_from_beginning`` makes ``from-latest`` on a source with no job
        write the :data:`FROM_BEGINNING` mark instead of raising, so a fresh
        install indexes its first jobs rather than waiting for a seed by hand.

        Scans read from the checkpoint inclusively, so a job at exactly the
        seeded instant is indexed.

        Raises:
            LineageSeedError: When ``from-latest`` finds no job (unless
                ``empty_from_beginning``), or ``spec`` is not a timestamp.
        """
        key = self._get_checkpoint_key()
        existing = self.read_checkpoint(storage)
        if existing is not None and not force:
            logger.info(
                "Lineage index checkpoint %s already exists (%s); ignoring the "
                "requested seed (%s).",
                key,
                existing,
                spec,
            )
            return False
        timestamp = self._resolve_seed(storage, spec, empty_from_beginning)
        self._write_timestamp(storage, timestamp)
        if existing is not None:
            logger.warning(
                "Overwrote lineage index checkpoint %s: %s -> %s. Jobs between the "
                "two are re-indexed if it moved back, or skipped for good if it "
                "moved forward.",
                key,
                existing,
                timestamp,
            )
        else:
            logger.info("Seeded lineage index checkpoint %s at %s.", key, timestamp)
        return True

    # -- Scanning ------------------------------------------------------------

    def scan_once(self, storage: Optional[SingletonAdminStorage] = None) -> int:
        """Index every job since the checkpoint; return how many were written.

        A job that fails is retried on the next scan, up to ``_MAX_JOB_ATTEMPTS``
        times, and then skipped with an error log.
        """
        storage = storage or get_admin_storage()
        checkpoint = self.read_checkpoint(storage)
        if checkpoint is None and is_standalone() and self._seeds_when_standalone:
            self._seed_latest_in_standalone(storage)
            return 0
        if checkpoint is None and not is_standalone():
            # Re-read every scan, so seeding takes effect without a restart.
            if not self._missing_checkpoint_logged:
                logger.info(
                    "No lineage index checkpoint under %s; indexing nothing until "
                    "one is seeded (gbserver lineage-index-init --base-timestamp).",
                    self._get_checkpoint_key(),
                )
                self._missing_checkpoint_logged = True
            return 0
        self._missing_checkpoint_logged = False
        if checkpoint is not None and self._is_from_beginning(checkpoint):
            # A full read, never a SQL filter on the sentinel date.
            checkpoint = None
        self._expected_checkpoint = self.read_checkpoint(storage)
        self._checkpoint_moved = False
        mark: Optional[Tuple[datetime, List[str]]] = (
            (
                self._parse_job_ts(checkpoint["timestamp"]),
                list(checkpoint.get("item_ids") or []),
            )
            if checkpoint
            else None
        )
        indexed = 0
        # The unwritten checkpoint, as [timestamp, ids, done_ahead].
        pending: List[Any] = []
        try:
            indexed = self._scan(storage, checkpoint, mark, pending)
        finally:
            if pending:
                self._advance_checkpoint(storage, *pending)
        return indexed

    # Whether a standalone server with no checkpoint seeds ``from-latest`` instead
    # of indexing from the beginning. Off for local sources; on for W&B, whose
    # project is shared, so "everything" is every other deployment's history.
    _seeds_when_standalone = False

    def _seed_latest_in_standalone(self, storage: SingletonAdminStorage) -> None:
        self.seed_if_absent(storage, SEED_FROM_LATEST, empty_from_beginning=True)
        logger.info(
            "No lineage index checkpoint under %s; seeded it from the latest job "
            "rather than back-filling the whole shared project. Run gbserver "
            "lineage-index-init --base-timestamp all --force to index history.",
            self._get_checkpoint_key(),
        )

    def _scan(
        self,
        storage: SingletonAdminStorage,
        checkpoint: Optional[dict],
        mark: Optional[Tuple[datetime, List[str]]],
        pending: List[Any],
    ) -> int:
        """Index jobs in order, holding the checkpoint behind a pending one.

        A pending job (:meth:`_is_pending`: still being written) does not stop the
        scan. The checkpoint stays just behind it, and the jobs after it are
        indexed and remembered as ``done_ahead`` in the checkpoint, so one stuck
        writer delays only its own job. When the held job is done, the checkpoint
        moves across it and the ``done_ahead`` jobs it reaches. An unsettled job
        still stops the scan: anything after it may not have committed yet.
        """
        indexed = 0
        unwritten = 0
        stored = self._expected_checkpoint or {}
        base = (
            (stored.get("timestamp"), list(stored.get("item_ids") or []))
            if stored.get("timestamp")
            else None
        )
        done_ahead: Dict[str, str] = dict((checkpoint or {}).get("done_ahead") or {})
        held = False
        changed = False
        for job in self._jobs_since(storage, checkpoint and checkpoint["timestamp"]):
            if self.stop_event.is_set() or self._checkpoint_moved:
                break
            item_id = self._item_id(job)
            job_ts = self._parse_job_ts(self._timestamp(job))
            if mark is not None and (
                job_ts < mark[0] or (job_ts == mark[0] and item_id in mark[1])
            ):
                # Behind the mark, or on it and already done.
                continue
            if not self._is_settled(job):
                # Jobs come in timestamp order, so every later one waits too.
                logger.debug("Job %s not settled; stopping the scan", item_id)
                break
            if item_id in done_ahead:
                if held:
                    continue
                # Done on an earlier scan while held: the mark now reaches it.
                del done_ahead[item_id]
                changed = True
            elif self._is_pending(job):
                logger.debug("Job %s still pending; holding the checkpoint", item_id)
                held = True
                continue
            else:
                if held and len(done_ahead) >= _MAX_DONE_AHEAD:
                    logger.warning(
                        "%d lineage jobs indexed past a held one; stopping the scan "
                        "until it completes.",
                        len(done_ahead),
                    )
                    break
                try:
                    if self._index(storage, job):
                        indexed += 1
                except Exception:
                    if not self._record_failure(item_id):
                        break
                self._failed_attempts.pop(item_id, None)
                if held:
                    done_ahead[item_id] = self._timestamp(job)
                    changed = True
            if not held:
                if mark is not None and job_ts == mark[0]:
                    ids = mark[1] + [item_id]
                else:
                    ids = [item_id]
                mark = (job_ts, ids)
                base = (self._timestamp(job), ids)
                changed = True
            if changed and base is not None:
                pending[:] = [base[0], base[1], dict(done_ahead)]
                unwritten += 1
            if unwritten >= _CHECKPOINT_EVERY:
                if not self._advance_checkpoint(storage, *pending):
                    pending.clear()
                    break
                pending.clear()
                unwritten = 0
                changed = False
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


def _targets_page(
    storage: SingletonAdminStorage, page_index: int, cutoff: Optional[datetime]
) -> List[StoredTargetRun]:
    """One newest-finished-first page of successful targets, narrowed in SQL.

    With a cutoff the storage prefilters on completion time; it may return rows
    before the cutoff but never drops one at or after it, so the caller's exact
    check stays the answer. Without one (the first scan) every target is read.
    """
    if cutoff is None:
        return _successful_targets_page(storage, page_index)
    page = storage.target_storage.get_successful_finished_since(
        as_aware(cutoff).astimezone(timezone.utc), page_index, _SCAN_PAGE_SIZE
    )
    return [t for t in page if isinstance(t, StoredTargetRun)]


def _recorded_at_form(epoch_seconds: float) -> str:
    """An instant in ``recorded_at``'s fixed-width form (``utc_now_iso``)."""
    return datetime.fromtimestamp(epoch_seconds, timezone.utc).isoformat(
        timespec="microseconds"
    )


def _finished_after_page(
    storage: SingletonAdminStorage,
    cutoff: Optional[datetime],
    after: Optional[Tuple[Any, str]],
) -> List[Tuple[Any, StoredTargetRun]]:
    """One keyset page of successful targets, oldest finished first.

    Raises:
        NotImplementedError: When the storage cannot order by instant in SQL.
    """
    page = storage.target_storage.get_successful_finished_after(
        as_aware(cutoff).astimezone(timezone.utc) if cutoff else None,
        after,
        _SCAN_PAGE_SIZE,
    )
    # Not filtered by item type here: the caller's page-size stop and cursor
    # must see the page as the storage returned it.
    return page


def _in_instant_order(targets: List[StoredTargetRun]) -> List[StoredTargetRun]:
    """``targets`` (all finished) sorted by the exact ``(instant, uuid)``."""

    def key(target: StoredTargetRun) -> Tuple[datetime, str]:
        assert target.finished_at is not None
        return as_aware(target.finished_at), target.uuid

    return sorted(targets, key=key)


def _targets_by_uuid(
    storage: SingletonAdminStorage, uuids: List[str]
) -> List[Optional[StoredTargetRun]]:
    """The targets for ``uuids``, in that order; ``None`` where one is gone."""
    found = storage.target_storage.get_by_uuid(list(uuids))
    assert isinstance(found, list)  # a list in gives a list out
    return [t if isinstance(t, StoredTargetRun) else None for t in found]


class TargetLineageIndexer(JobLineageIndexer):
    """Standalone source: successful ``gb_targets`` rows by ``finished_at``.

    Where the storage can order by instant in SQL (SQLite through an expression
    index, Postgres through the column), targets are read by keyset a page at a
    time (:meth:`_keyset_targets`). Otherwise, and the rest of this describes
    that fallback, every key since the checkpoint is listed first:

    Past the first scan, the storage narrows the read to targets finished at or
    after the checkpoint (see :func:`_targets_page`). Every page of that is read,
    never stopping at the first row behind the checkpoint:
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

    PROVIDER = LINEAGE_PROVIDER_NONE

    def _timestamp(self, job: StoredTargetRun) -> str:
        return as_aware(job.finished_at).isoformat()

    def _item_id(self, job: StoredTargetRun) -> str:
        return job.uuid

    def _keys_since(
        self, storage: SingletonAdminStorage, timestamp: Optional[str]
    ) -> List[Tuple[datetime, str]]:
        """``(instant, uuid)`` of every successful target finished at or after
        ``timestamp`` (all when ``None``), oldest first.

        Only the keys are kept, not the targets: a first ``all`` scan lists the
        whole table, and a target carries its full artifacts. Each page is dropped
        once its keys are taken; :meth:`_jobs_since` loads the targets a batch at
        a time. A target listed twice (paging by offset while rows are added) is
        kept once.
        """
        cutoff = _parse_ts(timestamp) if timestamp else None
        keys: Dict[str, datetime] = {}
        page_index = 0
        while True:
            page = _targets_page(storage, page_index, cutoff)
            if not page:
                break
            for target in page:
                # Unfinished rows are skipped, never a stop: NULLs can interleave.
                if target.finished_at is None:
                    continue
                instant = as_aware(target.finished_at)
                if cutoff is not None and instant < cutoff:
                    continue
                keys.setdefault(target.uuid, instant)
            if len(page) < _SCAN_PAGE_SIZE:
                break
            page_index += 1
        # The DB sort can disagree with instant order (SQLite stores wall-clock
        # text), so order by the aware instant here; uuid only makes ties stable.
        return sorted((instant, uuid) for uuid, instant in keys.items())

    def _jobs_since(
        self, storage: SingletonAdminStorage, timestamp: Optional[str]
    ) -> Iterator[StoredTargetRun]:
        """The targets of :meth:`_keys_since`, loaded ``_CHECKPOINT_EVERY`` at a
        time and only as the scan reaches them, so memory is bounded by a batch
        and a scan that stops early (an unsettled or failing job) loads no more.
        """
        cutoff = _parse_ts(timestamp) if timestamp else None
        try:
            # Probe the first page here, so a backend without the keyset read
            # falls back before anything is yielded.
            first = _finished_after_page(storage, cutoff, None)
        except NotImplementedError as exc:
            if not self._keyset_fallback_logged:
                logger.info(
                    "Lineage index scan of gb_targets lists every key first: %s", exc
                )
                self._keyset_fallback_logged = True
        else:
            yield from self._keyset_targets(storage, cutoff, first)
            return
        keys = self._keys_since(storage, timestamp)
        for start in range(0, len(keys), _CHECKPOINT_EVERY):
            uuids = [uuid for _, uuid in keys[start : start + _CHECKPOINT_EVERY]]
            for target in _targets_by_uuid(storage, uuids):
                # Deleted, or no longer finished, since it was listed.
                if target is not None and target.finished_at is not None:
                    yield target

    _keyset_fallback_logged = False

    def _keyset_targets(
        self,
        storage: SingletonAdminStorage,
        cutoff: Optional[datetime],
        page: List[Tuple[Any, StoredTargetRun]],
    ) -> Iterator[StoredTargetRun]:
        """Targets at or after ``cutoff`` in exact instant order, a page at a time.

        Memory is one page and one run of equal keys, never the whole range. The
        storage key never orders two instants against their true order but may tie
        them (SQLite keeps milliseconds), and the scan skips any job behind the
        last one it took, so the targets of one key are held until the key changes
        and then released sorted by the exact ``(instant, uuid)``.
        """
        held: List[StoredTargetRun] = []
        held_key: Any = None
        while page:
            for key, target in page:
                if not isinstance(target, StoredTargetRun):
                    continue
                if held and key != held_key:
                    yield from _in_instant_order(held)
                    held = []
                held_key = key
                if target.finished_at is None:
                    continue
                if cutoff is not None and as_aware(target.finished_at) < cutoff:
                    continue
                held.append(target)
            if len(page) < _SCAN_PAGE_SIZE:
                break
            last_key, last = page[-1]
            page = _finished_after_page(storage, cutoff, (last_key, last.uuid))
        yield from _in_instant_order(held)

    def _index(self, storage: SingletonAdminStorage, job: StoredTargetRun) -> bool:
        if not job.input_artifacts and not any(job.output_artifacts.values()):
            return False
        # gb_targets IS the job store for this source, so nothing is copied into
        # gb_lineage_job: the rows are stamped TARGETS and their ``retrieve`` keys
        # (build_id, target_run_uuid) are what GET /lineage/jobs/{id} follows back
        # to the target run and its steps -- see job_detail's per-store dispatch.
        # Writing a job record too would duplicate, under this index's own notion
        # of a job, data the admin DB already owns and keeps current.
        try:
            # Passing no build lets the builder resolve the target's own, and
            # raise when it is gone: a target whose build vanished has no
            # provenance to index, and the rows' retrieve keys would point at a
            # build the detail route cannot fetch.
            events, _ = self._sink.create_jobstats_for_target(storage, job)
        except ValueError as exc:
            logger.warning(
                "Target run %s could not be built into lineage entries; "
                "not indexed: %s",
                job.uuid,
                exc,
            )
            return False
        indexed = False
        for event in events:
            indexed |= self._rows.index_job_entry(
                event,
                build_id=job.build_id,
                target_run_uuid=job.uuid,
                # The build's own tags are the user's; they are not in the job
                # entry, which the shared builder shapes for W&B.
                extra_tags=_build_tags(storage, job.build_id),
                job_store=JobStore.TARGETS,
            )
        return indexed

    def _latest_job(self, storage: SingletonAdminStorage) -> Optional[StoredTargetRun]:
        page_index = 0
        while True:
            page = _targets_page(storage, page_index, None)
            if not page:
                return None
            finished = [t for t in page if t.finished_at is not None]
            if finished:
                return max(finished, key=lambda t: as_aware(t.finished_at))
            page_index += 1

    def _format_timestamp(self, instant: datetime) -> str:
        # Same form as _timestamp: the aware isoformat, offset kept as given.
        return as_aware(instant).isoformat()


class LineageJobIndexer(JobLineageIndexer):
    """``db`` source: ``gb_lineage_job`` records by ``recorded_at``.

    The ``db`` sink records only the job, keeping the emitted entry in its
    attributes; this turns each record into index rows. ``recorded_at`` is UTC,
    fixed-width ISO-8601, so ordering by the string is ordering by instant, and the
    sink moves it forward when a merge changes a record so the change is re-read.

    Every record read is deduplicated by the rows' unique indexes (``upsert_row``),
    so a reset checkpoint re-reads everything without duplicating or losing rows.
    """

    PROVIDER = LINEAGE_PROVIDER_DB

    # A few seconds, not the default. recorded_at is stamped by the database
    # inside the write statement (see SQLLineageJobStorage._db_now), so a retried
    # write carries the time of the attempt that commits. What is left is the gap
    # between that statement and its COMMIT, and concurrent writers committing out
    # of stamp order within it: milliseconds. Not read from the environment: the
    # variable sizes gb_targets' grace, whose finished_at predates the write.
    settle_grace_s: float = 5.0

    _thread_name = "lineage-indexer-db"

    def _job_storage(self, storage: SingletonAdminStorage):
        return storage.lineage_job_storage

    def _timestamp(self, job: StoredLineageJob) -> str:
        return job.recorded_at

    def _item_id(self, job: StoredLineageJob) -> str:
        return job.job_id

    def _keys_since(
        self, storage: SingletonAdminStorage, timestamp: Optional[str]
    ) -> List[Tuple[datetime, str, str]]:
        """``(instant, job_id, recorded_at)`` of every record at or after
        ``timestamp``, oldest first.

        Only the keys are kept, not the records: a first ``all`` scan lists the
        whole table, and each record carries its full job entry. The records are
        loaded a batch at a time by :meth:`_scan`.

        Paging is newest first by offset. A merge only moves ``recorded_at``
        forward, which moves a record *up* the list: rows behind it shift down a
        page and are read twice (deduplicated here), never skipped; the moved
        record itself is newer than the checkpoint, so the next scan reads it.
        """
        cutoff = _parse_ts(timestamp) if timestamp else None
        keys: Dict[str, Tuple[datetime, str, str]] = {}
        for page in self._newest_first(storage):
            reached_cutoff = False
            for job in page:
                if not job.recorded_at:
                    continue
                instant = _parse_ts(job.recorded_at)
                if cutoff is not None and instant < cutoff:
                    reached_cutoff = True
                    continue
                keys.setdefault(job.job_id, (instant, job.job_id, job.recorded_at))
            # Newest first, so once a page reaches behind the cutoff every later
            # page is older still: a scan reads what is new, not all of history.
            if reached_cutoff:
                break
        return sorted(keys.values())

    def _jobs_since(
        self, storage: SingletonAdminStorage, timestamp: Optional[str]
    ) -> Iterator[StoredLineageJob]:
        """Records at or after ``timestamp``, oldest first, a page at a time.

        By keyset on ``(recorded_at, job_id)`` where the storage has it, so memory
        is one page whatever the range. Otherwise every key is listed first
        (:meth:`_keys_since`) and the records are loaded ``_CHECKPOINT_EVERY`` at a
        time.
        """
        cutoff = _parse_ts(timestamp) if timestamp else None
        job_storage = self._job_storage(storage)
        # In recorded_at's own form, a second early: the exact check is below.
        cutoff_text = (
            _recorded_at_form(cutoff.timestamp() - 1) if cutoff is not None else None
        )
        try:
            page = job_storage.get_recorded_after(cutoff_text, None, _SCAN_PAGE_SIZE)
        except NotImplementedError:
            page = None
        if page is not None:
            while page:
                for job in page:
                    if cutoff is None or _parse_ts(job.recorded_at) >= cutoff:
                        yield job
                if len(page) < _SCAN_PAGE_SIZE:
                    return
                last = page[-1]
                page = job_storage.get_recorded_after(
                    cutoff_text, (last.recorded_at, last.job_id), _SCAN_PAGE_SIZE
                )
            return
        keys = self._keys_since(storage, timestamp)
        for start in range(0, len(keys), _CHECKPOINT_EVERY):
            batch = keys[start : start + _CHECKPOINT_EVERY]
            records = job_storage.get_jobs_by_id([job_id for _, job_id, _ in batch])
            for _, job_id, _ in batch:
                # A record deleted since it was listed has nothing left to index.
                if job_id in records:
                    yield records[job_id]

    def _index(self, storage: SingletonAdminStorage, job: StoredLineageJob) -> bool:
        return self._rows.index_job_record(job)

    def _scan(
        self,
        storage: SingletonAdminStorage,
        checkpoint: Optional[dict],
        mark: Optional[Tuple[datetime, List[str]]],
        pending: List[Any],
    ) -> int:
        """Index in batches of ``_CHECKPOINT_EVERY``, checkpointing after each.

        Same rules as the base loop, a batch at a time: the scan stops at the
        first record not settled yet (a later one taken past it would move the
        checkpoint over it), and a batch that fails is retried next scan. When the
        failure is the batch's own, not an outage, its records are indexed one by
        one so a single bad record is counted, and given up on, alone instead of
        blocking every record behind it.
        """
        indexed = 0
        pairs: List[Tuple[Tuple[datetime, str, str], StoredLineageJob]] = []
        for job in self._jobs_since(storage, checkpoint and checkpoint["timestamp"]):
            instant = _parse_ts(job.recorded_at)
            if mark is not None and (
                instant < mark[0] or (instant == mark[0] and job.job_id in mark[1])
            ):
                continue
            if not self._is_settled_at(job.recorded_at):
                break
            pairs.append(((instant, job.job_id, job.recorded_at), job))
            if len(pairs) < _CHECKPOINT_EVERY:
                continue
            done, mark = self._index_batch(storage, pairs, mark)
            indexed += done
            pairs = []
            if mark is None:
                return indexed
        if pairs:
            done, _ = self._index_batch(storage, pairs, mark)
            indexed += done
        return indexed

    def _index_batch(
        self,
        storage: SingletonAdminStorage,
        pairs: List[Tuple[Tuple[datetime, str, str], StoredLineageJob]],
        mark: Optional[Tuple[datetime, List[str]]],
    ) -> Tuple[int, Optional[Tuple[datetime, List[str]]]]:
        """Index one batch and checkpoint past what was done.

        Returns how many were written and the new mark, ``None`` when the scan
        must stop (a failure retried next scan, or the checkpoint not moved).
        """
        if self.stop_event.is_set() or self._checkpoint_moved:
            return 0, None
        batch_keys = [key for key, _ in pairs]
        batch = [record for _, record in pairs]
        try:
            return self._rows.index_job_records(batch), self._checkpoint_batch(
                storage, batch_keys, mark
            )
        except Exception:
            if _is_transient_error(sys.exc_info()[1]):
                logger.warning(
                    "Transient failure indexing lineage job records; retrying "
                    "next scan.",
                    exc_info=True,
                )
                return 0, None
        indexed, stopped_at = self._index_one_by_one(storage, batch)
        if stopped_at is not None:
            # Stopped at a record still being retried: checkpoint up to it.
            if stopped_at:
                self._checkpoint_batch(storage, batch_keys[:stopped_at], mark)
            return indexed, None
        return indexed, self._checkpoint_batch(storage, batch_keys, mark)

    def _index_one_by_one(
        self, storage: SingletonAdminStorage, batch: List[StoredLineageJob]
    ) -> Tuple[int, Optional[int]]:
        """Index ``batch`` record by record after the batch write failed.

        Returns how many were written and, when the scan must stop at a record
        that will be retried, its position in the batch (else ``None``).
        """
        indexed = 0
        for position, job in enumerate(batch):
            try:
                if self._index(storage, job):
                    indexed += 1
            except Exception:
                if not self._record_failure(job.job_id):
                    return indexed, position
            self._failed_attempts.pop(job.job_id, None)
        return indexed, None

    def _checkpoint_batch(
        self,
        storage: SingletonAdminStorage,
        batch_keys: List[Tuple[datetime, str, str]],
        mark: Optional[Tuple[datetime, List[str]]],
    ) -> Optional[Tuple[datetime, List[str]]]:
        """Move the checkpoint past ``batch_keys``; ``None`` when it was not."""
        last_ts, _, last_recorded_at = batch_keys[-1]
        ids = [job_id for instant, job_id, _ in batch_keys if instant == last_ts]
        if mark is not None and last_ts == mark[0]:
            ids = mark[1] + ids
        if not self._advance_checkpoint(storage, last_recorded_at, ids):
            return None
        return (last_ts, ids)

    def _is_settled_at(self, recorded_at: str) -> bool:
        age = datetime.now(timezone.utc) - _parse_ts(recorded_at)
        return age.total_seconds() >= self.settle_grace_s

    def _newest_first(self, storage: SingletonAdminStorage):
        """Pages of job records by ``recorded_at`` descending, sorted in SQL.

        ``recorded_at`` is fixed-width UTC text, so the column's string order is
        instant order and the database can sort it.
        """
        job_storage = self._job_storage(storage)
        page_index = 0
        while True:
            page = job_storage.get_by_where(
                None,
                query_control=QueryControl(
                    pagination=Pagination(index=page_index, size=_SCAN_PAGE_SIZE),
                    # job_id breaks ties, so offset paging cannot skip a record.
                    sort_orders=[
                        SortOrder(column="recorded_at", ascending=False),
                        SortOrder(column="job_id", ascending=False),
                    ],
                ),
            )
            if not page:
                return
            yield page
            if len(page) < _SCAN_PAGE_SIZE:
                return
            page_index += 1

    def _latest_job(self, storage: SingletonAdminStorage) -> Optional[StoredLineageJob]:
        latest = None
        # The first page holds the newest record; the max is taken over it rather
        # than trusting its first item, in case a backend sorts differently.
        for page in self._newest_first(storage):
            for job in page:
                if job.recorded_at and (
                    latest is None
                    or _parse_ts(job.recorded_at) > _parse_ts(latest.recorded_at)
                ):
                    latest = job
            break
        return latest

    def _format_timestamp(self, instant: datetime) -> str:
        # recorded_at's own form (utc_now_iso).
        return instant.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _tags_of(run: Any) -> Dict[str, str]:
    tags: Dict[str, str] = {}
    for tag in run.tags or []:
        if "=" in tag:
            key, value = tag.split("=", 1)
            tags[key] = value
    return tags


def wandb_run_to_job(run: Any) -> Optional[dict]:
    """Rebuild the job entry a W&B lineage run was recorded from.

    The inverse of ``WandBLineageService.emit_event``, in the job-entry shape the
    admin-DB path builds, so both sources decompose identically.

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

    **Running runs are pending.** A W&B run executes nothing, but ``emit_event``
    creates it with ``wandb.init`` and only then logs its artifacts and job_id,
    asynchronously until ``finish()``. Indexing it in that window would record
    some of its edges, or none, and move the checkpoint past it; so a run still
    ``running`` stops the scan, up to ``_WANDB_RUNNING_MAX_AGE_S``.

    **Several runs, one job.** The W&B sink writes one run per output artifact, all
    carrying the target's ``job_id``. Each is indexed on its own and adds its own
    rows under the shared job, which is why dedup here is by the sink's unique
    indexes and not by "does this job already have rows".
    """

    PROVIDER = LINEAGE_PROVIDER_WANDB

    _thread_name = "lineage-indexer-wandb"

    # No settle grace. The grace exists for sources whose timestamp is stamped
    # before the row commits (gb_targets' finished_at, gb_lineage_job's
    # recorded_at), so rows can appear out of order. A W&B run's createdAt is set
    # by the W&B server when the run is created, and the run is listed from that
    # moment: nothing commits later behind it. A run still being written is held
    # by _is_pending instead. GBSERVER_LINEAGE_INDEX_SETTLE_GRACE_S does not apply.
    settle_grace_s: float = 0.0

    _seeds_when_standalone = True

    def __init__(
        self,
        monitoring_interval: float = 30.0,
        sink: Optional[DBLineageStore] = None,
        rows: Optional[LineageRowIndexer] = None,
        api: Any = None,
    ) -> None:
        super().__init__(monitoring_interval=monitoring_interval, sink=sink, rows=rows)
        self._api = api
        self._logged_in = False
        # Run ids already warned about as pinning the scan, so it is said once.
        self._pinned_warned: set = set()

    def _wandb_api(self) -> Any:
        """A fresh ``wandb.Api`` per scan, unless one was injected.

        ``Api.runs`` caches its result per path, filters and order and returns the
        cached list on a repeat call, so a long-lived Api re-iterates the same runs
        once the checkpoint stops moving and never sees new ones. It also keeps
        every list it built, so memory would grow with each checkpoint.
        """
        if self._api is not None:
            return self._api
        import wandb

        from gbserver.types.constants import (
            GBSERVER_WANDB_API_KEY,
            GBSERVER_WANDB_BASE_URL,
        )

        if not self._logged_in:
            wandb.login(key=GBSERVER_WANDB_API_KEY, host=GBSERVER_WANDB_BASE_URL)
            self._logged_in = True
        return wandb.Api()

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

    def _parse_job_ts(self, value: str) -> datetime:
        # W&B's createdAt comes zoneless (UTC) from the API, and a seed is spelled
        # with a trailing Z; both are UTC, never local time.
        return _parse_utc_ts(value)

    def _is_pending(self, job: Any) -> bool:
        if getattr(job, "state", None) != "running":
            return False
        # The project is shared: only a lineage run is worth waiting on, or any
        # long training run logged there would stall indexing. A lineage run is
        # known by its job_type (set at init) or, from older writers, its job_id.
        config = getattr(job, "config", None) or {}
        if getattr(job, "job_type", None) != _WANDB_LINEAGE_RUN_JOB_TYPE and not (
            isinstance(config, dict) and config.get("job_id")
        ):
            return False
        age = datetime.now(timezone.utc) - self._parse_job_ts(self._timestamp(job))
        age_s = age.total_seconds()
        if age_s >= _WANDB_RUNNING_MAX_AGE_S:
            self._pinned_warned.discard(job.id)
            return False
        if age_s >= _WANDB_PINNED_WARN_AFTER_S and job.id not in self._pinned_warned:
            self._pinned_warned.add(job.id)
            logger.warning(
                "W&B lineage run %s has been running for %.0f minutes and holds "
                "the lineage index checkpoint behind it (later runs are still "
                "indexed); its writer may have died. It is waited on for %.0f "
                "more minutes, then indexed as it is.",
                job.id,
                age_s / 60,
                (_WANDB_RUNNING_MAX_AGE_S - age_s) / 60,
            )
        return True

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
        build_id = tags.get("build_id", "")
        target_run_uuid = tags.get("target_id", "")
        # Only the index is written. W&B is the job store here: the rows are
        # stamped WANDB and the job's detail is read back from W&B when asked for
        # (see job_detail's per-store dispatch), so nothing is copied into
        # gb_lineage_job, which is the db provider's store alone.
        self._rows.index_job_entry(
            entry,
            build_id=build_id,
            target_run_uuid=target_run_uuid,
            job_store=JobStore.WANDB,
        )
        # True regardless: a run that yields no edges has nothing more to index,
        # so the scan must advance past it rather than retry it forever.
        return True

    def _first_run(self, **kwargs: Any) -> Optional[Any]:
        return next(iter(self._wandb_api().runs(self._project_path(), **kwargs)), None)

    def _latest_job(self, storage: SingletonAdminStorage) -> Optional[Any]:
        return self._first_run(order="-created_at", per_page=1)

    def _format_timestamp(self, instant: datetime) -> str:
        # createdAt is UTC (the API returns it without an offset); the $gte filter
        # compares against it, so spell a seed in UTC with a trailing Z.
        return instant.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
