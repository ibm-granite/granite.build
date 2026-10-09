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

"""The sink that writes lineage into the local index.

The write half of the lineage index: ``DBLineageService`` reads the table, this
fills it. Same ``ILineageStore`` interface the W&B sink implements, so the
reconciler and the watcher drive it unchanged.

**The job entries are not built here.** ``create_jobstats_for_target`` in
``wandb_jobstats`` already turns a target run into job entries -- resolving input
and output artifacts, collecting step configs, redacting secret-named keys -- and
``_add_jobstats_mirror_fields`` already exposes them as top-level
``sources``/``targets``, which is exactly the shape ``to_lineage_rows`` consumes.
Reusing it means the two sinks cannot disagree about what a build's lineage *is*;
re-deriving it here would fork that logic and let them drift. Only the module
functions are reused, never ``WandBLineageStore`` itself, so nothing here needs
``wandb`` installed or configured.

What this sink adds is the decomposition: one job entry with N inputs and M
outputs becomes N*M flat rows sharing a ``job_id``. That is lossy only in
appearance -- ``group_by_job`` recovers which inputs and which outputs an
execution had -- and it is what makes each row an independently indexable edge.

Dedup is by presence of ``job_id``, not by row count. A job either has
its rows or it does not. It deliberately does not compare against the reconciler's
``expected_counts``, which counts one W&B run per output artifact: that number
never equals an N*M row count, so comparing would report every target as
unrecorded forever and re-record on every scan.
"""

import logging
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from gbserver.lineage.attributes import (
    JOB,
    JOB_COMPLETED_AT,
    JOB_NAMESPACE,
    JOB_OWNER,
    JOB_STARTED_AT,
    JOB_STATUS,
    build_index_attributes,
    build_job_attributes,
    job_detail,
    origin_id,
)
from gbserver.lineage.decompose import LineageDecomposeError, to_lineage_rows
from gbserver.lineage.jobstats import ILineageStore
from gbserver.lineage.merge import UPDATED, rows_to_add, upsert_job, upsert_row
from gbserver.storage.artifact_registration import ArtifactRegistration
from gbserver.storage.lineage_job_storage import ILineageJobStorage
from gbserver.storage.lineage_row_storage import ILineageRowStorage, tags_to_map
from gbserver.storage.singleton_storage import SingletonAdminStorage
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_lineage_job import StoredLineageJob
from gbserver.storage.stored_lineage_row import (
    TERMINAL,
    JobStore,
    StoredLineageRow,
    utc_now_iso,
)
from gbserver.storage.stored_target_run import StoredTargetRun

logger = logging.getLogger(__name__)

# Longest tag kept. Tags now live in the row blob, so this is no column's width; it
# only keeps a runaway value out of every row of a job.
MAX_TAG_LENGTH = 256


def is_storable_tag(tag: object) -> bool:
    """Whether ``tag`` is kept: a non-empty string no longer than the limit."""
    return isinstance(tag, str) and 0 < len(tag) <= MAX_TAG_LENGTH

# Names the system that produced a row, so rows this sink derived stay
# distinguishable from rows an importer supplied. Surfaced on a run node as
# ``source_system``; an importer passes its own name.
SOURCE_SYSTEM = "granite.build"

# Key in a job record's ``attributes`` holding the job entry as emitted, so the
# lineage indexer can decompose it into index rows later. Only the record-only
# sink (``index_rows=False``, the ``db`` provider) writes it.
ENTRY_ATTRIBUTE = "entry"


class DBLineageStore(ILineageStore):
    """Record lineage into the local lineage index.

    Each recorded execution produces two writes: the N*M lineage rows, and one job
    record holding the metadata a row does not carry (the large payloads, and the
    fields promoted to job columns). Both are idempotent under their own unique
    index, so re-recording is a no-op rather than a duplicate.

    Args:
        storage: the lineage row storage to write. Defaults to the process-wide
            admin storage, resolved lazily so importing this module does not
            require a configured database.
        job_storage: the lineage job storage to write. Resolved the same way, but
            only when ``storage`` was not given -- see :attr:`job_storage`. Without
            it the rows are still written; only the job record is skipped.
        index_rows: write the index rows along with the job record. ``False`` is
            the ``db`` provider's sink: it records only ``gb_lineage_job``, with the
            entry kept under :data:`ENTRY_ATTRIBUTE`, and the lineage indexer
            builds the rows from it later (:meth:`index_job_record`).
    """

    def __init__(
        self,
        storage: Optional[ILineageRowStorage] = None,
        job_storage: Optional[ILineageJobStorage] = None,
        index_rows: bool = True,
    ) -> None:
        self._row_storage = storage
        self._job_storage = job_storage
        self._index_rows = index_rows

    @property
    def row_storage(self) -> ILineageRowStorage:
        """The lineage row storage, resolved on first use."""
        if self._row_storage is None:
            from gbserver.storage.singleton_storage import get_admin_storage

            self._row_storage = get_admin_storage().lineage_row_storage
        return self._row_storage

    @property
    def job_storage(self) -> Optional[ILineageJobStorage]:
        """The lineage job storage, resolved on first use, or ``None``.

        ``None`` when there is nothing to resolve it from. The rows are what the graph
        is built from; the job record enriches them, so a caller that supplied only a
        row storage still records usable lineage rather than failing.

        A caller that passed an explicit ``storage`` and no ``job_storage`` gets
        ``None`` without the singleton being consulted at all. Reaching for it would
        open a database connection that caller never asked for -- and, against a
        configured-but-unreachable backend, block rather than fail.
        """
        if self._job_storage is None:
            if self._row_storage is not None:
                return None

            from gbserver.storage.singleton_storage import get_admin_storage

            try:
                self._job_storage = get_admin_storage().lineage_job_storage
            except Exception as exc:
                # A warning, not debug: rows are still written without it, and a
                # row with no job record has no status, owner or step params.
                logger.warning("No lineage job storage available: %s", exc)
                return None
        return self._job_storage

    # -- Recording -----------------------------------------------------------

    def add_jobstats_for_build(
        self, storage: SingletonAdminStorage, build_id: str
    ) -> None:
        """Record every target of a build.

        Raises:
            ValueError: if the build does not exist, or has no targets. Both match
                the W&B sink, so the reconciler sees one behaviour regardless of
                which sink is configured.
        """
        build = self._require_build(storage, build_id)
        targets = storage.target_storage.get_by_where({"build_id": build_id})
        if not targets:
            raise ValueError(f"Zero targets found in build with id {build_id}")

        for target in targets:
            self._record_target(storage, build, target)

    def add_jobstats_for_build_target(
        self, storage: SingletonAdminStorage, build_id: str, target_id: str
    ) -> None:
        """Record one target of a build -- the single recording leaf.

        Idempotent: re-recording a target that already has rows is a no-op, so the
        reconciler can call this for an already-recorded target harmlessly.

        Raises:
            ValueError: if the build or the target does not exist.
        """
        build = self._require_build(storage, build_id)
        targets = storage.target_storage.get_by_where(
            {"build_id": build_id, "uuid": target_id}
        )
        if not targets:
            raise ValueError(f"Zero targets found in build with id {build_id}")

        for target in targets:
            self._record_target(storage, build, target)

    def add_jobstats_for_original_artifact(
        self,
        artifact: ArtifactRegistration,
        sources: list[ArtifactRegistration],
    ) -> None:
        """Record lineage for a registered artifact and the sources it came from.

        This path has no target run, so the rows carry an empty
        ``target_run_uuid``; ``build_id`` holds the artifact's uuid instead, which
        is what the W&B sink also does -- its own comment says the "release_id" for
        a registered artifact *is* the artifact uuid, and that is the column
        :meth:`count_release_ids` queries. The ``(job_id, input, output)`` unique
        still protects against duplicates, so having no target run costs nothing.

        One consequence worth knowing: these rows are never skipped by the
        presence-based dedup, which keys on ``target_run_uuid``. They do not need
        to be -- the unique index makes a re-record a no-op -- but they do cost a
        write attempt each time.
        """
        job = self.create_jobstats_for_original_artifact(artifact, sources)
        if not job:
            return
        # build_id carries the artifact uuid so count_release_ids(artifact.uuid)
        # finds these rows -- the same convention the W&B sink uses, where a
        # registered artifact's "release_id" IS its uuid.
        self._write_job(job, build_id=artifact.uuid, target_run_uuid="")

    def _record_target(
        self,
        storage: SingletonAdminStorage,
        build: StoredBuild,
        target: StoredTargetRun,
    ) -> None:
        """Decompose one target run's job entries into rows and store them."""
        if not isinstance(target, StoredTargetRun):
            return

        if self.row_storage.has_rows_for_job(target.uuid) and self._has_job_record(
            target.uuid
        ):
            # Already recorded. Presence, not count: see the module docstring for
            # why a count comparison would re-record forever.
            #
            # Keyed on target.uuid because that IS the job id of build lineage --
            # the event builder sets ``job_details.job_id = targetrun.uuid``
            # (``wandb_jobstats.py:274``), so per-job dedup is per-target dedup here
            # without the index needing a column for a target run.
            #
            # Rows alone are not enough: a job upsert that failed left rows with no
            # job record, and skipping on rows would never write it. The rewrite is
            # idempotent, so the rows merge and only the record is added.
            logger.debug("Target run %s already has lineage rows", target.uuid)
            return

        events, _ = self.create_jobstats_for_target(storage, target, build)
        for job in events:
            # The build's own tags are the user's; they are not in the job entry,
            # which the shared builder shapes for W&B, so they are passed alongside.
            self._write_job(
                job,
                build_id=build.uuid,
                target_run_uuid=target.uuid,
                extra_tags=build.tags,
            )

    def write_job(
        self,
        job: dict,
        build_id: str,
        target_run_uuid: str,
        extra_tags: Optional[List[str]] = None,
        job_store: JobStore = JobStore.OTHER,
    ) -> None:
        """Index one already-built job entry.

        The entry point for a caller that has the job entry in hand rather than a
        target run to build it from -- the lineage indexer, reading entries back
        out of another lineage store. Idempotent under the same unique indexes as
        every other write here, so re-indexing an entry is a no-op.

        Args:
            extra_tags: tags to attach beyond those derived from the entry, e.g. a
                source's own labels. Free-form; see :func:`job_tags`.
            job_store: where the entry's full data lives, e.g. ``WANDB`` for an
                entry read back out of W&B.
        """
        self._write_job(
            job,
            build_id=build_id,
            target_run_uuid=target_run_uuid,
            extra_tags=extra_tags,
            job_store=job_store,
        )

    def _write_job(
        self,
        job: dict,
        build_id: str,
        target_run_uuid: str,
        extra_tags: Optional[List[str]] = None,
        job_store: JobStore = JobStore.TARGETS,
    ) -> None:
        """Decompose one job entry and add its rows.

        A job that cannot be decomposed is logged and skipped rather than aborting
        the scan: one unrecordable target must not stop the rest of a build's
        lineage from landing. The reason is logged with it -- a bare "skipping"
        makes lineage loss undiagnosable, and lineage this index misses is not
        """
        try:
            drafts = to_lineage_rows(_normalized_job(job))
        except LineageDecomposeError as exc:
            logger.warning(
                "Job entry could not be decomposed into lineage rows; skipping "
                "(build=%s, target_run=%s): %s",
                build_id,
                target_run_uuid,
                exc,
            )
            return

        # The job record first: it holds the metadata the rows no longer carry
        # (the large payloads, and the three fields promoted to job columns), so it
        # is written even if every draft below turns out to be unstorable.
        entry = None
        if not self._index_rows:
            entry = {
                "event": job,
                "build_id": build_id,
                "target_run_uuid": target_run_uuid,
                "extra_tags": list(extra_tags or []),
            }
        self._add_job(
            _normalized_job(job),
            build_id=build_id,
            target_run_uuid=target_run_uuid,
            entry=entry,
        )
        if not self._index_rows:
            # The indexer writes the rows from the record; see index_job_record.
            return
        self._write_rows(
            job,
            drafts,
            build_id=build_id,
            target_run_uuid=target_run_uuid,
            extra_tags=extra_tags,
            job_store=job_store,
        )

    def index_job_record(self, record: StoredLineageJob) -> bool:
        """Write the index rows for a job record in ``gb_lineage_job``.

        The record's :data:`ENTRY_ATTRIBUTE` is either ``{"event": ...}``, the job
        entry the ``db`` sink emitted (decomposed here), or ``{"rows": [...]}``,
        index rows an importer already built.

        The job record itself is not rewritten: it is the source here. Idempotent
        under the rows' unique indexes, so re-indexing after a checkpoint reset
        writes nothing new. Returns whether the record carried an entry to index.
        """
        entry = (record.attributes or {}).get(ENTRY_ATTRIBUTE) or {}
        if entry.get("rows"):
            # Already-built rows, from an importer whose source is one row per
            # edge (Lakehouse): written as given, nothing to decompose.
            for data in entry["rows"]:
                try:
                    row = StoredLineageRow.model_validate(data)
                    row.job_store = JobStore.LINEAGE_JOB
                    _fill_job_from_record(row, record)
                    upsert_row(self.row_storage, row)
                except Exception:
                    logger.debug(
                        "Lineage row from job record could not be added or merged "
                        "(job=%s, row=%r)",
                        record.job_id,
                        data,
                        exc_info=True,
                    )
            return True
        job = entry.get("event")
        if not job:
            return False
        build_id = entry.get("build_id") or ""
        target_run_uuid = entry.get("target_run_uuid") or ""
        try:
            drafts = to_lineage_rows(_normalized_job(job))
        except LineageDecomposeError as exc:
            logger.warning(
                "Lineage job record could not be decomposed into rows; skipping "
                "(job=%s): %s",
                record.job_id,
                exc,
            )
            return False
        self._write_rows(
            job,
            drafts,
            build_id=build_id,
            target_run_uuid=target_run_uuid,
            extra_tags=entry.get("extra_tags"),
            job_store=JobStore.LINEAGE_JOB,
        )
        return True

    def index_job_records(self, records: List[StoredLineageJob]) -> int:
        """Batched :meth:`index_job_record`; return how many records had an entry.

        One write per row, each its own commit, made the indexer ~3 ms a row. Here
        a batch costs one lookup of the rows already stored for its jobs and one
        bulk add of the new ones. Jobs that already have rows go through
        :func:`upsert_row` one by one, so the dedup and merge are unchanged, and a
        bulk add that fails falls back to the same per-row path.
        """
        prebuilt: List[StoredLineageRow] = []
        indexed = 0
        for record in records:
            entry = (record.attributes or {}).get(ENTRY_ATTRIBUTE) or {}
            if not entry.get("rows"):
                indexed += self.index_job_record(record)
                continue
            indexed += 1
            for data in entry["rows"]:
                try:
                    row = StoredLineageRow.model_validate(data)
                    row.job_store = JobStore.LINEAGE_JOB
                    _fill_job_from_record(row, record)
                    prebuilt.append(row)
                except Exception:
                    logger.debug(
                        "Invalid lineage row in job record (job=%s, row=%r)",
                        record.job_id,
                        data,
                        exc_info=True,
                    )
        if not prebuilt:
            return indexed

        stored_jobs = {
            row.job_id
            for row in self.row_storage.get_rows_by_jobs(
                list({row.job_id for row in prebuilt})
            )
        }
        fresh = rows_to_add([row for row in prebuilt if row.job_id not in stored_jobs])
        merging = [row for row in prebuilt if row.job_id in stored_jobs]
        if fresh:
            try:
                self.row_storage.add(fresh)
            except Exception:
                logger.debug(
                    "Bulk add of %d lineage rows failed; adding one by one",
                    len(fresh),
                    exc_info=True,
                )
                merging.extend(fresh)
        for row in merging:
            try:
                upsert_row(self.row_storage, row)
            except Exception:
                logger.debug(
                    "Lineage row could not be added or merged "
                    "(job=%s, input=%r, output=%r)",
                    row.job_id,
                    row.input,
                    row.output,
                    exc_info=True,
                )
        return indexed

    def _write_rows(
        self,
        job: dict,
        drafts,
        build_id: str,
        target_run_uuid: str,
        extra_tags: Optional[List[str]] = None,
        job_store: JobStore = JobStore.TARGETS,
    ) -> None:
        # Tags ride on every row of the job, in ``attributes.job.tags``.
        tags = job_tags(
            job,
            build_id=build_id,
            target_run_uuid=target_run_uuid,
            extra_tags=extra_tags,
        )

        for draft in drafts:
            if not draft.input and not draft.output:
                # Both endpoints unidentifiable: the row would be terminal on both
                # sides, which identifies nothing and would join unrelated jobs.
                continue
            self._add_row(
                draft,
                build_id=build_id,
                target_run_uuid=target_run_uuid,
                tags=tags,
                job_store=job_store,
            )

    def _add_job(
        self,
        job_metadata: dict,
        build_id: str,
        target_run_uuid: str,
        entry: Optional[dict] = None,
    ) -> None:
        """Store one job record, merging into one another writer already stored.

        The same execution can arrive from more than one source -- a target run the
        Lakehouse importer wrote first, say -- so a collision on ``job_id`` fills
        what the stored copy lacks rather than being dropped. See
        :mod:`gbserver.lineage.merge`.

        With ``entry`` (the record-only sink), a merge that changed the record
        also moves ``recorded_at`` forward: the indexer reads by it, and would
        otherwise never see what the merge added.
        """
        storage = self.job_storage
        if storage is None:
            return
        job = _job_from_metadata(
            job_metadata,
            build_id=build_id,
            target_run_uuid=target_run_uuid,
        )
        if not job:
            return
        if entry is not None:
            job.attributes[ENTRY_ATTRIBUTE] = entry
        try:
            if upsert_job(storage, job) == UPDATED and entry is not None:
                stored = storage.get_job(job.job_id)
                if stored is not None:
                    storage.update_fields(stored.uuid, {"recorded_at": utc_now_iso()})
        except Exception:
            # A warning: the rows are written anyway, so this is the only trace of
            # a job left without its record (the next scan backfills it, see
            # _has_job_record).
            logger.warning(
                "Lineage job could not be added or merged (job=%s)",
                job.job_id,
                exc_info=True,
            )

    def _add_row(
        self,
        draft,
        build_id: str,
        target_run_uuid: str,
        tags: Optional[List[str]] = None,
        job_store: JobStore = JobStore.TARGETS,
    ) -> None:
        """Store one decomposed row, merging into the same edge already stored.

        Re-ingest stays idempotent -- a merge that adds nothing writes nothing -- and
        an edge another source recorded first gains what this one knows. See
        :mod:`gbserver.lineage.merge`.
        """
        row = _row_from_draft(
            draft,
            build_id=build_id,
            target_run_uuid=target_run_uuid,
            job_store=job_store,
        )
        if tags:
            row.attributes.setdefault("job", {})["tags"] = tags_to_map(tags)
        try:
            upsert_row(self.row_storage, row)
        except Exception:
            logger.debug(
                "Lineage row could not be added or merged "
                "(job=%s, input=%r, output=%r)",
                row.job_id,
                row.input,
                row.output,
            )

    # -- Building (delegated, so both sinks agree on what lineage is) --------

    def create_jobstats_for_target(
        self,
        storage: SingletonAdminStorage,
        targetrun: StoredTargetRun,
        build: Optional[StoredBuild] = None,
    ) -> Tuple[List[dict], Dict[str, List[dict]]]:
        """Build the job entries for a target run.

        Delegates to the shared builder rather than re-deriving, so this sink and
        the W&B sink cannot disagree about a build's lineage. Only module functions
        are used, so ``wandb`` is neither imported nor required.
        """
        from gbserver.lineage.wandb_jobstats import WandBLineageStore

        # Called unbound with self=None: both builders are pure -- neither touches
        # self -- so no W&B instance (and no wandb import) is needed. Passing None
        # rather than binding makes that dependency explicit and fails loudly if a
        # future edit starts reaching for instance state.
        return WandBLineageStore._build_events_for_target(
            None,
            storage,
            self._require_build_of(storage, targetrun, build),
            targetrun,
        )

    def create_jobstats_for_original_artifact(
        self,
        artifact: ArtifactRegistration,
        sources: list[ArtifactRegistration],
    ) -> dict:
        """Build the job entry for a registered artifact. See above for delegation."""
        from gbserver.lineage.wandb_jobstats import WandBLineageStore

        return WandBLineageStore._build_event_for_artifact(None, artifact, sources)

    # -- Completeness queries -----------------------------------------------

    def count_release_ids(
        self, release_id: str, target_id: Optional[str] = None
    ) -> int:
        """Count the lineage rows recorded for a release.

        ``release_id`` is a ``build_id`` for build lineage and the artifact's uuid
        for a registered artifact -- W&B's convention, kept so the same argument
        works against either sink.

        Neither of those is a column here: the index is keyed by artifact URI and
        job, and a build id lives in the unqueryable ``attributes`` blob. So this
        pages and filters in Python. It is only used by W&B-shaped callers (no
        production caller outside that sink), and a scan is the honest cost of
        answering a question this schema is not organized around -- as opposed to
        adding a column that would be blank on every imported row.

        Args:
            release_id: the build uuid, or the artifact uuid.
            target_id: optional target run to narrow to.

        Returns:
            How many rows are recorded. ``0`` for an unknown release.
        """
        if not release_id:
            return 0

        job_ids = self._job_ids_for_release(release_id, target_id)
        if job_ids:
            return self.row_storage.count({"job_id": sorted(job_ids)})

        # No tagged job: either the release is unknown, or its rows predate job
        # tags. Only the scan can tell those apart, so it stays as the fallback.
        matched = 0
        for page in self.row_storage.get_paged():
            for row in page:
                if origin_id(row.attributes, "build_id") != release_id:
                    continue
                if target_id and (
                    origin_id(row.attributes, "target_run_uuid") != target_id
                ):
                    continue
                matched += 1
        return matched

    def _job_ids_for_release(
        self, release_id: str, target_id: Optional[str]
    ) -> Set[str]:
        """The jobs tagged with a release (and target), by indexed tag lookup.

        The tags are the ones :func:`job_tags` derives from the same ``ids`` the
        scan compares, so both answer the same question.
        """
        required = [f"target_run_uuid={target_id}"] if target_id else None
        try:
            return self.row_storage.get_job_ids_by_tags(
                [f"build_id={release_id}"], all_of=required
            )
        except Exception as exc:
            logger.debug("Lineage job tag lookup failed: %s", exc)
            return set()

    def does_release_id_exist(
        self, release_id: str, expected_count: int, target_id: Optional[str] = None
    ) -> bool:
        """Whether a release has exactly ``expected_count`` rows recorded.

        Kept for interface compatibility. Mind the unit: ``expected_count`` is
        compared against a ROW count, and W&B creates one run per (target, output
        artifact) while this sink writes one row per (input, output) pair -- a build
        with 2 inputs and 5 outputs is 5 runs there and 10 rows here. A count
        computed in W&B's shape will not match. That same mismatch is why recording
        dedup is presence-based rather than count-based.
        """
        return self.count_release_ids(release_id, target_id) == expected_count

    def filter_unrecorded(
        self,
        target_ids: set[str],
        expected_counts: Optional[dict[str, int]] = None,
        on_query_error: Optional[Callable[[Exception], None]] = None,
    ) -> set[str]:
        """Return the candidates that have no rows yet.

        ``expected_counts`` is accepted and **ignored**: it counts one W&B run per
        output artifact, a shape that never equals an N*M row count, so honouring
        it would mark every target unrecorded forever. Completeness is by presence
        instead -- a target's rows are written together, so it either has them or
        it does not.

        The candidates are target run uuids, and they are looked up as *job* ids,
        which is correct rather than a coincidence: the event builder stamps
        ``job_details.job_id = targetrun.uuid`` (``wandb_jobstats.py:274``), so a
        target run and its job share one identifier. That is what lets the index
        drop its ``target_run_uuid`` column without weakening dedup.

        Fails **open**, and invokes ``on_query_error``: on a query failure every
        candidate is reported unrecorded. Re-recording is idempotent, so that is
        harmless, and the reconciler is what decides not to record at all (it fails
        closed on the callback). Without the callback this would silently duplicate
        work rather than skip it.
        """
        if not target_ids:
            return set()
        try:
            recorded = self.recorded_by_self(
                self.row_storage.get_recorded_jobs(list(target_ids))
            )
        except Exception as exc:
            logger.warning("Lineage dedup query failed; treating all as unrecorded")
            if on_query_error is not None:
                on_query_error(exc)
            return set(target_ids)
        return {target_id for target_id in target_ids if target_id not in recorded}

    def recorded_by_self(self, job_ids: Iterable[str]) -> Set[str]:
        """Narrow ``job_ids`` that have rows to those granite.build itself recorded.

        A target run another source imported first -- Lakehouse keeps granite.build
        runs under ``job_id = targetrun.uuid`` -- has rows, but not this system's
        view of it: no namespace, which the read path needs to authorize it. Such a
        job is reported unrecorded so the scan writes it, and the write merges into
        the imported copy rather than duplicating it.

        Dedup is by job_id alone: if the job record exists, it was already written
        by some source. The merge logic fills in any blanks the first writer missed.
        """
        job_ids = set(job_ids)
        storage = self.job_storage
        if storage is None or not job_ids:
            return job_ids
        jobs = storage.get_jobs_by_id(list(job_ids))
        return {job_id for job_id in job_ids if job_id in jobs}

    def _has_job_record(self, job_id: str) -> bool:
        """Whether the job table holds ``job_id``; ``True`` with no job storage.

        With no job storage there is nothing to backfill, so presence of rows
        decides, as in :meth:`recorded_by_self`.
        """
        storage = self.job_storage
        if storage is None:
            return True
        return job_id in storage.get_jobs_by_id([job_id])

    # -- Helpers -------------------------------------------------------------

    @staticmethod
    def _require_build(storage: SingletonAdminStorage, build_id: str) -> StoredBuild:
        build = storage.build_storage.get_by_uuid(build_id)
        if build is None:
            raise ValueError(f"Build with id {build_id} was not found")
        if not isinstance(build, StoredBuild):
            raise ValueError(f"Build with id {build_id} was not a build")
        return build

    def _require_build_of(
        self,
        storage: SingletonAdminStorage,
        targetrun: StoredTargetRun,
        build: Optional[StoredBuild],
    ) -> StoredBuild:
        """Resolve the target's build, validating it matches when one is given."""
        if build is None:
            return self._require_build(storage, targetrun.build_id)
        if targetrun.build_id != build.uuid:
            raise ValueError(
                f"target's build id ({targetrun.build_id}) does not match that "
                f"of the given build ({build.uuid})"
            )
        return build


def _normalized_job(job: dict) -> dict:
    """Flatten a jobstats event into the shape ``to_lineage_rows`` expects.

    The shared builders emit an OpenLineage-shaped event: ``sources``/``targets``
    are mirrored to the top level by ``_add_jobstats_mirror_fields``, but the job
    identity and its metadata live nested under ``job_details``. Decomposition
    wants both at the top level, so the nested block is lifted here rather than
    taught to the decomposer -- which stays independent of the W&B event shape and
    so remains usable by an importer with its own.

    Three further blocks are lifted for the same reason: the namespace from ``job``,
    and the large payloads from ``run.facets``. Each is nested somewhere the mirror
    does not reach, and each is needed by something downstream -- authorization for
    the first, the job record for the rest.

    ``job_details.job_id`` is the target run's uuid for build lineage and the
    artifact's uuid for a registered artifact, so it is a stable per-execution
    identity either way -- exactly what the rows of one job must share.
    """
    details = job.get("job_details") or {}
    normalized = {**details, **job}
    # job_details wins for the identity keys: the top level either lacks them
    # (job_id) or mirrors the same values.
    if details.get("job_id"):
        normalized["job_id"] = details["job_id"]

    # The namespace lives under the event's "job" block as
    # f"{space_name}/{build_name}" (_build_events_for_target), a third nesting the
    # decomposer does not know about. Lifted because the read path splits it to
    # recover the space and prune nodes the caller cannot see: a row without it
    # fails closed, so losing it here makes every node built from this event vanish
    # from every graph -- authorization working correctly on absent provenance,
    # which looks exactly like an empty index.
    job_block = job.get("job") or {}
    namespace = job_block.get("namespace")
    if namespace and not normalized.get("job_namespace"):
        normalized["job_namespace"] = namespace
    if job_block.get("name") and not normalized.get("job_name"):
        normalized["job_name"] = job_block["name"]

    # The large payloads sit under run.facets, a fourth nesting: the mirror lifts
    # only job_details (and job_output_stats lives inside THAT), so
    # job_input_params, execution_stats and source_code stay behind. They are
    # lifted here because the job record carries them -- one copy per execution,
    # which is what a row could not do -- and without this they would silently be
    # absent and the record would look like a source that never reported them.
    #
    # ``source_code`` is renamed to ``source_code_details``, the name the attributes
    # contract and the wire model both use.
    facets = (job.get("run") or {}).get("facets") or {}
    for facet_key, flat_key in (
        ("job_input_params", "job_input_params"),
        ("execution_stats", "execution_stats"),
        ("source_code", "source_code_details"),
    ):
        value = facets.get(facet_key)
        if value and not normalized.get(flat_key):
            normalized[flat_key] = value

    # The run's own identity and the event envelope, for the job record's RUN
    # group: kept so the record is as close as possible to what was emitted.
    run_block = job.get("run") or {}
    for flat_key, value in (
        ("run_id", run_block.get("runId")),
        ("run_tags", facets.get("tags")),
        ("event_type", job.get("eventType")),
        ("event_time", job.get("eventTime")),
    ):
        if value and not normalized.get(flat_key):
            normalized[flat_key] = value
    return normalized


def job_tags(
    job: dict,
    build_id: str,
    target_run_uuid: str,
    extra_tags: Optional[Iterable[str]] = None,
) -> List[str]:
    """The tags a job is filterable by, as ``k=v`` strings.

    Three sources, all free-form -- a build is just one kind of tag:

    - the originating ids (``build_id``, ``target_run_uuid``), the same values the
      attributes blob carries under ``origin.ids``, which is not queryable;
    - the entry's ``run.facets.tags``, the map the W&B sink turns into run tags
      (``build_id``, ``target_id``, ``username``, ``space_name``, ``output_id``, ...),
      so a filter that works against W&B works here too;
    - ``extra_tags`` verbatim, e.g. the user's build tags.

    Empty values are dropped rather than stored as ``k=``. A tag that does not fit
    its column is logged and dropped, never truncated.
    """
    tags: List[str] = []
    for key, value in (("build_id", build_id), ("target_run_uuid", target_run_uuid)):
        if value:
            tags.append(f"{key}={value}")
    facet_tags = ((job.get("run") or {}).get("facets") or {}).get("tags") or {}
    if isinstance(facet_tags, dict):
        for key, value in facet_tags.items():
            if value:
                tags.append(f"{key}={value}")
    tags.extend(extra_tags or [])

    storable = []
    for tag in dict.fromkeys(tags):
        if is_storable_tag(tag):
            storable.append(tag)
        else:
            logger.warning("Dropping unstorable lineage job tag %r", tag)
    return storable


def _job_from_metadata(
    job_metadata: dict,
    build_id: str,
    target_run_uuid: str,
    source_system: str = SOURCE_SYSTEM,
) -> Optional[StoredLineageJob]:
    """Turn a normalized job dict into the stored job record.

    One record per execution, which is what lets it carry the four large payloads --
    ``job_input_params``, ``execution_stats``, ``job_output_stats``,
    ``source_code_details`` -- that a *row* cannot, because a row is one of N*M and
    would hold N*M copies of each.

    ``space_name`` is derived from ``job_namespace``, which the producers spell
    ``"<space_name>/<build_name>"``: the space is the part before the first ``/``, the
    same split the read path uses to prune what a caller cannot see. It is stored as
    its own column rather than recomputed per read, but the namespace is kept
    verbatim too, so the derived value can always be checked against its source.

    Returns:
        The record, or ``None`` when the metadata carries no ``job_id`` -- there is
        no identity to store it under, and a blank key would collide with every other
        identity-less job under the unique index.
    """
    job_id = str(job_metadata.get("job_id") or "")
    if not job_id:
        return None

    namespace = str(job_metadata.get("job_namespace") or "")
    space_name = namespace.split("/", 1)[0] if namespace else ""

    return StoredLineageJob(
        job_id=job_id,
        job_namespace=namespace,
        space_name=space_name,
        owner=str(job_metadata.get("owner") or ""),
        status=str(job_metadata.get("job_status") or ""),
        started_at=str(job_metadata.get("job_started_at") or ""),
        attributes=build_job_attributes(
            job_metadata=job_metadata,
            source_system=source_system,
            ids={"build_id": build_id, "target_run_uuid": target_run_uuid},
        ),
    )


def _fill_job_from_record(row: StoredLineageRow, record: StoredLineageJob) -> None:
    """Copy the record's job fields onto a prebuilt row's ``job`` group.

    An importer that builds rows itself (Lakehouse) leaves them at ``{name, type,
    id}``, while the record holds the namespace, owner, status and span. The
    ``lineage-index`` routes read the index alone, and the access filter reads
    ``namespace`` off the row, so they are carried here. What the row already
    says wins; values keep the record's own form.
    """
    record_job = job_detail(record.attributes)
    fill = {
        JOB_NAMESPACE: record.job_namespace or record_job.get(JOB_NAMESPACE),
        JOB_OWNER: record.owner or record_job.get(JOB_OWNER),
        JOB_STATUS: record.status or record_job.get(JOB_STATUS),
        JOB_STARTED_AT: record.started_at or record_job.get(JOB_STARTED_AT),
        JOB_COMPLETED_AT: record_job.get(JOB_COMPLETED_AT),
    }
    attributes = dict(row.attributes or {})
    job = dict(attributes.get(JOB) or {})
    for key, value in fill.items():
        if value and not job.get(key):
            job[key] = str(value)
    attributes[JOB] = job
    row.attributes = attributes


def _row_from_draft(
    draft,
    build_id: str,
    target_run_uuid: str,
    source_system: str = SOURCE_SYSTEM,
    job_store: JobStore = JobStore.TARGETS,
) -> StoredLineageRow:
    """Turn a decomposed draft into the stored row.

    An empty endpoint stays :data:`TERMINAL` (``""``), not NULL: in SQL, NULL never
    equals NULL, so NULL endpoints would slip past the unique index and leave
    creation/deletion rows as the only ones a re-ingest could duplicate.

    Everything that is not ``job_id``/``input``/``output`` goes into the row's
    ``attributes`` blob, whose shape is defined by
    :mod:`gbserver.lineage.attributes` -- including the two process ids, which are
    deliberately not columns: a build and a target run are granite.build's own
    concepts and are empty for every imported source, so indexing them would index
    blanks over most of the table.

    Args:
        draft: the decomposed row.
        build_id: the build this row came from; empty for lineage with no build.
        target_run_uuid: the target run this row came from; empty when there is
            none.
        source_system: which system produced the job. Not written to the row:
            the slim shape leaves it on the job record. Kept so callers need not
            change.
        job_store: where the job's full data lives; see :class:`JobStore`.
    """
    metadata = draft.metadata or {}
    return StoredLineageRow(
        job_id=draft.job_id,
        input=draft.input or TERMINAL,
        output=draft.output or TERMINAL,
        job_store=job_store,
        attributes=build_index_attributes(
            job_id=draft.job_id,
            job_name=str(metadata.get("job_name") or ""),
            job_type=str(metadata.get("job_type") or ""),
            job_namespace=str(metadata.get("job_namespace") or ""),
            owner=str(metadata.get("owner") or ""),
            job_status=str(metadata.get("job_status") or ""),
            job_started_at=str(metadata.get("job_started_at") or ""),
            job_completed_at=str(metadata.get("job_completed_at") or ""),
            input_uri=draft.input or "",
            input_artifact=draft.input_artifact,
            output_uri=draft.output or "",
            output_artifact=draft.output_artifact,
            retrieve={
                "build_id": build_id,
                "target_run_uuid": target_run_uuid,
            }
            if build_id or target_run_uuid
            else {"job_id": draft.job_id},
        ),
    )
