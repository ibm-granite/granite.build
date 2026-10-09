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

"""The ``db`` lineage store: records executions into ``gb_lineage_job``.

One source, this store's own table. That is the ``ILineageStore`` contract --
each store consults exactly one source, this one its job table, the W&B store the
W&B API, the no-op store nothing -- and it is why nothing here touches
``gb_lineage_index``. The index is a *derived* artifact keyed by URI, owned and
written by the lineage indexer (:mod:`gbserver.lineage.row_indexing`), which
reads these records back and decomposes them into edges.

Each recorded execution is one job record, with the emitted entry kept verbatim
under :data:`ENTRY_ATTRIBUTE` so the indexer has something to derive from. The
write is idempotent under the table's unique index on ``job_id``, so re-recording
merges rather than duplicating.

**The job entries are not built here.** ``create_jobstats_for_target`` in
``wandb_jobstats`` already turns a target run into job entries -- resolving input
and output artifacts, collecting step configs, redacting secret-named keys -- and
``_add_jobstats_mirror_fields`` already exposes them as top-level
``sources``/``targets``, which is exactly the shape ``to_lineage_rows`` consumes.
Reusing it means the two sinks cannot disagree about what a build's lineage *is*;
re-deriving it here would fork that logic and let them drift. Only the module
functions are reused, never ``WandBLineageStore`` itself, so nothing here needs
``wandb`` installed or configured.

Dedup is by presence of ``job_id``, not by count: a record is written whole, so
there is no partial state for a count to detect. The reconciler's
``expected_counts`` is therefore accepted and ignored.
"""

import logging
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from gbserver.lineage.attributes import (
    build_job_attributes,
    origin_id,
)
from gbserver.lineage.decompose import LineageDecomposeError, to_lineage_rows
from gbserver.lineage.jobstats import ILineageStore
from gbserver.lineage.merge import ADDED, UPDATED, upsert_job
from gbserver.storage.artifact_registration import ArtifactRegistration
from gbserver.storage.lineage_job_storage import ILineageJobStorage
from gbserver.storage.singleton_storage import SingletonAdminStorage
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_lineage_job import StoredLineageJob
from gbserver.storage.stored_lineage_row import JobStore, utc_now_iso
from gbserver.storage.stored_target_run import StoredTargetRun

logger = logging.getLogger(__name__)

# Longest tag kept. Tags live in a JSON blob, so this is no column's width; it
# only keeps a runaway value out of a job's record.
MAX_TAG_LENGTH = 256


def is_storable_tag(tag: object) -> bool:
    """Whether ``tag`` is kept: a non-empty string no longer than the limit."""
    return isinstance(tag, str) and 0 < len(tag) <= MAX_TAG_LENGTH


# Names the system that produced a job, so what this store recorded stays
# distinguishable from what an importer supplied. Surfaced on a run node as
# ``source_system``; an importer passes its own name.
SOURCE_SYSTEM = "granite.build"

# Key in a job record's ``attributes`` holding the job entry as emitted, so the
# lineage indexer can decompose it into index rows later. Every record this store
# writes carries it -- deriving the index is the indexer's only way in.
ENTRY_ATTRIBUTE = "entry"


class DBLineageStore(ILineageStore):
    """Record lineage into the local lineage index.

    Each recorded execution produces two writes: the N*M lineage rows, and one job
    record holding the metadata a row does not carry (the large payloads, and the
    fields promoted to job columns). Both are idempotent under their own unique
    index, so re-recording is a no-op rather than a duplicate.

    Args:
        job_storage: the lineage job storage to write. Defaults to the
            process-wide admin storage, resolved lazily so importing this module
            does not require a configured database.
    """

    def __init__(
        self,
        job_storage: Optional[ILineageJobStorage] = None,
    ) -> None:
        self._job_storage = job_storage

    @property
    def job_storage(self) -> ILineageJobStorage:
        """The lineage job storage, resolved on first use.

        Raises rather than degrading: this table is the store's *only* source, so
        a missing one leaves nothing to record into and nothing to answer a query
        from. It used to return ``None`` and log a warning, which was tenable
        only while the index rows were also written here -- lineage still landed,
        minus the record's status, owner and step params. With the rows now the
        indexer's (:mod:`gbserver.lineage.row_indexing`), a silent ``None`` would
        make every write a no-op and every count zero, which reads as "no lineage
        yet" rather than as a broken store.
        """
        if self._job_storage is None:
            from gbserver.storage.singleton_storage import get_admin_storage

            self._job_storage = get_admin_storage().lineage_job_storage
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

        if self._has_job_record(target.uuid):
            # Already recorded. Presence, not count: see the module docstring for
            # why a count comparison would re-record forever.
            #
            # Keyed on target.uuid because that IS the job id of build lineage --
            # the event builder sets ``job_details.job_id = targetrun.uuid``
            # (``wandb_jobstats.py:274``), so per-job dedup is per-target dedup
            # here without any table needing a column for a target run.
            #
            # The job record alone decides, because it is the only thing this
            # store writes. The index rows are the indexer's, derived from this
            # record afterwards, so their presence or absence says nothing about
            # whether *this* store has recorded the target.
            logger.debug("Target run %s already has a lineage job record", target.uuid)
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
        """Record one job entry into ``gb_lineage_job``.

        The entry is kept verbatim under :data:`ENTRY_ATTRIBUTE` so the lineage
        indexer can derive the index rows from it later; this store writes no
        rows itself.

        A job that cannot be decomposed is logged and skipped rather than aborting
        the scan: one unrecordable target must not stop the rest of a build's
        lineage from landing. The reason is logged with it -- a bare "skipping"
        makes lineage loss undiagnosable. The decomposition is discarded here and
        redone by the indexer; it runs as a *gate*, so an entry that could never
        become rows is rejected at the door rather than stored to fail on every
        later scan.

        ``job_store`` is accepted for interface symmetry with :meth:`write_job`
        and is not stored: where a job's full data lives is a property of the
        index row, which the indexer sets when it derives one.
        """
        try:
            to_lineage_rows(_normalized_job(job))
        except LineageDecomposeError as exc:
            logger.warning(
                "Job entry could not be decomposed into lineage rows; skipping "
                "(build=%s, target_run=%s): %s",
                build_id,
                target_run_uuid,
                exc,
            )
            return

        self._add_job(
            _normalized_job(job),
            build_id=build_id,
            target_run_uuid=target_run_uuid,
            entry={
                "events": [job],
                "build_id": build_id,
                "target_run_uuid": target_run_uuid,
                "extra_tags": list(extra_tags or []),
            },
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

        With ``entry``, a merge that changed the record also moves ``recorded_at``
        forward: the indexer reads by it, and would otherwise never see what the
        merge added.

        **The entry is written over, not merged.** Everything else fills blanks and
        keeps what is stored (see :func:`merge_attributes`), which is right for
        provenance that only accrues -- but the entry is a snapshot of the latest
        emission, and its lists are *values*, not blanks. Merging it would keep the
        first version forever: re-recording a target with a tag added would store
        the old tag list, the indexer would derive rows from that, and the new tag
        would vanish with nothing logged.

        So a changed entry is written over after the merge, and on its own account:
        the merge reports ``UNCHANGED`` precisely when the entry was the *only*
        difference, so keying this off ``UPDATED`` would skip the one case it exists
        for. ``recorded_at`` moves with it, because that is what the indexer reads
        by -- without it the new entry would sit behind the checkpoint, never
        re-read, and the index would keep serving rows derived from the old one.

        **The entry's events accumulate.** One execution can emit several entries
        under a single ``job_id`` -- the builder emits one per output artifact, so a
        target with two outputs arrives as two calls here -- while the table holds
        one record per ``job_id``. Replacing the entry would keep only the last
        event and the index would lose every other output's edges silently. So the
        events are unioned, and the indexer derives rows from all of them.
        """
        storage = self.job_storage
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
            result = upsert_job(storage, job)
            if result == ADDED or entry is None:
                return
            stored = storage.get_job(job.job_id)
            if stored is None:
                return
            merged = _merge_entry((stored.attributes or {}).get(ENTRY_ATTRIBUTE), entry)
            fields: dict = {}
            if merged != (stored.attributes or {}).get(ENTRY_ATTRIBUTE):
                attributes = dict(stored.attributes or {})
                attributes[ENTRY_ATTRIBUTE] = merged
                fields["attributes"] = attributes
            if fields or result == UPDATED:
                fields["recorded_at"] = utc_now_iso()
                storage.update_fields(stored.uuid, fields)
        except Exception:
            # A warning, and the only trace: the record is all this store writes,
            # so a failure here means the execution went unrecorded entirely and
            # the index will have nothing to derive from. The next scan retries it
            # (see _has_job_record), which is why this does not raise.
            logger.warning(
                "Lineage job could not be added or merged (job=%s)",
                job.job_id,
                exc_info=True,
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
        """Count the lineage **jobs** recorded for a release.

        ``release_id`` is a ``build_id`` for build lineage and the artifact's uuid
        for a registered artifact -- W&B's convention, kept so the same argument
        works against either sink.

        Counted against ``gb_lineage_job``, this store's only source. Neither id
        is a column there -- both live in the ``attributes`` blob under
        ``origin.ids`` -- so this pages and filters in Python. The scan is the
        honest cost of answering a question this schema is not organized around,
        as opposed to adding a column that would be blank on every imported row.
        Measured at ~1s over 323k records, against an occasional caller.

        **The unit is one job per execution**, which is W&B's run shape, so a
        count computed there is comparable to this one. It used to count index
        *rows* (one per input/output pair), which never matched: a build with 2
        inputs and 5 outputs is 5 runs in W&B and was 10 rows here.

        Args:
            release_id: the build uuid, or the artifact uuid.
            target_id: optional target run to narrow to.

        Returns:
            How many jobs are recorded. ``0`` for an unknown release.
        """
        if not release_id:
            return 0

        matched = 0
        for page in self.job_storage.get_paged():
            for job in page:
                if origin_id(job.attributes, "build_id") != release_id:
                    continue
                if target_id and (
                    origin_id(job.attributes, "target_run_uuid") != target_id
                ):
                    continue
                matched += 1
        return matched

    def does_release_id_exist(
        self, release_id: str, expected_count: int, target_id: Optional[str] = None
    ) -> bool:
        """Whether a release has exactly ``expected_count`` jobs recorded.

        ``expected_count`` is the caller's W&B-shaped count (one run per output
        artifact), and :meth:`count_release_ids` now answers in that same unit, so
        the comparison is meaningful rather than merely type-correct.
        """
        return self.count_release_ids(release_id, target_id) == expected_count

    def filter_unrecorded(
        self,
        target_ids: set[str],
        expected_counts: Optional[dict[str, int]] = None,
        on_query_error: Optional[Callable[[Exception], None]] = None,
    ) -> set[str]:
        """Return the candidates that have no job record yet.

        ``expected_counts`` is accepted and **ignored**: completeness is by
        presence. A job record is written once per execution, whole, so it either
        exists or it does not -- there is no partial state for a count to detect.

        The candidates are target run uuids, and they are looked up as *job* ids,
        which is correct rather than a coincidence: the event builder stamps
        ``job_details.job_id = targetrun.uuid`` (``wandb_jobstats.py:274``), so a
        target run and its job share one identifier. That is what lets dedup work
        without any table carrying a ``target_run_uuid`` column.

        Resolved by indexed ``job_id`` lookup (``uq_gb_lineage_job_job_id``), which
        matters: this runs on every watcher tick and is the only thing preventing
        duplicate records.

        Fails **open**, and invokes ``on_query_error``: on a query failure every
        candidate is reported unrecorded. Re-recording is idempotent, so that is
        harmless, and the reconciler is what decides not to record at all (it fails
        closed on the callback). Without the callback this would silently duplicate
        work rather than skip it.
        """
        if not target_ids:
            return set()
        try:
            recorded = self.recorded_by_self(target_ids)
        except Exception as exc:
            logger.warning("Lineage dedup query failed; treating all as unrecorded")
            if on_query_error is not None:
                on_query_error(exc)
            return set(target_ids)
        return {target_id for target_id in target_ids if target_id not in recorded}

    def recorded_by_self(self, job_ids: Iterable[str]) -> Set[str]:
        """Narrow ``job_ids`` to those already holding a job record.

        A target run another source imported first -- Lakehouse keeps granite.build
        runs under ``job_id = targetrun.uuid`` -- may already be recorded without
        this system's view of it: no namespace, which the read path needs to
        authorize it. The write merges into the imported copy rather than
        duplicating it, so a record's existence is enough to skip re-recording.

        Dedup is by job_id alone: if the job record exists, it was already written
        by some source. The merge logic fills in any blanks the first writer missed.
        """
        job_ids = set(job_ids)
        if not job_ids:
            return job_ids
        jobs = self.job_storage.get_jobs_by_id(list(job_ids))
        return {job_id for job_id in job_ids if job_id in jobs}

    def _has_job_record(self, job_id: str) -> bool:
        """Whether the job table holds ``job_id``."""
        return job_id in self.job_storage.get_jobs_by_id([job_id])

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


def _entry_events(entry: Optional[dict]) -> List[dict]:
    """The events an entry carries, in either shape.

    ``events`` is the current one. ``event`` is what records written before the
    accumulation fix hold, and they are still in the table, so both are read; a
    record is never rewritten just to change its shape.
    """
    if not entry:
        return []
    events = entry.get("events")
    if isinstance(events, list):
        return [event for event in events if event]
    single = entry.get("event")
    return [single] if single else []


def _merge_entry(stored: Optional[dict], incoming: dict) -> dict:
    """Union ``incoming``'s events into ``stored``'s, keeping the newest metadata.

    One ``job_id`` can receive several entries (one per output artifact), so the
    events accumulate rather than replace -- see :meth:`DBLineageStore._add_job`.
    An event already present is not duplicated, which is what keeps re-recording
    idempotent; equality is on the whole event, so a *changed* event is kept
    alongside rather than silently dropped, and the rows it decomposes to merge
    under their own unique index.
    """
    events = _entry_events(stored)
    for event in _entry_events(incoming):
        if event not in events:
            events.append(event)
    merged = dict(incoming)
    merged.pop("event", None)
    merged["events"] = events
    return merged


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
