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

"""Writes the lineage index (``gb_lineage_index``) from job entries and records.

This is the indexer's half of the lineage write path. ``DBLineageStore`` records
executions into ``gb_lineage_job`` and nothing else; the index rows are *derived*
from those records, which is this module's job.

Keeping the two apart is deliberate, and it is what the ``ILineageStore``
contract means: a store consults exactly one source -- the ``db`` store its job
table, the W&B store the W&B API, the no-op store nothing. The index is not a
store's source at all, it is a derived artifact keyed by URI, so a store that
wrote it would reach across that line (and did: the indexer used to call
``index_job_record`` *on the sink*, making the store own writes to a table it has
no business knowing).

What this adds to a job entry is the decomposition: one entry with N inputs and M
outputs becomes N*M flat rows sharing a ``job_id``. That is lossy only in
appearance -- ``group_by_job`` recovers which inputs and which outputs an
execution had -- and it is what makes each row an independently indexable edge.

Every write is idempotent under the rows' unique index, so re-indexing after a
checkpoint reset writes nothing new rather than duplicating.
"""

import logging
from typing import List, Optional

from gbserver.lineage.attributes import (
    JOB,
    JOB_COMPLETED_AT,
    JOB_NAMESPACE,
    JOB_OWNER,
    JOB_STARTED_AT,
    JOB_STATUS,
    build_index_attributes,
    job_detail,
)
from gbserver.lineage.db_jobstats import (
    ENTRY_ATTRIBUTE,
    _entry_events,
    _normalized_job,
    job_tags,
)
from gbserver.lineage.decompose import LineageDecomposeError, to_lineage_rows
from gbserver.lineage.merge import rows_to_add, upsert_row
from gbserver.storage.lineage_row_storage import ILineageRowStorage, tags_to_map
from gbserver.storage.stored_lineage_job import StoredLineageJob
from gbserver.storage.stored_lineage_row import TERMINAL, JobStore, StoredLineageRow

logger = logging.getLogger(__name__)


class LineageRowIndexer:
    """Derives and writes ``gb_lineage_index`` rows.

    Owns the :class:`ILineageRowStorage` so no lineage *store* has to. The row
    storage is resolved lazily, so importing this module does not require a
    configured database.
    """

    def __init__(self, storage: Optional[ILineageRowStorage] = None) -> None:
        self._row_storage = storage

    @property
    def row_storage(self) -> ILineageRowStorage:
        """The lineage row storage, resolved on first use."""
        if self._row_storage is None:
            from gbserver.storage.singleton_storage import get_admin_storage

            self._row_storage = get_admin_storage().lineage_row_storage
        return self._row_storage

    def has_rows_for_job(self, job_id: str) -> bool:
        """Whether the index already holds rows for ``job_id``."""
        return self.row_storage.has_rows_for_job(job_id)

    def index_job_record(self, record: StoredLineageJob) -> bool:
        """Write the index rows for a job record in ``gb_lineage_job``.

        The record's ``entry`` attribute holds either ``events`` (the job entries
        the ``db`` store emitted, decomposed here) or ``rows`` (index rows an
        importer already built).

        **Every event is indexed, not just the first.** One execution emits one
        entry per output artifact under a single ``job_id``, and the store unions
        them into the record, so a target with two outputs has two events whose
        edges both belong in the index.

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
                    fill_job_from_record(row, record)
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
        events = _entry_events(entry)
        if not events:
            return False
        build_id = entry.get("build_id") or ""
        target_run_uuid = entry.get("target_run_uuid") or ""
        indexed = False
        for job in events:
            try:
                drafts = to_lineage_rows(_normalized_job(job))
            except LineageDecomposeError as exc:
                # One unindexable event must not drop the others: a record's events
                # are separate outputs of the same execution.
                logger.warning(
                    "Lineage job record could not be decomposed into rows; skipping "
                    "(job=%s): %s",
                    record.job_id,
                    exc,
                )
                continue
            self.write_rows(
                job,
                drafts,
                build_id=build_id,
                target_run_uuid=target_run_uuid,
                extra_tags=entry.get("extra_tags"),
                job_store=JobStore.LINEAGE_JOB,
            )
            indexed = True
        return indexed

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
                    fill_job_from_record(row, record)
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

    def index_job_entry(
        self,
        job: dict,
        build_id: str,
        target_run_uuid: str,
        extra_tags: Optional[List[str]] = None,
        job_store: JobStore = JobStore.TARGETS,
    ) -> bool:
        """Decompose one job entry and write its rows; whether it yielded any.

        The entry point for a caller holding an entry rather than a stored record
        -- the W&B source, which reads a run and indexes it directly. An entry that
        cannot be decomposed is logged and skipped, not raised: it may still be a
        valid job record, just not one with identifiable edges.
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
            return False
        self.write_rows(
            job,
            drafts,
            build_id=build_id,
            target_run_uuid=target_run_uuid,
            extra_tags=extra_tags,
            job_store=job_store,
        )
        return True

    def write_rows(
        self,
        job: dict,
        drafts,
        build_id: str,
        target_run_uuid: str,
        extra_tags: Optional[List[str]] = None,
        job_store: JobStore = JobStore.TARGETS,
    ) -> None:
        """Decompose-and-store every draft of one job entry."""
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
        row = row_from_draft(
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


def fill_job_from_record(row: StoredLineageRow, record: StoredLineageJob) -> None:
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


def row_from_draft(
    draft,
    build_id: str,
    target_run_uuid: str,
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
        job_store: which store holds the job's full content; see
            :class:`JobStore`. This is the row's one provenance column -- there is
            no ``origin`` or ``source_system`` column, and the producing system
            stays on the job record under ``origin.system``.
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
            retrieve=(
                {
                    "build_id": build_id,
                    "target_run_uuid": target_run_uuid,
                }
                if build_id or target_run_uuid
                else {"job_id": draft.job_id}
            ),
        ),
    )
