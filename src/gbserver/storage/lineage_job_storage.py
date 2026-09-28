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

"""Base storage interface and implementation for lineage job records.

Every query here is keyed on ``job_id``, which is indexed and unique, so each is a
single indexed lookup. The read paths fetch jobs for a set of rows they already hold,
so the batched :meth:`BaseLineageJobStorage.get_jobs` is the one that matters: a
graph walk resolves all of a level's jobs in one query rather than one per row.
"""

from typing import Dict, List, Optional

from gbserver.storage.storage import BaseItemStorage, IItemStorage
from gbserver.storage.stored_lineage_job import StoredLineageJob
from gbserver.types.constants import GB_LINEAGE_JOB_TABLE_NAME


class ILineageJobStorage(IItemStorage[StoredLineageJob]):
    """Interface for lineage job storage implementations."""

    def get_job(self, job_id: str) -> Optional[StoredLineageJob]:
        """Return the job record for ``job_id``, or ``None``."""
        raise NotImplementedError

    def get_jobs(self, job_ids: List[str]) -> List[StoredLineageJob]:
        """Return the job records for ``job_ids``, in one batched query."""
        raise NotImplementedError

    def get_jobs_by_id(self, job_ids: List[str]) -> Dict[str, StoredLineageJob]:
        """Return the job records for ``job_ids``, keyed by ``job_id``."""
        raise NotImplementedError

    def has_job(self, job_id: str) -> bool:
        """Whether a job record is already stored for ``job_id``."""
        raise NotImplementedError


class BaseLineageJobStorage(BaseItemStorage[StoredLineageJob], ILineageJobStorage):
    """Base storage implementation for lineage job records.

    Provides the shared query logic across backends (SQL, SQLite).
    """

    def __init__(self, **kwargs) -> None:
        kwargs["item_class"] = StoredLineageJob
        if kwargs.get("table_name") is None:
            kwargs["table_name"] = GB_LINEAGE_JOB_TABLE_NAME
        super().__init__(**kwargs)

    def _get_column_values(self, item: StoredLineageJob) -> dict:
        """Extract the queryable columns from a job record.

        Every field returned here becomes a real column; everything else lives in the
        JSON blob, which is Text and therefore neither queryable nor indexable. The
        set is returned unconditionally -- a conditionally omitted key would be
        missing from the schema derived from the sample item.

        **Every promoted column is a string, by construction.** ``get_by_where``
        builds an ``IN`` clause only for string-typed columns and otherwise degrades
        *silently* to ``column == [list]`` -- a meaningless predicate that returns
        plausible but wrong rows with no error. Keeping the promoted set all-text
        makes that failure unreachable rather than merely avoided by convention.
        """
        fields_to_include = {
            "job_id",
            "job_namespace",
            "space_name",
            "owner",
            "source_system",
            "status",
            "started_at",
            "recorded_at",
        }
        return item.model_dump(include=fields_to_include)

    @classmethod
    def _get_sample_item(cls) -> StoredLineageJob:
        """Return a sample record used to derive the table schema.

        Every column must be present and of its real type here: the SQL layer infers
        each column's type from this item's values. So this is the schema, not an
        example of one.
        """
        return StoredLineageJob(
            job_id="sample-job",
            job_namespace="sample-space/sample-build",
            space_name="sample-space",
            owner="sample-user",
            source_system="granite.build",
            status="SUCCEEDED",
            started_at="2026-01-01 00:00:00",
            recorded_at="2000-01-01T00:00:00.000000+00:00",
        )

    def get_job(self, job_id: str) -> Optional[StoredLineageJob]:
        """Return the job record for ``job_id``, or ``None`` if there is none.

        ``job_id`` is unique, so at most one row matches; a second row for the same
        job cannot exist to be chosen between.
        """
        if not job_id:
            return None
        jobs = self.get_by_where({"job_id": job_id})
        return jobs[0] if jobs else None

    def get_jobs(self, job_ids: List[str]) -> List[StoredLineageJob]:
        """Return the job records for ``job_ids`` in one batched, indexed query.

        Args:
            job_ids: the executions to fetch. Empty values are dropped and duplicates
                collapsed, so a caller can pass the ``job_id`` of every row in a
                frontier without filtering first.

        Returns:
            The records that exist. A ``job_id`` with no record is simply absent --
            rows can outlive or precede their job record, and a missing job is not an
            error for a caller that only wanted to enrich what it has.
        """
        wanted = self._batchable(job_ids)
        if not wanted:
            return []
        return self.get_by_where({"job_id": wanted})

    def get_jobs_by_id(self, job_ids: List[str]) -> Dict[str, StoredLineageJob]:
        """Return the job records for ``job_ids``, keyed by ``job_id``.

        The shape every read path actually wants: it holds rows and needs each row's
        job, so it looks up by key rather than scanning a list per row.
        """
        return {job.job_id: job for job in self.get_jobs(job_ids) if job.job_id}

    @staticmethod
    def _batchable(identifiers: List[str]) -> List[str]:
        """Drop empty ids and duplicates from a batch.

        Sorted so the generated query is stable, which keeps it comparable in logs
        and cacheable by the backend.
        """
        return sorted({value for value in identifiers if value})

    def has_job(self, job_id: str) -> bool:
        """Whether a job record is already stored for ``job_id``.

        The sink's presence check: a job record is written whole, so it either exists
        or it does not. Keyed on ``job_id`` for the same reason the row table's dedup
        is -- it is the only identifier every lineage source has by definition.
        """
        if not job_id:
            return False
        return bool(self.get_by_where({"job_id": job_id}))
