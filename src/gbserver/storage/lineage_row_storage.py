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

"""Base storage interface and implementation for lineage rows.

The queries here are the ones the traversal and the sink need: one batched hop in
each direction, seeding by build, and a presence check for dedup. Each is a single
indexed query, so a graph walk costs one query per level rather than a scan.
"""

from typing import List

from gbserver.storage.storage import BaseItemStorage, IItemStorage
from gbserver.storage.stored_lineage_row import TERMINAL, StoredLineageRow
from gbserver.types.constants import GB_LINEAGE_TABLE_NAME


class ILineageRowStorage(IItemStorage[StoredLineageRow]):
    """Interface for lineage row storage implementations."""

    def get_rows_by_input(self, inputs: List[str]) -> List[StoredLineageRow]:
        """Return rows whose ``input`` is one of ``inputs`` (descendant hop)."""
        raise NotImplementedError

    def get_rows_by_output(self, outputs: List[str]) -> List[StoredLineageRow]:
        """Return rows whose ``output`` is one of ``outputs`` (ancestor hop)."""
        raise NotImplementedError

    def get_rows_by_job(self, job_id: str) -> List[StoredLineageRow]:
        """Return every row of one job execution."""
        raise NotImplementedError

    def has_rows_for_job(self, job_id: str) -> bool:
        """Whether any row is already recorded for a job."""
        raise NotImplementedError

    def get_recorded_jobs(self, job_ids: List[str]) -> set:
        """Return which of ``job_ids`` already have rows."""
        raise NotImplementedError


class BaseLineageRowStorage(BaseItemStorage[StoredLineageRow], ILineageRowStorage):
    """Base storage implementation for lineage rows.

    Provides the shared query logic across backends (SQL, SQLite).
    """

    def __init__(self, **kwargs) -> None:
        kwargs["item_class"] = StoredLineageRow
        if kwargs.get("table_name") is None:
            kwargs["table_name"] = GB_LINEAGE_TABLE_NAME
        super().__init__(**kwargs)

    def _get_column_values(self, item: StoredLineageRow) -> dict:
        """Extract the queryable columns from a row.

        Every field returned here becomes a real column; everything else lives in
        the JSON blob, which is Text and therefore neither queryable nor
        indexable. The set is returned unconditionally -- a conditionally omitted
        key would be missing from the schema derived from the sample item.

        **Every promoted column is a string, by construction.** ``get_by_where``
        builds an ``IN`` clause only for string-typed columns and otherwise
        degrades *silently* to ``column == [list]`` -- a meaningless predicate that
        returns plausible but wrong rows with no error. Keeping the promoted set
        all-text makes that failure unreachable rather than merely avoided by
        convention.
        """
        fields_to_include = {
            "job_id",
            "input",
            "output",
            "recorded_at",
        }
        return item.model_dump(include=fields_to_include)

    @classmethod
    def _get_sample_item(cls) -> StoredLineageRow:
        """Return a sample row used to derive the table schema.

        Every column must be present and of its real type here: the SQL layer
        infers each column's type from this item's values. So this is the schema,
        not an example of one.
        """
        return StoredLineageRow(
            job_id="sample-job",
            input="lh://prod/ns/models/tbl/sample",
            output="hf://huggingface.co/models/org/sample",
        )

    def get_rows_by_input(self, inputs: List[str]) -> List[StoredLineageRow]:
        """Return rows whose ``input`` is one of ``inputs``.

        One batched, indexed query -- the descendant hop of a level-order walk.

        Args:
            inputs: normalized URIs of the current frontier. The terminal
                marker is dropped: a creation row's input identifies no artifact,
                so matching on it would join unrelated creations together.

        Returns:
            The matching rows, or an empty list when nothing is left to match.
        """
        wanted = self._batchable(inputs)
        if not wanted:
            return []
        return self.get_by_where({"input": wanted})

    def get_rows_by_output(self, outputs: List[str]) -> List[StoredLineageRow]:
        """Return rows whose ``output`` is one of ``outputs``.

        The ancestor hop; see :meth:`get_rows_by_input`.
        """
        wanted = self._batchable(outputs)
        if not wanted:
            return []
        return self.get_by_where({"output": wanted})

    @staticmethod
    def _batchable(identifiers: List[str]) -> List[str]:
        """Drop empty URIs and duplicates from a hop's frontier.

        The terminal marker must never reach a query: every creation row shares it,
        so a hop matching on it would treat unrelated creations as one node. The
        walk stops on a terminal instead of following it.
        """
        return sorted({value for value in identifiers if value and value != TERMINAL})

    def get_rows_by_job(self, job_id: str) -> List[StoredLineageRow]:
        """Return every row of one job execution.

        This is what makes the N*M decomposition lossless: the rows sharing a
        ``job_id`` still say which inputs and outputs that execution had, which is
        how the read path rebuilds a single run node from several rows.
        """
        if not job_id:
            return []
        return self.get_by_where({"job_id": job_id})

    def has_rows_for_job(self, job_id: str) -> bool:
        """Whether any row is already recorded for a job."""
        if not job_id:
            return False
        return bool(self.get_by_where({"job_id": job_id}))

    def get_recorded_jobs(self, job_ids: List[str]) -> set:
        """Return which of ``job_ids`` already have rows.

        The sink's dedup is presence-based: a job's rows are written together, so a
        job either has them or it does not. It deliberately does not compare a row
        count against the reconciler's ``expected_counts``, which counts one W&B run
        per output artifact -- a shape that never matches an N*M row count, and
        would report every job as unrecorded forever.

        Keyed on ``job_id`` rather than on any process id because ``job_id`` is the
        only identifier every lineage source has by definition. A build or a target
        run is granite.build's own concept and is absent from every imported row, so
        deduping on one would leave imported sources with no dedup at all.

        Args:
            job_ids: the job executions to check.

        Returns:
            The subset that already has at least one row.
        """
        wanted = self._batchable(list(job_ids))
        if not wanted:
            return set()
        rows = self.get_by_where({"job_id": wanted})
        return {row.job_id for row in rows if row.job_id}
