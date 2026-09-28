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

"""SQL storage implementation for lineage job records."""

from gbserver.storage.lineage_job_storage import (
    BaseLineageJobStorage,
    ILineageJobStorage,
)
from gbserver.storage.sql.sql_storage import BaseSQLItemStorage
from gbserver.storage.stored_lineage_job import StoredLineageJob


class SQLLineageJobStorage(
    BaseSQLItemStorage[StoredLineageJob],
    BaseLineageJobStorage,
    ILineageJobStorage,
):
    """SQL-based storage implementation for lineage job records.

    Schema decisions, and why each column is indexed:

    - ``job_id`` is the identity and every lookup's key -- a point fetch, and the
      batched ``WHERE job_id IN (...)`` a read path uses to enrich a set of rows.
      Unique as well as indexed: a job record is written whole, so a second row for
      the same execution is a duplicate rather than a variant.
    - ``space_name`` and ``owner`` are indexed because they are how a job listing is
      scoped, and because they are what an access filter narrows on -- a predicate
      that has to be cheap since it applies to every request that reaches this table.
    - ``started_at`` is indexed to order a listing by recency without a scan. It is
      text, not a timestamp, holding the source's own spelling (see
      :mod:`gbserver.storage.stored_lineage_job`), so ordering is lexicographic --
      correct for the zero-padded forms the producers emit, and the reason the column
      must not be "helpfully" converted to ``DateTime``.
    - ``recorded_at`` is this index's own write time, UTC ISO-8601, and the default
      pagination order. Indexed because it is what a future high-water-mark
      incremental import will range over. Deliberately not in the unique key: it
      differs on every write, so including it would defeat the dedup.

    There is no autoincrement ``index`` column: nothing reads it, so ``uuid`` is the
    primary key.

    ``indexed_columns`` is single-column only; this layer has no composite
    non-unique index, and a composite is reachable only as a side effect of
    ``unique_columns`` with a tuple key. Nothing here needs one.

    Every indexed column is text. ``get_by_where`` builds an ``IN`` clause only for
    string-typed columns and silently degrades to ``column == [list]`` otherwise, so a
    non-text indexed column is a latent wrong-results bug rather than merely a slow
    one.

    No column is named ``input`` or ``output``: those names are in the SQL layer's
    ``_WIDE_STRING_COLUMNS``, which is keyed on column name and applies to every table
    in the system, so using one here would silently widen it to the row table's URI
    width.
    """

    def __init__(self, **kwargs) -> None:
        kwargs["indexed_columns"] = [
            "job_id",
            "space_name",
            "owner",
            "started_at",
            "recorded_at",
        ]
        # One row per execution. A second write for the same job is a no-op rather
        # than a duplicate, which is what makes re-recording idempotent and lets the
        # sink's presence check be an optimization rather than a correctness
        # requirement.
        #
        # Note that unique indexes are only created WITH the table, so this cannot be
        # added later without recreating it -- and __create_unique_indexes only
        # *warns* on failure, so an index the backend rejects costs idempotence
        # silently rather than loudly. That is why the promoted columns are held to
        # the inferred 256-char width instead of being widened.
        kwargs["unique_columns"] = {"job_id": None}
        kwargs["default_pagination_sort_by_column"] = "recorded_at"
        super().__init__(**kwargs)
