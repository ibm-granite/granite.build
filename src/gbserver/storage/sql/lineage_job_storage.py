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

from typing import List, Optional, Tuple

from sqlalchemy import and_, or_

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
        # Tags are stored as "tag1,tag2,tag3", so the generic where-clause path
        # narrows with %like% in SQL and then re-checks exact membership in Python
        # -- the same handling gb_artifacts and gb_builds use for theirs. Declared
        # here so a caller passing {"tags": [...]} cannot silently get the
        # column == [list] degradation instead of a filter.
        kwargs["exact_liked_list_columns"] = {"tags": "tags"}
        kwargs["default_pagination_sort_by_column"] = "recorded_at"
        super().__init__(**kwargs)

    def _ensure_table(self) -> bool:
        """Initialize the model if needed; whether the table exists to query.

        ``__initialize_storage`` is name-mangled private, so this replicates it
        through the protected API, as the row storage and ``SQLSpaceUserStorage``
        do.
        """
        if self._sql_alchemy_model is None:
            sample = self._convert_item_to_row_dict(self._get_sample_item())
            self._create_or_adjust_schema_item_dict(sample)
        return self._does_table_exist()

    def search_by_tags(
        self,
        any_of: List[str],
        all_of: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[int, List[StoredLineageJob]]:
        """Match tags in SQL: a disjunction for ``any_of``, conjunction for ``all_of``.

        Not ``get_by_where``'s list handling, which chains one filter per value and
        is therefore an AND. ``any_of`` has to be an OR to mean what W&B's ``$in``
        means, so the clauses are built here: ``or_()`` over ``any_of``, chained
        ``and_()`` over ``all_of``.

        ``LIKE`` over the joined column over-matches -- ``%team=nlp%`` also hits
        ``team=nlp-eval`` -- so SQL only narrows the candidates, and each one is
        re-checked against its own tag list here. The total counts the re-checked
        set, which is why it is computed here rather than taken from SQL.

        SQL selects ids, not whole rows, and the page is hydrated through
        :meth:`get_jobs`: the same path every other read uses, so a record comes
        back built the one way rather than two.
        """
        wanted_any = [tag for tag in any_of if tag]
        wanted_all = [tag for tag in (all_of or []) if tag]
        if not self._ensure_table():
            return 0, []
        model = self._sql_alchemy_model
        assert model is not None
        column = model.tags

        conditions = [column.like(f"%{tag}%") for tag in wanted_all]
        if wanted_any:
            conditions.append(or_(*(column.like(f"%{tag}%") for tag in wanted_any)))

        session = self._get_session_without_retry()
        try:
            query = session.query(model.job_id, model.tags)
            if conditions:
                query = query.filter(and_(*conditions))
            candidates = query.order_by(model.recorded_at.desc()).all()
        finally:
            session.close()

        exact_any, exact_all = set(wanted_any), set(wanted_all)
        job_ids = []
        for job_id, joined in candidates:
            tags = {tag for tag in (joined or "").split(",") if tag}
            if not exact_all <= tags:
                continue
            if exact_any and exact_any.isdisjoint(tags):
                continue
            job_ids.append(job_id)

        page = job_ids[offset : offset + limit]
        by_id = self.get_jobs_by_id(page)
        return len(job_ids), [by_id[j] for j in page if j in by_id]
