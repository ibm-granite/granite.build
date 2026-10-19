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

import json
from typing import List, Optional, Tuple

from sqlalchemy import and_, func, literal_column, or_

from gbserver.storage.lineage_job_storage import (
    BaseLineageJobStorage,
    ILineageJobStorage,
)
from gbserver.storage.sql.sql_storage import BaseSQLItemStorage, retry_read
from gbserver.storage.storage import JSON_COLUMN_NAME
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

    No composite index: every query here filters on one indexed column (or scans
    ``tags``, which no index serves). ``composite_indexes`` is there if one is
    ever needed.

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
        # The joined tag list has no natural width: build and target-run uuids,
        # username, space and user labels together pass 256 routinely, which
        # Postgres rejects (SQLite ignores VARCHAR lengths, so only Postgres
        # loses the record). Unindexed, so Text costs nothing.
        kwargs["text_columns"] = ["tags"]
        kwargs["default_pagination_sort_by_column"] = "recorded_at"
        super().__init__(**kwargs)

    # -- recorded_at: stamped by the database, kept only in its column -------
    #
    # The indexer checkpoints on recorded_at, so it must not be older than the
    # moment the row becomes visible. A stamp taken in Python before the write is
    # carried unchanged through every retry (up to minutes), and a row that
    # commits that late lands behind a checkpoint already moved past it. Here the
    # value is a SQL expression in the INSERT/UPDATE itself, evaluated by the
    # database on each attempt, in utc_now_iso's fixed-width form so it still
    # orders as a string.

    def _db_now(self):
        """The database's current UTC time as ``YYYY-MM-DDTHH:MM:SS.ffffff+00:00``."""
        if self._engine is not None and self._engine.dialect.name == "postgresql":
            # clock_timestamp(), not now(): now() is the transaction's start.
            return func.to_char(
                func.timezone("UTC", func.clock_timestamp()),
                'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"',
            )
        # SQLite: %f is seconds with milliseconds (SS.SSS); padded to microseconds
        # so the string keeps utc_now_iso's width.
        return literal_column("strftime('%Y-%m-%dT%H:%M:%f', 'now') || '000+00:00'")

    def _convert_item_to_row_dict(self, item: StoredLineageJob) -> dict:
        """The row, with ``recorded_at`` out of the JSON blob: its column is the
        one copy, so the two cannot disagree."""
        row = super()._convert_item_to_row_dict(item)
        blob = json.loads(row[JSON_COLUMN_NAME])
        blob.pop("recorded_at", None)
        row[JSON_COLUMN_NAME] = json.dumps(blob)
        return row

    def _convert_row_dict_to_item(self, item_dict: dict) -> StoredLineageJob:
        item = super()._convert_row_dict_to_item(item_dict)
        recorded_at = item_dict.get("recorded_at")
        if recorded_at:
            item.recorded_at = str(recorded_at)
        return item

    def _add_item_dicts(self, items: list) -> None:
        # Re-run on each retry of the base method, so every attempt is stamped.
        for row in items:
            row["recorded_at"] = self._db_now()
        super()._add_item_dicts(items)

    def _on_update_row(self, db_item) -> None:
        # Stamped in the same transaction as the change, on every retry: a
        # merged record must move past the checkpoint exactly when its new
        # content commits, or the indexer never re-reads it.
        db_item.recorded_at = self._db_now()

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

        conditions = [_contains(column, tag) for tag in wanted_all]
        if wanted_any:
            conditions.append(or_(*(_contains(column, tag) for tag in wanted_any)))
        candidates = self._tag_candidates(conditions)

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

    @retry_read
    def get_recorded_after(
        self,
        cutoff: Optional[str],
        after: Optional[Tuple[str, str]],
        limit: int,
    ) -> List[StoredLineageJob]:
        """Keyset page by ``(recorded_at, job_id)``: a range on the ``recorded_at``
        index, ``job_id`` breaking ties. A merge only moves ``recorded_at``
        forward, which moves a record ahead of the cursor, so it is read later and
        nothing behind the cursor shifts or is skipped (offset paging cannot say
        that)."""
        if not self._ensure_table_without_retry():
            return []
        model = self._sql_alchemy_model
        session = self._get_session_without_retry()
        try:
            query = session.query(model).filter(
                model.recorded_at.isnot(None), model.recorded_at != ""
            )
            if cutoff:
                query = query.filter(model.recorded_at >= cutoff)
            if after is not None:
                query = query.filter(
                    or_(
                        model.recorded_at > after[0],
                        and_(model.recorded_at == after[0], model.job_id > after[1]),
                    )
                )
            rows = query.order_by(model.recorded_at, model.job_id).limit(limit).all()
            assert self._column_types is not None
            return [
                self._convert_row_dict_to_item(
                    {name: getattr(row, name) for name in self._column_types}
                )
                for row in rows
            ]
        finally:
            session.close()

    @retry_read
    def _tag_candidates(self, conditions: list) -> List[Tuple[str, str]]:
        """``(job_id, joined tags)`` of the rows ``conditions`` narrow to, newest first."""
        model = self._sql_alchemy_model
        session = self._get_session_without_retry()
        try:
            query = session.query(model.job_id, model.tags)
            if conditions:
                query = query.filter(and_(*conditions))
            return list(query.order_by(model.recorded_at.desc()).all())
        finally:
            session.close()


_LIKE_ESCAPE = "\\"


def _contains(column, tag: str):
    """``column LIKE %tag%`` with the tag's own ``%`` and ``_`` matched literally.

    Unescaped, a tag like ``run_1`` would also narrow to ``runX1``; the exact
    re-check would drop it, but ``%`` alone would match every row.
    """
    escaped = (
        tag.replace(_LIKE_ESCAPE, _LIKE_ESCAPE * 2)
        .replace("%", _LIKE_ESCAPE + "%")
        .replace("_", _LIKE_ESCAPE + "_")
    )
    return column.like(f"%{escaped}%", escape=_LIKE_ESCAPE)
