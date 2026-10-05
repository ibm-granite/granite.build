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

"""SQL storage implementation for lineage rows."""

from contextlib import contextmanager
from typing import List, Optional, Set

from sqlalchemy import and_, cast, distinct, func, or_
from sqlalchemy.dialects.postgresql import JSONB

from gbserver.storage.lineage_row_storage import (
    DOWNSTREAM,
    UPSTREAM,
    BaseLineageRowStorage,
    GroupedEdge,
    ILineageRowStorage,
    _tag_pairs,
    endpoint_pair,
)
from gbserver.storage.sql.sql_storage import BaseSQLItemStorage
from gbserver.storage.stored_lineage_row import TERMINAL, StoredLineageRow


class SQLLineageRowStorage(
    BaseSQLItemStorage[StoredLineageRow],
    BaseLineageRowStorage,
    ILineageRowStorage,
):
    """SQL-based storage implementation for lineage rows.

    Schema decisions, and why each column is indexed:

    - ``input`` / ``output`` carry the graph traversal. Every hop is
      ``WHERE input IN (frontier)`` or ``WHERE output IN (frontier)``, so without
      these indexes a walk degrades to a full scan per level. They are the
      composite ``(input, output)`` and ``(output, input)``, so the leading column
      serves the hop and the pair serves an edge lookup and the graph's per-edge
      ``GROUP BY`` without reading the blob. They hold
      normalized URIs, which is also what makes the *root* lookup an index seek:
      a request names an artifact by URI, so there is finally something indexed to
      match it against.
    - ``job_id`` groups the rows of one execution, which is what makes an N*M
      decomposition regroupable, and it also backs the sink's presence-based dedup,
      run on every scan. The prototype joins and groups on it but leaves it
      unindexed -- a gap corrected here.
    - ``origin`` says where the job's full content lives (granite.build, db,
      wandb, other); indexed so a migration or re-import can select by source.
    - ``recorded_at`` is this index's own write time, UTC ISO-8601, and the default
      pagination order. Indexed because it is what a future high-water-mark
      incremental import will range over. Deliberately not in the unique key: it
      differs on every write, so including it would defeat the dedup.

    There is no autoincrement ``index`` column: nothing reads it, so ``uuid`` is the
    primary key.

    There is deliberately no ``build_id`` or ``target_run_uuid`` column. Those are
    granite.build's process concepts, empty on every imported row, so indexing them
    would index blanks for most of the table. A caller wanting a process-scoped view
    resolves that scope in its own system and seeds the walk with the resulting
    URIs, which keeps this index answering exactly one question: what is the lineage
    of this artifact.

    Every indexed column is text. ``get_by_where`` builds an ``IN`` clause only for
    string-typed columns and silently degrades to ``column == [list]`` otherwise, so
    a non-text indexed column is a latent wrong-results bug rather than merely a
    slow one. ``source_system`` lives in the JSON blob rather than in a column: it is
    read to label a run node, never to filter one.
    """

    def __init__(self, **kwargs) -> None:
        kwargs["indexed_columns"] = [
            "job_id",
            "origin",
            "recorded_at",
        ]
        # (input, output) serves downstream hops and edge drill-in, (output, input)
        # upstream hops; both also answer the per-edge GROUP BY of the graph from
        # the index alone. They replace the single-column input/output indexes.
        kwargs["composite_indexes"] = [("input", "output"), ("output", "input")]
        # One row per (job, input, output). A second source reporting the same
        # relation is a no-op, which is what makes re-ingest idempotent -- a
        # property the prototype lacks entirely (it has no key at all and
        # executemany's without a guard, so it duplicates silently).
        #
        # This is why input/output hold the "" sentinel rather than NULL for
        # terminals: in SQL, NULL never equals NULL, so NULL endpoints would slip
        # past this index and leave creation/deletion rows unprotected. Note also
        # that unique indexes are only created with the table, so this cannot be
        # added later without recreating it -- and that __create_unique_indexes
        # only *warns* on failure, so an index too wide for the backend's key
        # limit costs idempotence silently. That is why the URI columns are 512
        # and not 1024 (see MAX_LINEAGE_URI_LENGTH).
        kwargs["unique_columns"] = {("job_id", "input", "output"): None}
        kwargs["default_pagination_sort_by_column"] = "recorded_at"
        super().__init__(**kwargs)

    def count_jobs_touching(
        self,
        uri: str,
        self_loop: bool = False,
        output: Optional[str] = None,
        terminal: Optional[str] = None,
    ) -> int:
        """Count the distinct jobs touching ``uri`` in one ``COUNT(DISTINCT)``.

        The artifact this exists for has 68,905 runs; counting them by reading the
        rows cost seconds per request, whatever the page size.
        """
        with self._touching(uri, self_loop, output, terminal) as query:
            if query is None:
                return 0
            return int(query.with_entities(func.count(distinct(self._job_id))).scalar())

    def get_job_ids_touching(
        self,
        uri: str,
        limit: int,
        offset: int,
        self_loop: bool = False,
        output: Optional[str] = None,
        terminal: Optional[str] = None,
    ) -> List[str]:
        """Return one page of the distinct jobs touching ``uri``, in SQL."""
        with self._touching(uri, self_loop, output, terminal) as query:
            if query is None:
                return []
            page = (
                query.with_entities(self._job_id)
                .distinct()
                .order_by(self._job_id)
                .limit(limit)
                .offset(offset)
            )
            return [job_id for (job_id,) in page.all()]

    def grouped_edges(
        self, frontier: List[str], direction: str, limit: Optional[int] = None
    ) -> List[GroupedEdge]:
        """Return one level of the walk as stacked edges, in one ``GROUP BY``.

        ``COUNT(DISTINCT job_id)`` rather than ``COUNT(*)`` for parity with the
        fallback; the ``(job_id, input, output)`` unique key makes them equal.
        """
        if direction not in (DOWNSTREAM, UPSTREAM):
            raise ValueError(f"Unknown direction: {direction!r}")
        wanted = self._batchable(frontier)
        if not wanted or not self._ensure_table():
            return []
        session = self._BaseSQLItemStorage__get_session_without_retry()
        try:
            model = self._sql_alchemy_model
            side = model.input if direction == DOWNSTREAM else model.output
            last = func.max(model.recorded_at)
            query = (
                session.query(
                    model.input,
                    model.output,
                    func.count(distinct(model.job_id)),
                    last,
                    func.min(model.job_id),
                )
                .filter(side.in_(wanted))
                .group_by(model.input, model.output)
                .order_by(last.desc(), model.input.desc(), model.output.desc())
            )
            if limit is not None:
                query = query.limit(limit)
            return [
                GroupedEdge(i, o, int(count), recorded or "", sample or "")
                for i, o, count, recorded, sample in query.all()
            ]
        finally:
            session.close()

    def get_job_ids_by_tags(
        self, any_of: List[str], all_of: Optional[List[str]] = None
    ) -> Set[str]:
        """Match the tags on the JSON blob in SQL, one ``SELECT DISTINCT job_id``.

        Reads ``attributes.job.tags.<key>`` from the ``json`` text column with the
        dialect's JSON operator. A dialect without one falls back to the base scan.
        """
        wanted_any, wanted_all = _tag_pairs(any_of), _tag_pairs(all_of or [])
        if not wanted_any and not wanted_all:
            return set()
        if not self._ensure_table():
            return set()
        dialect = self._engine.dialect.name
        if dialect not in ("sqlite", "postgresql"):
            return super().get_job_ids_by_tags(any_of, all_of)
        model = self._sql_alchemy_model

        def tag(key: str):
            if dialect == "postgresql":
                return cast(model.json, JSONB)["attributes"]["job"]["tags"][key].astext
            # The key is quoted so a dot or space in it stays one path segment.
            escaped = key.replace("\\", "\\\\").replace('"', '\\"')
            return func.json_extract(model.json, f'$.attributes.job.tags."{escaped}"')

        conditions = [tag(key) == value for key, value in wanted_all]
        if wanted_any:
            conditions.append(or_(*(tag(key) == value for key, value in wanted_any)))
        session = self._BaseSQLItemStorage__get_session_without_retry()
        try:
            query = session.query(model.job_id).filter(and_(*conditions)).distinct()
            return {job_id for (job_id,) in query.all() if job_id}
        finally:
            session.close()

    @property
    def _job_id(self):
        return self._sql_alchemy_model.job_id

    @contextmanager
    def _touching(
        self,
        uri: str,
        self_loop: bool = False,
        output: Optional[str] = None,
        terminal: Optional[str] = None,
    ):
        """The rows with ``uri`` on either side, or ``None`` when none can exist.

        ``input = u OR output = u`` over two indexed columns, which every supported
        backend answers from both indexes (bitmap OR, index merge, OR-by-union)
        rather than by a scan.
        """
        if not uri or uri == TERMINAL or not self._ensure_table():
            yield None
            return
        session = self._BaseSQLItemStorage__get_session_without_retry()
        try:
            model = self._sql_alchemy_model
            pair = endpoint_pair(uri, self_loop, output, terminal)
            if pair:
                condition = and_(model.input == pair[0], model.output == pair[1])
            else:
                condition = or_(model.input == uri, model.output == uri)
            yield session.query(model).filter(condition)
        finally:
            session.close()

    def _ensure_table(self) -> bool:
        """Initialize the model if needed; whether the table exists to query.

        ``__initialize_storage`` is name-mangled private, so this replicates it
        through the protected API, as ``SQLSpaceUserStorage`` does.
        """
        if self._sql_alchemy_model is None:
            sample = self._convert_item_to_row_dict(self._get_sample_item())
            self._create_or_adjust_schema_item_dict(sample)
        return self._does_table_exist()
