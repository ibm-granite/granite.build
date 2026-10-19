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
from typing import List, Optional

from sqlalchemy import String, and_, case, cast, distinct, func, literal, or_
from sqlalchemy.dialects.postgresql import JSONB

from gbserver.storage.lineage_row_storage import (
    DOWNSTREAM,
    UPSTREAM,
    BaseLineageRowStorage,
    GroupedEdge,
    ILineageRowStorage,
    endpoint_pair,
    tag_pairs,
)
from gbserver.storage.sql.sql_storage import BaseSQLItemStorage, retry_read
from gbserver.storage.storage import JSON_COLUMN_NAME, UUID_COLUMN_NAME
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
    - ``job_store`` says which store holds the job's full content (the lineage job
      table, this server's builds and target runs, W&B, other); indexed so a
      migration or re-import can select by store, and read per row by the job
      detail route to know where to fetch from. It replaced an earlier ``origin``
      column, which named the producing *system* rather than the store to follow;
      the producing system is still carried in the blob under ``origin.system``,
      where it is readable but not queryable.
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
    slow one.
    """

    def __init__(self, **kwargs) -> None:
        kwargs["indexed_columns"] = [
            "job_id",
            "job_store",
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

    @retry_read
    def count_jobs_touching(
        self,
        uri: str,
        self_loop: bool = False,
        output: Optional[str] = None,
        terminal: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> int:
        """Count the distinct jobs touching ``uri`` in one ``COUNT(DISTINCT)``.

        The artifact this exists for has 68,905 runs; counting them by reading the
        rows cost seconds per request, whatever the page size.
        """
        if not self._can_filter_tags(tags):
            return super().count_jobs_touching(uri, self_loop, output, terminal, tags)
        with self._touching(uri, self_loop, output, terminal, tags) as query:
            if query is None:
                return 0
            return int(query.with_entities(func.count(distinct(self._job_id))).scalar())

    @retry_read
    def get_job_ids_touching(
        self,
        uri: str,
        limit: int,
        offset: int,
        self_loop: bool = False,
        output: Optional[str] = None,
        terminal: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[str]:
        """Return one page of the distinct jobs touching ``uri``, in SQL."""
        if not self._can_filter_tags(tags):
            return super().get_job_ids_touching(
                uri, limit, offset, self_loop, output, terminal, tags
            )
        with self._touching(uri, self_loop, output, terminal, tags) as query:
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

    def get_rows_by_input(
        self, inputs: List[str], self_loops: bool = True, limit: Optional[int] = None
    ) -> List[StoredLineageRow]:
        """The descendant hop; ``self_loops=False`` and ``limit`` applied in SQL."""
        if self_loops and limit is None:
            return super().get_rows_by_input(inputs)
        return self._hop_rows(inputs, DOWNSTREAM, self_loops, limit)

    def get_rows_by_output(
        self, outputs: List[str], self_loops: bool = True, limit: Optional[int] = None
    ) -> List[StoredLineageRow]:
        """The ancestor hop; ``self_loops=False`` and ``limit`` applied in SQL."""
        if self_loops and limit is None:
            return super().get_rows_by_output(outputs)
        return self._hop_rows(outputs, UPSTREAM, self_loops, limit)

    @retry_read
    def grouped_self_loops(self, uris: List[str]) -> List[GroupedEdge]:
        """Fold each artifact's self-loops into one edge, in one ``GROUP BY``.

        Constrains both ``input`` and ``output`` to ``uris`` so the
        ``(input, output)`` index seeks straight to the ``uri -> uri`` range. Never
        reads the blob: the artifact this exists for has 68,905 self-loops, about
        67 MB of JSON that the graph collapses into one node anyway.
        """
        wanted = self._batchable(uris)
        if not wanted or not self._ensure_table_without_retry():
            return []
        session = self._get_session_without_retry()
        try:
            model = self._sql_alchemy_model
            query = (
                session.query(
                    model.input,
                    func.count(distinct(model.job_id)),
                    func.max(model.recorded_at),
                    func.min(model.job_id),
                )
                .filter(
                    model.input.in_(wanted),
                    model.output.in_(wanted),
                    model.input == model.output,
                )
                .group_by(model.input)
            )
            return [
                GroupedEdge(uri, uri, int(count), recorded or "", sample or "")
                for uri, count, recorded, sample in query.all()
            ]
        finally:
            session.close()

    @retry_read
    def _hop_rows(
        self,
        frontier: List[str],
        direction: str,
        self_loops: bool,
        limit: Optional[int],
    ) -> List[StoredLineageRow]:
        """One hop's rows, deserializing only those it returns.

        Ordered by the matched side, then the neighbour, then ``job_id``, so a
        ``limit`` that cuts a hub keeps the same rows every time; the matching
        ``(input, output)`` / ``(output, input)`` index yields the first two keys
        already in order.
        """
        wanted = self._batchable(frontier)
        if not wanted or not self._ensure_table_without_retry():
            return []
        session = self._get_session_without_retry()
        try:
            model = self._sql_alchemy_model
            downstream = direction == DOWNSTREAM
            side = model.input if downstream else model.output
            neighbour = model.output if downstream else model.input
            query = session.query(model.json).filter(side.in_(wanted))
            if not self_loops:
                query = query.filter(model.input != model.output)
            if limit is not None:
                query = query.order_by(side, neighbour, model.job_id).limit(limit)
            return [
                self._convert_row_dict_to_item({JSON_COLUMN_NAME: blob})
                for (blob,) in query.all()
            ]
        finally:
            session.close()

    @retry_read
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
        if not wanted or not self._ensure_table_without_retry():
            return []
        session = self._get_session_without_retry()
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

    def add_replacing(self, row: StoredLineageRow, stale_uuids: List[str]) -> None:
        """The insert and the deletes in one transaction: both or neither.

        A unique-key conflict (another writer added the same row) rolls back the
        deletes too and raises ``IntegrityError``, as ``add`` does.
        """
        if not stale_uuids:
            self.add(row)
            return
        if not self._ensure_table_without_retry():
            self.add(row)  # creates the table; nothing stored can be stale yet
            return
        model = self._sql_alchemy_model
        session = self._get_session_without_retry()
        try:
            session.add(model(**self._convert_item_to_row_dict(row)))
            session.query(model).filter(
                getattr(model, UUID_COLUMN_NAME).in_(stale_uuids)
            ).delete(synchronize_session=False)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def delete_all_rows(self) -> int:
        """One ``DELETE`` in one transaction; the table and its indexes stay."""
        if not self._ensure_table_without_retry():
            return 0
        session = self._get_session_without_retry()
        try:
            deleted = session.query(self._sql_alchemy_model).delete(
                synchronize_session=False
            )
            session.commit()
            return int(deleted or 0)
        except Exception:
            session.rollback()
            raise
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
        tags: Optional[List[str]] = None,
    ):
        """The rows with ``uri`` on either side, or ``None`` when none can exist.

        ``input = u OR output = u`` over two indexed columns, which every supported
        backend answers from both indexes (bitmap OR, index merge, OR-by-union)
        rather than by a scan.

        ``tags`` adds one JSON-path equality per tag on ``attributes.job.tags``
        (see :meth:`_tag_condition`). No index serves it: with a ``uri`` it only
        narrows the rows the endpoint index found; without one it scans the table,
        which is the accepted cost of a tag-only listing.
        """
        # TERMINAL is "", so an empty uri is "no artifact filter" only with tags.
        if not (uri or tags) or not self._ensure_table_without_retry():
            yield None
            return
        session = self._get_session_without_retry()
        try:
            model = self._sql_alchemy_model
            conditions = [
                self._tag_condition(key, value) for key, value in tag_pairs(tags or [])
            ]
            if uri:
                pair = endpoint_pair(uri, self_loop, output, terminal)
                if pair:
                    conditions.append(
                        and_(model.input == pair[0], model.output == pair[1])
                    )
                else:
                    conditions.append(or_(model.input == uri, model.output == uri))
            yield session.query(model).filter(and_(*conditions))
        finally:
            session.close()

    def _can_filter_tags(self, tags: Optional[List[str]]) -> bool:
        """Whether this backend can put ``tags`` in SQL; otherwise the rows are read.

        SQLite and Postgres only. SQLite's ``->>`` reads a right operand starting
        with ``$`` as a path rather than a key, so such a key is matched in Python.
        """
        if not tags:
            return True
        assert self._engine is not None
        dialect = self._engine.dialect.name
        if dialect == "postgresql":
            return True
        if dialect == "sqlite":
            return not any(key.startswith("$") for key, _ in tag_pairs(tags))
        return False

    def _tag_condition(self, key: str, value: str):
        """``attributes.job.tags[key] = value``, read from the JSON blob.

        The blob is ``Text``, so Postgres casts it to ``jsonb``; SQLite guards with
        ``json_valid`` so one malformed row cannot fail the query. The key is a
        bound parameter, never spliced into the path, so a key with a dot or a
        quote is matched as itself.
        """
        blob = getattr(self._sql_alchemy_model, JSON_COLUMN_NAME)
        assert self._engine is not None
        if self._engine.dialect.name == "postgresql":
            # A "->" chain, not subscripts: those need Postgres 14.
            tags = cast(blob, JSONB)
            for step in ("attributes", "job", "tags"):
                tags = tags.op("->")(literal(step, String))
            return tags.op("->>")(literal(key, String)) == value
        tags = blob.op("->")(literal("$.attributes.job.tags"))
        extracted = case(
            (func.json_valid(blob), tags.op("->>")(literal(key, String))),
            else_=None,
        )
        return extracted == value
