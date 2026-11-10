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

"""Shared tests for the lineage index table, ``gb_lineage_index``.

The storage class is ``*LineageRowStorage`` (each item is a ``StoredLineageRow``);
the table it backs is ``gb_lineage_index``. The bodies here are backend-agnostic and
run against a prefixed copy of the admin tables, so they never touch the real one:

- test/unit/storage/sqlite/test_sqlite_lineage_index_storage.py (SQLite)
- test/unit/storage/postgres/test_postgres_lineage_index_storage.py (local PostgreSQL)
- test/integration/ibm/storage/sql/test_sql_lineage_index_storage.py (hosted PostgreSQL)
"""

from typing import Any

import pytest
from libgbtest.storage.storage import AbstractStorageTest, AbstractStorageTestSupport
from sqlalchemy.exc import IntegrityError

from gbserver.lineage.attributes import build_attributes
from gbserver.storage.lineage_row_storage import (
    DOWNSTREAM,
    UPSTREAM,
    BaseLineageRowStorage,
    GroupedEdge,
)
from gbserver.storage.storage import BaseItemStorage, BaseStoredItem
from gbserver.storage.stored_lineage_row import (
    MAX_LINEAGE_URI_LENGTH,
    TERMINAL,
    JobStore,
    StoredLineageRow,
)


def row(
    job_id: str = "J",
    input: str = "lh://prod/ns/models/t/a",
    output: str = "lh://prod/ns/datasets/t/b",
    **kwargs,
) -> StoredLineageRow:
    """A row with URI endpoints.

    ``source_system`` is not a row column; it only seeds ``attributes.origin``.
    """
    source_system = kwargs.pop("source_system", "granite.build")
    if "attributes" in kwargs:
        attributes = dict(kwargs.pop("attributes") or {})
    else:
        attributes = build_attributes(source_system=source_system)
    assert not kwargs, f"unhandled row() keywords: {sorted(kwargs)}"
    return StoredLineageRow(
        job_id=job_id,
        input=input,
        output=output,
        attributes=attributes,
    )


class LineageIndexStorageTestSupport(AbstractStorageTestSupport):

    def __init__(self):
        super().__init__(sort_column="job_id")

    def _get_test_item(self, index: int) -> BaseStoredItem:
        # Zero-padded so the job_id sort order matches the index order.
        return row(
            job_id=f"job{index:04d}",
            input=f"lh://prod/ns/tables/in{index}",
            output=f"lh://prod/ns/models/t/out{index}",
        )


class BaseLineageIndexStorageTest(AbstractStorageTest):
    """``gb_lineage_index`` via ``*LineageRowStorage``."""

    @classmethod
    def _get_test_config(cls) -> AbstractStorageTestSupport:
        return LineageIndexStorageTestSupport()

    def _get_tested_storage(self) -> BaseLineageRowStorage:
        return self.storage.lineage_row_storage

    def _get_where_search_columns(
        self, storage: BaseItemStorage, item: BaseStoredItem
    ) -> dict[str, Any]:
        columns = super()._get_where_search_columns(storage, item)
        # Unique per item (see _get_where_test_item), so a search on it can't match more than one.
        del columns["job_id"]
        # Stamped at construction, so two items built from one index still differ.
        del columns["recorded_at"]
        return columns

    def _get_where_test_item(self, index: int) -> BaseStoredItem:
        """The where tests add the same index twice; (job_id, input, output) is unique."""
        item = super()._get_where_test_item(index)
        assert isinstance(item, StoredLineageRow)
        item.job_id = item.uuid
        return item

    def _add(self, job_id, input, output, recorded_at):
        stored = row(job_id=job_id, input=input, output=output)
        stored.recorded_at = recorded_at
        self._get_tested_storage().add(stored)

    # get_rows_by_input / get_rows_by_output: one indexed, batched query per traversal level.

    def test_get_rows_by_input(self):
        storage = self._get_tested_storage()
        storage.add(row(input="a", output="b"))
        storage.add(row(job_id="J2", input="b", output="c"))
        found = storage.get_rows_by_input(["a"])
        assert [r.output for r in found] == ["b"]

    def test_get_rows_by_output(self):
        storage = self._get_tested_storage()
        storage.add(row(input="a", output="b"))
        storage.add(row(job_id="J2", input="b", output="c"))
        found = storage.get_rows_by_output(["c"])
        assert [r.input for r in found] == ["b"]

    def test_get_rows_by_input_batches_a_frontier(self):
        storage = self._get_tested_storage()
        for i in range(5):
            storage.add(row(job_id=f"J{i}", input=f"s{i}", output=f"t{i}"))
        found = storage.get_rows_by_input(["s0", "s2", "s4"])
        assert sorted(r.input for r in found) == ["s0", "s2", "s4"]

    def test_get_rows_by_input_empty_frontier(self):
        storage = self._get_tested_storage()
        storage.add(row())
        assert storage.get_rows_by_input([]) == []
        assert storage.get_rows_by_output([]) == []

    def test_get_rows_by_input_ignores_terminal(self):
        """Every creation row shares the terminal marker, so a hop matching on it
        would treat unrelated creations as one node.
        """
        storage = self._get_tested_storage()
        storage.add(row(job_id="C1", input=TERMINAL, output="x"))
        storage.add(row(job_id="C2", input=TERMINAL, output="y"))
        storage.add(row(job_id="N", input="real", output="z"))

        # Both creations would match on the marker, collapsing them into one node.
        assert storage.get_rows_by_input([TERMINAL]) == []
        # A real identifier alongside the marker still matches, and only itself.
        found = storage.get_rows_by_input([TERMINAL, "real"])
        assert [r.job_id for r in found] == ["N"]

    def test_get_rows_by_input_dedups_frontier(self):
        storage = self._get_tested_storage()
        storage.add(row(input="a", output="b"))
        assert len(storage.get_rows_by_input(["a", "a", "a"])) == 1

    def test_get_rows_by_uri(self):
        """The read path resolves a root this way, with one query and no scan."""
        storage = self._get_tested_storage()
        storage.add(row(job_id="U", input="s3://bkt/in", output="s3://bkt/out"))
        assert [r.job_id for r in storage.get_rows_by_input(["s3://bkt/in"])] == ["U"]
        assert [r.job_id for r in storage.get_rows_by_output(["s3://bkt/out"])] == ["U"]

    def test_get_rows_without_self_loops(self):
        storage = self._get_tested_storage()
        self._add("A", "t", "t", "2026-01-01")
        self._add("B", "t", "u", "2026-01-01")
        self._add("C", "s", "t", "2026-01-01")
        assert [
            r.job_id for r in storage.get_rows_by_input(["t"], self_loops=False)
        ] == ["B"]
        assert [
            r.job_id for r in storage.get_rows_by_output(["t"], self_loops=False)
        ] == ["C"]
        assert len(storage.get_rows_by_input(["t"])) == 2

    # get_rows_by_job: a job's rows are recoverable as a unit.
    #
    # This is the only grouping the index offers, and deliberately so: ``job_id`` is
    # the one identifier every lineage source has. A build or a target run is
    # granite.build's own concept, so a process-scoped view resolves its scope in
    # that system and seeds the walk with URIs instead.

    def test_get_rows_by_job(self):
        storage = self._get_tested_storage()
        for src in ("lh://prod/ns/tables/i1", "lh://prod/ns/tables/i2"):
            storage.add(row(job_id="J", input=src))
        assert len(storage.get_rows_by_job("J")) == 2

    def test_get_rows_by_job_empty_job_id(self):
        storage = self._get_tested_storage()
        storage.add(row())
        assert storage.get_rows_by_job("") == []

    # has_rows_for_job / get_recorded_jobs: presence-based dedup, keyed on job_id.
    #
    # The sink asks "do these jobs already have rows?" on every watcher tick. It is
    # the only thing standing between a re-selected job and duplicate rows, so it
    # has to be an indexed lookup rather than a scan.

    def test_has_rows_for_job(self):
        storage = self._get_tested_storage()
        storage.add(row(job_id="J"))
        assert storage.has_rows_for_job("J")
        assert not storage.has_rows_for_job("OTHER")

    def test_has_rows_for_job_empty_job_id(self):
        assert not self._get_tested_storage().has_rows_for_job("")

    def test_get_recorded_jobs(self):
        storage = self._get_tested_storage()
        storage.add(row(job_id="J1"))
        storage.add(row(job_id="J2"))
        assert storage.get_recorded_jobs(["J1", "J2", "J3"]) == {"J1", "J2"}

    def test_get_recorded_jobs_ignores_row_count(self):
        """One row is enough: a job's rows are written together.

        The reconciler's ``expected_counts`` counts one W&B run per output artifact,
        which never equals an N*M row count -- honouring it would report every job
        unrecorded forever and re-record on every scan.
        """
        storage = self._get_tested_storage()
        for tgt in ("s3://b/o1", "s3://b/o2", "s3://b/o3"):
            storage.add(row(job_id="MANY", output=tgt))
        assert storage.get_recorded_jobs(["MANY"]) == {"MANY"}

    def test_get_recorded_jobs_empty_batch(self):
        assert self._get_tested_storage().get_recorded_jobs([]) == set()

    # Uniqueness on (job_id, input, output): re-ingest idempotency.
    #
    # The prototype has no key at all and executemany's without a guard, so it
    # duplicates silently.

    def test_uniqueness_enforcement(self):
        storage = self._get_tested_storage()
        storage.add(row(job_id="J", input="a", output="b"))
        with pytest.raises(Exception):
            storage.add(row(job_id="J", input="a", output="b"))

    def test_uniqueness_enforcement_creation_terminal(self):
        """The reason source/target hold "" and not NULL: in SQL NULL never equals
        NULL, so a NULL endpoint would slip past the unique and leave creation rows
        as the only duplicable ones.
        """
        storage = self._get_tested_storage()
        storage.add(row(job_id="C", input=TERMINAL, output="x"))
        with pytest.raises(IntegrityError):
            storage.add(row(job_id="C", input=TERMINAL, output="x"))

    def test_uniqueness_enforcement_deletion_terminal(self):
        storage = self._get_tested_storage()
        storage.add(row(job_id="D", input="x", output=TERMINAL))
        with pytest.raises(Exception):
            storage.add(row(job_id="D", input="x", output=TERMINAL))

    def test_uniqueness_ignores_attributes(self):
        # Row identity is (job_id, input, output). Anything in the blob is
        # outside it, so re-ingesting a row whose metadata changed must not
        # double the edge.
        storage = self._get_tested_storage()
        storage.add(row(attributes={"job_name": "one"}))
        with pytest.raises(Exception):
            storage.add(row(attributes={"job_name": "two"}))

    def test_uniqueness_allows_same_job_different_endpoints(self):
        """An N*M job writes several rows under one job_id."""
        storage = self._get_tested_storage()
        storage.add(row(job_id="J", input="i1", output="o1"))
        storage.add(row(job_id="J", input="i1", output="o2"))
        storage.add(row(job_id="J", input="i2", output="o1"))
        assert len(storage.get_rows_by_job("J")) == 3

    def test_uniqueness_allows_same_endpoints_different_job(self):
        """Two executions can relate the same pair of artifacts."""
        storage = self._get_tested_storage()
        storage.add(row(job_id="J1", input="a", output="b"))
        storage.add(row(job_id="J2", input="a", output="b"))
        assert len(storage.get_rows_by_input(["a"])) == 2

    # Columns: what is, and is not, promoted out of the JSON blob.

    def test_schema_no_process_id_columns(self):
        """A build or target run id may be carried, but only in the blob.

        An indexed column that is empty on every imported row indexes nothing, and
        that is what these were: the importer contract sets both to "" for every
        source that is not granite.build.
        """
        storage = self._get_tested_storage()
        storage.add(
            row(job_id="P", attributes={"build_id": "b1", "target_run_uuid": "tr1"})
        )
        names = storage.get_column_names()
        assert "build_id" not in names
        assert "target_run_uuid" not in names
        # Still readable, just not queryable.
        assert storage.get_rows_by_job("P")[0].attributes["build_id"] == "b1"

    def test_schema_no_separate_uri_columns(self):
        """The endpoints hold URIs, which is what collapses the old schema.

        A canonical identifier did not encode a scheme, so the old row carried a
        separate ``source_uri``/``target_uri`` pair to keep the real URI at all. With
        the URI *as* the identity those columns are redundant, and the root lookup
        becomes an index seek instead of a paged full scan.
        """
        storage = self._get_tested_storage()
        storage.add(row())
        names = storage.get_column_names()
        assert "source_uri" not in names
        assert "target_uri" not in names

    def test_schema_attributes_not_queryable(self):
        """The blob is Text, so nothing inside it can be a predicate.

        Asserted rather than assumed: it is the reason the rebuild filters in
        Python, and the reason anything needing a filter has to become a column.
        """
        storage = self._get_tested_storage()
        storage.add(row(job_id="A", source_system="lh"))
        assert "source_system" not in storage.get_column_names()
        # and the attributes blob itself is still not queryable
        assert "attributes" not in storage.get_column_names()

    # Round trips through the real table.

    def test_round_trip_all_fields(self):
        storage = self._get_tested_storage()
        original = StoredLineageRow(
            job_id="J",
            input="lh://prod/ns/models/t/a",
            output="hf://huggingface.co/models/org/b",
            job_store=JobStore.LINEAGE_JOB,
            attributes={
                "job_name": "train",
                "owner": "someone",
                "source_system": "lh",
                "source_kind": "model",
                "target_kind": "model",
                "space_name": "sp",
                "build_id": "B",
                "target_run_uuid": "TR",
            },
        )
        storage.add(original)
        stored = storage.get_rows_by_job("J")[0]

        assert stored.input == original.input
        assert stored.output == original.output
        assert stored.job_store == original.job_store
        # Everything else lives in the JSON blob, whole -- including the process
        # ids, which are not columns.
        assert stored.attributes == original.attributes

    def test_round_trip_origin_always_recorded(self):
        """``origin`` is never absent: provenance is not an optional detail.

        A row whose system is unknown could not be scoped by a rebuild -- it would
        be deleted as granite.build's or preserved as an importer's by accident. The
        other groups ARE omitted when empty, so a reader can tell "not recorded"
        from "recorded empty"; this one is not.
        """
        storage = self._get_tested_storage()
        storage.add(row(job_id="BARE"))
        attributes = storage.get_rows_by_job("BARE")[0].attributes
        assert attributes["origin"]["system"] == "granite.build"
        # Nothing described the endpoints or the job, so those groups are absent.
        assert "source" not in attributes
        assert "job" not in attributes

    def test_round_trip_default_attributes_empty(self):
        """The field itself defaults empty; ``build_attributes`` is what fills it."""
        storage = self._get_tested_storage()
        storage.add(StoredLineageRow(job_id="RAW", input="s3://b/x", output="s3://b/y"))
        assert storage.get_rows_by_job("RAW")[0].attributes == {}

    def test_round_trip_terminal_rows(self):
        storage = self._get_tested_storage()
        storage.add(row(job_id="C", input=TERMINAL, output="x"))
        storage.add(row(job_id="D", input="x", output=TERMINAL))
        storage.add(row(job_id="S", input="tbl", output="tbl"))

        assert storage.get_rows_by_job("C")[0].input == TERMINAL
        assert storage.get_rows_by_job("D")[0].output == TERMINAL
        assert storage.get_rows_by_job("S")[0].is_self_loop()

    def test_round_trip_creation_row_not_self_loop(self):
        """Two terminal rows both have source == target == "" but are not loops."""
        storage = self._get_tested_storage()
        storage.add(row(job_id="C", input=TERMINAL, output=TERMINAL))
        assert not storage.get_rows_by_job("C")[0].is_self_loop()

    def test_round_trip_max_length_uri(self):
        """Proves the width landed: at 256 this would truncate silently."""
        storage = self._get_tested_storage()
        prefix = "hf://huggingface.co/models/org/"
        long_uri = prefix + "x" * (MAX_LINEAGE_URI_LENGTH - len(prefix))
        storage.add(row(job_id="LONG", input=long_uri))
        stored = storage.get_rows_by_input([long_uri])
        assert stored, "a maximum-length URI did not round trip"
        assert stored[0].input == long_uri

    def test_get_by_where_endpoints_hold_uris(self):
        storage = self._get_tested_storage()
        storage.add(row(job_id="U", input="s3://bkt/in", output="hf:///org/out"))
        stored = storage.get_by_where({"job_id": "U"})[0]
        assert stored.input == "s3://bkt/in"
        assert stored.output == "hf:///org/out"

    # job_store: where a job's full data lives is a column, so the read path can
    # dispatch on it.

    def test_job_store_default(self):
        storage = self._get_tested_storage()
        storage.add(row())
        assert storage.get_rows_by_job("J")[0].job_store == JobStore.OTHER

    @pytest.mark.parametrize("job_store", list(JobStore))
    def test_job_store_round_trip(self, job_store):
        storage = self._get_tested_storage()
        stored = row()
        stored.job_store = job_store
        storage.add(stored)
        assert storage.get_rows_by_job("J")[0].job_store == job_store

    def test_get_by_where_job_store(self):
        storage = self._get_tested_storage()
        first, second = row(job_id="A"), row(job_id="B")
        first.job_store = JobStore.WANDB
        second.job_store = JobStore.LINEAGE_JOB
        storage.add(first)
        storage.add(second)
        found = storage.get_by_where({"job_store": JobStore.WANDB.value})
        assert [r.job_id for r in found] == ["A"]

    # walk_lineage over the real table, not a fake.
    #
    # The fake in test_walk.py proves the algorithm; these prove the algorithm and
    # the storage agree -- that a batched hop really produces an IN over the indexed
    # column, and that the terminal sentinel survives a round trip through SQL.
    # get_by_where only builds an IN for string columns; on a non-string column it
    # degrades to ``column == [list]`` instead of raising, so a dialect mismatch
    # shows up as an empty graph rather than an error.

    def test_walk_lineage_chain(self):
        from gbserver.lineage.walk import Direction, walk_lineage

        storage = self._get_tested_storage()
        storage.add(row(job_id="J1", input="a", output="b"))
        storage.add(row(job_id="J2", input="b", output="c"))

        graph = walk_lineage(storage, ["a"], Direction.DESCENDANTS)
        assert graph.depths == {"a": 0, "b": 1, "c": 2}

        back = walk_lineage(storage, ["c"], Direction.ANCESTORS)
        assert back.depths == {"c": 0, "b": 1, "a": 2}

    def test_walk_lineage_terminal(self):
        from gbserver.lineage.walk import Direction, walk_lineage

        storage = self._get_tested_storage()
        storage.add(row(job_id="C", input=TERMINAL, output="a"))
        storage.add(row(job_id="J", input="a", output="b"))

        graph = walk_lineage(storage, ["b"], Direction.ANCESTORS)
        assert graph.depths == {"b": 0, "a": 1}
        assert TERMINAL not in graph.depths
        assert any(r.input == TERMINAL for r in graph.rows)

    def test_walk_lineage_cartesian_job(self):
        from gbserver.lineage.walk import Direction, walk_lineage

        storage = self._get_tested_storage()
        for src in ("i1", "i2", "i3"):
            for tgt in ("o1", "o2"):
                storage.add(row(job_id="J", input=src, output=tgt))

        graph = walk_lineage(storage, ["i1"], Direction.DESCENDANTS)
        assert graph.depths == {"i1": 0, "o1": 1, "o2": 1}

        # From an output, every input of that execution is one hop back.
        back = walk_lineage(storage, ["o1"], Direction.ANCESTORS)
        assert back.depths == {"o1": 0, "i1": 1, "i2": 1, "i3": 1}

    def test_walk_lineage_collapsed_self_loops(self):
        from gbserver.lineage.graph_builder import build_graph_dict
        from gbserver.lineage.walk import Direction, walk_lineage

        storage = self._get_tested_storage()
        for job_id in ("A", "B", "C"):
            storage.add(row(job_id=job_id, input="t", output="t"))
        storage.add(row(job_id="J", input="t", output="u"))

        full = walk_lineage(storage, ["t"], Direction.BOTH)
        collapsed = walk_lineage(
            storage, ["t"], Direction.BOTH, collapse_self_loops=True
        )
        assert collapsed.depths == full.depths
        assert sorted((r.job_id, r.input, r.output) for r in collapsed.rows) == [
            ("A", "t", "t"),
            ("J", "t", "u"),
        ]
        assert collapsed.self_loop_runs == {"t": 3}
        # Same graph either way.
        assert build_graph_dict(collapsed, root_uri="t") == build_graph_dict(
            full, root_uri="t"
        )

    # grouped_edges / grouped_self_loops: one level of the graph walk, where every
    # job between two nodes is one edge.

    def test_grouped_edges(self):
        for i, job_id in enumerate(["J3", "J1", "J2"]):
            self._add(job_id, "x", "y", f"2026-01-0{i + 1}")
        self._add("K", "x", "z", "2026-01-01")
        edges = self._get_tested_storage().grouped_edges(["x"], DOWNSTREAM)
        assert edges == [
            GroupedEdge("x", "y", 3, "2026-01-03", "J1"),
            GroupedEdge("x", "z", 1, "2026-01-01", "K"),
        ]

    def test_grouped_edges_upstream(self):
        self._add("A", "x", "y", "2026-01-01")
        self._add("B", "w", "y", "2026-01-02")
        edges = self._get_tested_storage().grouped_edges(["y"], UPSTREAM)
        assert [(e.input, e.output) for e in edges] == [("w", "y"), ("x", "y")]

    def test_grouped_edges_self_loops_collapse(self):
        for job_id in ("A", "B", "C"):
            self._add(job_id, "t", "t", "2026-01-01")
        edges = self._get_tested_storage().grouped_edges(["t"], DOWNSTREAM)
        assert [e.job_count for e in edges] == [3]

    def test_grouped_edges_limit(self):
        self._add("A", "x", "old", "2026-01-01")
        self._add("B", "x", "new", "2026-02-01")
        edges = self._get_tested_storage().grouped_edges(["x"], DOWNSTREAM, limit=1)
        assert [e.output for e in edges] == ["new"]

    def test_grouped_edges_empty_frontier(self):
        storage = self._get_tested_storage()
        self._add("A", TERMINAL, "y", "2026-01-01")
        assert storage.grouped_edges([], DOWNSTREAM) == []
        assert storage.grouped_edges([TERMINAL], DOWNSTREAM) == []

    def test_grouped_edges_unknown_direction(self):
        with pytest.raises(ValueError):
            self._get_tested_storage().grouped_edges(["x"], "sideways")

    def test_grouped_self_loops(self):
        storage = self._get_tested_storage()
        for job_id in ("B", "A", "C"):
            self._add(job_id, "t", "t", f"2026-01-0{ord(job_id) - 64}")
        self._add("D", "u", "u", "2026-01-01")
        self._add("E", "t", "u", "2026-01-09")
        edges = sorted(
            storage.grouped_self_loops(["t", "u", "nothing"]), key=lambda e: e.input
        )
        assert edges == [
            GroupedEdge("t", "t", 3, "2026-01-03", "A"),
            GroupedEdge("u", "u", 1, "2026-01-01", "D"),
        ]
        assert storage.grouped_self_loops([TERMINAL]) == []

    # Writes the indexer and --rebuild issue.

    def test_delete_all_rows_keeps_the_table(self):
        """--rebuild empties the index under a running process that already resolved
        the table: that process must still read and write it afterwards."""
        storage = self._get_tested_storage()
        other = self._get_storage_factory().create_lineage_row_storage(
            table_name=storage.table_name
        )
        other.add(row(job_id="J1"))
        assert other.get_rows_by_job("J1")
        assert storage.delete_all_rows() == 1
        assert not other.get_rows_by_job("J1")
        other.add(row(job_id="J2"))
        assert [r.job_id for r in other.get_rows_by_job("J2")] == ["J2"]
        assert storage.delete_all_rows() == 1

    def test_add_replacing(self):
        storage = self._get_tested_storage()
        terminal = row(job_id="J", output=TERMINAL)
        storage.add(terminal)
        edge = row(job_id="J")
        storage.add_replacing(edge, [terminal.uuid])
        assert [(r.input, r.output) for r in storage.get_rows_by_job("J")] == [
            (edge.input, edge.output)
        ]

    def test_add_replacing_conflict_rolls_back(self):
        """A lost race rolls the deletes back with the insert."""
        storage = self._get_tested_storage()
        terminal = row(job_id="J", output=TERMINAL)
        storage.add(terminal)
        storage.add(row(job_id="J"))  # another writer got the edge in first
        with pytest.raises(IntegrityError):
            storage.add_replacing(row(job_id="J"), [terminal.uuid])
        assert len(storage.get_rows_by_job("J")) == 2
