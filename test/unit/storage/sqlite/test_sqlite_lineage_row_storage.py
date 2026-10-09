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

"""Storage-level tests for lineage rows against SQLite.

These exercise the parts that only a real table can prove: that the declared
indexes and the composite unique actually exist, that the unique protects
terminal rows too, and that a batched hop produces an ``IN`` rather than silently
degrading.
"""

import os
import sqlite3
import uuid as uuid_module

import pytest

from gbserver.lineage.attributes import build_attributes
from gbserver.storage.lineage_row_storage import DOWNSTREAM, UPSTREAM, GroupedEdge
from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory
from gbserver.storage.stored_lineage_row import (
    TERMINAL,
    JobStore,
    StoredLineageRow,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)


@pytest.fixture(name="storage")
def storage_fixture():
    """A lineage row storage on a table unique to this test."""
    table = f"t_lin_{uuid_module.uuid4().hex[:8]}"
    return SqliteStorageFactory().create_lineage_row_storage(table_name=table)


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


class TestHops:
    """One indexed, batched query per traversal level."""

    def test_hop_by_source_finds_descendant_rows(self, storage):
        storage.add(row(input="a", output="b"))
        storage.add(row(job_id="J2", input="b", output="c"))
        found = storage.get_rows_by_input(["a"])
        assert [r.output for r in found] == ["b"]

    def test_hop_by_target_finds_ancestor_rows(self, storage):
        storage.add(row(input="a", output="b"))
        storage.add(row(job_id="J2", input="b", output="c"))
        found = storage.get_rows_by_output(["c"])
        assert [r.input for r in found] == ["b"]

    def test_hop_batches_a_whole_frontier(self, storage):
        for i in range(5):
            storage.add(row(job_id=f"J{i}", input=f"s{i}", output=f"t{i}"))
        found = storage.get_rows_by_input(["s0", "s2", "s4"])
        assert sorted(r.input for r in found) == ["s0", "s2", "s4"]

    def test_empty_frontier_queries_nothing(self, storage):
        storage.add(row())
        assert storage.get_rows_by_input([]) == []
        assert storage.get_rows_by_output([]) == []

    def test_terminal_marker_never_reaches_a_query(self, storage):
        """Every creation row shares the terminal marker, so a hop matching on it
        would treat unrelated creations as one node.
        """
        storage.add(row(job_id="C1", input=TERMINAL, output="x"))
        storage.add(row(job_id="C2", input=TERMINAL, output="y"))
        storage.add(row(job_id="N", input="real", output="z"))

        # Both creations would match on the marker, collapsing them into one node.
        assert storage.get_rows_by_input([TERMINAL]) == []
        # A real identifier alongside the marker still matches, and only itself.
        found = storage.get_rows_by_input([TERMINAL, "real"])
        assert [r.job_id for r in found] == ["N"]

    def test_frontier_is_deduplicated(self, storage):
        storage.add(row(input="a", output="b"))
        assert len(storage.get_rows_by_input(["a", "a", "a"])) == 1


class TestJobGrouping:
    """A job's rows are recoverable as a unit.

    This is the only grouping the index offers, and deliberately so: ``job_id`` is
    the one identifier every lineage source has. A build or a target run is
    granite.build's own concept, so a process-scoped view resolves its scope in that
    system and seeds the walk with URIs instead.
    """

    def test_rows_by_job_recovers_the_whole_execution(self, storage):
        for src in ("lh://prod/ns/tables/i1", "lh://prod/ns/tables/i2"):
            storage.add(row(job_id="J", input=src))
        assert len(storage.get_rows_by_job("J")) == 2

    def test_empty_job_id_returns_nothing(self, storage):
        storage.add(row())
        assert storage.get_rows_by_job("") == []

    def test_process_ids_are_not_columns(self, storage):
        """A build or target run id may be carried, but only in the blob.

        An indexed column that is empty on every imported row indexes nothing, and
        that is what these were: the importer contract sets both to "" for every
        source that is not granite.build.
        """
        storage.add(
            row(job_id="P", attributes={"build_id": "b1", "target_run_uuid": "tr1"})
        )
        names = storage.get_column_names()
        assert "build_id" not in names
        assert "target_run_uuid" not in names
        # Still readable, just not queryable.
        assert storage.get_rows_by_job("P")[0].attributes["build_id"] == "b1"


class TestDedupSupport:
    """Presence-based dedup, keyed on job_id.

    The sink asks "do these jobs already have rows?" on every watcher tick. It is
    the only thing standing between a re-selected job and duplicate rows, so it has
    to be an indexed lookup rather than a scan.
    """

    def test_has_rows_for_job(self, storage):
        storage.add(row(job_id="J"))
        assert storage.has_rows_for_job("J")
        assert not storage.has_rows_for_job("OTHER")

    def test_empty_job_is_not_recorded(self, storage):
        assert not storage.has_rows_for_job("")

    def test_get_recorded_jobs_filters(self, storage):
        storage.add(row(job_id="J1"))
        storage.add(row(job_id="J2"))
        assert storage.get_recorded_jobs(["J1", "J2", "J3"]) == {"J1", "J2"}

    def test_presence_is_independent_of_row_count(self, storage):
        """One row is enough: a job's rows are written together.

        The reconciler's ``expected_counts`` counts one W&B run per output artifact,
        which never equals an N*M row count -- honouring it would report every job
        unrecorded forever and re-record on every scan.
        """
        for tgt in ("s3://b/o1", "s3://b/o2", "s3://b/o3"):
            storage.add(row(job_id="MANY", output=tgt))
        assert storage.get_recorded_jobs(["MANY"]) == {"MANY"}

    def test_empty_batch_queries_nothing(self, storage):
        assert storage.get_recorded_jobs([]) == set()


class TestUniqueConstraint:
    """Re-ingest idempotency -- a property added here, not inherited.

    The prototype has no key at all and executemany's without a guard, so it
    duplicates silently.
    """

    def test_same_triple_is_rejected(self, storage):
        storage.add(row(job_id="J", input="a", output="b"))
        with pytest.raises(Exception):
            storage.add(row(job_id="J", input="a", output="b"))

    def test_creation_terminal_is_protected(self, storage):
        """The reason source/target hold "" and not NULL: in SQL NULL never equals
        NULL, so a NULL endpoint would slip past the unique and leave creation rows
        as the only duplicable ones.
        """
        storage.add(row(job_id="C", input=TERMINAL, output="x"))
        with pytest.raises(Exception):
            storage.add(row(job_id="C", input=TERMINAL, output="x"))

    def test_deletion_terminal_is_protected(self, storage):
        storage.add(row(job_id="D", input="x", output=TERMINAL))
        with pytest.raises(Exception):
            storage.add(row(job_id="D", input="x", output=TERMINAL))

    def test_same_job_different_endpoints_is_allowed(self, storage):
        """An N*M job writes several rows under one job_id."""
        storage.add(row(job_id="J", input="i1", output="o1"))
        storage.add(row(job_id="J", input="i1", output="o2"))
        storage.add(row(job_id="J", input="i2", output="o1"))
        assert len(storage.get_rows_by_job("J")) == 3

    def test_same_endpoints_different_job_is_allowed(self, storage):
        """Two executions can relate the same pair of artifacts."""
        storage.add(row(job_id="J1", input="a", output="b"))
        storage.add(row(job_id="J2", input="a", output="b"))
        assert len(storage.get_rows_by_input(["a"])) == 2


class TestSchema:
    """The declared indexes and unique must actually exist in the table."""

    @staticmethod
    def _query_schema(storage, sql: str, parameters: tuple = ()) -> list:
        connection = sqlite3.connect(str(storage._get_db_file_path()))
        try:
            return connection.execute(sql, parameters).fetchall()
        finally:
            connection.close()

    @classmethod
    def _index_statements(cls, storage) -> list:
        return cls._query_schema(
            storage,
            "SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name=?",
            (storage.table_name,),
        )

    @pytest.mark.parametrize(
        "column",
        ["job_id", "job_store", "recorded_at"],
    )
    def test_column_is_indexed(self, storage, column):
        storage.add(row())
        statements = " ".join(sql or "" for _, sql in self._index_statements(storage))
        assert f"({column})" in statements, f"{column} is not indexed: {statements}"

    @pytest.mark.parametrize("columns", ["(input, output)", "(output, input)"])
    def test_edge_composite_index_exists(self, storage, columns):
        """Both directions of the walk, and the per-edge GROUP BY, are index-only."""
        storage.add(row())
        statements = [sql or "" for _, sql in self._index_statements(storage)]
        assert any(columns in s and "UNIQUE" not in s for s in statements), statements

    def test_composite_indexes_are_idempotent(self, storage):
        """A schema adjust on an existing table re-runs creation without error."""
        storage.add(row())
        before = sorted(name for name, _ in self._index_statements(storage))
        storage._create_or_adjust_schema_item_dict(
            storage._convert_item_to_row_dict(storage._get_sample_item())
        )
        after = sorted(name for name, _ in self._index_statements(storage))
        assert before == after

    def test_composite_unique_exists(self, storage):
        """It can only be created with the table, never added later."""
        storage.add(row())
        statements = [sql or "" for _, sql in self._index_statements(storage)]
        unique = [s for s in statements if "UNIQUE" in s and "job_id" in s]
        assert unique, statements
        assert "(job_id, input, output)" in unique[0]

    def test_every_promoted_column_is_text(self, storage):
        """get_by_where only builds an IN clause for string columns; anything else
        silently degrades to ``column == [list]`` -- a predicate that returns
        plausible but wrong rows with no error.

        So the invariant is stronger than "the batched ones are text": *every*
        promoted column is text, which makes that failure unreachable rather than
        merely avoided by convention: there is no non-text promoted column left for
        it to happen to.
        """
        storage.add(row())
        columns = {
            name: kind
            for _, name, kind, *_ in self._query_schema(
                storage, f"PRAGMA table_info('{storage.table_name}')"
            )
        }
        # The producer is on the job record, not on every row.
        assert "source_system" not in columns
        # build_id and target_run_uuid stay in the blob - they're GB-specific
        assert "build_id" not in columns
        assert "target_run_uuid" not in columns

        for column in ("input", "output", "job_id", "job_store"):
            assert columns[column].startswith("VARCHAR"), (column, columns[column])

        non_text = {
            name: kind
            for name, kind in columns.items()
            # "json" is the blob; it is never a query predicate. (There is no
            # autoincrement "index" column -- uuid is the primary key.)
            if name != "json" and not kind.startswith("VARCHAR")
        }
        assert not non_text, f"a non-text promoted column is a latent bug: {non_text}"

    def test_uri_columns_are_wide_enough_for_a_real_uri(self, storage):
        """The endpoints are 512, matching MAX_LINEAGE_URI_LENGTH.

        A truncated URI would merge two distinct artifacts sharing a prefix, so the
        normalizer drops anything longer rather than letting the column cut it.
        """
        from gbserver.storage.stored_lineage_row import MAX_LINEAGE_URI_LENGTH

        storage.add(row())
        columns = {
            name: kind
            for _, name, kind, *_ in self._query_schema(
                storage, f"PRAGMA table_info('{storage.table_name}')"
            )
        }
        for column in ("input", "output"):
            assert columns[column] == f"VARCHAR({MAX_LINEAGE_URI_LENGTH})"

    def test_a_maximum_length_uri_survives_a_round_trip(self, storage):
        """Proves the width landed: at 256 this would truncate silently."""
        from gbserver.storage.stored_lineage_row import MAX_LINEAGE_URI_LENGTH

        prefix = "hf://huggingface.co/models/org/"
        long_uri = prefix + "x" * (MAX_LINEAGE_URI_LENGTH - len(prefix))
        storage.add(row(job_id="LONG", input=long_uri))
        stored = storage.get_rows_by_input([long_uri])
        assert stored, "a maximum-length URI did not round trip"
        assert stored[0].input == long_uri


class TestRoundTrip:
    def test_all_fields_survive_storage(self, storage):
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

    def test_a_row_always_records_its_origin(self, storage):
        """``origin`` is never absent: provenance is not an optional detail.

        A row whose system is unknown could not be scoped by a rebuild -- it would
        be deleted as granite.build's or preserved as an importer's by accident. The
        other groups ARE omitted when empty, so a reader can tell "not recorded"
        from "recorded empty"; this one is not.
        """
        storage.add(row(job_id="BARE"))
        attributes = storage.get_rows_by_job("BARE")[0].attributes
        assert attributes["origin"]["system"] == "granite.build"
        # Nothing described the endpoints or the job, so those groups are absent.
        assert "source" not in attributes
        assert "job" not in attributes

    def test_the_model_default_for_attributes_is_an_empty_dict(self, storage):
        """The field itself defaults empty; ``build_attributes`` is what fills it."""
        storage.add(StoredLineageRow(job_id="RAW", input="s3://b/x", output="s3://b/y"))
        assert storage.get_rows_by_job("RAW")[0].attributes == {}

    def test_attributes_are_not_queryable(self, storage):
        """The blob is Text, so nothing inside it can be a predicate.

        Asserted rather than assumed: it is the reason the rebuild filters in
        Python, and the reason anything needing a filter has to become a column.
        """
        storage.add(row(job_id="A", source_system="lh"))
        assert "source_system" not in storage.get_column_names()
        # and the attributes blob itself is still not queryable
        assert "attributes" not in storage.get_column_names()

    def test_terminal_helpers_survive_storage(self, storage):
        storage.add(row(job_id="C", input=TERMINAL, output="x"))
        storage.add(row(job_id="D", input="x", output=TERMINAL))
        storage.add(row(job_id="S", input="tbl", output="tbl"))

        assert storage.get_rows_by_job("C")[0].is_creation()
        assert storage.get_rows_by_job("D")[0].is_deletion()
        assert storage.get_rows_by_job("S")[0].is_self_loop()

    def test_creation_row_is_not_a_self_loop(self, storage):
        """Two terminal rows both have source == target == "" but are not loops."""
        storage.add(row(job_id="C", input=TERMINAL, output=TERMINAL))
        assert not storage.get_rows_by_job("C")[0].is_self_loop()


class TestWalkAgainstRealStorage:
    """The traversal over the real table, not a fake.

    The fake in test_walk.py proves the algorithm; this proves the algorithm and
    the storage agree -- that a batched hop really produces an IN over the indexed
    column, and that the terminal sentinel survives a round trip through SQL.
    """

    def test_walks_a_chain_stored_in_sqlite(self, storage):
        from gbserver.lineage.walk import Direction, walk_lineage

        storage.add(row(job_id="J1", input="a", output="b"))
        storage.add(row(job_id="J2", input="b", output="c"))

        graph = walk_lineage(storage, ["a"], Direction.DESCENDANTS)
        assert graph.depths == {"a": 0, "b": 1, "c": 2}

        back = walk_lineage(storage, ["c"], Direction.ANCESTORS)
        assert back.depths == {"c": 0, "b": 1, "a": 2}

    def test_terminal_survives_a_round_trip(self, storage):
        from gbserver.lineage.walk import Direction, walk_lineage

        storage.add(row(job_id="C", input=TERMINAL, output="a"))
        storage.add(row(job_id="J", input="a", output="b"))

        graph = walk_lineage(storage, ["b"], Direction.ANCESTORS)
        assert graph.depths == {"b": 0, "a": 1}
        assert TERMINAL not in graph.depths
        assert any(r.is_creation() for r in graph.rows)

    def test_cartesian_job_walks_from_any_endpoint(self, storage):
        from gbserver.lineage.walk import Direction, walk_lineage

        for src in ("i1", "i2", "i3"):
            for tgt in ("o1", "o2"):
                storage.add(row(job_id="J", input=src, output=tgt))

        graph = walk_lineage(storage, ["i1"], Direction.DESCENDANTS)
        assert graph.depths == {"i1": 0, "o1": 1, "o2": 1}

        # From an output, every input of that execution is one hop back.
        back = walk_lineage(storage, ["o1"], Direction.ANCESTORS)
        assert back.depths == {"o1": 0, "i1": 1, "i2": 1, "i3": 1}


class TestUriIsTheIdentity:
    """The endpoints hold URIs, which is what collapses the old schema.

    A canonical identifier did not encode a scheme, so the old row carried a
    separate ``source_uri``/``target_uri`` pair to keep the real URI at all. With
    the URI *as* the identity those columns are redundant, and the root lookup
    becomes an index seek instead of a paged full scan.
    """

    def test_the_separate_uri_columns_are_gone(self, storage):
        storage.add(row())
        names = storage.get_column_names()
        assert "source_uri" not in names
        assert "target_uri" not in names

    def test_endpoints_hold_real_uris(self, storage):
        storage.add(row(job_id="U", input="s3://bkt/in", output="hf:///org/out"))
        stored = storage.get_by_where({"job_id": "U"})[0]
        assert stored.input == "s3://bkt/in"
        assert stored.output == "hf:///org/out"

    def test_a_uri_is_found_by_an_indexed_lookup(self, storage):
        """The read path resolves a root this way, with one query and no scan."""
        storage.add(row(job_id="U", input="s3://bkt/in", output="s3://bkt/out"))
        assert [r.job_id for r in storage.get_rows_by_input(["s3://bkt/in"])] == ["U"]
        assert [r.job_id for r in storage.get_rows_by_output(["s3://bkt/out"])] == ["U"]

    def test_differing_attributes_do_not_admit_a_duplicate_row(self, storage):
        # Row identity is (job_id, input, output). Anything in the blob is
        # outside it, so re-ingesting a row whose metadata changed must not
        # double the edge.
        storage.add(row(attributes={"job_name": "one"}))
        with pytest.raises(Exception):
            storage.add(row(attributes={"job_name": "two"}))


class TestJobStore:
    """Where a job's full data lives is a column, so the read path can dispatch on it."""

    def test_default_is_other(self, storage):
        storage.add(row())
        assert storage.get_rows_by_job("J")[0].job_store == JobStore.OTHER

    @pytest.mark.parametrize("job_store", list(JobStore))
    def test_round_trip(self, storage, job_store):
        stored = row()
        stored.job_store = job_store
        storage.add(stored)
        assert storage.get_rows_by_job("J")[0].job_store == job_store

    def test_is_queryable(self, storage):
        first, second = row(job_id="A"), row(job_id="B")
        first.job_store = JobStore.WANDB
        second.job_store = JobStore.LINEAGE_JOB
        storage.add(first)
        storage.add(second)
        found = storage.get_by_where({"job_store": JobStore.WANDB.value})
        assert [r.job_id for r in found] == ["A"]


class TestGroupedEdges:
    """One level of the graph walk: every job between two nodes is one edge."""

    def _add(self, storage, job_id, input, output, recorded_at):
        stored = row(job_id=job_id, input=input, output=output)
        stored.recorded_at = recorded_at
        storage.add(stored)

    def test_jobs_between_two_nodes_stack_into_one_edge(self, storage):
        for i, job_id in enumerate(["J3", "J1", "J2"]):
            self._add(storage, job_id, "x", "y", f"2026-01-0{i + 1}")
        self._add(storage, "K", "x", "z", "2026-01-01")
        edges = storage.grouped_edges(["x"], DOWNSTREAM)
        assert edges == [
            GroupedEdge("x", "y", 3, "2026-01-03", "J1"),
            GroupedEdge("x", "z", 1, "2026-01-01", "K"),
        ]

    def test_upstream_groups_by_output(self, storage):
        self._add(storage, "A", "x", "y", "2026-01-01")
        self._add(storage, "B", "w", "y", "2026-01-02")
        edges = storage.grouped_edges(["y"], UPSTREAM)
        assert [(e.input, e.output) for e in edges] == [("w", "y"), ("x", "y")]

    def test_self_loops_collapse(self, storage):
        for job_id in ("A", "B", "C"):
            self._add(storage, job_id, "t", "t", "2026-01-01")
        assert [e.job_count for e in storage.grouped_edges(["t"], DOWNSTREAM)] == [3]

    def test_limit_keeps_most_recent(self, storage):
        self._add(storage, "A", "x", "old", "2026-01-01")
        self._add(storage, "B", "x", "new", "2026-02-01")
        edges = storage.grouped_edges(["x"], DOWNSTREAM, limit=1)
        assert [e.output for e in edges] == ["new"]

    def test_empty_and_terminal_frontier_queries_nothing(self, storage):
        self._add(storage, "A", TERMINAL, "y", "2026-01-01")
        assert storage.grouped_edges([], DOWNSTREAM) == []
        assert storage.grouped_edges([TERMINAL], DOWNSTREAM) == []

    def test_unknown_direction_is_rejected(self, storage):
        with pytest.raises(ValueError):
            storage.grouped_edges(["x"], "sideways")
