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

"""Tests for the lineage graph traversal.

The cross-check tests compare against ``reference_walk``, an independent
re-derivation written against the specification. They assert on ``(node, depth)``
pairs, not just which nodes were reached: the prototype's equivalent check compares
only key sets and skips depths -- a gap its own docstring admits -- and depth is
precisely what "shortest path" promises.
"""

import random

import pytest

from gbserver.lineage.walk import (
    DEFAULT_MAX_DEPTH,
    Direction,
    LineageGraph,
    walk_lineage,
)
from gbserver.storage.lineage_row_storage import GroupedEdge
from gbserver.storage.stored_lineage_row import TERMINAL, StoredLineageRow

from .reference_walk import reference_walk


class FakeStorage:
    """In-memory stand-in implementing only the two hop queries.

    Counts queries, so a test can assert the walk costs one query per level rather
    than one per node.
    """

    def __init__(self, rows: list) -> None:
        self.rows = rows
        self.queries = 0

    def get_rows_by_input(self, sources: list, self_loops: bool = True) -> list:
        self.queries += 1
        wanted = {s for s in sources if s and s != TERMINAL}
        return [
            r
            for r in self.rows
            if r.input in wanted and (self_loops or r.input != r.output)
        ]

    def get_rows_by_output(self, targets: list, self_loops: bool = True) -> list:
        self.queries += 1
        wanted = {t for t in targets if t and t != TERMINAL}
        return [
            r
            for r in self.rows
            if r.output in wanted and (self_loops or r.input != r.output)
        ]

    def get_rows_by_jobs(self, job_ids: list) -> list:
        self.queries += 1
        return [r for r in self.rows if r.job_id in set(job_ids)]

    def grouped_self_loops(self, uris: list) -> list:
        self.queries = getattr(self, "queries", 0) + 1
        jobs: dict = {}
        for r in self.rows:
            if r.input == r.output and r.input in set(uris):
                jobs.setdefault(r.input, set()).add(r.job_id)
        return [
            GroupedEdge(uri, uri, len(ids), "", min(ids)) for uri, ids in jobs.items()
        ]


def row(job_id: str, input: str, output: str) -> StoredLineageRow:
    return StoredLineageRow(job_id=job_id, input=input, output=output)


def chain(*nodes: str) -> list:
    """Rows forming a linear chain a -> b -> c ..."""
    return [row(f"J{i}", nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]


def keys(graph: LineageGraph) -> set:
    return {(r.job_id, r.input, r.output) for r in graph.rows}


class TestDirections:
    def test_descendants_follows_source_to_target(self):
        storage = FakeStorage(chain("a", "b", "c"))
        graph = walk_lineage(storage, ["a"], Direction.DESCENDANTS)
        assert graph.depths == {"a": 0, "b": 1, "c": 2}

    def test_ancestors_follows_target_to_source(self):
        storage = FakeStorage(chain("a", "b", "c"))
        graph = walk_lineage(storage, ["c"], Direction.ANCESTORS)
        assert graph.depths == {"c": 0, "b": 1, "a": 2}

    def test_descendants_does_not_walk_backward(self):
        storage = FakeStorage(chain("a", "b", "c"))
        graph = walk_lineage(storage, ["c"], Direction.DESCENDANTS)
        assert graph.depths == {"c": 0}
        assert not graph.rows

    def test_both_reaches_each_side(self):
        storage = FakeStorage(chain("a", "b", "c", "d"))
        graph = walk_lineage(storage, ["c"], Direction.BOTH)
        assert graph.depths == {"c": 0, "b": 1, "a": 2, "d": 1}


class TestBothDeduplicates:
    def test_a_row_reached_both_ways_appears_once(self):
        """Seeding at both ends of one row: each direction reaches it."""
        storage = FakeStorage([row("J", "a", "b")])
        graph = walk_lineage(storage, ["a", "b"], Direction.BOTH)
        assert len(graph.rows) == 1

    def test_diamond_counts_each_row_once(self):
        rows = [
            row("J1", "top", "left"),
            row("J2", "top", "right"),
            row("J3", "left", "bottom"),
            row("J4", "right", "bottom"),
        ]
        storage = FakeStorage(rows)
        graph = walk_lineage(storage, ["top", "bottom"], Direction.BOTH)
        assert len(graph.rows) == 4
        assert len(keys(graph)) == 4


class TestTerminals:
    def test_creation_row_is_included_but_not_followed(self):
        rows = [row("C", TERMINAL, "a")] + chain("a", "b")
        storage = FakeStorage(rows)
        graph = walk_lineage(storage, ["b"], Direction.ANCESTORS)
        assert ("C", TERMINAL, "a") in keys(graph)
        # The terminal marker is not a node.
        assert TERMINAL not in graph.depths
        assert graph.depths == {"b": 0, "a": 1}

    def test_deletion_row_is_included_but_not_followed(self):
        rows = chain("a", "b") + [row("D", "b", TERMINAL)]
        storage = FakeStorage(rows)
        graph = walk_lineage(storage, ["a"], Direction.DESCENDANTS)
        assert ("D", "b", TERMINAL) in keys(graph)
        assert TERMINAL not in graph.depths

    def test_two_unrelated_creations_do_not_join(self):
        """Every creation row shares the terminal marker; following it would make
        them one node and invent provenance.
        """
        rows = [row("C1", TERMINAL, "a"), row("C2", TERMINAL, "b")]
        storage = FakeStorage(rows)
        graph = walk_lineage(storage, ["a"], Direction.BOTH)
        assert "b" not in graph.depths

    def test_terminal_seed_yields_empty_graph(self):
        storage = FakeStorage([row("C", TERMINAL, "a")])
        graph = walk_lineage(storage, [TERMINAL], Direction.BOTH)
        assert not graph.rows
        assert graph.depths == {}


class TestSelfLoops:
    def test_self_loop_row_is_included_once(self):
        storage = FakeStorage([row("S", "tbl", "tbl")])
        graph = walk_lineage(storage, ["tbl"], Direction.DESCENDANTS)
        assert len(graph.rows) == 1

    def test_self_loop_does_not_chain_through(self):
        """Legitimate for unversioned entities -- several runs rewriting one table
        converge on a node -- so the walk must not loop forever.
        """
        rows = [row("S", "tbl", "tbl"), row("J", "tbl", "next")]
        storage = FakeStorage(rows)
        graph = walk_lineage(storage, ["tbl"], Direction.DESCENDANTS)
        assert graph.depths == {"tbl": 0, "next": 1}

    def test_cycle_terminates(self):
        rows = chain("a", "b", "c") + [row("Jx", "c", "a")]
        storage = FakeStorage(rows)
        graph = walk_lineage(storage, ["a"], Direction.DESCENDANTS)
        assert graph.depths == {"a": 0, "b": 1, "c": 2}
        # All three rows are on the path, including the one closing the cycle; "a"
        # is simply not re-expanded.
        assert keys(graph) == {("J0", "a", "b"), ("J1", "b", "c"), ("Jx", "c", "a")}


class TestShortestPath:
    def test_depth_is_the_shortest_path(self):
        """Two routes to one node: the shorter wins."""
        rows = [
            row("long1", "a", "x"),
            row("long2", "x", "y"),
            row("long3", "y", "z"),
            row("short", "a", "z"),
        ]
        storage = FakeStorage(rows)
        graph = walk_lineage(storage, ["a"], Direction.DESCENDANTS)
        assert graph.depths["z"] == 1

    def test_both_keeps_the_shorter_of_two_directions(self):
        rows = [row("J1", "seed", "x"), row("J2", "x", "seed")]
        storage = FakeStorage(rows)
        graph = walk_lineage(storage, ["seed"], Direction.BOTH)
        assert graph.depths["x"] == 1


class TestLimits:
    def test_max_depth_truncates(self):
        storage = FakeStorage(chain("a", "b", "c", "d", "e"))
        graph = walk_lineage(storage, ["a"], Direction.DESCENDANTS, max_depth=2)
        assert graph.depths == {"a": 0, "b": 1, "c": 2}
        assert graph.truncated

    def test_exact_depth_is_not_truncated(self):
        storage = FakeStorage(chain("a", "b"))
        graph = walk_lineage(storage, ["a"], Direction.DESCENDANTS, max_depth=5)
        assert not graph.truncated

    def test_zero_depth_returns_seeds_only(self):
        storage = FakeStorage(chain("a", "b"))
        graph = walk_lineage(storage, ["a"], Direction.DESCENDANTS, max_depth=0)
        assert graph.depths == {"a": 0}
        assert not graph.rows

    def test_wide_level_truncates(self):
        rows = [row(f"J{i}", "a", f"out{i}") for i in range(20)]
        storage = FakeStorage(rows)
        graph = walk_lineage(
            storage, ["a"], Direction.DESCENDANTS, max_nodes_per_level=5
        )
        assert graph.truncated


class TestUnexpandedFrontier:
    """``unexpanded`` turns ``truncated`` into a magnitude.

    A bare boolean let the UI claim "partially displayed" without saying how much
    was missing, which read as "click twice more" on a graph whose remainder was in
    the thousands. Both stopping conditions report their leftover frontier.
    """

    def test_a_complete_walk_reports_nothing_unexpanded(self):
        storage = FakeStorage(chain("a", "b"))
        graph = walk_lineage(storage, ["a"], Direction.DESCENDANTS, max_depth=5)
        assert not graph.truncated
        assert graph.unexpanded == 0

    def test_depth_limit_reports_the_frontier_it_stopped_on(self):
        storage = FakeStorage(chain("a", "b", "c", "d", "e"))
        graph = walk_lineage(storage, ["a"], Direction.DESCENDANTS, max_depth=2)
        # Stopped holding "c", whose descendants were never queried.
        assert graph.truncated
        assert graph.unexpanded == 1

    def test_depth_limit_counts_a_wide_frontier(self):
        rows = [row(f"J{i}", "a", f"m{i}") for i in range(7)]
        rows += [row(f"K{i}", f"m{i}", f"end{i}") for i in range(7)]
        storage = FakeStorage(rows)
        graph = walk_lineage(storage, ["a"], Direction.DESCENDANTS, max_depth=1)
        assert graph.truncated
        assert graph.unexpanded == 7

    def test_wide_level_reports_the_frontier_it_refused(self):
        rows = [row(f"J{i}", "a", f"out{i}") for i in range(20)]
        storage = FakeStorage(rows)
        graph = walk_lineage(
            storage, ["a"], Direction.DESCENDANTS, max_depth=3, max_nodes_per_level=5
        )
        # The 20-node level exceeded the cap, so all 20 went unexpanded.
        assert graph.truncated
        assert graph.unexpanded == 20

    def test_both_sums_each_direction(self):
        """A BOTH walk shares one graph, so each side adds its own remainder."""
        storage = FakeStorage(chain("up2", "up1", "a", "down1", "down2", "down3"))
        graph = walk_lineage(storage, ["a"], Direction.BOTH, max_depth=1)
        # Holding "up1"'s ancestor side and "down1"'s descendant side.
        assert graph.truncated
        assert graph.unexpanded == 2

    def test_a_terminal_frontier_is_not_unexpanded(self):
        """Reaching the end of the graph is not truncation, even at the depth limit."""
        storage = FakeStorage(chain("a", "b"))
        graph = walk_lineage(storage, ["a"], Direction.DESCENDANTS, max_depth=1)
        assert not graph.truncated
        assert graph.unexpanded == 0


class TestRaisingThePerLevelCeiling:
    """A caller can raise the per-level cap for an explicit "full graph" request.

    The point of raising it is that a wide level stops being truncation. The ceiling
    is still a ceiling: there is deliberately no value meaning "no limit".
    """

    def test_a_raised_ceiling_stops_truncating_a_wide_level(self):
        rows = [row(f"J{i}", "a", f"out{i}") for i in range(20)]
        storage = FakeStorage(rows)
        graph = walk_lineage(
            storage, ["a"], Direction.DESCENDANTS, max_nodes_per_level=50
        )
        assert not graph.truncated
        assert graph.unexpanded == 0
        assert len(graph.depths) == 21

    def test_the_same_level_truncates_under_a_low_ceiling(self):
        """The contrast: identical graph, only the ceiling differs."""
        rows = [row(f"J{i}", "a", f"out{i}") for i in range(20)]
        storage = FakeStorage(rows)
        graph = walk_lineage(
            storage, ["a"], Direction.DESCENDANTS, max_depth=3, max_nodes_per_level=5
        )
        assert graph.truncated
        assert graph.unexpanded == 20


class TestQueryCost:
    def test_one_query_per_level_not_per_node(self):
        rows = [row(f"J{i}", "a", f"m{i}") for i in range(10)]
        rows += [row(f"K{i}", f"m{i}", "end") for i in range(10)]
        storage = FakeStorage(rows)
        graph = walk_lineage(storage, ["a"], Direction.DESCENDANTS)
        # 3 levels expanded (a -> m*, m* -> end, end -> nothing), 20 rows.
        assert storage.queries == 3
        assert len(graph.rows) == 20

    def test_the_truncation_probe_is_one_query_not_one_per_node(self):
        """Confirming a frontier has more graph must not cost a query per node.

        The probe that decides `unexpanded` runs on the level the walk abandons. A
        naive per-node version reintroduced the N-queries-per-level cost that _hop
        exists to avoid, and no existing test caught it because none exhausted the
        depth limit on a wide frontier.
        """
        rows = [row(f"J{i}", "a", f"m{i}") for i in range(10)]
        rows += [row(f"K{i}", f"m{i}", f"end{i}") for i in range(10)]
        storage = FakeStorage(rows)
        graph = walk_lineage(storage, ["a"], Direction.DESCENDANTS, max_depth=1)
        assert graph.unexpanded == 10
        # One query to expand "a", one to probe the 10-node frontier.
        assert storage.queries == 2

    def test_both_costs_each_direction_separately(self):
        storage = FakeStorage(chain("a", "b"))
        walk_lineage(storage, ["a"], Direction.BOTH)
        assert storage.queries >= 2


class TestScopingIsBySeedNotByFilter:
    """A scoped walk is expressed by choosing seeds, not by filtering rows.

    ``walk_lineage`` used to take a ``build_id`` that it applied in Python *after*
    each level's query. The column is gone (a build is granite.build's own concept,
    empty on every imported row), and with it the filter: a caller wanting a scoped
    view resolves that scope in its own system and passes the URIs it produced.
    """

    def test_walk_lineage_takes_no_build_scope(self):
        import inspect

        assert "build_id" not in inspect.signature(walk_lineage).parameters

    def test_seeding_a_subset_bounds_the_graph(
        self,
    ):
        rows = [row("J1", "a", "b"), row("J2", "c", "d")]
        storage = FakeStorage(rows)
        graph = walk_lineage(storage, ["a"], Direction.DESCENDANTS)
        assert graph.depths == {"a": 0, "b": 1}


class TestEmptyCases:
    def test_artifact_with_no_lineage_is_an_empty_graph_not_an_error(self):
        storage = FakeStorage([])
        graph = walk_lineage(storage, ["orphan"], Direction.BOTH)
        assert graph.depths == {"orphan": 0}
        assert not graph.rows
        assert not graph.truncated

    def test_no_seeds_queries_nothing(self):
        storage = FakeStorage(chain("a", "b"))
        graph = walk_lineage(storage, [], Direction.BOTH)
        assert storage.queries == 0
        assert graph.depths == {}

    def test_multiple_seeds_all_start_at_zero(self):
        storage = FakeStorage(chain("a", "b") + chain("x", "y"))
        graph = walk_lineage(storage, ["a", "x"], Direction.DESCENDANTS)
        assert graph.depths == {"a": 0, "x": 0, "b": 1, "y": 1}


class TestGraphResult:
    def test_nodes_excludes_terminals(self):
        storage = FakeStorage([row("C", TERMINAL, "a")])
        graph = walk_lineage(storage, ["a"], Direction.ANCESTORS)
        assert graph.nodes == {"a"}

    def test_max_depth_reached(self):
        storage = FakeStorage(chain("a", "b", "c"))
        graph = walk_lineage(storage, ["a"], Direction.DESCENDANTS)
        assert graph.max_depth_reached() == 2

    def test_empty_graph_max_depth_is_zero(self):
        assert LineageGraph().max_depth_reached() == 0


def random_graph(seed: int, nodes: int = 12, edges: int = 20) -> list:
    """Build a random graph, deliberately including the hard shapes.

    Cycles, self-loops, N*M fan-out, terminals and multi-edges all arise, because
    those are exactly where two traversals are most likely to disagree.
    """
    rng = random.Random(seed)
    names = [f"n{i}" for i in range(nodes)]
    rows = []
    for i in range(edges):
        kind = rng.random()
        if kind < 0.08:
            source, target = TERMINAL, rng.choice(names)  # creation
        elif kind < 0.16:
            source, target = rng.choice(names), TERMINAL  # deletion
        elif kind < 0.24:
            node = rng.choice(names)
            source, target = node, node  # self-loop
        else:
            source, target = rng.choice(names), rng.choice(names)
        rows.append(row(f"J{i}", source, target))
    return rows


class TestCrossCheckAgainstReference:
    """The implementation must agree with an independent re-derivation.

    Both the rows reached and every node's depth are compared -- the depth half is
    what the prototype's own cross-check omits.
    """

    # Ten seeds, not thirty: at 12 nodes and 20 edges the generator saturates its
    # structural variety early -- thirty seeds yield only seven distinct
    # shape signatures (creation / deletion / self-loop / multi-edge present or
    # not), and these ten already cover six of them. The seventh, a graph with no
    # self-loop, is pinned separately below rather than left to chance.
    @pytest.mark.parametrize("seed", range(10))
    @pytest.mark.parametrize(
        "direction",
        [Direction.ANCESTORS, Direction.DESCENDANTS, Direction.BOTH],
        ids=lambda d: d.value,
    )
    def test_matches_reference_on_random_graphs(self, seed, direction):
        rows = random_graph(seed)
        storage = FakeStorage(rows)
        start = ["n0", "n5"]

        graph = walk_lineage(storage, start, direction, max_depth=DEFAULT_MAX_DEPTH)
        ref_keys, ref_depths, _ = reference_walk(
            rows, start, direction, max_depth=DEFAULT_MAX_DEPTH
        )

        assert keys(graph) == ref_keys
        assert graph.depths == ref_depths

    @pytest.mark.parametrize("max_depth", [1, 2, 3, 5])
    def test_matches_reference_under_a_depth_limit(self, max_depth):
        rows = random_graph(7, nodes=15, edges=30)
        storage = FakeStorage(rows)
        graph = walk_lineage(storage, ["n0"], Direction.BOTH, max_depth=max_depth)
        ref_keys, ref_depths, _ = reference_walk(
            rows, ["n0"], Direction.BOTH, max_depth=max_depth
        )
        assert keys(graph) == ref_keys
        assert graph.depths == ref_depths

    @pytest.mark.parametrize(
        "rows,label",
        [
            (random_graph(24), "no-self-loop"),
            ([row("S", "n", "n")], "only-a-self-loop"),
            ([row("C", TERMINAL, "a"), row("D", "a", TERMINAL)], "only-terminals"),
        ],
        ids=lambda value: value if isinstance(value, str) else "",
    )
    def test_matches_reference_on_pinned_shapes(self, rows, label):
        """Shapes the random generator reaches rarely or never."""
        storage = FakeStorage(rows)
        graph = walk_lineage(storage, ["a", "n", "n0"], Direction.BOTH, max_depth=10)
        ref_keys, ref_depths, _ = reference_walk(
            rows, ["a", "n", "n0"], Direction.BOTH, max_depth=10
        )
        assert keys(graph) == ref_keys
        assert graph.depths == ref_depths

    def test_matches_reference_on_a_cartesian_job(self):
        """A true N*M row set, which the traversal must still handle.

        ``to_lineage_rows`` now rejects such a job, so granite.build will not
        write this shape -- but the walk reads rows, not jobs, and an importer or
        rows predating the guard can still present it. Built directly for that
        reason.
        """
        rows = [
            row("J", src, tgt) for src in ("i1", "i2", "i3") for tgt in ("o1", "o2")
        ]
        storage = FakeStorage(rows)
        graph = walk_lineage(storage, ["i1"], Direction.BOTH, max_depth=10)
        ref_keys, ref_depths, _ = reference_walk(
            rows, ["i1"], Direction.BOTH, max_depth=10
        )
        assert keys(graph) == ref_keys
        assert graph.depths == ref_depths


class TestCollapseSelfLoops:
    """A collapsing walk reads one row per self-looping artifact, plus its count."""

    def test_keeps_one_sample_and_the_count(self):
        rows = [row(f"S{i}", "a", "a") for i in range(5)] + chain("a", "b", "c")
        storage = FakeStorage(rows)
        graph = walk_lineage(storage, ["a"], Direction.BOTH, collapse_self_loops=True)
        assert graph.depths == walk_lineage(FakeStorage(rows), ["a"]).depths
        assert keys(graph) == {("S0", "a", "a"), ("J0", "a", "b"), ("J1", "b", "c")}
        assert graph.self_loop_runs == {"a": 5}

    def test_self_loops_deeper_in_the_graph_are_counted(self):
        rows = chain("a", "b") + [row("S1", "b", "b"), row("S2", "b", "b")]
        graph = walk_lineage(
            FakeStorage(rows), ["a"], Direction.DESCENDANTS, collapse_self_loops=True
        )
        assert graph.self_loop_runs == {"b": 2}
        assert ("S1", "b", "b") in keys(graph)

    def test_default_walk_reads_every_self_loop(self):
        rows = [row("S1", "a", "a"), row("S2", "a", "a")]
        graph = walk_lineage(FakeStorage(rows), ["a"], Direction.BOTH)
        assert len(graph.rows) == 2
        assert graph.self_loop_runs == {}
