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

"""Level-order traversal of the lineage graph.

One query per level, batched over an indexed text column, identical on SQLite and
Postgres. The traversal runs here rather than in SQL because there is no raw-SQL
seam to run a ``WITH RECURSIVE`` through -- ``get_by_where``'s string branch
raises ``NotImplementedError`` -- and because a Python walk needs no per-dialect
SQL.

Breadth-first is not an implementation detail: it is what makes the depth reported
for a node the length of its *shortest* path. The recursive SQL this replaces gets
the same result by grouping on ``max(depth)``/``min(depth)`` after the fact.
"""

import logging
from enum import Enum
from typing import Iterable, Optional

from gbserver.storage.lineage_row_storage import ILineageRowStorage
from gbserver.storage.stored_lineage_row import TERMINAL, StoredLineageRow

logger = logging.getLogger(__name__)

# Matches the prototype's Java-derived limit (its literals are -100/100).
DEFAULT_MAX_DEPTH = 100

# Ceiling on the nodes one level may add. A wide graph can otherwise turn one hop
# into a query with an unbounded IN list and a response with every edge of a hub;
# the walk stops adding nodes to a level once it holds this many, and the result is
# marked truncated rather than silently partial.
#
# Raised from 1000 so an ordinary artifact loads its whole graph instead of
# tripping the cap on the first hop. It stays a real ceiling for hub artifacts --
# a shared base model in model_shared fans out into the thousands per hop, and
# there the omitted runs are genuinely different jobs feeding different
# artifacts, so capping does cost reachability (unlike the self-loop collapse in
# graph_builder). Such a graph reports the reached-but-dropped nodes, and the
# frontier nodes that still had graph behind them, via `unexpanded`, rather than
# pretending a few more clicks would finish it.
DEFAULT_MAX_NODES_PER_LEVEL = 5000
# Rows one hop may read per node the level can take: many runs share an edge, so a
# node usually comes with several rows, but past this the hop is a hub and what is
# shown is marked truncated rather than read whole.
ROWS_PER_NODE = 10

# Hard ceiling a caller can raise the per-level cap to, for an explicit "show the
# full graph" request. Deliberately NOT unbounded: a shared base model's closure
# runs into six figures, each level becomes one IN list that wide, and the SVG
# renderer draws every node with no virtualization. This is the point past which
# the answer stops being a graph anyone can read -- so "full" means "as full as is
# survivable", and the response still reports `unexpanded` when it stops.
ABSOLUTE_MAX_NODES_PER_LEVEL = 50_000


class Direction(Enum):
    """Which way to walk.

    Named for what they walk toward, not for the wire values. The prototype's
    ``downstream``/``upstream`` are counter-intuitive -- its ``downstream`` seeds on
    ``target`` and walks toward origins -- but the live W&B backend already treats
    ``downstream`` as ``used_by()``, i.e. toward descendants, and the frontend sends
    it with that meaning. The wire mapping lives in the API layer; nothing here
    inherits the prototype's inversion.
    """

    ANCESTORS = "ancestors"
    DESCENDANTS = "descendants"
    BOTH = "both"


class LineageGraph:
    """The result of a walk.

    Attributes:
        rows: every row on a path from the seed, deduplicated. A row is included
            when it is reached, including a terminal or a self-loop row.
        depths: node identifier -> depth of its shortest path from the seed. Seeds
            are at 0. Ancestors and descendants both count upward, so a depth is a
            distance and never a signed direction.
        truncated: whether a limit stopped the walk before it ran out of graph, so
            a caller can tell "this is all of it" from "this is as far as we went".
        unexpanded: how many distinct nodes a limit kept out of the graph: nodes
            a level reached past its cap, plus frontier nodes the depth limit left
            that still had graph behind them (a leaf is not counted). Each node is
            counted once, even when both directions of a ``BOTH`` walk stop at it.
            Turns ``truncated`` from a bare flag into a magnitude, so a caller can
            say "N not expanded" instead of implying a few more clicks will finish.
            0 whenever ``truncated`` is False.
        self_loop_runs: artifact URI -> how many jobs rewrote it in place, filled
            only by a walk that collapses self-loops. There ``rows`` holds one
            sample self-loop row per artifact, and this holds the real count.
    """

    __slots__ = ("rows", "depths", "truncated", "unexpanded", "self_loop_runs")

    def __init__(
        self,
        rows: Optional[list] = None,
        depths: Optional[dict] = None,
        truncated: bool = False,
        unexpanded: int = 0,
        self_loop_runs: Optional[dict] = None,
    ) -> None:
        self.rows = rows if rows is not None else []
        self.depths = depths if depths is not None else {}
        self.truncated = truncated
        self.unexpanded = unexpanded
        self.self_loop_runs = self_loop_runs if self_loop_runs is not None else {}

    @property
    def nodes(self) -> set:
        """Every artifact identifier reached, terminals excluded."""
        return set(self.depths)

    def __repr__(self) -> str:
        return (
            f"LineageGraph(rows={len(self.rows)}, nodes={len(self.depths)}, "
            f"truncated={self.truncated}, unexpanded={self.unexpanded})"
        )


def _row_key(row: StoredLineageRow) -> tuple:
    """Identity of a row for deduplication.

    Mirrors the storage unique index, so two reads of one row collapse and two
    genuinely different rows never do. This is also what keeps ``BOTH`` from
    double-counting the row that both directions reach.
    """
    return (row.job_id, row.input, row.output)


def walk_lineage(
    storage: ILineageRowStorage,
    seeds: Iterable[str],
    direction: Direction = Direction.BOTH,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_nodes_per_level: int = DEFAULT_MAX_NODES_PER_LEVEL,
    collapse_self_loops: bool = False,
) -> LineageGraph:
    """Walk the lineage graph outward from ``seeds``.

    Args:
        storage: the lineage row storage to query.
        seeds: canonical identifiers to start from. Terminal markers and empty
            strings are dropped -- they identify no artifact.
        direction: which way to walk. ``BOTH`` runs each direction and merges,
            keeping the shorter depth for a node reachable both ways.
        max_depth: how many levels to expand. 0 returns an empty graph.
        max_nodes_per_level: ceiling on the nodes one level adds. A level that
            reaches more keeps the first ones (in the hop's deterministic order:
            matched node, neighbour, job) and drops the rest,
            with their rows, marking the result truncated. Seeds beyond it are
            not walked at all.
        collapse_self_loops: read one sample row per self-looping artifact and
            count the rest in SQL (``graph.self_loop_runs``), instead of reading
            every ``uri -> uri`` row. For a caller that folds self-loops into one
            node anyway: one dataset has 68,905 of them, about 67 MB of JSON per
            direction that the graph then draws as a single node.

    Returns:
        The reachable subgraph. An artifact with no lineage yields an empty graph
        rather than an error -- "nothing recorded" is a real answer.
    """
    frontier = sorted({seed for seed in seeds if seed and seed != TERMINAL})
    graph = LineageGraph(depths={seed: 0 for seed in frontier})
    if not frontier or max_depth <= 0:
        return graph

    directions = (
        (Direction.ANCESTORS, Direction.DESCENDANTS)
        if direction == Direction.BOTH
        else (direction,)
    )

    seen_rows: set = set()
    # Shared across directions so a node both directions stop at counts once.
    unexpanded: set = set()
    # Self-loops do not depend on direction, so a BOTH walk counts each artifact's
    # once, whichever direction reaches it first.
    self_loops_checked: Optional[set] = set() if collapse_self_loops else None
    for one_way in directions:
        _walk_one_direction(
            storage=storage,
            seeds=frontier,
            direction=one_way,
            max_depth=max_depth,
            max_nodes_per_level=max_nodes_per_level,
            graph=graph,
            seen_rows=seen_rows,
            self_loops_checked=self_loops_checked,
            unexpanded=unexpanded,
        )
    graph.unexpanded = len(unexpanded)
    graph.truncated = graph.truncated or bool(unexpanded)
    return graph


def _walk_one_direction(
    storage: ILineageRowStorage,
    seeds: list,
    direction: Direction,
    max_depth: int,
    max_nodes_per_level: int,
    graph: LineageGraph,
    seen_rows: set,
    self_loops_checked: Optional[set] = None,
    unexpanded: Optional[set] = None,
) -> None:
    """Expand one direction level by level, accumulating into ``graph``.

    Both directions of a ``BOTH`` walk share ``graph`` and ``seen_rows``, which is
    what merges them: a row reached both ways is stored once, and a node reachable
    both ways keeps its shorter depth.

    ``self_loops_checked`` is ``None`` for a walk that reads every row, and
    otherwise the artifacts whose self-loops are already counted; see
    :func:`_self_loop_samples`.

    ``unexpanded`` collects the nodes a limit kept out, shared across directions;
    the caller sets ``graph.unexpanded`` from it.
    """
    if unexpanded is None:
        unexpanded = set()
    frontier = list(seeds)
    visited = set(seeds)
    depth = 0

    if len(frontier) > max_nodes_per_level:
        # Only seeds can get here: a level never adds more than the cap.
        logger.warning(
            "Lineage walk not started: %d seeds exceed the %d-node limit",
            len(frontier),
            max_nodes_per_level,
        )
        graph.truncated = True
        unexpanded.update(frontier)
        return

    while frontier and depth < max_depth:
        depth += 1

        row_limit = max_nodes_per_level * ROWS_PER_NODE
        if self_loops_checked is None:
            rows = _hop(storage, frontier, direction, limit=row_limit)
        else:
            # Leaving self-loops out of the hop changes nothing below: the walk
            # never continues through one.
            rows = _hop(storage, frontier, direction, self_loops=False, limit=row_limit)
        if len(rows) >= row_limit:
            # Some of this level's rows were never read: what it shows is partial.
            logger.warning(
                "Lineage walk truncated at depth %d: hop hit the %d-row limit",
                depth,
                row_limit,
            )
            graph.truncated = True
            unexpanded.update(_cut_off(rows, frontier, direction))
        if self_loops_checked is not None:
            rows += _self_loop_samples(storage, frontier, graph, self_loops_checked)

        next_frontier: list = []
        dropped: set = set()
        for row in rows:
            reached = _continuation(row, direction)

            # The level is full: a node it has not taken yet is left out, row and
            # all, so a hub's first hop cannot return every edge it has.
            if (
                len(next_frontier) >= max_nodes_per_level
                and reached
                and reached != TERMINAL
                and row.input != row.output
                and reached not in visited
            ):
                dropped.add(reached)
                continue

            key = _row_key(row)
            if key not in seen_rows:
                seen_rows.add(key)
                graph.rows.append(row)

            # Stop at a terminal: a creation row has no input and a deletion row no
            # output, so there is nothing further along that path. The row itself is
            # already recorded above -- "this artifact had no input" is information.
            if reached == TERMINAL or not reached:
                continue

            # Stop at a self-loop. Legitimate for unversioned entities: several runs
            # rewriting one table converge on a single node. The row is included, but
            # chaining *through* it would loop forever.
            if row.input == row.output:
                continue

            if reached in visited:
                continue
            visited.add(reached)
            # First time reached, and levels are expanded in order, so this depth is
            # the shortest path. min() keeps the shorter one when the other
            # direction of a BOTH walk already found it.
            existing = graph.depths.get(reached)
            graph.depths[reached] = depth if existing is None else min(existing, depth)
            next_frontier.append(reached)

        if dropped:
            logger.warning(
                "Lineage walk truncated at depth %d: %d nodes past the %d-node limit",
                depth,
                len(dropped),
                max_nodes_per_level,
            )
            graph.truncated = True
            unexpanded.update(dropped)
        frontier = next_frontier

    if frontier:
        # Ran out of depth still holding a frontier -- but holding a node is not the
        # same as there being more graph behind it. A leaf sits on the frontier
        # exactly like a hub does; the difference costs one more query to learn.
        #
        # Without this check every leaf counted as unseen graph, so a shallow walk
        # reported truncation it could not justify and no amount of expanding ever
        # cleared it. One extra hop, only on the level we are about to abandon,
        # buys an honest answer -- and it is ONE query for the whole frontier, not one
        # per node: a per-node probe would reintroduce exactly the
        # N-queries-per-level cost that _hop exists to avoid.
        held = set(frontier)
        remainder = set()
        probe_limit = max_nodes_per_level * ROWS_PER_NODE
        # Self-loops are skipped below, so there is no reason to read them.
        probed = _hop(storage, frontier, direction, self_loops=False, limit=probe_limit)
        if len(probed) >= probe_limit:
            # The probe was cut too: the nodes it never reached are unknown, so
            # count them rather than call them leaves.
            remainder.update(_cut_off(probed, frontier, direction))
        for r in probed:
            reached = _continuation(r, direction)
            # Same stopping rules the main loop applies, so a frontier node whose only
            # rows are terminals or self-loops is correctly not counted as more graph.
            if reached == TERMINAL or not reached or reached in visited:
                continue
            if r.input == r.output:
                continue
            matched = _matched_on(r, direction)
            # The batched query can return rows for nodes beyond this frontier; keep
            # only the ones it was asked about.
            if matched in held:
                remainder.add(matched)
        if remainder:
            graph.truncated = True
            unexpanded.update(remainder)


def _cut_off(rows: list, frontier: list, direction: Direction) -> set:
    """The frontier nodes a row-limited hop may not have read in full.

    The hop is ordered by the matched node, so the last row's node may be partly
    read; those with no row at all may be leaves, or may sort after the cut. Which
    of the two cannot be told without the collation, so both count as unexpanded.
    """
    matched = {_matched_on(row, direction) for row in rows}
    cut = {node for node in frontier if node not in matched}
    if rows:
        cut.add(_matched_on(rows[-1], direction))
    return cut


def _hop(
    storage: ILineageRowStorage,
    frontier: list,
    direction: Direction,
    self_loops: bool = True,
    limit: Optional[int] = None,
) -> list:
    """Fetch one level's rows with a single batched, indexed query.

    ``limit`` bounds the rows read in SQL, so a hub's hop cannot load every edge
    it has into memory before the per-level node cap trims them.
    """
    if direction == Direction.DESCENDANTS:
        return storage.get_rows_by_input(frontier, self_loops=self_loops, limit=limit)
    return storage.get_rows_by_output(frontier, self_loops=self_loops, limit=limit)


def _self_loop_samples(
    storage: ILineageRowStorage,
    frontier: list,
    graph: LineageGraph,
    checked: set,
) -> list:
    """One sample self-loop row per frontier artifact, recording the real counts.

    Two queries, whatever the number of self-loops: a ``GROUP BY`` for the counts
    and the sample job of each, then those sample jobs' rows. Artifacts already in
    ``checked`` are skipped, so a BOTH walk does not count one twice.
    """
    pending = [uri for uri in frontier if uri not in checked]
    checked.update(pending)
    if not pending:
        return []
    edges = storage.grouped_self_loops(pending)
    if not edges:
        return []
    samples = set()
    for edge in edges:
        graph.self_loop_runs[edge.input] = edge.job_count
        samples.add((edge.sample_job_id, edge.input))
    return [
        row
        for row in storage.get_rows_by_jobs(sorted({job for job, _ in samples}))
        if row.input == row.output and (row.job_id, row.input) in samples
    ]


def _matched_on(row: StoredLineageRow, direction: Direction) -> str:
    """Return the identifier a row was matched *on*, the mirror of _continuation.

    Walking toward descendants, ``_hop`` queries by ``input``, so a returned row
    belongs to the frontier node in its ``input``.
    """
    return row.input if direction == Direction.DESCENDANTS else row.output


def _continuation(row: StoredLineageRow, direction: Direction) -> str:
    """Return the identifier a walk continues from, for one row.

    Walking toward descendants, a row matched on its ``input`` continues from its
    ``output``, and vice versa.
    """
    return row.output if direction == Direction.DESCENDANTS else row.input
