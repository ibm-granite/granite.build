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

# Ceiling on the frontier of a single level. A wide graph can otherwise turn one
# hop into a query with an unbounded IN list; when it trips, the result is marked
# truncated rather than silently partial.
#
# Raised from 1000 so an ordinary artifact loads its whole graph instead of
# tripping the cap on the first hop. It stays a real ceiling for hub artifacts --
# a shared base model in model_shared fans out into the thousands per hop, and
# there the omitted runs are genuinely different jobs feeding different
# artifacts, so capping does cost reachability (unlike the self-loop collapse in
# graph_builder). Such a graph reports its leftover frontier via `unexpanded`
# rather than pretending a few more clicks would finish it.
DEFAULT_MAX_NODES_PER_LEVEL = 5000

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
        unexpanded: how many nodes were left on the frontier when a limit stopped
            the walk. Turns ``truncated`` from a bare flag into a magnitude, so a
            caller can say "N not expanded" instead of implying a few more clicks
            will finish. 0 whenever ``truncated`` is False.
    """

    __slots__ = ("rows", "depths", "truncated", "unexpanded")

    def __init__(
        self,
        rows: Optional[list] = None,
        depths: Optional[dict] = None,
        truncated: bool = False,
        unexpanded: int = 0,
    ) -> None:
        self.rows = rows if rows is not None else []
        self.depths = depths if depths is not None else {}
        self.truncated = truncated
        self.unexpanded = unexpanded

    @property
    def nodes(self) -> set:
        """Every artifact identifier reached, terminals excluded."""
        return set(self.depths)

    def max_depth_reached(self) -> int:
        """The deepest level any node was found at, or 0 for an empty graph."""
        return max(self.depths.values(), default=0)

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
) -> LineageGraph:
    """Walk the lineage graph outward from ``seeds``.

    Args:
        storage: the lineage row storage to query.
        seeds: canonical identifiers to start from. Terminal markers and empty
            strings are dropped -- they identify no artifact.
        direction: which way to walk. ``BOTH`` runs each direction and merges,
            keeping the shorter depth for a node reachable both ways.
        max_depth: how many levels to expand. 0 returns an empty graph.
        max_nodes_per_level: ceiling on one level's frontier; exceeding it marks
            the result truncated.

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
    for one_way in directions:
        _walk_one_direction(
            storage=storage,
            seeds=frontier,
            direction=one_way,
            max_depth=max_depth,
            max_nodes_per_level=max_nodes_per_level,
            graph=graph,
            seen_rows=seen_rows,
        )
    return graph


def _walk_one_direction(
    storage: ILineageRowStorage,
    seeds: list,
    direction: Direction,
    max_depth: int,
    max_nodes_per_level: int,
    graph: LineageGraph,
    seen_rows: set,
) -> None:
    """Expand one direction level by level, accumulating into ``graph``.

    Both directions of a ``BOTH`` walk share ``graph`` and ``seen_rows``, which is
    what merges them: a row reached both ways is stored once, and a node reachable
    both ways keeps its shorter depth.
    """
    frontier = list(seeds)
    visited = set(seeds)
    depth = 0

    while frontier and depth < max_depth:
        depth += 1

        if len(frontier) > max_nodes_per_level:
            logger.warning(
                "Lineage walk truncated at depth %d: frontier of %d exceeds the "
                "%d-node limit",
                depth,
                len(frontier),
                max_nodes_per_level,
            )
            graph.truncated = True
            graph.unexpanded += len(frontier)
            return

        rows = _hop(storage, frontier, direction)

        next_frontier: list = []
        for row in rows:
            key = _row_key(row)
            if key not in seen_rows:
                seen_rows.add(key)
                graph.rows.append(row)

            reached = _continuation(row, direction)

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
        for r in _hop(storage, frontier, direction):
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
            graph.unexpanded += len(remainder)


def _hop(
    storage: ILineageRowStorage,
    frontier: list,
    direction: Direction,
) -> list:
    """Fetch one level's rows with a single batched, indexed query."""
    if direction == Direction.DESCENDANTS:
        return storage.get_rows_by_input(frontier)
    return storage.get_rows_by_output(frontier)


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
