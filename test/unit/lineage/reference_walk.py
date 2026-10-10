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

"""An independent re-derivation of the lineage traversal, for cross-checking.

Written against the *specification* rather than against ``walk.py``, and
deliberately in a different style: this one materializes the whole edge list and
runs a textbook BFS with a queue, where the implementation batches a frontier per
level and queries storage. Two implementations agreeing is only evidence if they
do not share structure.

The specification it encodes:

1. A row is ``(job_id, input, output)``; ``input``/``output`` are artifact
   identifiers, or the terminal marker for a creation/deletion.
2. Descendants: from node X, follow rows whose ``source`` is X, arriving at their
   ``target``. Ancestors: the mirror.
3. A row that is reached is part of the result, terminals and self-loops included.
4. A terminal is not followed: there is nothing beyond it.
5. A self-loop row is not chained through.
6. A node's reported depth is the length of its shortest path from a seed.
7. ``both`` is the union of the two directions, with each row counted once and
   each node keeping its shorter depth.
8. Exceeding a limit marks the result truncated.

The prototype's own cross-check compares only key sets, not depths -- a gap its
docstring admits. Depth is exactly what property 6 promises, so this reference
returns it and the tests assert on it.
"""

from collections import deque
from typing import Optional

from gbserver.lineage.walk import Direction
from gbserver.storage.stored_lineage_row import TERMINAL


def reference_walk(
    rows: list,
    seeds: list,
    direction: Direction,
    max_depth: int,
    max_nodes_per_level: Optional[int] = None,
) -> tuple:
    """Re-derive the reachable subgraph from a full row list.

    Args:
        rows: every row in the graph, as ``StoredLineageRow`` objects. Taking the
            whole list up front is the point: no storage, no batching, no shared
            code with the implementation.
        seeds: starting identifiers.
        direction: which way to walk.
        max_depth: level limit.
        max_nodes_per_level: optional frontier ceiling.

    Returns:
        ``(row_keys, depths, truncated)`` -- the row identity tuples reached, the
        node -> shortest-depth map, and whether a limit stopped the walk.
    """

    # Adjacency built explicitly, both ways, before any walking happens.
    out_edges: dict = {}
    in_edges: dict = {}
    for row in rows:
        out_edges.setdefault(row.input, []).append(row)
        in_edges.setdefault(row.output, []).append(row)

    start = sorted({s for s in seeds if s and s != TERMINAL})
    depths: dict = {s: 0 for s in start}
    if not start or max_depth <= 0:
        return set(), depths, False

    ways = (
        [Direction.ANCESTORS, Direction.DESCENDANTS]
        if direction == Direction.BOTH
        else [direction]
    )

    row_keys: set = set()
    truncated = False

    for way in ways:
        adjacency = out_edges if way == Direction.DESCENDANTS else in_edges
        # Textbook BFS with an explicit queue of (node, depth) pairs, rather than
        # level-batched frontiers.
        queue = deque((node, 0) for node in start)
        seen = set(start)
        level_counts: dict = {}

        while queue:
            node, depth = queue.popleft()
            if depth >= max_depth:
                truncated = True
                continue

            for row in adjacency.get(node, []):
                row_keys.add((row.job_id, row.input, row.output))

                nxt = row.output if way == Direction.DESCENDANTS else row.input
                if not nxt or nxt == TERMINAL:
                    continue  # property 4
                if row.input == row.output:
                    continue  # property 5
                if nxt in seen:
                    continue

                seen.add(nxt)
                level = depth + 1
                level_counts[level] = level_counts.get(level, 0) + 1
                if (
                    max_nodes_per_level is not None
                    and level_counts[level] > max_nodes_per_level
                ):
                    truncated = True
                    continue
                if nxt in depths:
                    depths[nxt] = min(depths[nxt], level)  # property 7
                else:
                    depths[nxt] = level  # property 6
                queue.append((nxt, level))

    return row_keys, depths, truncated
