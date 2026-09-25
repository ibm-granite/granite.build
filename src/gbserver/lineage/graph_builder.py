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

"""Turn walked lineage rows into the graph dict the API already speaks.

The read path's shape is not ours to choose: ``api/lineage.get_artifact_graph``
consumes ``{root_id, nodes, edges, truncated}`` with artifact and run nodes and
directed edges between them, and re-projects that into the run-centred
``ArtifactGraphResponse`` the frontend renders. Serving the index through the same
dict is what lets a second provider land with no frontend change.

The shape conversion is the real work here, because the storage model and the wire
model disagree about what a node is. A stored row is one flat
``(source, job_id, target)`` triple -- the job is a *column*, not a node. The wire
graph is **bipartite**: artifacts and runs are both nodes, and every edge joins one
of each (``artifact -> run`` for a consumed input, ``run -> artifact`` for a
produced output). So one row expands into up to two edges through a shared run
node, and the rows of one ``job_id`` converge on that single run node -- which is
what reassembles the N*M decomposition into "this execution had these inputs and
these outputs".

Terminals do not become nodes. A creation row's source and a deletion row's target
are the empty marker, identifying no artifact; each contributes only the half-edge
it does have. Emitting a node for them would collapse every creation in the graph
into one shared "nothing" node.
"""

import logging
from typing import Optional

from gbcommon.uri.lh import URLSEGMENT_FILES, URLSEGMENT_MODELS as LH_URLSEGMENT_MODELS
from gbserver.lineage.attributes import (
    SOURCE,
    TARGET,
    endpoint_kind,
    endpoint_name,
    job_detail,
    origin_id,
    origin_system,
)
from gbserver.lineage.walk import LineageGraph
from gbserver.storage.stored_lineage_row import TERMINAL, StoredLineageRow

logger = logging.getLogger(__name__)

# Wire node_type values, matching what the W&B provider emits and what
# api/lineage.get_artifact_graph branches on.
NODE_TYPE_ARTIFACT = "artifact"
NODE_TYPE_RUN = "run"

# The run-node metadata keys the API handler reads to build an ArtifactRunEntry,
# mapped from the ``job`` group of the row's attributes blob. Listed as data so the
# handler's expected names and the blob's contract are translated in one visible
# place rather than by a chain of lookups.
#
# Four of the handler's fields are absent from a ROW on purpose: job_input_params,
# execution_stats, job_output_stats and source_code_details are large and identical
# across every row of one job, so they live on the job record instead --
# gb_lineage_job, one row per job_id, under the PAYLOAD group (see
# gbserver.lineage.attributes). They still default to {} here, because this projects
# rows; a caller wanting them reads the job record, or GET /lineage/target/{id}.
_RUN_METADATA_FROM_JOB = {
    "name": "job_name",
    "namespace": "job_namespace",
    "type": "job_type",
    "status": "job_status",
    "started_at": "job_started_at",
    "completed_at": "job_completed_at",
    "category": "category",
    "owner": "owner",
}


def build_graph_dict(
    graph: LineageGraph,
    root_uri: str = "",
    root_is_artifact: bool = True,
) -> dict:
    """Project a walked graph into the API's graph dict.

    Args:
        graph: the walk result, as returned by
            :func:`~gbserver.lineage.walk.walk_lineage`.
        root_uri: the normalized URI the graph is *about*. It becomes ``root_id``,
            because a node's URI is its identity -- there is no separate identifier
            to reconcile it against any more.
        root_is_artifact: whether ``root_uri`` names an artifact. ``False`` for a
            build-seeded graph, which has several roots and no single one to flag:
            nothing is marked ``is_root`` and no node is synthesized. Defaulting
            this to ``True`` would mint a bogus artifact node named after the build.

    Returns:
        ``{root_id, nodes, edges, truncated}``. An empty artifact graph yields the
        root node alone with no edges: "nothing recorded" is a real answer and must
        not read as an error.

    **Self-loops are collapsed.** A row whose source equals its target is an in-place
    rewrite -- an append to a dataset, a table refreshed in place -- and real data is
    full of them: 30.4% of an imported Lakehouse graph, with one dataset appended
    68,905 times. Rendering one run node per such row produced a 55 MB response
    describing a single artifact.

    Collapsing costs nothing in reachability, which is what makes it safe to do here
    rather than as a cap: the traversal already refuses to chain *through* a self-loop
    (see :func:`~gbserver.lineage.walk._walk_one_direction`), so those runs expand no
    frontier and reach no artifact the graph would otherwise miss. They are pure
    volume.

    The collapsed node keeps a ``run_count`` and the id of one representative run, and
    the full list stays available from ``GET /lineage/runs?uri=...`` -- indexed, paged,
    and never truncated. Nothing is lost, only moved off the graph response.
    """
    artifact_nodes: dict[str, dict] = {}
    run_nodes: dict[str, dict] = {}
    # Self-looped rows, grouped by the artifact they rewrite. Collected first so one
    # collapsed node can carry the count, rather than emitting a node per row.
    self_loop_rows: dict[str, list] = {}
    edges: list[dict] = []
    edge_keys: set[tuple[str, str]] = set()

    def add_edge(source: str, target: str) -> None:
        key = (source, target)
        if key in edge_keys:
            return
        edge_keys.add(key)
        edges.append({"source": source, "target": target})

    for row in graph.rows:
        if row.is_self_loop():
            self_loop_rows.setdefault(row.source, []).append(row)
            continue

        run_id = _run_node_id(row)
        if run_id not in run_nodes:
            run_nodes[run_id] = _run_node(row, run_id)

        if row.source != TERMINAL:
            _ensure_artifact_node(
                artifact_nodes,
                uri=row.source,
                kind=endpoint_kind(row.attributes, SOURCE),
                name=endpoint_name(row.attributes, SOURCE),
                depth=graph.depths.get(row.source),
            )
            add_edge(row.source, run_id)

        if row.target != TERMINAL:
            _ensure_artifact_node(
                artifact_nodes,
                uri=row.target,
                kind=endpoint_kind(row.attributes, TARGET),
                name=endpoint_name(row.attributes, TARGET),
                depth=graph.depths.get(row.target),
            )
            add_edge(run_id, row.target)

    # One node per self-rewritten artifact, in place of one per row.
    for uri, rows in self_loop_rows.items():
        _ensure_artifact_node(
            artifact_nodes,
            uri=uri,
            kind=endpoint_kind(rows[0].attributes, SOURCE),
            name=endpoint_name(rows[0].attributes, SOURCE),
            depth=graph.depths.get(uri),
        )
        run_id = _self_loop_node_id(uri)
        run_nodes[run_id] = _self_loop_node(rows, run_id)
        # Both directions, so the rewrite reads as a cycle on the artifact rather than
        # a dangling node. The edge set dedups, so this is two edges however many rows
        # collapsed into it.
        add_edge(uri, run_id)
        add_edge(run_id, uri)

    if root_is_artifact and root_uri:
        # The root may appear in no row -- an artifact with no lineage recorded yet.
        # It still has to be in the graph, or the response would describe a
        # different artifact than the one that was asked about.
        if root_uri not in artifact_nodes:
            _ensure_artifact_node(
                artifact_nodes,
                uri=root_uri,
                kind="",
                name="",
                depth=graph.depths.get(root_uri, 0),
            )
        artifact_nodes[root_uri]["is_root"] = True

    return {
        "root_id": root_uri,
        "nodes": list(artifact_nodes.values()) + list(run_nodes.values()),
        "edges": edges,
        "truncated": graph.truncated,
    }


def _ensure_artifact_node(
    nodes: dict[str, dict],
    uri: str,
    kind: str,
    name: str,
    depth: Optional[int] = None,
) -> None:
    """Add an artifact node for ``uri`` if it is not already present.

    Keyed by the normalized URI, which is both the graph's identity and what the
    frontend deduplicates by -- the two used to be different things, and keeping
    them in step was the reason a separate URI column existed.

    **First writer wins.** A URI has one artifact type by decision, so the first row
    to mention it settles what it is; nothing here reconciles a later disagreement.
    Kind and name come from the ``attributes`` blob, so a row written by a producer
    that recorded neither leaves them to the URI-derived fallback rather than
    showing an unnamed node.

    Args:
        nodes: the accumulator, keyed by URI.
        uri: the artifact's normalized URI.
        kind: its artifact type, if the row carried one.
        name: its display name, if the row carried one.
        depth: hops from the seed, when the walk reached it.
    """
    if uri in nodes:
        return

    nodes[uri] = {
        "id": uri,
        "node_type": NODE_TYPE_ARTIFACT,
        "name": name or _name_from_uri(uri),
        "artifact_type": kind or None,
        "is_root": False,
        "depth": depth,
        "metadata": {"uri": uri},
    }


def _name_from_uri(uri: str) -> str:
    """A display name for a URI whose row carried none.

    The last non-empty path segment, which is the artifact's own name in every
    scheme this index stores (a model label, a table name, an object key) --
    except an ``lh://`` model or fileset, whose last segment is a revision or
    version, not a name (see ``uri_normalize._normalize_lh``); there the name is
    the segment before it. Falls back to the whole URI rather than to an empty
    label: a node with no name is worse to look at than a long one.
    """
    if not uri:
        return ""
    without_scheme = uri.split("://", 1)[-1]
    segments = [segment for segment in without_scheme.split("/") if segment]
    if not segments:
        return uri
    if uri.startswith("lh://") and len(segments) >= 5 and segments[2] in (
        LH_URLSEGMENT_MODELS,
        URLSEGMENT_FILES,
    ):
        return segments[-2]
    return segments[-1]


def _self_loop_node_id(uri: str) -> str:
    """Identity of the node standing in for every in-place rewrite of one artifact.

    Keyed by artifact rather than by job, which is the whole point: the rows being
    collapsed have distinct ``job_id``s and that is exactly the multiplicity being
    removed. Prefixed like a run node so it cannot collide with an artifact URI in the
    shared id space, and distinctly from ``run:`` so a client can tell a collapsed node
    from a real one without inspecting metadata.
    """
    return f"runs:{uri}"


def _self_loop_node(rows: list, run_id: str) -> dict:
    """Build the collapsed node for one artifact's in-place rewrites.

    Carries the count and one representative job id. The representative is the first
    row walked, not a choice of "most recent" -- ordering rows by time would need a
    timestamp the blob does not promise, and claiming a "latest" that is not one is
    worse than not claiming it.
    """
    representative = rows[0]
    metadata = {
        "run_count": len(rows),
        "collapsed": True,
        "representative_job_id": representative.job_id,
        "source_system": origin_system(representative.attributes),
    }
    job = job_detail(representative.attributes)
    if job.get("namespace"):
        # Kept so the per-run space filter on POST /artifact still has something to
        # read; without it a collapsed node fails closed and vanishes from that route.
        metadata["job_namespace"] = job["namespace"]
    if job.get("owner"):
        metadata["owner"] = job["owner"]

    return {
        "id": run_id,
        "node_type": NODE_TYPE_RUN,
        "name": f"{len(rows)} in-place rewrites",
        "artifact_type": None,
        "is_root": False,
        "metadata": metadata,
    }


def _run_node_id(row: StoredLineageRow) -> str:
    """Identity of the run node a row hangs off.

    ``job_id`` is the same value on every row of one execution, which is exactly
    the grouping a run node needs: the N*M rows of a job with 3 inputs and 2
    outputs converge on one run with 3 inbound and 2 outbound edges.

    Prefixed so a run node id can never equal an artifact node id. They share one
    id space in the wire graph -- edges reference plain strings -- and a job_id that
    happened to look like an artifact URI would otherwise fuse a run and an
    artifact into one node.
    """
    return f"run:{row.job_id}"


def _run_node(row: StoredLineageRow, run_id: str) -> dict:
    """Build the run node for a row's job execution.

    The job's detail lives in the ``job`` group of the row's attributes blob; only
    the keys the API handler reads are copied out, under the names it expects.

    ``job_namespace`` is load-bearing beyond display: the handler splits it on the
    first ``/`` to recover the space name and drops runs the caller cannot see. A run
    with neither namespace nor owner therefore fails closed, which is the intended
    behaviour for a row whose provenance is unknown.
    """
    job = job_detail(row.attributes)
    node_metadata = {
        handler_key: job[job_key]
        for job_key, handler_key in _RUN_METADATA_FROM_JOB.items()
        if job.get(job_key)
    }

    # Identity of the execution, for a client correlating back to the index.
    node_metadata.setdefault("job_id", row.job_id)

    # The originating system's own ids, when it had any. Prefixed so a client cannot
    # mistake them for the index's own identity, which is the URI.
    build_id = origin_id(row.attributes, "build_id")
    if build_id:
        node_metadata.setdefault("gb_build_id", build_id)
        # The handler surfaces this as release_id, which IS the build id -- the two
        # were separate fields holding one value before.
        node_metadata.setdefault("release_id", build_id)
    target_run_uuid = origin_id(row.attributes, "target_run_uuid")
    if target_run_uuid:
        node_metadata.setdefault("gb_target_run_uuid", target_run_uuid)

    node_metadata.setdefault("source_system", origin_system(row.attributes))

    return {
        "id": run_id,
        "node_type": NODE_TYPE_RUN,
        "name": job.get("name") or row.job_id,
        "artifact_type": None,
        "is_root": False,
        "metadata": node_metadata,
    }
