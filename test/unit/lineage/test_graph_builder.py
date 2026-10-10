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

"""Tests for projecting walked rows into the API's graph dict.

The conversion is between two disagreeing models: a stored row keeps the job in a
*column*, while the wire graph is bipartite and the job is a *node*. Most of what
is checked here is that the disagreement is resolved without inventing or losing
structure -- especially that terminals do not become nodes and that one job's rows
converge on one run node.
"""

from gbserver.lineage.attributes import build_attributes, build_index_attributes
from gbserver.lineage.graph_builder import build_graph_dict
from gbserver.lineage.walk import LineageGraph
from gbserver.storage.stored_lineage_row import TERMINAL, StoredLineageRow


def row(
    job_id: str,
    input: str,
    output: str,
    **kwargs,
) -> StoredLineageRow:
    """A row whose endpoints are URIs.

    Per-endpoint detail (``source_kind``, ``target_name``, ...) and job metadata go
    into ``attributes``, the JSON blob, since neither is a column. Accepted as
    keywords here and folded in, so the tests stay readable.
    """
    if "attributes" in kwargs:
        attributes = dict(kwargs.pop("attributes") or {})
    else:
        attributes = build_attributes(
            job_metadata=kwargs.pop("metadata", None),
            input_artifact=_artifact_of(kwargs, "source"),
            output_artifact=_artifact_of(kwargs, "target"),
            source_system=kwargs.pop("source_system", "granite.build"),
            ids={
                "build_id": kwargs.pop("build_id", ""),
                "target_run_uuid": kwargs.pop("target_run_uuid", ""),
            },
        )
    assert not kwargs, f"unhandled row() keywords: {sorted(kwargs)}"
    return StoredLineageRow(
        job_id=job_id,
        input=input,
        output=output,
        attributes=attributes,
    )


def _artifact_of(kwargs: dict, side: str) -> dict:
    """Build the artifact dict for one side from ``<side>_kind``/``<side>_name``."""
    artifact = {}
    kind = kwargs.pop(f"{side}_kind", "")
    name = kwargs.pop(f"{side}_name", "")
    if kind:
        artifact["artifact_type"] = kind
    if name:
        artifact["name"] = name
    return artifact


def graph_of(*rows) -> LineageGraph:
    depths = {}
    for r in rows:
        for endpoint in (r.input, r.output):
            if endpoint and endpoint != TERMINAL:
                depths.setdefault(endpoint, 1)
    return LineageGraph(rows=list(rows), depths=depths)


def nodes_by_type(result: dict, node_type: str) -> list:
    return [n for n in result["nodes"] if n["node_type"] == node_type]


def edge_pairs(result: dict) -> set:
    return {(e["source"], e["target"]) for e in result["edges"]}


# Endpoints are normalized URIs -- the node's identity and what the frontend
# deduplicates by are now the same string.
A = "lh://prod/ns/tables/a"
B = "lh://prod/ns/tables/b"
C = "lh://prod/ns/tables/c"


class TestBipartiteShape:
    def test_one_row_becomes_two_edges_through_a_run(self):
        result = build_graph_dict(graph_of(row("J1", A, B)), A)
        assert edge_pairs(result) == {(A, "run:J1"), ("run:J1", B)}

    def test_artifact_and_run_nodes_are_both_present(self):
        result = build_graph_dict(graph_of(row("J1", A, B)), A)
        assert {n["id"] for n in nodes_by_type(result, "artifact")} == {A, B}
        assert {n["id"] for n in nodes_by_type(result, "run")} == {"run:J1"}

    def test_run_node_id_cannot_collide_with_an_artifact_id(self):
        # A job_id shaped like a canonical identifier would otherwise fuse a run
        # node and an artifact node into one.
        result = build_graph_dict(graph_of(row(A, A, B)), A)
        run_ids = {n["id"] for n in nodes_by_type(result, "run")}
        artifact_ids = {n["id"] for n in nodes_by_type(result, "artifact")}
        assert not run_ids & artifact_ids


class TestJobRegrouping:
    def test_n_by_m_rows_converge_on_one_run_node(self):
        # 2 inputs x 2 outputs is 4 rows, but one execution, so one run node with
        # 2 in and 2 out. Rows are built directly: the decompose guard rejects a
        # single 2x2 job, while a real target run reaches this shape as two
        # per-output jobs sharing one job_id (test_lineage_index_roundtrip).
        rows = [
            row("J1", "table:prod/ns::i1", "table:prod/ns::o1"),
            row("J1", "table:prod/ns::i1", "table:prod/ns::o2"),
            row("J1", "table:prod/ns::i2", "table:prod/ns::o1"),
            row("J1", "table:prod/ns::i2", "table:prod/ns::o2"),
        ]
        result = build_graph_dict(graph_of(*rows), "table:prod/ns::i1")
        assert len(nodes_by_type(result, "run")) == 1
        inbound = {e for e in edge_pairs(result) if e[1] == "run:J1"}
        outbound = {e for e in edge_pairs(result) if e[0] == "run:J1"}
        assert len(inbound) == 2 and len(outbound) == 2

    def test_different_jobs_get_different_run_nodes(self):
        result = build_graph_dict(graph_of(row("J1", A, B), row("J2", B, C)), A)
        assert len(nodes_by_type(result, "run")) == 2

    def test_edges_are_deduplicated(self):
        result = build_graph_dict(graph_of(row("J1", A, B), row("J1", A, B)), A)
        assert len(result["edges"]) == len(edge_pairs(result))


class TestTerminals:
    def test_a_creation_contributes_no_source_node(self):
        # Every creation row shares the terminal marker; emitting a node for it
        # would collapse all creations in the graph into one shared node.
        result = build_graph_dict(graph_of(row("J1", TERMINAL, B)), B)
        assert {n["id"] for n in nodes_by_type(result, "artifact")} == {B}
        assert edge_pairs(result) == {("run:J1", B)}

    def test_a_deletion_contributes_no_target_node(self):
        result = build_graph_dict(graph_of(row("J1", A, TERMINAL)), A)
        assert {n["id"] for n in nodes_by_type(result, "artifact")} == {A}
        assert edge_pairs(result) == {(A, "run:J1")}

    def test_two_creations_do_not_share_a_node(self):
        result = build_graph_dict(
            graph_of(row("J1", TERMINAL, B), row("J2", TERMINAL, C)), B
        )
        assert {n["id"] for n in nodes_by_type(result, "artifact")} == {B, C}


class TestRoot:
    def test_root_is_flagged(self):
        result = build_graph_dict(graph_of(row("J1", A, B)), A)
        roots = [n for n in result["nodes"] if n["is_root"]]
        assert [n["id"] for n in roots] == [A]

    def test_root_id_is_echoed(self):
        result = build_graph_dict(graph_of(row("J1", A, B)), A)
        assert result["root_id"] == A

    def test_an_artifact_with_no_lineage_still_yields_its_own_node(self):
        # "Nothing recorded" is a real answer and must not read as an error, nor
        # describe a different artifact than the one asked about.
        result = build_graph_dict(LineageGraph(), A)
        assert result["edges"] == []
        assert [n["id"] for n in result["nodes"]] == [A]
        assert result["nodes"][0]["is_root"] is True

    def test_only_one_node_is_root(self):
        result = build_graph_dict(graph_of(row("J1", A, B), row("J2", B, C)), B)
        assert sum(1 for n in result["nodes"] if n["is_root"]) == 1


class TestNodeIdentity:
    def test_an_artifact_reached_twice_is_one_node(self):
        # A diamond: two paths converge on C.
        rows = [
            row("J1", A, B),
            row("J2", A, C),
            row("J3", B, "table:prod/ns::d"),
            row("J4", C, "table:prod/ns::d"),
        ]
        result = build_graph_dict(graph_of(*rows), A)
        ids = [n["id"] for n in nodes_by_type(result, "artifact")]
        assert len(ids) == len(set(ids))

    def test_the_node_id_and_its_uri_are_the_same_string(self):
        """There is no separate stored URI to keep in step any more.

        The old row carried ``source_uri``/``target_uri`` because a canonical
        identifier did not encode a scheme, so the real URI had to travel beside it.
        With the URI as the identity, the node id *is* the URI.
        """
        result = build_graph_dict(graph_of(row("J1", A, B)), root_uri=A)
        node = next(n for n in result["nodes"] if n["id"] == A)
        assert node["metadata"]["uri"] == A == node["id"]

    def test_nodes_without_a_stored_uri_still_differ(self):
        # What the frontend deduplicates by, so equal URIs would merge them.
        result = build_graph_dict(graph_of(row("J1", A, B)), A)
        uris = [n["metadata"]["uri"] for n in nodes_by_type(result, "artifact")]
        assert len(uris) == len(set(uris))
        assert all(u for u in uris)

    def test_name_falls_back_to_the_last_uri_segment(self):
        """A producer that recorded no name still yields a readable node.

        The last path segment is the artifact's own name in every scheme stored here
        (a model label, a table name, an object key), so it beats showing the whole
        URI -- and showing nothing at all is worse than either.
        """
        result = build_graph_dict(graph_of(row("J1", A, B)), root_uri=A)
        node = next(n for n in result["nodes"] if n["id"] == A)
        assert node["name"] == "a"

    def test_artifact_type_is_absent_when_no_row_recorded_one(self):
        """It is not guessed from the URI shape.

        The blob is the only source for a node's type. Inferring ``table`` from an
        ``lh://.../tables/...`` path would be a second, silently diverging opinion
        about what an artifact is.
        """
        result = build_graph_dict(graph_of(row("J1", A, B)), root_uri=A)
        node = next(n for n in result["nodes"] if n["id"] == A)
        assert node["artifact_type"] is None

    def test_recorded_detail_is_preferred_over_the_uri_fallback(self):
        result = build_graph_dict(
            graph_of(row("J1", A, B, source_name="Display Name", source_kind="model")),
            root_uri=A,
        )
        node = next(n for n in result["nodes"] if n["id"] == A)
        assert node["name"] == "Display Name"
        assert node["artifact_type"] == "model"


class TestRunMetadata:
    def test_job_namespace_survives_for_the_authorization_filter(self):
        # The API handler splits this on "/" to recover the space and drop runs the
        # caller cannot see, so losing it would fail every run closed.
        result = build_graph_dict(
            graph_of(
                row(
                    "J1",
                    A,
                    B,
                    metadata={"job_namespace": "my-space/my-build", "owner": "alice"},
                )
            ),
            A,
        )
        run = nodes_by_type(result, "run")[0]
        assert run["metadata"]["job_namespace"] == "my-space/my-build"
        assert run["metadata"]["owner"] == "alice"

    def test_run_name_comes_from_job_name_when_present(self):
        result = build_graph_dict(
            graph_of(row("J1", A, B, metadata={"job_name": "train"})), A
        )
        assert nodes_by_type(result, "run")[0]["name"] == "train"

    def test_run_name_falls_back_to_job_id(self):
        result = build_graph_dict(graph_of(row("J1", A, B)), A)
        assert nodes_by_type(result, "run")[0]["name"] == "J1"

    def test_unknown_metadata_keys_are_not_copied(self):
        result = build_graph_dict(
            graph_of(row("J1", A, B, metadata={"surprise": "x"})), A
        )
        assert "surprise" not in nodes_by_type(result, "run")[0]["metadata"]

    def test_build_provenance_is_exposed(self):
        result = build_graph_dict(
            graph_of(row("J1", A, B, build_id="BLD", target_run_uuid="TR")), A
        )
        metadata = nodes_by_type(result, "run")[0]["metadata"]
        assert metadata["gb_build_id"] == "BLD"
        assert metadata["gb_target_run_uuid"] == "TR"


class TestTruncation:
    def test_truncated_is_propagated(self):
        graph = graph_of(row("J1", A, B))
        graph.truncated = True
        assert build_graph_dict(graph, A)["truncated"] is True

    def test_untruncated_is_propagated(self):
        assert build_graph_dict(graph_of(row("J1", A, B)), A)["truncated"] is False


class TestSelfLoopCollapse:
    """In-place rewrites collapse to one node, not one per row.

    Real data made this necessary rather than nice: 30.4% of an imported Lakehouse
    graph is self-loops, and one dataset was appended 68,905 times. A run node per row
    produced a 55 MB response describing a single artifact (2.3 KB after this).

    It is safe to collapse *here*, rather than cap, because the traversal already
    refuses to chain through a self-loop -- those rows expand no frontier and reach no
    artifact the graph would otherwise miss. They are pure volume.
    """

    def test_many_rewrites_become_one_node(self):
        rows = [row(f"J{i}", A, A) for i in range(500)]
        result = build_graph_dict(graph_of(*rows), root_uri=A)
        runs = nodes_by_type(result, "run")
        assert len(runs) == 1
        assert runs[0]["metadata"]["run_count"] == 500

    def test_the_collapsed_node_is_flagged(self):
        """A client must be able to tell a looped node from a single run."""
        result = build_graph_dict(graph_of(row("J1", A, A)), root_uri=A)
        run = nodes_by_type(result, "run")[0]
        assert run["metadata"]["self_loop"] is True
        assert run["id"].startswith("runs:")

    def test_it_is_named_after_the_first_job(self):
        """Same as a repeated A -> B: the jobs share a step, so its name is theirs."""
        rows = [
            row("J1", A, A, metadata={"job_name": "append"}),
            row("J2", A, A, metadata={"job_name": "other"}),
        ]
        run = nodes_by_type(build_graph_dict(graph_of(*rows), root_uri=A), "run")[0]
        assert run["name"] == "append"
        assert run["metadata"]["run_count"] == 2

    def test_it_says_how_to_list_its_jobs(self):
        result = build_graph_dict(graph_of(row("J1", A, A)), root_uri=A)
        metadata = nodes_by_type(result, "run")[0]["metadata"]
        assert metadata["source_uri"] == metadata["target_uri"] == A
        assert metadata["jobs_query"] == {"uri": A, "output": A}

    def test_it_is_drawn_as_a_loop(self):
        result = build_graph_dict(graph_of(row("J1", A, A)), root_uri=A)
        run_id = nodes_by_type(result, "run")[0]["id"]
        assert {"source": A, "target": run_id} in result["edges"]
        assert {"source": run_id, "target": A} in result["edges"]

    def test_it_names_a_representative_job(self):
        """The count needs somewhere to lead; this is the caller's starting point."""
        result = build_graph_dict(graph_of(row("J7", A, A)), root_uri=A)
        run = nodes_by_type(result, "run")[0]
        assert run["metadata"]["representative_job_id"] == "J7"

    def test_the_collapsed_id_cannot_collide_with_a_real_run(self):
        result = build_graph_dict(
            graph_of(row("J1", A, A), row("J2", A, B)), root_uri=A
        )
        ids = {n["id"] for n in nodes_by_type(result, "run")}
        assert ids == {f"runs:{A}", "run:J2"}

    def test_the_rewritten_artifact_still_appears(self):
        result = build_graph_dict(graph_of(row("J1", A, A)), root_uri=A)
        assert A in {n["id"] for n in nodes_by_type(result, "artifact")}

    def test_the_rewrite_reads_as_a_cycle_on_the_artifact(self):
        """Both edges, so it is not a dangling node; deduped however many rows."""
        rows = [row(f"J{i}", A, A) for i in range(100)]
        result = build_graph_dict(graph_of(*rows), root_uri=A)
        assert len(result["edges"]) == 2
        assert {(e["source"], e["target"]) for e in result["edges"]} == {
            (A, f"runs:{A}"),
            (f"runs:{A}", A),
        }

    def test_two_rewritten_artifacts_get_a_node_each(self):
        result = build_graph_dict(
            graph_of(row("J1", A, A), row("J2", B, B)), root_uri=A
        )
        assert len(nodes_by_type(result, "run")) == 2

    def test_real_runs_are_untouched(self):
        """A graph with no self-loops must render exactly as before."""
        rows = [row("J1", A, B), row("J2", B, C)]
        result = build_graph_dict(graph_of(*rows), root_uri=A)
        runs = nodes_by_type(result, "run")
        assert {r["id"] for r in runs} == {"run:J1", "run:J2"}
        assert all("collapsed" not in r["metadata"] for r in runs)

    def test_the_namespace_survives_for_authorization(self):
        """POST /artifact filters runs per space, so a collapsed node needs it.

        Without it the collapsed node fails closed and vanishes from that route.
        """
        result = build_graph_dict(
            graph_of(row("J1", A, A, metadata={"job_namespace": "sp/build"})),
            root_uri=A,
        )
        assert (
            nodes_by_type(result, "run")[0]["metadata"]["job_namespace"] == "sp/build"
        )


class TestSameEndpointGrouping:
    """Jobs with the same single input and output share one node, like self-loops."""

    def test_parallel_jobs_become_one_node(self):
        rows = [
            row(f"J{i}", A, B, metadata={"job_name": f"fdedup{i}"}) for i in range(12)
        ]
        result = build_graph_dict(graph_of(*rows), root_uri=A)
        runs = nodes_by_type(result, "run")
        assert len(runs) == 1
        assert runs[0]["name"] == "fdedup0"
        metadata = runs[0]["metadata"]
        assert metadata["run_count"] == 12
        assert metadata["self_loop"] is False
        assert metadata["jobs_query"] == {"uri": A, "output": B}
        assert runs[0]["id"] == f"runs:{A} \u2192 {B}"
        assert {(e["source"], e["target"]) for e in result["edges"]} == {
            (A, runs[0]["id"]),
            (runs[0]["id"], B),
        }

    def test_a_single_job_is_untouched(self):
        result = build_graph_dict(graph_of(row("J1", A, B)), root_uri=A)
        assert [n["id"] for n in nodes_by_type(result, "run")] == ["run:J1"]

    def test_jobs_with_more_endpoints_are_not_grouped(self):
        """The pair filter would over-list them, so they stay separate."""
        rows = [row("J1", A, B), row("J1", C, B), row("J2", A, B), row("J2", C, B)]
        result = build_graph_dict(graph_of(*rows), root_uri=A)
        assert {n["id"] for n in nodes_by_type(result, "run")} == {"run:J1", "run:J2"}

    def test_ungrouped_returns_one_node_per_job(self):
        rows = [row("J1", A, B), row("J2", A, B), row("J3", A, A), row("J4", A, A)]
        result = build_graph_dict(graph_of(*rows), root_uri=A, group_runs=False)
        runs = nodes_by_type(result, "run")
        assert {n["id"] for n in runs} == {"run:J1", "run:J2", "run:J3", "run:J4"}
        assert all("run_count" not in n["metadata"] for n in runs)
        assert {"source": "run:J3", "target": A} in result["edges"]

    def test_different_pairs_get_a_node_each(self):
        rows = [row("J1", A, B), row("J2", A, B), row("J3", A, C), row("J4", A, C)]
        result = build_graph_dict(graph_of(*rows), root_uri=A)
        assert len(nodes_by_type(result, "run")) == 2

    def test_jobs_with_no_input_become_one_node(self):
        rows = [row(f"J{i}", TERMINAL, B) for i in range(5)]
        result = build_graph_dict(graph_of(*rows), root_uri=B)
        runs = nodes_by_type(result, "run")
        assert len(runs) == 1
        assert runs[0]["metadata"]["run_count"] == 5
        assert runs[0]["metadata"]["jobs_query"] == {"uri": B, "terminal": "input"}
        assert edge_pairs(result) == {(runs[0]["id"], B)}
        assert {n["id"] for n in nodes_by_type(result, "artifact")} == {B}

    def test_jobs_with_no_output_become_one_node(self):
        rows = [row(f"J{i}", A, TERMINAL) for i in range(3)]
        result = build_graph_dict(graph_of(*rows), root_uri=A)
        runs = nodes_by_type(result, "run")
        assert len(runs) == 1
        assert runs[0]["metadata"]["jobs_query"] == {"uri": A, "terminal": "output"}
        assert edge_pairs(result) == {(A, runs[0]["id"])}


class TestSlimRows:
    """Rows in the slim shape (every writer now) draw the same graph as old rows."""

    def _slim(self):
        return row(
            "J1",
            A,
            B,
            attributes=build_index_attributes(
                job_id="J1",
                job_name="train",
                job_namespace="space-a/proj",
                owner="a@x.com",
                job_status="COMPLETED",
                job_started_at="t0",
                job_completed_at="t1",
                input_uri=A,
                input_artifact={"artifact_type": "dataset"},
                output_uri=B,
                output_artifact={"artifact_type": "model"},
                retrieve={"build_id": "BLD", "target_run_uuid": "TR"},
            ),
        )

    def test_artifact_type_is_read_from_type(self):
        result = build_graph_dict(graph_of(self._slim()), A)
        kinds = {n["id"]: n["artifact_type"] for n in nodes_by_type(result, "artifact")}
        assert kinds == {A: "dataset", B: "model"}

    def test_run_node_carries_auth_fields_status_and_times(self):
        result = build_graph_dict(graph_of(self._slim()), A)
        (run,) = nodes_by_type(result, "run")
        metadata = run["metadata"]
        assert metadata["job_namespace"] == "space-a/proj"
        assert metadata["owner"] == "a@x.com"
        assert metadata["job_status"] == "COMPLETED"
        assert metadata["job_started_at"] == "t0"
        assert metadata["job_completed_at"] == "t1"
        assert metadata["gb_build_id"] == "BLD"
        assert metadata["gb_target_run_uuid"] == "TR"
        assert "source_system" not in metadata

    def test_old_rows_still_read(self):
        old = row("J1", A, B, source_kind="dataset", build_id="BLD")
        result = build_graph_dict(graph_of(old), A)
        (run,) = nodes_by_type(result, "run")
        assert run["metadata"]["gb_build_id"] == "BLD"
        kinds = {n["id"]: n["artifact_type"] for n in nodes_by_type(result, "artifact")}
        assert kinds[A] == "dataset"
