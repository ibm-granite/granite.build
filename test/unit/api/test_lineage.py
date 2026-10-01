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

"""Unit tests for per-run access filtering in the OpenLineage-backed routes.

POST /lineage/search and POST /lineage/artifact both query an external
lineage backend that is not itself space-scoped, so each returned run is
filtered by has_space_member_access using the owner/space_name recovered
from that run's own facets (see gbserver.api.lineage for exactly how those
are recovered from each backend's response shape). These tests stub the
backend service only — the real has_space_member_access / space_access_check
/ is_super_admin path runs unmocked except for the space-role lookup itself,
so a change to the filtering logic or the facet shape it depends on would
fail here.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from gbserver.api import lineage as lineage_mod
from gbserver.lineage.openlineage_models import (
    ArtifactGraphRequest,
    LineageQueryRequest,
    TagSearchRequest,
)

MY_SPACE = "my-space"
OTHER_SPACE = "other-space"


def _fake_request(login: str, email: str) -> SimpleNamespace:
    return SimpleNamespace(
        state=SimpleNamespace(data={"user": SimpleNamespace(login=login, email=email)})
    )


def _member_of(space_name: str):
    """Patch the underlying space-role lookup: caller is a plain member of
    `space_name` only, not an admin anywhere, not the runs' owner."""
    return (
        patch("gbserver.api.utils.is_super_admin", return_value=False),
        patch(
            "gbserver.api.utils.space_access_check",
            side_effect=lambda username, space: space == space_name,
        ),
    )


def _search_run(space_name: str, owner: str = "someone_else@example.com"):
    return {
        "run": {
            "runId": f"run-{space_name}",
            "facets": {
                "job_details": {"owner": owner},
                "tags": {"space_name": space_name},
                "job_input_params": {"SECRET": "should-not-leak"},
            },
        }
    }


def _graph_node(node_id: str, space_name: str, owner: str, name: str = "a-run"):
    return {
        "id": node_id,
        "node_type": "run",
        "name": name,
        "metadata": {"job_namespace": f"{space_name}/some-build", "owner": owner},
    }


# ------------------------------------------------ shared redaction accessor


def test_get_redacted_job_input_params():
    """The shared accessor masks secret values, keeps non-secret data, empties absent."""
    assert lineage_mod.get_redacted_job_input_params({}) == {}
    assert lineage_mod.get_redacted_job_input_params({"job_input_params": None}) == {}
    out = lineage_mod.get_redacted_job_input_params(
        {"job_input_params": {"SECRET": "leak", "commit_hash": "abc123"}}
    )
    assert out == {"SECRET": "<redacted>", "commit_hash": "abc123"}


# --------------------------------------------------------------------- search


def test_search_lineage_events_excludes_cross_space_run():
    my_run = _search_run(MY_SPACE)
    other_run = _search_run(OTHER_SPACE)
    fake_service = SimpleNamespace(
        search_lineage_by_tags=lambda tags, limit, offset: (2, [my_run, other_run])
    )
    is_admin, is_member = _member_of(MY_SPACE)
    with (
        is_admin,
        is_member,
        patch.object(
            lineage_mod, "_get_openlineage_service", return_value=fake_service
        ),
    ):
        resp = lineage_mod.search_lineage_events(
            _fake_request("member", "member@example.com"), TagSearchRequest(tags=[])
        )
    run_ids = [r["run"]["runId"] for r in resp.runs]
    assert run_ids == [f"run-{MY_SPACE}"], run_ids
    assert resp.total == 1
    assert resp.count == 1


def test_search_lineage_events_includes_owned_run_from_any_space():
    # owner is compared against the caller's login (not email) — see
    # has_space_member_access's owner shortcut.
    owned_run = _search_run(OTHER_SPACE, owner="member")
    fake_service = SimpleNamespace(
        search_lineage_by_tags=lambda tags, limit, offset: (1, [owned_run])
    )
    is_admin, is_member = _member_of(MY_SPACE)  # not a member of OTHER_SPACE
    with (
        is_admin,
        is_member,
        patch.object(
            lineage_mod, "_get_openlineage_service", return_value=fake_service
        ),
    ):
        resp = lineage_mod.search_lineage_events(
            _fake_request("member", "member@example.com"), TagSearchRequest(tags=[])
        )
    assert len(resp.runs) == 1, "owner should see their own run regardless of space"


def test_search_lineage_events_redacts_job_input_params():
    my_run = _search_run(MY_SPACE)
    fake_service = SimpleNamespace(
        search_lineage_by_tags=lambda tags, limit, offset: (1, [my_run])
    )
    is_admin, is_member = _member_of(MY_SPACE)
    with (
        is_admin,
        is_member,
        patch.object(
            lineage_mod, "_get_openlineage_service", return_value=fake_service
        ),
    ):
        resp = lineage_mod.search_lineage_events(
            _fake_request("member", "member@example.com"), TagSearchRequest(tags=[])
        )
    # Redact-not-omit: the facet is retained so non-secret step data (e.g. a
    # commit_hash) still surfaces, but any secret-named key has its VALUE masked.
    # The key "SECRET" itself is preserved by design; only "should-not-leak" must go.
    assert "should-not-leak" not in str(resp.runs)
    params = resp.runs[0]["run"]["facets"]["job_input_params"]
    assert params == {"SECRET": "<redacted>"}


def test_search_lineage_events_redacts_step_config_and_metadata():
    """Search redacts secret-named keys in both step config and metadata (recursively).

    The whole job_input_params facet is redacted unconditionally on the read path:
    non-secret values (uri, repo, commit_hash) surface intact, while secret-named keys
    anywhere in the nested config/metadata have their values masked.
    """
    my_run = _search_run(MY_SPACE)
    my_run["run"]["facets"]["job_input_params"] = {
        "steps": [
            {
                "uri": "space://steps/byoc",
                "config": {
                    "byoc_config": {"repo": "https://example/r.git", "token": "sekret"}
                },
                "metadata": {"commit_hash": "deadbeef", "api_key": "supersecret"},
            }
        ]
    }
    fake_service = SimpleNamespace(
        search_lineage_by_tags=lambda tags, limit, offset: (1, [my_run])
    )
    is_admin, is_member = _member_of(MY_SPACE)
    with (
        is_admin,
        is_member,
        patch.object(
            lineage_mod, "_get_openlineage_service", return_value=fake_service
        ),
    ):
        resp = lineage_mod.search_lineage_events(
            _fake_request("member", "member@example.com"), TagSearchRequest(tags=[])
        )
    step = resp.runs[0]["run"]["facets"]["job_input_params"]["steps"][0]
    assert step["uri"] == "space://steps/byoc"
    # config surfaces (redacted), consistent with the by-id jobstats endpoints.
    assert step["config"]["byoc_config"]["repo"] == "https://example/r.git"
    assert step["config"]["byoc_config"]["token"] == "<redacted>"
    # metadata surfaces; non-secret value kept, secret-named key masked.
    assert step["metadata"]["commit_hash"] == "deadbeef"
    assert step["metadata"]["api_key"] == "<redacted>"


# ---------------------------------------------------------------- artifact graph


def _fake_graph_result(nodes, edges=None):
    return {
        "root_id": nodes[0]["id"] if nodes else "",
        "truncated": False,
        "nodes": nodes,
        "edges": edges or [],
    }


def test_get_artifact_graph_excludes_cross_space_run():
    nodes = [
        _graph_node("run-mine", MY_SPACE, "someone_else@example.com"),
        _graph_node("run-other", OTHER_SPACE, "someone_else@example.com"),
    ]
    fake_service = SimpleNamespace(
        get_artifact_graph=lambda **kw: _fake_graph_result(nodes)
    )
    is_admin, is_member = _member_of(MY_SPACE)
    with (
        is_admin,
        is_member,
        patch.object(
            lineage_mod, "_get_openlineage_service", return_value=fake_service
        ),
    ):
        resp = lineage_mod.get_artifact_graph(
            _fake_request("member", "member@example.com"),
            ArtifactGraphRequest(artifact_name="dataset-x", direction="both"),
        )
    namespaces = [r.job_namespace for r in resp.runs]
    assert namespaces == [f"{MY_SPACE}/some-build"], namespaces


def test_get_artifact_graph_includes_owned_run_from_any_space():
    # owner is compared against the caller's login (not email) — see
    # has_space_member_access's owner shortcut.
    nodes = [_graph_node("run-owned", OTHER_SPACE, "member")]
    fake_service = SimpleNamespace(
        get_artifact_graph=lambda **kw: _fake_graph_result(nodes)
    )
    is_admin, is_member = _member_of(MY_SPACE)  # not a member of OTHER_SPACE
    with (
        is_admin,
        is_member,
        patch.object(
            lineage_mod, "_get_openlineage_service", return_value=fake_service
        ),
    ):
        resp = lineage_mod.get_artifact_graph(
            _fake_request("member", "member@example.com"),
            ArtifactGraphRequest(artifact_name="dataset-x", direction="both"),
        )
    assert len(resp.runs) == 1, "owner should see their own run regardless of space"


def test_get_artifact_graph_redacts_job_input_params():
    """The artifact-graph read path masks job_input_params like search does.

    Both endpoints are member-readable, so redaction must be applied on each or a
    secret the write-side missed leaks on one path but not the other. Asserts the
    returned ArtifactRunEntry carries the masked value, not the raw secret.
    """
    node = _graph_node("run-mine", MY_SPACE, "someone_else@example.com")
    node["metadata"]["job_input_params"] = {"SECRET": "should-not-leak"}
    fake_service = SimpleNamespace(
        get_artifact_graph=lambda **kw: _fake_graph_result([node])
    )
    is_admin, is_member = _member_of(MY_SPACE)
    with (
        is_admin,
        is_member,
        patch.object(
            lineage_mod, "_get_openlineage_service", return_value=fake_service
        ),
    ):
        resp = lineage_mod.get_artifact_graph(
            _fake_request("member", "member@example.com"),
            ArtifactGraphRequest(artifact_name="dataset-x", direction="both"),
        )
    assert "should-not-leak" not in str(resp.runs)
    assert resp.runs[0].job_input_params == {"SECRET": "<redacted>"}


def test_get_artifact_graph_excludes_run_with_no_owner_or_namespace():
    """Fail closed: a run missing both signals must never be returned."""
    nodes = [
        {
            "id": "run-unknown",
            "node_type": "run",
            "name": "mystery",
            "metadata": {},
        }
    ]
    fake_service = SimpleNamespace(
        get_artifact_graph=lambda **kw: _fake_graph_result(nodes)
    )
    is_admin, is_member = _member_of(MY_SPACE)
    with (
        is_admin,
        is_member,
        patch.object(
            lineage_mod, "_get_openlineage_service", return_value=fake_service
        ),
    ):
        resp = lineage_mod.get_artifact_graph(
            _fake_request("member", "member@example.com"),
            ArtifactGraphRequest(artifact_name="dataset-x", direction="both"),
        )
    assert resp.runs == []


def _db_service(**kwargs):
    """A stub that passes the endpoint's isinstance(DBLineageService) check."""
    from gbserver.lineage.db_service import DBLineageService

    service = DBLineageService.__new__(DBLineageService)
    for name, value in kwargs.items():
        setattr(service, name, value)
    return service


# --------------------------------------------------------------- POST/GET /graph


def test_query_graph_rejects_an_unknown_direction():
    with pytest.raises(HTTPException) as caught:
        lineage_mod.query_lineage_graph(
            _fake_request("member", "member@example.com"),
            LineageQueryRequest(uri="s3://b/x", direction="sideways"),
        )
    assert caught.value.status_code == 400


def test_index_routes_read_the_index_whatever_the_provider():
    """The index is filled by the lineage-indexer, so a wandb/none provider still has data."""
    service = _db_service(
        query_graph=lambda **_kw: {"root_id": "", "nodes": [], "edges": []},
        list_jobs=lambda **_kw: {
            "jobs": [],
            "total": 0,
            "limit": 100,
            "offset": 0,
        },
    )
    req = _fake_request("member", "member@example.com")
    with (
        patch.object(lineage_mod, "_get_index_service", return_value=service),
        patch.object(
            lineage_mod, "_get_openlineage_service", side_effect=AssertionError
        ),
    ):
        lineage_mod.query_lineage_graph(req, LineageQueryRequest(uri="s3://b/x"))
        lineage_mod.list_lineage_jobs(
            req, uri="s3://b/x", tags=["build_id=B"], required_tags=[]
        )


def test_query_graph_requires_a_uri_or_a_job_id():
    """A graph needs somewhere to start; "what ran lately" is GET /jobs."""
    service = _db_service(query_graph=lambda **_kw: pytest.fail("must not walk"))
    with patch.object(lineage_mod, "_get_index_service", return_value=service):
        with pytest.raises(HTTPException) as caught:
            lineage_mod.query_lineage_graph_get(
                _fake_request("member", "member@example.com")
            )
    assert caught.value.status_code == 400


def test_query_graph_fills_in_its_defaults():
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return {"root_id": "", "nodes": [], "edges": [], "truncated": False}

    service = _db_service(query_graph=fake)
    with patch.object(lineage_mod, "_get_index_service", return_value=service):
        resp = lineage_mod.query_lineage_graph(
            _fake_request("member", "member@example.com"),
            LineageQueryRequest(job_id="J1"),
        )
    assert seen == {
        "uri": None,
        "job_id": "J1",
        "direction": "both",
        "max_depth": 10,
        # None, not a number: the per-level ceiling is the service's default unless a
        # caller explicitly raises it for a "show the full graph" request.
        "max_nodes_per_level": None,
        "group_runs": True,
    }
    assert resp.root_id == ""
    assert resp.nodes == []


def test_query_graph_passes_every_filter_through():
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return {"root_id": "s3://b/x", "nodes": [], "edges": [], "truncated": False}

    service = _db_service(query_graph=fake)
    with patch.object(lineage_mod, "_get_index_service", return_value=service):
        lineage_mod.query_lineage_graph(
            _fake_request("member", "member@example.com"),
            LineageQueryRequest(
                uri="s3://b/x",
                job_id="J1",
                direction="upstream",
                max_depth=3,
                max_nodes_per_level=25_000,
            ),
        )
    assert seen == {
        "uri": "s3://b/x",
        "job_id": "J1",
        "direction": "upstream",
        "max_depth": 3,
        "max_nodes_per_level": 25_000,
        "group_runs": True,
    }


def test_query_graph_does_not_404_on_an_empty_graph():
    """ "Nothing recorded" is a real answer and must not read as an error.

    This is the difference from ``POST /artifact``, whose ``None`` becomes the 404 the
    frontend renders as "lineage is not available".
    """
    service = _db_service(
        query_graph=lambda **_kw: {
            "root_id": "",
            "nodes": [],
            "edges": [],
            "truncated": False,
        }
    )
    with patch.object(lineage_mod, "_get_index_service", return_value=service):
        resp = lineage_mod.query_lineage_graph(
            _fake_request("member", "member@example.com"),
            LineageQueryRequest(uri="s3://b/absent"),
        )
    assert resp.nodes == []


def test_query_graph_carries_node_depth_to_the_response():
    """``depth`` is the field a client needs to lay the graph out."""
    service = _db_service(
        query_graph=lambda **_kw: {
            "root_id": "s3://b/x",
            "nodes": [
                {
                    "id": "s3://b/x",
                    "node_type": "artifact",
                    "name": "x",
                    "is_root": True,
                    "depth": 0,
                    "metadata": {"uri": "s3://b/x"},
                },
                {
                    "id": "s3://b/y",
                    "node_type": "artifact",
                    "name": "y",
                    "depth": 2,
                    "metadata": {"uri": "s3://b/y"},
                },
                _graph_node("run:J1", MY_SPACE, "someone_else@example.com"),
            ],
            "edges": [
                {"source": "s3://b/x", "target": "run:J1"},
                {"source": "run:J1", "target": "s3://b/y"},
            ],
            "truncated": False,
        }
    )
    is_admin, is_member = _member_of(MY_SPACE)
    with (
        is_admin,
        is_member,
        patch.object(lineage_mod, "_get_index_service", return_value=service),
    ):
        resp = lineage_mod.query_lineage_graph(
            _fake_request("member", "member@example.com"),
            LineageQueryRequest(uri="s3://b/x"),
        )
    depths = {n.id: n.depth for n in resp.nodes if n.node_type == "artifact"}
    assert depths == {"s3://b/x": 0, "s3://b/y": 2}


def test_query_graph_get_form_maps_its_query_params():
    """The GET form exists so a lineage view can be bookmarked and shared."""
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return {"root_id": "", "nodes": [], "edges": [], "truncated": False}

    service = _db_service(query_graph=fake)
    with patch.object(lineage_mod, "_get_index_service", return_value=service):
        lineage_mod.query_lineage_graph_get(
            _fake_request("member", "member@example.com"),
            uri="s3://b/x",
            job_id="J1",
            direction="downstream",
            depth=4,
        )
    # note: the wire calls it `depth`, the service `max_depth`
    assert seen["max_depth"] == 4
    assert seen["direction"] == "downstream"
    assert seen["uri"] == "s3://b/x"
    assert seen["job_id"] == "J1"


# --------------------------------------- the graph is cross-space, by decision


def _two_space_graph() -> dict:
    """A graph whose two runs live in different spaces.

    ``mine`` produced ``a_mine``; ``theirs`` produced ``a_theirs`` and also consumed
    ``a_shared``, which both runs touch.
    """
    return {
        "root_id": "a_shared",
        "nodes": [
            {"id": "a_shared", "node_type": "artifact", "name": "shared", "depth": 0},
            {"id": "a_mine", "node_type": "artifact", "name": "mine", "depth": 1},
            {"id": "a_theirs", "node_type": "artifact", "name": "theirs", "depth": 1},
            _graph_node("run:mine", MY_SPACE, "someone_else@example.com"),
            _graph_node("run:theirs", OTHER_SPACE, "someone_else@example.com"),
        ],
        "edges": [
            {"source": "a_shared", "target": "run:mine"},
            {"source": "run:mine", "target": "a_mine"},
            {"source": "a_shared", "target": "run:theirs"},
            {"source": "run:theirs", "target": "a_theirs"},
        ],
        "truncated": False,
    }


def _query_graph_as_member_of(space: str, graph: dict):
    service = _db_service(query_graph=lambda **_kw: graph)
    is_admin, is_member = _member_of(space)
    with (
        is_admin,
        is_member,
        patch.object(lineage_mod, "_get_index_service", return_value=service),
    ):
        return lineage_mod.query_lineage_graph(
            _fake_request("member", "member@example.com"),
            LineageQueryRequest(uri="a_shared"),
        )


def test_graph_is_not_filtered_per_space():
    """The graph crosses spaces, and that is the decision -- not an oversight.

    A lineage graph carries no access to any artifact: URIs, job names and edges
    only. Filtering it would make "what was my model trained on?" silently
    unanswerable whenever a chain crosses a space, which is the normal case for a
    shared dataset or a platform base model.

    This test exists so the behaviour cannot be changed by accident. Reversing it is a
    product decision about whether artifact URIs are themselves secret -- see the
    docstring on ``query_lineage_graph``.
    """
    resp = _query_graph_as_member_of(MY_SPACE, _two_space_graph())
    ids = {n.id for n in resp.nodes}
    assert {"a_shared", "a_mine", "a_theirs"} <= ids
    assert {"run:mine", "run:theirs"} <= ids


def test_graph_keeps_every_edge_regardless_of_space():
    resp = _query_graph_as_member_of(MY_SPACE, _two_space_graph())
    assert len(resp.edges) == 4


def test_graph_provenance_is_complete_across_a_space_boundary():
    """The use case the no-filtering decision protects.

    A model in my space, trained on a dataset curated by another team: the upstream
    chain must answer truthfully, or the index does not do its job.
    """
    graph = {
        "root_id": "s3://mine/finetune",
        "nodes": [
            {
                "id": "s3://mine/finetune",
                "node_type": "artifact",
                "name": "finetune",
                "is_root": True,
                "depth": 0,
            },
            {
                "id": "s3://curated/corpus",
                "node_type": "artifact",
                "name": "corpus",
                "depth": 1,
            },
            _graph_node("run:platform", "platform-team", "other@example.com"),
        ],
        "edges": [
            {"source": "s3://curated/corpus", "target": "run:platform"},
            {"source": "run:platform", "target": "s3://mine/finetune"},
        ],
        "truncated": False,
    }
    resp = _query_graph_as_member_of(MY_SPACE, graph)
    depths = {n.id: n.depth for n in resp.nodes if n.node_type == "artifact"}
    assert depths == {"s3://mine/finetune": 0, "s3://curated/corpus": 1}


def test_graph_keeps_a_run_with_no_provenance():
    """No space check means an unattributed run is not dropped either.

    ``POST /artifact`` and ``POST /search`` fail closed on a run with neither
    namespace nor owner because they authorize per run. This route does not authorize
    per node at all, so there is nothing to fail closed about -- worth pinning, since
    the two routes now differ.
    """
    graph = {
        "root_id": "a1",
        "nodes": [
            {"id": "a1", "node_type": "artifact", "name": "a"},
            {"id": "run:anon", "node_type": "run", "name": "anon", "metadata": {}},
        ],
        "edges": [{"source": "a1", "target": "run:anon"}],
        "truncated": False,
    }
    resp = _query_graph_as_member_of(MY_SPACE, graph)
    assert {n.id for n in resp.nodes} == {"a1", "run:anon"}


def test_artifact_graph_still_filters_per_space():
    """The contrast: POST /artifact DOES filter, and must keep doing so.

    It re-projects into run-centred entries and has always applied the per-run space
    check. Only the node/edge routes are cross-space, so a change to one must not be
    assumed to apply to the other.
    """
    my_run = _graph_node("run:mine", MY_SPACE, "someone_else@example.com")
    other_run = _graph_node("run:theirs", OTHER_SPACE, "someone_else@example.com")
    fake_service = SimpleNamespace(
        get_artifact_graph=lambda **_kw: {
            "root_id": "a1",
            "nodes": [
                {"id": "a1", "node_type": "artifact", "name": "a"},
                my_run,
                other_run,
            ],
            "edges": [
                {"source": "a1", "target": "run:mine"},
                {"source": "a1", "target": "run:theirs"},
            ],
            "truncated": False,
        }
    )
    is_admin, is_member = _member_of(MY_SPACE)
    with (
        is_admin,
        is_member,
        patch.object(
            lineage_mod, "_get_openlineage_service", return_value=fake_service
        ),
    ):
        resp = lineage_mod.get_artifact_graph(
            _fake_request("member", "member@example.com"),
            ArtifactGraphRequest(artifact_url="a1"),
        )
    assert [r.job_namespace.split("/", 1)[0] for r in resp.runs] == [MY_SPACE]


# ----------------------------------------------------------- GET /lineage/jobs


def test_jobs_accepts_a_request_with_no_filters():
    """An unfiltered listing is legitimate: it means "the most recent jobs"."""
    service = _db_service(
        list_jobs=lambda **_kw: {"jobs": [], "total": 0, "limit": 100, "offset": 0}
    )
    with patch.object(lineage_mod, "_get_index_service", return_value=service):
        resp = lineage_mod.list_lineage_jobs(
            _fake_request("member", "member@example.com"), tags=[], required_tags=[]
        )
    assert resp.total == 0


def test_jobs_passes_every_filter_and_its_paging_through():
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return {
            "jobs": [{"job_id": "J1", "tags": ["build_id=B", "team=nlp"]}],
            "total": 1,
            "limit": 10,
            "offset": 50,
        }

    service = _db_service(list_jobs=fake)
    with patch.object(lineage_mod, "_get_index_service", return_value=service):
        resp = lineage_mod.list_lineage_jobs(
            _fake_request("member", "member@example.com"),
            uri="s3://b/x",
            job_id="J1",
            tags=["build_id=B"],
            required_tags=["team=nlp"],
            limit=10,
            offset=50,
        )
    assert seen == {
        "uri": "s3://b/x",
        "job_id": "J1",
        "tags": ["build_id=B"],
        "required_tags": ["team=nlp"],
        "limit": 10,
        "offset": 50,
        "self_loop": False,
        "output": None,
        "terminal": None,
    }
    assert resp.limit == 10
    assert resp.offset == 50
    assert resp.jobs[0].tags == ["build_id=B", "team=nlp"]


def test_jobs_reports_the_total_so_a_caller_can_page():
    """The count is what makes a collapsed graph node expandable."""
    service = _db_service(
        list_jobs=lambda **_kw: {
            "jobs": [
                {
                    "job_id": "J1",
                    "inputs": ["s3://b/x"],
                    "outputs": ["s3://b/x"],
                    "job": {"name": "append"},
                    "source_system": "lakehouse",
                }
            ],
            "total": 68906,
            "limit": 1,
            "offset": 0,
        }
    )
    with patch.object(lineage_mod, "_get_index_service", return_value=service):
        resp = lineage_mod.list_lineage_jobs(
            _fake_request("member", "member@example.com"),
            uri="s3://b/x",
            tags=[],
            required_tags=[],
        )
    assert resp.total == 68906
    assert resp.jobs[0].inputs == resp.jobs[0].outputs == ["s3://b/x"]
    assert resp.jobs[0].job["name"] == "append"
