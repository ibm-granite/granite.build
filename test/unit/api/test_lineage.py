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

POST /lineage/search queries an external lineage backend that is not
itself space-scoped, so each returned run is
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


def test_search_lineage_events_filters_a_real_db_provider_result():
    """The db provider's own envelope must satisfy the shared route contract.

    Not a hand-written run dict like the tests above: this goes through
    DBLineageService, so it fails if db_responses ever stops emitting the two
    facets the route filters on. The route is provider-agnostic and fails closed,
    so a missing one silently empties every search instead of erroring.
    """
    import uuid as uuid_module

    from gbserver.lineage.db_service import DBLineageService
    from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory
    from gbserver.storage.stored_lineage_job import StoredLineageJob

    jobs = SqliteStorageFactory().create_lineage_job_storage(
        table_name=f"t_api_{uuid_module.uuid4().hex[:8]}"
    )
    for space in (MY_SPACE, OTHER_SPACE):
        jobs.add(
            StoredLineageJob(
                job_id=f"run-{space}",
                job_namespace=f"{space}/some-build",
                space_name=space,
                owner="someone_else@example.com",
                tags=["team=nlp"],
            )
        )
    service = DBLineageService(job_storage=jobs)

    is_admin, is_member = _member_of(MY_SPACE)
    with (
        is_admin,
        is_member,
        patch.object(lineage_mod, "_get_openlineage_service", return_value=service),
    ):
        resp = lineage_mod.search_lineage_events(
            _fake_request("member", "member@example.com"),
            TagSearchRequest(tags=["team=nlp"]),
        )
    assert [r["run"]["runId"] for r in resp.runs] == [f"run-{MY_SPACE}"]
    assert resp.total == 1


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


def _db_service(**kwargs):
    """A stub that passes the endpoint's isinstance(DBLineageService) check."""
    from gbserver.lineage.db_service import DBLineageService

    service = DBLineageService.__new__(DBLineageService)
    for name, value in kwargs.items():
        setattr(service, name, value)
    return service


# --------------------------------------------------------------- POST/GET /graph


def test_query_graph_rejects_an_unknown_direction():
    """A bad direction is a 422, as the route documents, like any invalid param."""
    with pytest.raises(HTTPException) as caught:
        lineage_mod.query_lineage_graph_get(
            _fake_request("member", "member@example.com"),
            uri="s3://b/x",
            direction="sideways",
        )
    assert caught.value.status_code == 422


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
        lineage_mod.list_lineage_jobs(req, uri="s3://b/x")


def test_query_graph_requires_a_uri_or_a_job_id():
    """A graph needs somewhere to start: 422, as GET /jobs answers."""
    service = _db_service(query_graph=lambda **_kw: pytest.fail("must not walk"))
    with patch.object(lineage_mod, "_get_index_service", return_value=service):
        with pytest.raises(HTTPException) as caught:
            lineage_mod.query_lineage_graph_get(
                _fake_request("member", "member@example.com")
            )
    assert caught.value.status_code == 422


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
    authorize_job = seen.pop("authorize_job")
    assert callable(authorize_job)
    assert seen == {
        "uri": None,
        "job_id": "J1",
        "job_ids": None,
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
    with (
        patch.object(lineage_mod, "_get_index_service", return_value=service),
        patch.object(lineage_mod, "is_super_admin", return_value=True),
    ):
        lineage_mod.query_lineage_graph(
            _fake_request("admin", "admin@example.com"),
            LineageQueryRequest(
                uri="s3://b/x",
                job_id="J1",
                job_ids=["J2", "J3"],
                direction="upstream",
                max_depth=30,
                max_nodes_per_level=25_000,
            ),
        )
    seen.pop("authorize_job")
    assert seen == {
        "uri": "s3://b/x",
        "job_id": "J1",
        "job_ids": ["J2", "J3"],
        "direction": "upstream",
        "max_depth": 30,
        "max_nodes_per_level": 25_000,
        "group_runs": True,
    }


@pytest.mark.parametrize(
    "kwargs",
    [{"max_depth": 21}, {"max_nodes_per_level": 5001}],
    ids=["depth", "nodes-per-level"],
)
def test_query_graph_holds_a_non_admin_to_the_user_limits(kwargs):
    """Only a super admin may ask for a deeper walk or a wider level."""
    service = _db_service(query_graph=lambda **_kw: pytest.fail("must not walk"))
    with (
        patch.object(lineage_mod, "_get_index_service", return_value=service),
        patch.object(lineage_mod, "is_super_admin", return_value=False),
    ):
        with pytest.raises(HTTPException) as caught:
            lineage_mod.query_lineage_graph(
                _fake_request("member", "member@example.com"),
                LineageQueryRequest(uri="s3://b/x", **kwargs),
            )
    assert caught.value.status_code == 422


def test_query_graph_accepts_job_ids_alone_as_a_seed():
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return {"root_id": "", "nodes": [], "edges": []}

    service = _db_service(query_graph=fake)
    with (
        patch.object(lineage_mod, "_get_index_service", return_value=service),
        patch.object(lineage_mod, "is_super_admin", return_value=False),
    ):
        lineage_mod.query_lineage_graph_get(
            _fake_request("member", "member@example.com"), job_ids=["J1", "J2"]
        )
    assert seen["job_ids"] == ["J1", "J2"]


def test_query_graph_gates_origin_ids_on_the_job_detail_check():
    """The graph's per-node check is the same function /jobs/{id} uses."""
    captured = {}

    def fake_graph(**kwargs):
        captured["graph"] = kwargs["authorize_job"]
        return {"root_id": "", "nodes": [], "edges": []}

    def fake_detail(job_id, authorize_build, authorize_entry):
        captured["detail"] = authorize_entry
        return None

    service = _db_service(query_graph=fake_graph, get_job_detail=fake_detail)
    req = _fake_request("member", "member@example.com")
    calls = []
    with (
        patch.object(lineage_mod, "_get_index_service", return_value=service),
        patch.object(lineage_mod, "is_super_admin", return_value=False),
        patch.object(
            lineage_mod,
            "_can_read_job",
            side_effect=lambda request, entry: calls.append(entry) or False,
        ),
    ):
        lineage_mod.query_lineage_graph(req, LineageQueryRequest(uri="s3://b/x"))
        with pytest.raises(HTTPException):
            lineage_mod.get_lineage_job_detail(req, "J1")
        entry = {"owner": "o", "space_name": "s"}
        assert captured["graph"](entry) is False
        assert captured["detail"](entry) is False
    assert calls == [entry, entry]


def test_query_graph_does_not_404_on_an_empty_graph():
    """ "Nothing recorded" is a real answer and must not read as an error."""
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


@pytest.mark.parametrize(
    "params", [{"depth": 100}, {"depth": 0}, {"max_nodes_per_level": 0}]
)
def test_query_graph_get_answers_422_on_out_of_range_params(params):
    """Bounds live on the request model; a violation is the client's error, not a 500."""
    with pytest.raises(HTTPException) as caught:
        lineage_mod.query_lineage_graph_get(
            _fake_request("member", "member@example.com"), uri="s3://b/x", **params
        )
    assert caught.value.status_code == 422


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

    ``POST /search`` fails closed on a run with neither namespace nor owner because
    it authorizes per run. This route does not authorize per node at all, so there is
    nothing to fail closed about -- worth pinning, since the two routes differ.
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


# ----------------------------------------------------------- GET /lineage/jobs


def test_jobs_requires_a_uri_or_a_job_id():
    """An unfiltered listing would aggregate the whole index: 422, not a scan."""
    service = _db_service(list_jobs=lambda **_kw: pytest.fail("must not list"))
    with patch.object(lineage_mod, "_get_index_service", return_value=service):
        with pytest.raises(HTTPException) as caught:
            lineage_mod.list_lineage_jobs(_fake_request("member", "member@example.com"))
    assert caught.value.status_code == 422


def test_jobs_lists_by_job_ids_alone():
    """A grouped graph node's jobs_query is a job_ids list; that is enough."""
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return {"jobs": [], "total": 0, "limit": 100, "offset": 0}

    service = _db_service(list_jobs=fake)
    with patch.object(lineage_mod, "_get_index_service", return_value=service):
        lineage_mod.list_lineage_jobs(
            _fake_request("member", "member@example.com"), job_ids=["J1", "J2"]
        )
    assert seen["job_ids"] == ["J1", "J2"]


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
            limit=10,
            offset=50,
        )
    assert seen == {
        "uri": "s3://b/x",
        "job_id": "J1",
        "limit": 10,
        "offset": 50,
        "self_loop": False,
        "output": None,
        "terminal": None,
        "job_ids": None,
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
        )
    assert resp.total == 68906
    assert resp.jobs[0].inputs == resp.jobs[0].outputs == ["s3://b/x"]
    assert resp.jobs[0].job["name"] == "append"


def test_search_lineage_events_reports_a_backend_failure():
    """dmg-ng reads this route: a DB failure must not look like "no matches"."""
    import pytest
    from fastapi import HTTPException

    def boom(tags, limit, offset):
        raise ConnectionError("db down")

    fake_service = SimpleNamespace(search_lineage_by_tags=boom)
    is_admin, is_member = _member_of(MY_SPACE)
    with (
        is_admin,
        is_member,
        patch.object(
            lineage_mod, "_get_openlineage_service", return_value=fake_service
        ),
        pytest.raises(HTTPException) as caught,
    ):
        lineage_mod.search_lineage_events(
            _fake_request("member", "member@example.com"), TagSearchRequest(tags=[])
        )
    assert caught.value.status_code == 503


def test_list_jobs_answers_503_when_the_index_cannot_be_read():
    """An empty page would read as "no jobs"; a storage failure says so instead."""

    def broken(**_kw):
        raise RuntimeError("database is locked")

    service = _db_service(list_jobs=broken)
    with patch.object(lineage_mod, "_get_index_service", return_value=service):
        with pytest.raises(HTTPException) as caught:
            lineage_mod.list_lineage_jobs(
                _fake_request("member", "member@example.com"), uri="s3://b/x"
            )
    assert caught.value.status_code == 503


def test_list_jobs_answers_422_for_an_invalid_page():
    def refuse(**_kw):
        raise ValueError("limit must be between 1 and 1000")

    service = _db_service(list_jobs=refuse)
    with patch.object(lineage_mod, "_get_index_service", return_value=service):
        with pytest.raises(HTTPException) as caught:
            lineage_mod.list_lineage_jobs(
                _fake_request("member", "member@example.com"), uri="s3://b/x"
            )
    assert caught.value.status_code == 422


def test_query_graph_answers_422_for_a_service_value_error():
    """Out-of-range input the service rejects is a 422, as the route documents."""

    def refuse(**_kw):
        raise ValueError("direction must be one of ...")

    service = _db_service(query_graph=refuse)
    with patch.object(lineage_mod, "_get_index_service", return_value=service):
        with pytest.raises(HTTPException) as caught:
            lineage_mod.query_lineage_graph(
                _fake_request("member", "member@example.com"),
                LineageQueryRequest(uri="s3://b/x"),
            )
    assert caught.value.status_code == 422


def test_query_graph_answers_503_when_the_index_cannot_be_read():
    """A failing job_id seed is a 503, never an empty graph that reads as "nothing"."""

    def broken(**_kw):
        raise RuntimeError("database is locked")

    service = _db_service(query_graph=broken)
    with patch.object(lineage_mod, "_get_index_service", return_value=service):
        with pytest.raises(HTTPException) as caught:
            lineage_mod.query_lineage_graph_get(
                _fake_request("member", "member@example.com"), job_id="J1"
            )
    assert caught.value.status_code == 503


def test_list_jobs_caps_job_ids_like_the_graph():
    service = _db_service(list_jobs=lambda **_kw: AssertionError("not reached"))
    with patch.object(lineage_mod, "_get_index_service", return_value=service):
        with pytest.raises(HTTPException) as caught:
            lineage_mod.list_lineage_jobs(
                _fake_request("member", "member@example.com"),
                job_ids=[f"J{i}" for i in range(1001)],
            )
    assert caught.value.status_code == 422


def test_job_detail_carries_detail_denied():
    """The response model must not drop detail_denied, or a denial reads as unavailable."""
    service = _db_service(
        get_job_detail=lambda job_id, **_kw: {
            "job_id": job_id,
            "detail_available": False,
            "detail_denied": True,
        }
    )
    with patch.object(lineage_mod, "_get_index_service", return_value=service):
        detail = lineage_mod.get_lineage_job_detail(
            _fake_request("member", "member@example.com"), "J1"
        )
    assert detail.detail_denied is True


def test_query_graph_post_form_walks_every_job_id_at_once():
    """A grouped node's ids go in one walk, so grouping spans the whole set."""
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return {"root_id": "", "nodes": [], "edges": []}

    ids = [f"J{i}" for i in range(300)]
    service = _db_service(query_graph=fake)
    with (
        patch.object(lineage_mod, "_get_index_service", return_value=service),
        patch.object(lineage_mod, "is_super_admin", return_value=False),
    ):
        lineage_mod.query_lineage_graph_post(
            _fake_request("member", "member@example.com"),
            LineageQueryRequest(job_ids=ids, direction="upstream"),
        )
    assert seen["job_ids"] == ids
    assert seen["direction"] == "upstream"


def test_query_graph_keeps_an_http_error_raised_inside_the_walk():
    """An HTTPException from the auth callback is not an index failure (503)."""

    def fake(**kwargs):
        raise HTTPException(status_code=403, detail="nope")

    service = _db_service(query_graph=fake)
    with (
        patch.object(lineage_mod, "_get_index_service", return_value=service),
        pytest.raises(HTTPException) as caught,
    ):
        lineage_mod.query_lineage_graph_get(
            _fake_request("member", "member@example.com"), uri="s3://b/x"
        )
    assert caught.value.status_code == 403


# -- POST /lineage/: attribution, space namespaces, no takeover ----------------

from gbserver.api import utils as _api_utils  # noqa: E402
from gbserver.lineage.openlineage_models import LineageEvent  # noqa: E402
from gbserver.storage.stored_lineage_row import StoredLineageRow  # noqa: E402

# Captured at import, before conftest's mock-mode bypass patches it per test.
_REAL_HAS_WRITE = _api_utils._has_existing_item_write_access


def _event(namespace="ext-system/pipeline", job_id="EXT-1", owner=None, **facets):
    details = {"job_id": job_id}
    if owner:
        details["owner"] = owner
    details.update(facets)
    return LineageEvent(
        eventType="COMPLETE",
        eventTime="2026-10-16T00:00:00Z",
        run={"runId": "r1", "facets": {"job_details": details}},
        job={"namespace": namespace, "name": "j"},
        producer="external",
    )


def _ingest(event, login="alice", spaces=(), member_of=(), rows=(), admin=False):
    emitted = []
    storage = SimpleNamespace(
        space_storage=SimpleNamespace(
            get_by_name=lambda name: object() if name in spaces else None
        ),
        lineage_row_storage=SimpleNamespace(
            get_rows_by_job=lambda job_id: [r for r in rows if r.job_id == job_id]
        ),
    )
    service = SimpleNamespace(emit_event=emitted.append)
    with (
        patch.object(lineage_mod, "get_admin_storage", return_value=storage),
        patch.object(lineage_mod, "_get_openlineage_service", return_value=service),
        patch.object(_api_utils, "_has_existing_item_write_access", _REAL_HAS_WRITE),
        patch.object(_api_utils, "is_super_admin", return_value=admin),
        patch.object(_api_utils, "is_space_admin", return_value=admin),
        patch.object(
            _api_utils,
            "space_access_check",
            side_effect=lambda username, space: space in member_of,
        ),
    ):
        lineage_mod.ingest_lineage_event(
            _fake_request(login, f"{login}@example.com"), event
        )
    return emitted


def _row(job_id, submitted_by=None):
    job = {"submitted_by": submitted_by} if submitted_by else {}
    return StoredLineageRow(
        job_id=job_id, input="s3://b/a", output="s3://b/b", attributes={"job": job}
    )


def test_ingest_stamps_the_caller_as_submitter_over_what_the_event_claims():
    (emitted,) = _ingest(_event(submitted_by="mallory"), login="alice")
    assert emitted["run"]["facets"]["job_details"]["submitted_by"] == "alice"


def test_ingest_accepts_an_external_namespace_from_anyone():
    assert _ingest(_event(namespace="ext-system/pipeline"), spaces=("team",))


def test_ingest_into_a_space_requires_membership():
    with pytest.raises(HTTPException) as caught:
        _ingest(_event(namespace="team/build"), spaces=("team",), member_of=())
    assert caught.value.status_code == 401
    assert _ingest(
        _event(namespace="team/build"), spaces=("team",), member_of=("team",)
    )


def test_ingest_into_a_space_cannot_claim_another_owner():
    with pytest.raises(HTTPException) as caught:
        _ingest(
            _event(namespace="team/build", owner="bob"),
            spaces=("team",),
            member_of=("team",),
        )
    assert caught.value.status_code == 401


def test_ingest_refuses_a_job_id_already_recorded_by_another_source():
    for existing in (_row("J1"), _row("J1", submitted_by="bob")):
        with pytest.raises(HTTPException) as caught:
            _ingest(_event(job_id="J1"), login="alice", rows=[existing])
        assert caught.value.status_code == 409


def test_ingest_lets_the_same_submitter_extend_its_own_job():
    assert _ingest(_event(job_id="J1"), login="alice", rows=[_row("J1", "alice")])
