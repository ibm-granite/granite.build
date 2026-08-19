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

from gbserver.api import lineage as lineage_mod
from gbserver.lineage.openlineage_models import ArtifactGraphRequest, TagSearchRequest

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
