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

"""Unit tests for build-log query authorization (HackerOne 3875453).

POST /logquery had no authorization at all; POST /logquery/server/{build_id}
checked access against the PATH build_id while the actual Cloud Logs filter
came from the BODY's queryDef.queryParams.jsonObject, which could reference a
different, unauthorized build. Both are fixed in gbserver/api/logs.py.

A first pass at these fixes only checked the build-id label when it happened
to be present under jsonObject as a dict — which left two gaps a review
caught: (1) a caller filtering by any other dimension (or omitting the
build-id key) skipped the check entirely on the plain route, and (2) a
non-dict jsonObject skipped the build-id override on the server/{build_id}
route, re-opening the smuggling bypass. Both routes now require jsonObject to
be a dict (rejecting otherwise) and require the plain route's query to be
scoped to an accessible build id rather than defaulting to allow.

There are three functions literally named `logquery` in that module (one per
route decorator), so plain `from gbserver.api.logs import logquery` or
`logs_module.logquery` only resolves the last one defined. Each is pulled
directly off the FastAPI route table instead.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from gbserver.api import logs as logs_module
from gbserver.types.logs import Item, QueryDef, QueryParams

BUILD_LABEL_KEY = "kubernetes.labels.granite-dot-build/build-id"

_plain_logquery = next(
    r.endpoint for r in logs_module.logs_api.routes if r.path == "/logquery"
)
_server_logquery = next(
    r.endpoint
    for r in logs_module.logs_api.routes
    if r.path == "/logquery/server/{build_id}"
)


def _fake_request(login: str, email: str):
    from types import SimpleNamespace

    return SimpleNamespace(
        state=SimpleNamespace(data={"user": SimpleNamespace(login=login, email=email)})
    )


def _item(json_object, query_type="freeText"):
    return Item(
        queryDef=QueryDef(
            startDate=0,
            endDate=1,
            type=query_type,
            queryParams=QueryParams(metadata={}, jsonObject=json_object),
        )
    )


def _mocked_access_manager(accessible_build_id: str):
    mgr = MagicMock()
    mgr.has_build_access.side_effect = lambda username, bid: bid == accessible_build_id
    return patch(
        "gbserver.spaces.user_spaces_list.get_space_access_manager",
        return_value=mgr,
    )


# --------------------------------------------------------------- plain /logquery


def test_plain_logquery_rejects_victim_build_id_filter():
    with (
        _mocked_access_manager(accessible_build_id="attacker-build"),
        patch.object(logs_module, "get_log_manager", return_value=MagicMock()),
    ):
        resp = _plain_logquery(
            _fake_request("attacker_a", "attacker_a@example.com"),
            _item({BUILD_LABEL_KEY: ["victim-build"]}),
        )
    assert resp.status_code == 401


def test_plain_logquery_allows_own_build_id_filter():
    log_manager = MagicMock(query_cloud_logquery=lambda q: "ok")
    with (
        _mocked_access_manager(accessible_build_id="attacker-build"),
        patch.object(logs_module, "get_log_manager", return_value=log_manager),
    ):
        resp = _plain_logquery(
            _fake_request("attacker_a", "attacker_a@example.com"),
            _item({BUILD_LABEL_KEY: ["attacker-build"]}),
        )
    assert resp == "ok"


def test_plain_logquery_pins_query_type_to_freetext():
    """A caller-supplied queryDef.type is overwritten server-side, so an
    unpinned type can't select a different query mode with unknown
    combination semantics between jsonObject and any free-text filter."""
    captured = {}

    def fake_query(q):
        captured["type"] = q.queryDef.type
        return "ok"

    with (
        _mocked_access_manager(accessible_build_id="attacker-build"),
        patch.object(
            logs_module,
            "get_log_manager",
            return_value=MagicMock(query_cloud_logquery=fake_query),
        ),
    ):
        resp = _plain_logquery(
            _fake_request("attacker_a", "attacker_a@example.com"),
            _item({BUILD_LABEL_KEY: ["attacker-build"]}, query_type="somethingElse"),
        )
    assert resp == "ok"
    assert captured["type"] == "freeText"


def test_plain_logquery_rejects_query_with_no_build_id_filter():
    """A caller could otherwise skip the check entirely by filtering on any
    dimension other than the build-id label (or none at all) — every query
    must be scoped to an accessible build id, not just checked when one
    happens to be present."""
    with (
        _mocked_access_manager(accessible_build_id="attacker-build"),
        patch.object(logs_module, "get_log_manager", return_value=MagicMock()),
    ):
        with pytest.raises(HTTPException) as exc:
            _plain_logquery(
                _fake_request("attacker_a", "attacker_a@example.com"), _item({})
            )
        assert exc.value.status_code == 400


def test_plain_logquery_rejects_query_filtering_by_other_dimension_only():
    with (
        _mocked_access_manager(accessible_build_id="attacker-build"),
        patch.object(logs_module, "get_log_manager", return_value=MagicMock()),
    ):
        with pytest.raises(HTTPException) as exc:
            _plain_logquery(
                _fake_request("attacker_a", "attacker_a@example.com"),
                _item({"stream": ["stdout"]}),
            )
        assert exc.value.status_code == 400


def test_plain_logquery_rejects_non_dict_json_object():
    with (
        _mocked_access_manager(accessible_build_id="attacker-build"),
        patch.object(logs_module, "get_log_manager", return_value=MagicMock()),
    ):
        with pytest.raises(HTTPException) as exc:
            _plain_logquery(
                _fake_request("attacker_a", "attacker_a@example.com"),
                _item(["not", "a", "dict"]),
            )
        assert exc.value.status_code == 400


def test_plain_logquery_rejects_non_list_build_id_value():
    with (
        _mocked_access_manager(accessible_build_id="attacker-build"),
        patch.object(logs_module, "get_log_manager", return_value=MagicMock()),
    ):
        with pytest.raises(HTTPException) as exc:
            _plain_logquery(
                _fake_request("attacker_a", "attacker_a@example.com"),
                _item({BUILD_LABEL_KEY: "victim-build"}),
            )
        assert exc.value.status_code == 400


# ---------------------------------------------------- /logquery/server/{build_id}


def test_server_logquery_pins_query_type_to_freetext():
    captured = {}

    def fake_query(q):
        captured["type"] = q.queryDef.type
        return "ok"

    with (
        _mocked_access_manager(accessible_build_id="attacker-build"),
        patch.object(
            logs_module,
            "get_log_server_manager",
            return_value=MagicMock(query_cloud_logquery=fake_query),
        ),
    ):
        resp = _server_logquery(
            _fake_request("attacker_a", "attacker_a@example.com"),
            "attacker-build",
            _item({}, query_type="somethingElse"),
        )
    assert resp == "ok"
    assert captured["type"] == "freeText"


def test_server_logquery_overrides_smuggled_body_build_id():
    """Attacker passes their own build_id in the path (authorized) while
    smuggling a victim build_id into the body filter. The forwarded filter
    must be forced to the path-authorized build, not the smuggled one, while
    other legitimate filter keys are preserved."""
    captured = {}

    def fake_query(query):
        captured["jsonObject"] = query.queryDef.queryParams.jsonObject
        return "ok"

    with (
        _mocked_access_manager(accessible_build_id="attacker-build"),
        patch.object(
            logs_module,
            "get_log_server_manager",
            return_value=MagicMock(query_cloud_logquery=fake_query),
        ),
    ):
        resp = _server_logquery(
            _fake_request("attacker_a", "attacker_a@example.com"),
            "attacker-build",
            _item(
                {
                    BUILD_LABEL_KEY: ["victim-build"],
                    "kubernetes.labels.granite-dot-build/build-step-name": ["train"],
                }
            ),
        )

    assert resp == "ok"
    sent = captured["jsonObject"]
    assert sent[BUILD_LABEL_KEY] == ["attacker-build"], sent
    assert sent["kubernetes.labels.granite-dot-build/build-step-name"] == ["train"]


def test_server_logquery_rejects_non_dict_json_object():
    """A non-dict jsonObject previously skipped the build-id override
    entirely, forwarding whatever the body smuggled in unmodified."""
    with (
        _mocked_access_manager(accessible_build_id="attacker-build"),
        patch.object(logs_module, "get_log_server_manager", return_value=MagicMock()),
    ):
        with pytest.raises(HTTPException) as exc:
            _server_logquery(
                _fake_request("attacker_a", "attacker_a@example.com"),
                "attacker-build",
                _item(["not", "a", "dict"]),
            )
        assert exc.value.status_code == 400


def test_server_logquery_rejects_victim_build_id_in_path():
    with (
        _mocked_access_manager(accessible_build_id="attacker-build"),
        patch.object(logs_module, "get_log_server_manager", return_value=MagicMock()),
    ):
        resp = _server_logquery(
            _fake_request("attacker_a", "attacker_a@example.com"),
            "victim-build",
            _item({}),
        )
    assert resp.status_code == 401
