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

"""Unit tests for the ``build_continue`` service layer.

The key behaviour under test is that ``build_continue`` reports the *resolved
uuid* of the continued build via ``continued_from`` — even when the caller
passed a build URL. The CLI reads that field (not the raw argument) so a URL
never leaks into uuid-valued JSON output or a uuid-vs-URL root comparison.
"""

from unittest.mock import patch

import pytest

from gbcli.services import service_build

pytestmark = pytest.mark.standalone

PRIOR_UUID = "11111111-1111-1111-1111-111111111111"
NEW_UUID = "22222222-2222-2222-2222-222222222222"
ROOT_UUID = "33333333-3333-3333-3333-333333333333"
BUILD_URL = "https://example.com/builds/11111111-1111-1111-1111-111111111111"


def _server_response():
    # The server only returns the new build id and the resolved chain root; it
    # does not echo back which member was continued.
    return {"build_id": NEW_UUID, "root_build_id": ROOT_UUID}


def test_build_continue_reports_resolved_uuid_when_url_passed():
    """A URL is resolved to a uuid before the server call; continued_from must be
    that resolved uuid, not the URL the caller typed."""
    with (
        patch.object(
            service_build,
            "get_build_id_from_url",
            return_value=[{"uuid": PRIOR_UUID}],
        ) as resolve,
        patch.object(
            service_build, "make_gbserver_call", return_value=_server_response()
        ),
    ):
        result = service_build.build_continue(
            github_token="tok", build_id=BUILD_URL, id_format="url"
        )

    resolve.assert_called_once()
    assert result["continued_from"] == PRIOR_UUID
    assert result["continued_from"] != BUILD_URL
    # Server-provided fields are passed through untouched.
    assert result["build_id"] == NEW_UUID
    assert result["root_build_id"] == ROOT_UUID


def test_build_continue_reports_uuid_unchanged_when_uuid_passed():
    """When a uuid is passed there is nothing to resolve; continued_from is that
    same uuid, so the CLI's `root != continued_from` hint compares uuid-to-uuid."""
    with (
        patch.object(service_build, "get_build_id_from_url") as resolve,
        patch.object(
            service_build, "make_gbserver_call", return_value=_server_response()
        ),
    ):
        result = service_build.build_continue(
            github_token="tok", build_id=PRIOR_UUID, id_format="uuid"
        )

    resolve.assert_not_called()
    assert result["continued_from"] == PRIOR_UUID


def test_build_continue_returns_none_on_server_error():
    """A server/connection failure (make_gbserver_call -> None) yields None and no
    continued_from is fabricated."""
    with (
        patch.object(service_build, "get_build_id_from_url"),
        patch.object(service_build, "make_gbserver_call", return_value=None),
    ):
        result = service_build.build_continue(
            github_token="tok", build_id=PRIOR_UUID, id_format="uuid"
        )

    assert result is None
