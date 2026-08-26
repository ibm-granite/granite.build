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

"""Unit tests for the ``build_restart`` service layer.

The key behaviour under test is that ``build_restart`` reports the *resolved
uuid* of the restarted build via ``restarted_from`` — even when the caller
passed a build URL. The CLI reads that field (not the raw argument) so a URL
never leaks into uuid-valued JSON output.
"""

from unittest.mock import patch

import pytest

from gbcli.services import service_build

pytestmark = pytest.mark.standalone

PRIOR_UUID = "11111111-1111-1111-1111-111111111111"
# A restart reuses the same build id, so the server returns the same uuid it
# was given; a distinct constant here only makes the passthrough assertion clear.
NEW_UUID = "22222222-2222-2222-2222-222222222222"
BUILD_URL = "https://example.com/builds/11111111-1111-1111-1111-111111111111"


def _server_response():
    # A restart reuses the same build id, so the server returns just build_id.
    return {"build_id": NEW_UUID}


def test_build_restart_reports_resolved_uuid_when_url_passed():
    """A URL is resolved to a uuid before the server call; restarted_from must be
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
        result = service_build.build_restart(
            github_token="tok", build_id=BUILD_URL, id_format="url"
        )

    resolve.assert_called_once()
    assert result["restarted_from"] == PRIOR_UUID
    assert result["restarted_from"] != BUILD_URL
    # Server-provided fields are passed through untouched.
    assert result["build_id"] == NEW_UUID


def test_build_restart_reports_uuid_unchanged_when_uuid_passed():
    """When a uuid is passed there is nothing to resolve; restarted_from is that
    same uuid the caller supplied."""
    with (
        patch.object(service_build, "get_build_id_from_url") as resolve,
        patch.object(
            service_build, "make_gbserver_call", return_value=_server_response()
        ),
    ):
        result = service_build.build_restart(
            github_token="tok", build_id=PRIOR_UUID, id_format="uuid"
        )

    resolve.assert_not_called()
    assert result["restarted_from"] == PRIOR_UUID


def test_build_restart_returns_none_on_server_error():
    """A server/connection failure (make_gbserver_call -> None) yields None and no
    restarted_from is fabricated."""
    with (
        patch.object(service_build, "get_build_id_from_url"),
        patch.object(service_build, "make_gbserver_call", return_value=None),
    ):
        result = service_build.build_restart(
            github_token="tok", build_id=PRIOR_UUID, id_format="uuid"
        )

    assert result is None


def test_build_restart_unresolvable_url_returns_none_not_exception():
    """get_build_id_from_url returns None when a URL matches no build; build_restart
    must guard the deref and return None (surfacing the intended error) instead of
    raising a raw IndexError/TypeError — even with callback=None (no early exit)."""
    calls = []

    def _capture(callback_event, callback_args):
        calls.append((callback_event, callback_args))

    with (
        patch.object(service_build, "get_build_id_from_url", return_value=None),
        patch.object(service_build, "make_gbserver_call") as server_call,
    ):
        result = service_build.build_restart(
            github_token="tok",
            build_id=BUILD_URL,
            id_format="url",
            callback=_capture,
        )

    assert result is None
    # Never reached the server call — bailed out at the unresolved URL.
    server_call.assert_not_called()
    # Surfaced the intended error event rather than a raw exception.
    assert any(event == "error" for event, _ in calls)


def test_build_restart_unresolvable_url_none_callback_no_crash():
    """The deref guard must also hold with callback=None (a programmatic caller):
    it returns None rather than raising."""
    with (
        patch.object(service_build, "get_build_id_from_url", return_value=None),
        patch.object(service_build, "make_gbserver_call") as server_call,
    ):
        result = service_build.build_restart(
            github_token="tok", build_id=BUILD_URL, id_format="url", callback=None
        )

    assert result is None
    server_call.assert_not_called()
