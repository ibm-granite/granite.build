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

"""Standalone-mode token handling for ``gb build update`` (update_build).

In standalone mode an empty user token is legitimate (the local gbserver allows localhost
access when no GBSERVER_API_KEY is configured), so update_build must not reject it.
Outside standalone mode an empty token still raises "user not logged in".
"""

from types import SimpleNamespace

import pytest

from gbcli.services import service_build
from gbcli.utils.gbconstants import USER_NOT_LOGGED_IN_ERROR_MESSAGE


@pytest.fixture
def patched(monkeypatch):
    """Stub get_user (synthetic standalone user) and the gbserver update call."""
    calls = {}

    monkeypatch.setattr(
        service_build, "get_user", lambda token: SimpleNamespace(login="standalone")
    )

    def _fake_update(build_id, server_api, user_token, description, tags, append):
        calls["token"] = user_token
        return {"build_id": build_id, "description": description}

    monkeypatch.setattr(service_build, "update_build_gserver", _fake_update)
    return calls


class TestUpdateBuildStandaloneToken:
    def test_empty_token_allowed_in_standalone(self, monkeypatch, patched):
        monkeypatch.setenv("GB_ENVIRONMENT", "STANDALONE")

        result = service_build.update_build("", "bid", description="hi")

        assert result == {"build_id": "bid", "description": "hi"}
        assert patched["token"] == ""

    def test_empty_token_rejected_outside_standalone(self, monkeypatch, patched):
        monkeypatch.setenv("GB_ENVIRONMENT", "PROD")

        with pytest.raises(Exception) as exc_info:
            service_build.update_build("", "bid", description="hi")

        assert USER_NOT_LOGGED_IN_ERROR_MESSAGE in str(exc_info.value)
        assert "token" not in patched  # never reached the gbserver call

    def test_nonempty_token_works_outside_standalone(self, monkeypatch, patched):
        monkeypatch.setenv("GB_ENVIRONMENT", "PROD")

        result = service_build.update_build("real-token", "bid", description="hi")

        assert result == {"build_id": "bid", "description": "hi"}
        assert patched["token"] == "real-token"


class TestUpdateBuildGserverStandaloneGuard:
    """update_build_gserver must still issue the PUT with an empty token in standalone.

    The old ``if server_api and user_token:`` guard silently skipped the request (and
    returned None) when the token was empty -- so an update appeared to succeed but never
    reached gbserver. In standalone an empty token must still send the request.
    """

    def test_empty_token_still_sends_request_in_standalone(self, monkeypatch):
        from gbcli.utils import gbserver

        sent = {}

        def _fake_put(token, url, body):
            sent["url"] = url
            sent["body"] = body
            return {"build": {"build_id": "bid", "tags": body.get("tags")}}

        monkeypatch.setenv("GB_ENVIRONMENT", "STANDALONE")
        monkeypatch.setattr(gbserver, "gbserver_put", _fake_put)

        result = gbserver.update_build_gserver(
            build_id="bid",
            server_api="http://localhost:8080/api/v1/builds/",
            user_token="",
            tags=["tagA"],
        )

        assert sent, "PUT should be issued even with an empty token in standalone"
        assert sent["body"]["tags"] == {"set": ["tagA"]}
        assert result == {"build_id": "bid", "tags": {"set": ["tagA"]}}

    def test_empty_token_skips_request_outside_standalone(self, monkeypatch):
        from gbcli.utils import gbserver

        sent = {}

        def _fake_put(token, url, body):
            sent["url"] = url
            return {"build": {}}

        monkeypatch.setenv("GB_ENVIRONMENT", "PROD")
        monkeypatch.setattr(gbserver, "gbserver_put", _fake_put)

        result = gbserver.update_build_gserver(
            build_id="bid",
            server_api="http://host/api/v1/builds/",
            user_token="",
            tags=["tagA"],
        )

        # Outside standalone an empty token still short-circuits (no request, None).
        assert not sent
        assert result is None
