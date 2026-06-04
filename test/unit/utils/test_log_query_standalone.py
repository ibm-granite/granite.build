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

"""Standalone-mode token handling for ``build log`` (run_logquery).

In standalone mode the gbserver log query should work with an empty user token: the
local gbserver allows localhost access when no GBSERVER_API_KEY is configured. Outside
standalone mode an empty token must still be rejected as "not logged in".
"""

from types import SimpleNamespace

import pytest

from gbcli.utils import log_query
from gbcli.utils.gbconstants import USER_NOT_LOGGED_IN_ERROR_MESSAGE


@pytest.fixture
def fake_user(monkeypatch):
    """get_user() returns a synthetic standalone user (as it does in standalone mode)."""
    monkeypatch.setattr(
        log_query, "get_user", lambda token: SimpleNamespace(login="standalone")
    )


@pytest.fixture
def captured_post(monkeypatch):
    """Replace gbserver_post so no network call is made; record that it was reached."""
    calls = {}

    def _fake_post(token, url, payload):
        calls["token"] = token
        calls["url"] = url
        return {"status": 200, "result": []}

    monkeypatch.setattr(log_query, "gbserver_post", _fake_post)
    return calls


class TestRunLogqueryStandaloneToken:
    def test_empty_token_allowed_in_standalone(
        self, monkeypatch, fake_user, captured_post
    ):
        """An empty token must NOT raise in standalone mode; the query proceeds."""
        monkeypatch.setenv("GB_ENVIRONMENT", "STANDALONE")

        result = log_query.run_logquery(
            github_token="",
            start_epoch_in_s=0,
            end_epoch_in_s=1,
            callback=lambda **kwargs: None,
        )

        assert result == {"status": 200, "result": []}
        assert captured_post["token"] == ""  # forwarded as-is to the local gbserver

    def test_empty_token_rejected_outside_standalone(
        self, monkeypatch, fake_user, captured_post
    ):
        """An empty token must still raise 'not logged in' outside standalone mode."""
        monkeypatch.setenv("GB_ENVIRONMENT", "PROD")

        with pytest.raises(Exception) as exc_info:
            log_query.run_logquery(
                github_token="",
                start_epoch_in_s=0,
                end_epoch_in_s=1,
                callback=lambda **kwargs: None,
            )

        assert USER_NOT_LOGGED_IN_ERROR_MESSAGE in str(exc_info.value)
        assert "token" not in captured_post  # never reached the request

    def test_nonempty_token_works_outside_standalone(
        self, monkeypatch, fake_user, captured_post
    ):
        """A normal (non-empty) token still works outside standalone mode."""
        monkeypatch.setenv("GB_ENVIRONMENT", "PROD")

        result = log_query.run_logquery(
            github_token="some-token",
            start_epoch_in_s=0,
            end_epoch_in_s=1,
            callback=lambda **kwargs: None,
        )

        assert result == {"status": 200, "result": []}
        assert captured_post["token"] == "some-token"
