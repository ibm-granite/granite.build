#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""Standalone-mode token handling for ``gb version`` (get_gbserver_version).

In standalone mode the gbserver version query should work with an empty user token: the
local gbserver allows localhost access when no GBSERVER_API_KEY is configured. Outside
standalone mode an empty token must still be rejected as "not logged in".
"""

import pytest

from gbcli.services import service_version
from gbcli.utils.gbconstants import USER_NOT_LOGGED_IN_ERROR_MESSAGE


@pytest.fixture
def captured_call(monkeypatch):
    """Replace get_server_version so no network call is made; record the token used."""
    calls = {}

    def _fake_get_server_version(user_token, gbserver_instance):
        calls["token"] = user_token
        return {"git_commit": "abc1234"}

    monkeypatch.setattr(service_version, "get_server_version", _fake_get_server_version)
    # make_gbserver_call just invokes the thunk; keep it simple and deterministic.
    monkeypatch.setattr(
        service_version, "make_gbserver_call", lambda fn, callback: fn()
    )
    return calls


class TestGetGbserverVersionStandaloneToken:
    def test_empty_token_allowed_in_standalone(self, monkeypatch, captured_call):
        """An empty token must NOT raise in standalone mode; the query proceeds."""
        monkeypatch.setenv("GB_ENVIRONMENT", "STANDALONE")

        result = service_version.get_gbserver_version("", quiet=True, callback=None)

        assert result == "abc1234"
        assert captured_call["token"] == ""

    def test_empty_token_rejected_outside_standalone(self, monkeypatch, captured_call):
        """An empty token must still raise 'not logged in' outside standalone mode."""
        monkeypatch.setenv("GB_ENVIRONMENT", "PROD")

        with pytest.raises(Exception) as exc_info:
            service_version.get_gbserver_version("", quiet=True, callback=None)

        assert USER_NOT_LOGGED_IN_ERROR_MESSAGE in str(exc_info.value)
        assert "token" not in captured_call  # never reached the request

    def test_nonempty_token_works_outside_standalone(self, monkeypatch, captured_call):
        """A normal (non-empty) token still works outside standalone mode."""
        monkeypatch.setenv("GB_ENVIRONMENT", "PROD")

        result = service_version.get_gbserver_version(
            "some-token", quiet=True, callback=None
        )

        assert result == "abc1234"
        assert captured_call["token"] == "some-token"
