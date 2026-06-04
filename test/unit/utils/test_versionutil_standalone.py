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

"""Standalone-mode behavior of the CLI version check.

``check_current_and_latest_versions()`` runs at the top of most ``gb`` commands and
queries GitHub Enterprise (requiring GitHub auth). In standalone mode that auth is
unavailable, so the check must be skipped (return "") rather than raising
"user not logged in".
"""

import pytest

from gbcli.utils import versionutil
from gbcli.utils.gbconstants import USER_NOT_LOGGED_IN_ERROR_MESSAGE


class TestVersionCheckStandalone:
    def test_skipped_in_standalone(self, monkeypatch):
        """In standalone mode the check returns '' without touching credentials/network."""
        monkeypatch.setenv("GB_ENVIRONMENT", "STANDALONE")

        # Guard: if the check did NOT short-circuit, these would be reached and blow up,
        # making the test fail loudly rather than silently passing for the wrong reason.
        def _boom(*args, **kwargs):
            raise AssertionError("version check should not reach credentials/network")

        monkeypatch.setattr(versionutil, "GBCredentials", _boom)
        monkeypatch.setattr(versionutil, "get_latest_version", _boom)

        assert versionutil.check_current_and_latest_versions() == ""

    def test_requires_login_outside_standalone(self, monkeypatch):
        """Outside standalone, missing credentials still raise 'user not logged in'."""
        monkeypatch.setenv("GB_ENVIRONMENT", "PROD")

        class _FakeCreds:
            def check_values(self):
                return False

        monkeypatch.setattr(versionutil, "GBCredentials", lambda: _FakeCreds())

        with pytest.raises(Exception) as exc_info:
            versionutil.check_current_and_latest_versions()

        assert USER_NOT_LOGGED_IN_ERROR_MESSAGE in str(exc_info.value)
