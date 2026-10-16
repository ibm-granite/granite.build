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

"""Unit tests for the gbcli host / web-UI-URL environment overrides.

``GBSERVER_HOST`` and ``GB_WEB_UI_URL`` retarget the client at an external
deployment regardless of ``GB_ENVIRONMENT``. Both are resolved at import time, so
each test sets the env var and then reloads ``gbconstants``, reloading again in a
``finally`` so the module-level constants don't leak into other tests sharing the
xdist worker (same idiom as test/unit/gb_ui_backend/test_analytics_gating.py).
"""

import importlib

import pytest

from gbcli.utils import gbconstants
from gbcommon.types.gbenvconfig import gb_environment_config

pytestmark = pytest.mark.standalone

EXTERNAL_HOST = "https://gbserver.external.example.com"
EXTERNAL_WEB_UI = "https://dashboard.external.example.com"


def _reload(monkeypatch, **env):
    """Apply env vars, reload gbconstants, and restore it afterwards."""
    for name, value in env.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    importlib.reload(gbconstants)
    return gbconstants


class TestWebUiUrlOverride:
    """GB_WEB_UI_URL overrides the env config's web_ui_url."""

    def test_override_wins_when_set(self, monkeypatch):
        try:
            mod = _reload(monkeypatch, GB_WEB_UI_URL=EXTERNAL_WEB_UI)
            assert mod.WEB_UI_URL == EXTERNAL_WEB_UI
        finally:
            importlib.reload(gbconstants)

    def test_falls_back_to_env_config_when_unset(self, monkeypatch):
        try:
            mod = _reload(monkeypatch, GB_WEB_UI_URL=None, GB_ENVIRONMENT="PROD")
            assert mod.WEB_UI_URL == gb_environment_config("PROD").web_ui_url
        finally:
            importlib.reload(gbconstants)

    def test_build_links_use_the_override(self, monkeypatch):
        """The link builders bind WEB_UI_URL by value at their own import time, so
        the consumer module is reloaded too — a reload of gbconstants alone would
        leave the stale URL behind."""
        from gbcli.utils import utils

        try:
            _reload(monkeypatch, GB_WEB_UI_URL=EXTERNAL_WEB_UI)
            importlib.reload(utils)
            assert utils.get_build_lineage_url("b-123") == (
                f"{EXTERNAL_WEB_UI}/builds/b-123"
            )
        finally:
            importlib.reload(gbconstants)
            importlib.reload(utils)


class TestGbserverHostOverride:
    """Regression guard for the pre-existing GBSERVER_HOST override, now one of a
    symmetric pair with GB_WEB_UI_URL."""

    def test_override_wins_when_set(self, monkeypatch):
        try:
            mod = _reload(monkeypatch, GBSERVER_HOST=EXTERNAL_HOST)
            assert mod.GBSERVER_INSTANCE == EXTERNAL_HOST
        finally:
            importlib.reload(gbconstants)

    def test_falls_back_to_env_config_when_unset(self, monkeypatch):
        try:
            mod = _reload(monkeypatch, GBSERVER_HOST=None, GB_ENVIRONMENT="PROD")
            assert mod.GBSERVER_INSTANCE == gb_environment_config("PROD").gbserver_host
        finally:
            importlib.reload(gbconstants)

    def test_derived_api_urls_use_the_override(self, monkeypatch):
        """The service modules reach the external host through these derived
        constants, which is what makes GBClient honor the override too."""
        try:
            mod = _reload(monkeypatch, GBSERVER_HOST=EXTERNAL_HOST)
            assert mod.GBSERVER_BUILD_API == f"{EXTERNAL_HOST}/api/v1/builds/"
            assert mod.GBSERVER_ARTIFACT_API == f"{EXTERNAL_HOST}/api/v1/artifacts/"
            assert mod.GBSERVER_SECRETS_API == f"{EXTERNAL_HOST}/api/v1/secrets/"
            assert mod.GBSERVER_SPACES_API == f"{EXTERNAL_HOST}/api/v1/spaces/"
            assert mod.GBSERVER_LINEAGE_API == f"{EXTERNAL_HOST}/api/v1/lineage/"
            assert mod.GBSERVER_LOGS_API == f"{EXTERNAL_HOST}/api/v1/logs/"
        finally:
            importlib.reload(gbconstants)


class TestOverrideIsEnvironmentIndependent:
    """The point of the feature: an external host/web UI wins no matter which
    GB_ENVIRONMENT is set, not just in STANDALONE."""

    @pytest.mark.parametrize("gb_env", ["PROD", "STAGING", "DEV", "STANDALONE"])
    def test_override_wins_in_every_environment(self, monkeypatch, gb_env):
        try:
            mod = _reload(
                monkeypatch,
                GB_ENVIRONMENT=gb_env,
                GBSERVER_HOST=EXTERNAL_HOST,
                GB_WEB_UI_URL=EXTERNAL_WEB_UI,
            )
            assert mod.GBSERVER_INSTANCE == EXTERNAL_HOST
            assert mod.WEB_UI_URL == EXTERNAL_WEB_UI
            assert mod.GBSERVER_BUILD_API == f"{EXTERNAL_HOST}/api/v1/builds/"
        finally:
            importlib.reload(gbconstants)
