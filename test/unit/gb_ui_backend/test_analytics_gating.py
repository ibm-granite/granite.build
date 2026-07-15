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

"""Regression tests for analytics gating.

A deployed, API-only rest-server has the ``gb_ui_backend`` package installed but
no compiled UI assets. Analytics must stay off there — otherwise its startup
init opens a SQLite file at an unwritable path and crashes the whole server
(observed as a crashloop with ``sqlite3.OperationalError: unable to open
database file``). Covers:

  - analytics_is_enabled tri-state: explicit GB_UI_ANALYTICS_ENABLED wins,
    otherwise auto-detect off the presence of compiled UI assets.
  - _configure_analytics_env skips the SQLite GB_UI_DATABASE_URL fallback when
    analytics resolves off, so an API-only server never gets the crashing URL.
"""

import os

import pytest

from gb_ui_backend import config as gb_config
from gb_ui_backend.config import analytics_is_enabled
from gbserver.commands.command_rest_server import _configure_analytics_env


@pytest.fixture(autouse=True)
def _clear_config_cache():
    """get_config() is lru_cached; clear it so per-test env changes take effect."""
    gb_config.get_config.cache_clear()
    yield
    gb_config.get_config.cache_clear()


class TestAnalyticsIsEnabled:
    def test_auto_detect_off_when_no_ui_assets(self, monkeypatch):
        monkeypatch.delenv("GB_UI_ANALYTICS_ENABLED", raising=False)
        gb_config.get_config.cache_clear()
        assert analytics_is_enabled(ui_assets_present=False) is False

    def test_auto_detect_on_when_ui_assets_present(self, monkeypatch):
        monkeypatch.delenv("GB_UI_ANALYTICS_ENABLED", raising=False)
        gb_config.get_config.cache_clear()
        assert analytics_is_enabled(ui_assets_present=True) is True

    def test_explicit_false_overrides_present_ui(self, monkeypatch):
        monkeypatch.setenv("GB_UI_ANALYTICS_ENABLED", "false")
        gb_config.get_config.cache_clear()
        assert analytics_is_enabled(ui_assets_present=True) is False

    def test_explicit_true_overrides_absent_ui(self, monkeypatch):
        monkeypatch.setenv("GB_UI_ANALYTICS_ENABLED", "true")
        gb_config.get_config.cache_clear()
        assert analytics_is_enabled(ui_assets_present=False) is True

    def test_blank_value_treated_as_unset(self, monkeypatch):
        # Setting an env var to empty is a common "disable"/"unset" idiom in k8s
        # manifests and shells. It must not raise (which would crash startup in
        # the parent and every worker) — it falls through to auto-detect.
        for blank in ("", "   "):
            monkeypatch.setenv("GB_UI_ANALYTICS_ENABLED", blank)
            gb_config.get_config.cache_clear()
            assert analytics_is_enabled(ui_assets_present=True) is True
            gb_config.get_config.cache_clear()
            assert analytics_is_enabled(ui_assets_present=False) is False

    def test_whitespace_padded_value_parsed(self, monkeypatch):
        monkeypatch.setenv("GB_UI_ANALYTICS_ENABLED", " false ")
        gb_config.get_config.cache_clear()
        assert analytics_is_enabled(ui_assets_present=True) is False

    def test_unrecognized_nonblank_value_warns_and_is_true(self, monkeypatch, caplog):
        # A non-blank, non-bool value (typo like "enabled", "on-prod") must not
        # raise a ValidationError out of Config() — that would crash startup in
        # the parent and every worker. It resolves to True (anything set and not
        # a falsy token) and logs a warning so the likely typo is visible.
        monkeypatch.setenv("GB_UI_ANALYTICS_ENABLED", "enabled")
        gb_config.get_config.cache_clear()
        with caplog.at_level("WARNING"):
            assert analytics_is_enabled(ui_assets_present=False) is True
        assert any("Unrecognized boolean value" in r.message for r in caplog.records)


class TestConfigureAnalyticsEnv:
    """_configure_analytics_env must not set the SQLite fallback when analytics is off."""

    def _run(self, monkeypatch, tmp_path, *, ui_present, override):
        # Force the auto-detect signal via GBSERVER_UI_DIR: a real dir (present)
        # vs. a path that does not exist (absent, the API-only pod condition).
        ui_dir = str(tmp_path / "ui") if ui_present else str(tmp_path / "absent")
        if ui_present:
            os.makedirs(ui_dir, exist_ok=True)
        monkeypatch.setenv("GBSERVER_UI_DIR", ui_dir)
        if override is None:
            monkeypatch.delenv("GB_UI_ANALYTICS_ENABLED", raising=False)
        else:
            monkeypatch.setenv("GB_UI_ANALYTICS_ENABLED", override)
        # Clear every var _configure_analytics_env may write via os.environ[...] so
        # monkeypatch tracks and restores them — otherwise GB_UI_GBSERVER_URL (set
        # unconditionally) and GB_UI_GBSERVER_DB_URL leak into later tests.
        for var in (
            "GB_UI_DATABASE_URL",
            "GB_UI_GBSERVER_URL",
            "GB_UI_GBSERVER_DB_URL",
        ):
            monkeypatch.delenv(var, raising=False)
        gb_config.get_config.cache_clear()

        _configure_analytics_env(host="0.0.0.0", port=8080)

    def test_api_only_does_not_set_sqlite_url(self, monkeypatch, tmp_path):
        self._run(monkeypatch, tmp_path, ui_present=False, override=None)
        assert os.environ.get("GB_UI_DATABASE_URL") is None

    def test_explicit_off_does_not_set_sqlite_url(self, monkeypatch, tmp_path):
        self._run(monkeypatch, tmp_path, ui_present=True, override="false")
        assert os.environ.get("GB_UI_DATABASE_URL") is None

    def test_ui_mode_sets_sqlite_url(self, monkeypatch, tmp_path):
        self._run(monkeypatch, tmp_path, ui_present=True, override=None)
        url = os.environ.get("GB_UI_DATABASE_URL")
        assert url is not None and url.startswith("sqlite+aiosqlite:///")
