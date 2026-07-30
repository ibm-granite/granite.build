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


class TestDatabaseConnectArgsProperty:
    """Config.database_connect_args decodes GB_UI_DATABASE_CONNECT_ARGS (JSON-encoded
    since an ssl.SSLContext can't cross the env-var boundary as-is)."""

    def test_unset_is_empty_dict(self, monkeypatch):
        monkeypatch.delenv("GB_UI_DATABASE_CONNECT_ARGS", raising=False)
        gb_config.get_config.cache_clear()
        assert gb_config.get_config().database_connect_args == {}

    def test_set_decodes_json(self, monkeypatch):
        monkeypatch.setenv(
            "GB_UI_DATABASE_CONNECT_ARGS", '{"sslrootcert_file": "/tmp/root.pem"}'
        )
        gb_config.get_config.cache_clear()
        assert gb_config.get_config().database_connect_args == {
            "sslrootcert_file": "/tmp/root.pem"
        }


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
        # GBSERVER_METADATA_STORAGE unset here defaults to "sql", which since
        # TestDeriveAnalyticsDatabaseUrl below derives a postgres URL instead — pin
        # sqlite mode explicitly to keep asserting the SQLite path, and isolate
        # GB_HOME_DIR so this doesn't probe the real filesystem's home dir.
        monkeypatch.setenv("GBSERVER_METADATA_STORAGE", "sqlite")
        monkeypatch.setenv("GB_HOME_DIR", str(tmp_path / "gb_home"))
        import importlib

        from gbserver.types import constants

        importlib.reload(constants)
        try:
            self._run(monkeypatch, tmp_path, ui_present=True, override=None)
            url = os.environ.get("GB_UI_DATABASE_URL")
            assert url is not None and url.startswith("sqlite+aiosqlite:///")
        finally:
            importlib.reload(constants)

    def test_get_config_cache_cleared_after_env_mutation(self, monkeypatch, tmp_path):
        """Regression test for a lru_cache staleness bug found via live verification
        (not by any unit test): analytics_backend_enabled(), called at the top of
        _configure_analytics_env, constructs and caches a Config read from os.environ
        *before* GB_UI_DATABASE_URL is set later in the same function. Without
        clearing the cache afterward, every later consumer (init_analytics(),
        _get_engine()) would see that stale, pre-mutation instance with
        database_url="" forever — analytics would silently never get a database
        despite GB_UI_DATABASE_URL being set correctly in the process environment.
        """
        monkeypatch.setenv("GBSERVER_METADATA_STORAGE", "sqlite")
        monkeypatch.setenv("GB_HOME_DIR", str(tmp_path / "gb_home"))
        monkeypatch.delenv("GB_UI_DATABASE_URL", raising=False)
        # Force analytics on explicitly rather than relying on GBSERVER_UI_DIR
        # auto-detection: src/gbserver/static/ui is a gitignored build artifact
        # that may or may not exist on the machine running this test (it doesn't
        # in a fresh CI checkout), and analytics_backend_enabled() returning False
        # would make _configure_analytics_env() return before ever touching
        # GB_UI_DATABASE_URL — the exact thing this test needs to happen.
        monkeypatch.setenv("GB_UI_ANALYTICS_ENABLED", "true")
        # _configure_analytics_env sets these two directly via os.environ[...] (not
        # through monkeypatch), so without pre-registering them here monkeypatch
        # won't know to revert them — they'd leak into later tests sharing this
        # worker process, exactly like the leak _run() below already guards against.
        monkeypatch.delenv("GB_UI_GBSERVER_URL", raising=False)
        monkeypatch.delenv("GB_UI_GBSERVER_DB_URL", raising=False)
        import importlib

        from gbserver.types import constants

        importlib.reload(constants)
        try:
            # Prime the cache the same way analytics_backend_enabled()'s internal
            # get_config() call does, before _configure_analytics_env mutates env.
            gb_config.get_config.cache_clear()
            gb_config.get_config()
            _configure_analytics_env(host="0.0.0.0", port=8080)
            assert gb_config.get_config().database_url != ""
        finally:
            importlib.reload(constants)


class TestDeriveAnalyticsDatabaseUrl:
    """derive_analytics_database_url() inherits the main store's backend config
    instead of an independent SQLite default — see gbserver/types/constants.py.
    """

    def _reload_constants(self):
        import importlib

        from gbserver.types import constants

        importlib.reload(constants)
        return constants

    def _sql_env(self, monkeypatch, **overrides):
        env = {
            "GBSERVER_METADATA_STORAGE": "sql",
            "GBSERVER_SQL_SCHEME": "postgresql",
            "GBSERVER_SQL_HOST": "pg.example.com",
            "GBSERVER_SQL_PORT": "5432",
            "GBSERVER_SQL_USER": "gbui",
            "GBSERVER_SQL_PASSWD": "s3cr3t",
            "GBSERVER_SQL_DBNAME": "gbdb",
            **overrides,
        }
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        monkeypatch.delenv("GBSERVER_SQL_SSLROOT_CERT_FILE", raising=False)
        monkeypatch.delenv("GBSERVER_SQL_SSLROOT_CERT", raising=False)
        monkeypatch.delenv("GBSERVER_SQL_SSLROOT_CERT_BASE64", raising=False)

    def test_sql_mode_derives_postgres_asyncpg_url(self, monkeypatch):
        self._sql_env(monkeypatch)
        constants = self._reload_constants()
        try:
            assert (
                constants.derive_analytics_database_url()
                == "postgresql+asyncpg://gbui:s3cr3t@pg.example.com:5432/gbdb"
            )
        finally:
            self._reload_constants()

    def test_sql_mode_url_quotes_special_characters(self, monkeypatch):
        self._sql_env(
            monkeypatch, GBSERVER_SQL_USER="gb ui", GBSERVER_SQL_PASSWD="p@ss/word"
        )
        constants = self._reload_constants()
        try:
            url = constants.derive_analytics_database_url()
            assert "gb+ui" in url
            assert "p%40ss%2Fword" in url
        finally:
            self._reload_constants()

    def test_sql_mode_unsupported_scheme_returns_none(self, monkeypatch):
        self._sql_env(monkeypatch, GBSERVER_SQL_SCHEME="mysql")
        constants = self._reload_constants()
        try:
            assert constants.derive_analytics_database_url() is None
        finally:
            self._reload_constants()

    def test_sqlite_mode_uses_dashboard_analytics_db(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GBSERVER_METADATA_STORAGE", "sqlite")
        monkeypatch.setenv("GB_HOME_DIR", str(tmp_path))
        gb_config.get_config.cache_clear()
        constants = self._reload_constants()
        try:
            url = constants.derive_analytics_database_url()
            assert url == f"sqlite+aiosqlite:///{tmp_path / 'dashboard-analytics.db'}"
        finally:
            self._reload_constants()


class TestDeriveAnalyticsSqlConnectArgs:
    """derive_analytics_sql_connect_args() translates the main store's TLS cert
    (sslrootcert/sslmode, built for sync psycopg2) into the cert file path
    gb_ui_backend's db_schema.py turns into an ssl.SSLContext for asyncpg.
    """

    def _reset_cert_state(self, monkeypatch, *, cert_file=None, cert_base64=None):
        import gbserver.storage.sql.cert_file as cert_file_module

        # get_ssl_cert_file caches its result in a module global — reset it so each
        # test starts fresh regardless of what a previous test resolved.
        monkeypatch.setattr(cert_file_module, "_SSL_CERT_FILE", None)
        monkeypatch.setattr(
            cert_file_module, "GBSERVER_SQL_SSLROOT_CERT_FILE", cert_file
        )
        monkeypatch.setattr(
            cert_file_module, "GBSERVER_SQL_SSLROOT_CERT_BASE64", cert_base64
        )

    def test_no_cert_configured_returns_empty(self, monkeypatch):
        self._reset_cert_state(monkeypatch)
        from gbserver.types import constants

        assert constants.derive_analytics_sql_connect_args() == {}

    def test_cert_file_configured_returns_path_for_translation(
        self, monkeypatch, tmp_path
    ):
        cert_path = tmp_path / "root.pem"
        cert_path.write_text("dummy-cert-content")
        self._reset_cert_state(monkeypatch, cert_file=str(cert_path))
        from gbserver.types import constants

        assert constants.derive_analytics_sql_connect_args() == {
            "sslrootcert_file": str(cert_path)
        }
