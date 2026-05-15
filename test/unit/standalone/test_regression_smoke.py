"""Regression smoke tests for standalone mode development.

These tests lock in current import chains, factory wiring, CLI discovery,
and API routing so that lazy-import and conditional-singleton changes
(Milestones 1-5) cannot silently break existing code paths.

All tests are marked @pytest.mark.g4os and @pytest.mark.unit so they
run in both the existing CI pipeline and the g4os-mode pipeline.
"""

import pytest

# ---------------------------------------------------------------------------
# a) Module import smoke tests
# ---------------------------------------------------------------------------


class TestModuleImports:
    """Verify key modules import without errors."""

    def test_import_constants(self):
        from gbserver.types.constants import (
            API_BASE_PATH,
            GB_ENVIRONMENT,
            GB_METADATA_STORAGE,
            GBSERVER_GITHUB_TOKEN,
        )

    def test_import_env_config(self):
        from gbserver.types.gbserverenvconfig import gb_environment_config

    def test_import_singleton_storage(self):
        from gbserver.storage.singleton_storage import (
            get_admin_storage,
            get_storage_factory,
            set_storage_factory,
        )

    def test_import_sqlite_factory(self):
        from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory

    def test_import_sql_factory(self):
        from gbserver.storage.sql.storage_factory import SQLStorageFactory

    def test_import_root_api(self):
        try:
            from gbserver.api.root_api import root_api  # noqa: F401
        except ImportError:
            pytest.skip(
                "root_api requires kubernetes_asyncio (transitively via buildwatcher)"
            )

    def test_import_auth_middleware(self):
        from gbserver.api.auth import AuthMiddleware

    def test_import_space_access_manager(self):
        from gbserver.spaces.space_access_manager import (
            ISpaceAccessManager,
            get_space_access_manager,
            set_space_access_manager,
        )

    def test_import_jobstats(self):
        from gbserver.lineage.jobstats import get_lineage_store

    def test_import_cli(self):
        from gbserver.cli import GraniteBuildServerCLI, gbserver


# ---------------------------------------------------------------------------
# b) Storage factory instantiation
# ---------------------------------------------------------------------------


class TestStorageFactories:
    """Verify each storage factory can be instantiated and create storage objects."""

    def _assert_factory_creates_all_storages(self, factory):
        assert factory.create_build_storage(table_name="test_builds") is not None
        assert factory.create_target_storage(table_name="test_targets") is not None
        assert factory.create_step_storage(table_name="test_steps") is not None
        assert factory.create_space_storage(table_name="test_spaces") is not None
        assert factory.create_artifact_registry(table_name="test_artifacts") is not None
        assert factory.create_event_storage(table_name="test_events") is not None

    def test_sqlite_factory(self):
        from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory

        factory = SqliteStorageFactory()
        self._assert_factory_creates_all_storages(factory)

    def test_sql_factory(self):
        from gbserver.storage.sql.storage_factory import SQLStorageFactory

        factory = SQLStorageFactory()
        self._assert_factory_creates_all_storages(factory)

# ---------------------------------------------------------------------------
# c) CLI command discovery
# ---------------------------------------------------------------------------


class TestCLIDiscovery:
    """Verify the CLI discovers all expected subcommands."""

    EXPECTED_COMMANDS = [
        "admin-tables",
        "build",
        "build-runner",
        "build-watch",
        "create-spaces",
        "pr-watch",
        "rest-server",
        "rest-server-worker",
    ]

    def test_list_commands(self):
        from gbserver.cli import GraniteBuildServerCLI

        cli = GraniteBuildServerCLI(name="test")
        commands = cli.list_commands(ctx=None)
        for expected in self.EXPECTED_COMMANDS:
            assert expected in commands, f"CLI command '{expected}' not discovered"

    def test_command_count(self):
        from gbserver.cli import GraniteBuildServerCLI

        cli = GraniteBuildServerCLI(name="test")
        commands = cli.list_commands(ctx=None)
        assert len(commands) >= len(
            self.EXPECTED_COMMANDS
        ), f"Expected at least {len(self.EXPECTED_COMMANDS)} commands, got {len(commands)}"


# ---------------------------------------------------------------------------
# d) Environment type discovery
# ---------------------------------------------------------------------------


class TestEnvironmentDiscovery:
    """Verify the environment plugin system discovers built-in environment types."""

    def test_environment_types_loaded(self):
        from gbserver.environment import Environment

        assert len(Environment.environment_types) > 0, "No environment types discovered"

    def test_k8s_environment(self):
        from gbserver.environment import Environment

        try:
            import kubernetes_asyncio  # noqa: F401

            assert "K8s" in Environment.environment_types
            assert "k8s" in Environment.environment_types
        except ImportError:
            assert "K8s" not in Environment.environment_types

    def test_bash_environment(self):
        from gbserver.environment import Environment

        assert "Bash" in Environment.environment_types
        assert "bash" in Environment.environment_types

    def test_lsf_environment(self):
        from gbserver.environment import Environment

        assert "Lsf" in Environment.environment_types
        assert "lsf" in Environment.environment_types


# ---------------------------------------------------------------------------
# e) API route verification
# ---------------------------------------------------------------------------


class TestAPIRoutes:
    """Verify all expected sub-APIs are mounted on the root API."""

    EXPECTED_PATHS = [
        "/api/v1",
        "/api/v1/artifacts",
        "/api/v1/builds",
        "/api/v1/lineage",
        "/api/v1/logs",
        "/api/v1/secrets",
        "/api/v1/spaces",
    ]

    def test_api_routes_mounted(self):
        try:
            from gbserver.api.root_api import root_api
        except ImportError:
            pytest.skip(
                "root_api requires kubernetes_asyncio (transitively via buildwatcher)"
            )

        route_paths = set()
        for route in root_api.routes:
            if hasattr(route, "path"):
                route_paths.add(route.path)

        for expected in self.EXPECTED_PATHS:
            assert (
                expected in route_paths
            ), f"Route '{expected}' not found in root_api. Found: {sorted(route_paths)}"


# ---------------------------------------------------------------------------
# f) Constants regression
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify critical constants are accessible and have expected types."""

    def test_metadata_storage_is_string(self):
        from gbserver.types.constants import GB_METADATA_STORAGE

        assert isinstance(GB_METADATA_STORAGE, str)

    def test_environment_is_string(self):
        from gbserver.types.constants import GB_ENVIRONMENT

        assert isinstance(GB_ENVIRONMENT, str)

    def test_github_token_is_string(self):
        from gbserver.types.constants import GBSERVER_GITHUB_TOKEN

        assert isinstance(GBSERVER_GITHUB_TOKEN, str)

    def test_api_base_path_is_string(self):
        from gbserver.types.constants import API_BASE_PATH

        assert isinstance(API_BASE_PATH, str)
        assert API_BASE_PATH.startswith("/")


# ---------------------------------------------------------------------------
# g) Environment config regression
# ---------------------------------------------------------------------------


class TestEnvironmentConfig:
    """Verify all environment configs load without error."""

    def test_dev_config(self):
        from gbserver.types.gbserverenvconfig import gb_environment_config

        config = gb_environment_config("DEV")
        assert config.env == "DEV"

    def test_staging_config(self):
        from gbserver.types.gbserverenvconfig import gb_environment_config

        config = gb_environment_config("STAGING")
        assert config.env == "STAGING"

    def test_prod_config(self):
        from gbserver.types.gbserverenvconfig import gb_environment_config

        config = gb_environment_config("PROD")
        assert config.env == "PROD"

    def test_standalone_config(self):
        from gbserver.types.gbserverenvconfig import gb_environment_config

        config = gb_environment_config("STANDALONE")
        assert config.env == "STANDALONE"
        assert config.default_sql_schema == "standalone"
        assert config.default_pod_namespace == "default"
        assert config.lakehouse_environment == ""
        assert config.dmf_instance == ""


# ---------------------------------------------------------------------------
# h) Standalone environment defaults
# ---------------------------------------------------------------------------


class TestStandaloneEnvironmentDefaults:
    """Verify GB_ENVIRONMENT=STANDALONE sets expected env-var defaults."""

    def test_standalone_env_sets_defaults(self, monkeypatch):
        """When GB_ENVIRONMENT=STANDALONE, env defaults are applied."""
        import importlib

        from gbserver.types import constants

        # Clear any existing values so setdefault takes effect
        for key in [
            "GBSERVER_METADATA_STORAGE",
            "GBSERVER_DEFAULT_BUILDRUNNER_TYPE",
            "GBSERVER_PROCEED_WITHOUT_SECRETS",
        ]:
            monkeypatch.delenv(key, raising=False)

        monkeypatch.setenv("GB_ENVIRONMENT", "STANDALONE")
        importlib.reload(constants)

        try:
            import os

            assert os.environ.get("GBSERVER_METADATA_STORAGE") == "sqlite"
            assert os.environ.get("GBSERVER_DEFAULT_BUILDRUNNER_TYPE") == "thread"
            assert os.environ.get("GBSERVER_PROCEED_WITHOUT_SECRETS") == "true"
            assert constants.GB_ENVIRONMENT == "STANDALONE"
            assert constants.GBSERVER_PROCEED_WITHOUT_SECRETS is True
        finally:
            # Restore original module state
            importlib.reload(constants)

    def test_standalone_env_preserves_explicit_overrides(self, monkeypatch):
        """Explicit env vars are not overwritten by standalone defaults."""
        import importlib

        from gbserver.types import constants

        monkeypatch.setenv("GB_ENVIRONMENT", "STANDALONE")
        monkeypatch.setenv("GBSERVER_METADATA_STORAGE", "sql")

        # Clear others so setdefault takes effect on them
        monkeypatch.delenv("GBSERVER_DEFAULT_BUILDRUNNER_TYPE", raising=False)
        monkeypatch.delenv("GBSERVER_PROCEED_WITHOUT_SECRETS", raising=False)

        importlib.reload(constants)

        try:
            import os

            # Explicit override preserved
            assert os.environ.get("GBSERVER_METADATA_STORAGE") == "sql"
            # Defaults still applied where not overridden
            assert os.environ.get("GBSERVER_DEFAULT_BUILDRUNNER_TYPE") == "thread"
            assert os.environ.get("GBSERVER_PROCEED_WITHOUT_SECRETS") == "true"
        finally:
            importlib.reload(constants)
