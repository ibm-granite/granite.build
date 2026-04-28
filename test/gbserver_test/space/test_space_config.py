"""Tests for space configuration and environment YAML files (Issue #23)."""

import pathlib

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SPACE_DIR = REPO_ROOT / "space"


@pytest.mark.g4os
@pytest.mark.unit
class TestSpaceYaml:
    """Validate space/space.yaml."""

    def test_file_exists(self):
        assert (SPACE_DIR / "space.yaml").exists()

    def test_yaml_loads(self):
        with open(SPACE_DIR / "space.yaml") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)

    def test_name(self):
        with open(SPACE_DIR / "space.yaml") as f:
            data = yaml.safe_load(f)
        assert data["name"] == "standalone"

    def test_base_uris(self):
        with open(SPACE_DIR / "space.yaml") as f:
            data = yaml.safe_load(f)
        assert "file://." in data["base_uris"]
        assert "file://../assets" in data["base_uris"]

    def test_secret_manager(self):
        with open(SPACE_DIR / "space.yaml") as f:
            data = yaml.safe_load(f)
        assert data["secret_manager"]["type"] == "env"

    def test_default_environment_variable(self):
        with open(SPACE_DIR / "space.yaml") as f:
            data = yaml.safe_load(f)
        assert data["variables"]["DEFAULT_ENVIRONMENT"] == "skypilot"


@pytest.mark.g4os
@pytest.mark.unit
class TestSkypilotEnvironmentYaml:
    """Validate space/environments/skypilot/environment.yaml."""

    def test_file_exists(self):
        assert (SPACE_DIR / "environments" / "skypilot" / "environment.yaml").exists()

    def test_yaml_loads(self):
        with open(SPACE_DIR / "environments" / "skypilot" / "environment.yaml") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)

    def test_type_is_skypilot(self):
        with open(SPACE_DIR / "environments" / "skypilot" / "environment.yaml") as f:
            data = yaml.safe_load(f)
        assert data["type"] == "Skypilot"

    def test_default_cloud(self):
        with open(SPACE_DIR / "environments" / "skypilot" / "environment.yaml") as f:
            data = yaml.safe_load(f)
        assert data["config"]["default_cloud"] == "kubernetes"

    def test_assetstores_env_local(self):
        with open(SPACE_DIR / "environments" / "skypilot" / "environment.yaml") as f:
            data = yaml.safe_load(f)
        store = data["assetstores"][0]
        assert store["store_uri"] == "space://assetstores/env-local"
        assert store["load"][0]["mode"] == "env_local"
        assert store["push"][0]["mode"] == "env_local"


@pytest.mark.g4os
@pytest.mark.unit
class TestSkypilotManagedEnvironmentYaml:
    """Validate space/environments/skypilot-managed/environment.yaml."""

    def test_file_exists(self):
        assert (
            SPACE_DIR / "environments" / "skypilot-managed" / "environment.yaml"
        ).exists()

    def test_yaml_loads(self):
        with open(
            SPACE_DIR / "environments" / "skypilot-managed" / "environment.yaml"
        ) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)

    def test_type_is_skypilot_managed(self):
        with open(
            SPACE_DIR / "environments" / "skypilot-managed" / "environment.yaml"
        ) as f:
            data = yaml.safe_load(f)
        assert data["type"] == "Skypilot_managed"

    def test_default_cloud(self):
        with open(
            SPACE_DIR / "environments" / "skypilot-managed" / "environment.yaml"
        ) as f:
            data = yaml.safe_load(f)
        assert data["config"]["default_cloud"] == "kubernetes"

    def test_assetstores_env_local(self):
        with open(
            SPACE_DIR / "environments" / "skypilot-managed" / "environment.yaml"
        ) as f:
            data = yaml.safe_load(f)
        store = data["assetstores"][0]
        assert store["store_uri"] == "space://assetstores/env-local"
        assert store["load"][0]["mode"] == "env_local"
        assert store["push"][0]["mode"] == "env_local"
