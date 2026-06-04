"""Tests for the canonical in-repo space configuration files.

Validates the structure of `spaces/shared/` (the leaf primitives space) and
`spaces/standalone/public/` (the user-facing standalone space).
"""

import pathlib

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SPACES_DIR = REPO_ROOT / "spaces"
SHARED_DIR = SPACES_DIR / "shared"
STANDALONE_PUBLIC_DIR = SPACES_DIR / "standalone" / "public"


class TestSharedSpaceYaml:
    """Validate spaces/shared/space.yaml — the leaf primitives space referenced
    via base_uris from other spaces."""

    def test_file_exists(self):
        assert (SHARED_DIR / "space.yaml").exists()

    def test_yaml_loads(self):
        with open(SHARED_DIR / "space.yaml") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)

    def test_name(self):
        with open(SHARED_DIR / "space.yaml") as f:
            data = yaml.safe_load(f)
        assert data["name"] == "shared"

    def test_secret_manager(self):
        with open(SHARED_DIR / "space.yaml") as f:
            data = yaml.safe_load(f)
        assert data["secret_manager"]["type"] == "env"


class TestStandalonePublicSpaceYaml:
    """Validate spaces/standalone/public/space.yaml — the user-facing space
    consumed by `gbserver standalone` and the build tests."""

    def test_file_exists(self):
        assert (STANDALONE_PUBLIC_DIR / "space.yaml").exists()

    def test_yaml_loads(self):
        with open(STANDALONE_PUBLIC_DIR / "space.yaml") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)

    def test_name(self):
        with open(STANDALONE_PUBLIC_DIR / "space.yaml") as f:
            data = yaml.safe_load(f)
        assert data["name"] == "public"

    def test_base_uris_reference_shared(self):
        with open(STANDALONE_PUBLIC_DIR / "space.yaml") as f:
            data = yaml.safe_load(f)
        assert any("shared" in uri for uri in data["base_uris"])

    def test_secret_manager(self):
        with open(STANDALONE_PUBLIC_DIR / "space.yaml") as f:
            data = yaml.safe_load(f)
        assert data["secret_manager"]["type"] == "env"

    def test_default_environment_variable(self):
        with open(STANDALONE_PUBLIC_DIR / "space.yaml") as f:
            data = yaml.safe_load(f)
        assert data["variables"]["DEFAULT_ENVIRONMENT"] == "skypilot"


class TestSkypilotEnvironmentYaml:
    """Validate spaces/shared/environments/skypilot/environment.yaml."""

    ENV_PATH = SHARED_DIR / "environments" / "skypilot" / "environment.yaml"

    def test_file_exists(self):
        assert self.ENV_PATH.exists()

    def test_yaml_loads(self):
        with open(self.ENV_PATH) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)

    def test_type_is_skypilot(self):
        with open(self.ENV_PATH) as f:
            data = yaml.safe_load(f)
        assert data["type"] == "Skypilot"

    def test_default_cloud(self):
        with open(self.ENV_PATH) as f:
            data = yaml.safe_load(f)
        assert data["config"]["default_cloud"] == "kubernetes"

    def test_assetstores_env_local(self):
        with open(self.ENV_PATH) as f:
            data = yaml.safe_load(f)
        store = data["assetstores"][0]
        assert store["store_uri"] == "space://assetstores/env-local"
        assert store["load"][0]["mode"] == "env_local"
        assert store["push"][0]["mode"] == "env_local"


class TestSkypilotManagedEnvironmentYaml:
    """Validate spaces/shared/environments/skypilot-managed/environment.yaml."""

    ENV_PATH = SHARED_DIR / "environments" / "skypilot-managed" / "environment.yaml"

    def test_file_exists(self):
        assert self.ENV_PATH.exists()

    def test_yaml_loads(self):
        with open(self.ENV_PATH) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)

    def test_type_is_skypilot_managed(self):
        with open(self.ENV_PATH) as f:
            data = yaml.safe_load(f)
        assert data["type"] == "Skypilot_managed"

    def test_default_cloud(self):
        with open(self.ENV_PATH) as f:
            data = yaml.safe_load(f)
        assert data["config"]["default_cloud"] == "kubernetes"

    def test_assetstores_env_local(self):
        with open(self.ENV_PATH) as f:
            data = yaml.safe_load(f)
        store = data["assetstores"][0]
        assert store["store_uri"] == "space://assetstores/env-local"
        assert store["load"][0]["mode"] == "env_local"
        assert store["push"][0]["mode"] == "env_local"
