"""Tests for the canonical in-repo space configuration files.

Validates the structure of `configurations/assets/` (the leaf primitives
directory referenced via base_uris) and
`configurations/spaces/local/` (the user-facing standalone space
consumed by `gbserver standalone` and the build tests).
"""

import pathlib

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
CONFIGURATIONS_DIR = REPO_ROOT / "configurations"
ASSETS_DIR = CONFIGURATIONS_DIR / "assets"
LOCAL_SPACE_DIR = CONFIGURATIONS_DIR / "spaces" / "local"


class TestAssetsLayout:
    """Validate the configurations/assets/ tree — referenced via base_uris from
    spaces.  No space.yaml lives here; this is purely the primitives target."""

    def test_assetstores_dir_exists(self):
        assert (ASSETS_DIR / "assetstores").is_dir()

    def test_environments_dir_exists(self):
        assert (ASSETS_DIR / "environments").is_dir()

    def test_steps_dir_exists(self):
        assert (ASSETS_DIR / "steps").is_dir()

    def test_no_space_yaml_at_root(self):
        # assets/ is a base_uris target, not a space — there should be no
        # space.yaml claiming this directory as a standalone space.
        assert not (ASSETS_DIR / "space.yaml").exists()


class TestStandalonePublicSpaceYaml:
    """Validate configurations/spaces/local/space.yaml — the
    user-facing space consumed by `gbserver standalone` and the build tests."""

    def test_file_exists(self):
        assert (LOCAL_SPACE_DIR / "space.yaml").exists()

    def test_yaml_loads(self):
        with open(LOCAL_SPACE_DIR / "space.yaml") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)

    def test_name(self):
        with open(LOCAL_SPACE_DIR / "space.yaml") as f:
            data = yaml.safe_load(f)
        assert data["name"] == "public"

    def test_base_uris_reference_assets(self):
        with open(LOCAL_SPACE_DIR / "space.yaml") as f:
            data = yaml.safe_load(f)
        assert any("assets" in uri for uri in data["base_uris"])

    def test_secret_manager(self):
        with open(LOCAL_SPACE_DIR / "space.yaml") as f:
            data = yaml.safe_load(f)
        assert data["secret_manager"]["type"] == "env"

    def test_default_environment_variable(self):
        with open(LOCAL_SPACE_DIR / "space.yaml") as f:
            data = yaml.safe_load(f)
        assert data["variables"]["DEFAULT_ENVIRONMENT"] == "skypilot/kubernetes"


class TestSkyKubeEnvironmentYaml:
    """Validate configurations/assets/environments/skypilot/kubernetes/environment.yaml."""

    ENV_PATH = (
        ASSETS_DIR / "environments" / "skypilot" / "kubernetes" / "environment.yaml"
    )

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
    """Validate configurations/assets/environments/skypilot-managed/kubernetes/environment.yaml."""

    ENV_PATH = (
        ASSETS_DIR
        / "environments"
        / "skypilot-managed"
        / "kubernetes"
        / "environment.yaml"
    )

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
