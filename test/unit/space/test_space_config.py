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


class TestMergedQuickstartAssets:
    """Validate the assetstores, environments, and co-located steps merged in
    from the former standalone-quickstart sample.  These are what the sample's
    build.yaml resolves to through the configurations/spaces/local base_uris
    chain, so the quickstart can run without carrying its own copies."""

    ENVS_DIR = ASSETS_DIR / "environments"
    STORES_DIR = ASSETS_DIR / "assetstores"

    @staticmethod
    def _load(path):
        with open(path) as f:
            return yaml.safe_load(f)

    def test_local_store(self):
        data = self._load(self.STORES_DIR / "local" / "store.yaml")
        assert data["base_uri"] == "file:"

    def test_s3_store(self):
        data = self._load(self.STORES_DIR / "s3" / "store.yaml")
        assert data["base_uri"] == "s3://"
        assert "cos_access_key_id_secret_name" in data["config"]

    def test_runpod_environment(self):
        data = self._load(self.ENVS_DIR / "runpod" / "environment.yaml")
        assert data["type"] == "Runpod"
        assert any("s3" in s["store_uri"] for s in data["assetstores"])

    def test_skypilot_aws_environment(self):
        data = self._load(self.ENVS_DIR / "skypilot" / "aws" / "environment.yaml")
        assert data["type"] == "Skypilot"

    def test_bash_env_binds_local_and_hf(self):
        data = self._load(self.ENVS_DIR / "bash" / "environment.yaml")
        uris = {s["store_uri"] for s in data["assetstores"]}
        assert any("local" in u for u in uris)
        assert any("hf" in u for u in uris)

    def test_docker_env_binds_local(self):
        data = self._load(self.ENVS_DIR / "docker" / "environment.yaml")
        uris = {s["store_uri"] for s in data["assetstores"]}
        assert any("local" in u for u in uris)

    def test_colocated_hello_steps_exist(self):
        # bash/docker/runpod get co-located hello steps; the single Skypilot
        # hello (under skypilot/aws) resolves for other Skypilot envs via
        # env-class match.
        for env in ("bash", "docker", "runpod"):
            assert (self.ENVS_DIR / env / "steps" / "hello" / "step.yaml").exists()
        assert (
            self.ENVS_DIR / "skypilot" / "aws" / "steps" / "hello" / "step.yaml"
        ).exists()

    def test_bash_hello_step_has_command_script(self):
        script = (
            self.ENVS_DIR
            / "bash"
            / "steps"
            / "hello"
            / "bash_scripts"
            / "hello"
            / "command.sh"
        )
        assert script.exists()
