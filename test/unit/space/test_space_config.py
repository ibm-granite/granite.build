"""Tests for the canonical in-repo space configuration files.

Validates the structure of `configurations/assets/` (the leaf primitives
directory referenced via base_uris) and
`configurations/spaces/local/` (the user-facing standalone space
consumed by `gbserver standalone` and the build tests).
"""

import pathlib

import pytest
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
        # The standalone space uses the writable `local` backend so that
        # `gb secret create` works (the `env` backend is read-only). No
        # secrets_dir is configured, so LocalSpaceSecretManager defaults it
        # to <gb_home>/space_secrets.
        assert data["secret_manager"]["type"] == "local"
        assert data["secret_manager"]["config"] == {}

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

    def test_env_local_not_declared(self):
        """env:// is registered implicitly for every environment
        (Environment._register_default_envstore), so it must NOT be declared
        explicitly here — guards against re-adding the redundant entry."""
        with open(self.ENV_PATH) as f:
            data = yaml.safe_load(f)
        uris = {s["store_uri"] for s in (data.get("assetstores") or [])}
        assert "space://assetstores/env-local" not in uris

    def test_hf_pull_is_not_inline(self):
        """Inline pulls are being deprecated, so the generic env keeps main's separate
        hfpull step; hf inputs need the shared storage its commented block documents
        (skypilot/kubernetes/local is the working example)."""
        with open(self.ENV_PATH) as f:
            data = yaml.safe_load(f)
        hf = [s for s in data["assetstores"] if s["store_uri"].endswith("/hf")]
        assert len(hf) == 1
        assert not any(p["config"].get("inline") for p in hf[0]["pull"])

    def test_no_active_pod_config(self):
        """Active PVC wiring here would break current users: pods sit Pending when
        the claim is absent, and a top-level pod_config list would replace one the
        user keeps in ~/.sky/config.yaml. It ships commented out instead."""
        with open(self.ENV_PATH) as f:
            data = yaml.safe_load(f)
        assert "cloud_config" not in data["config"]
        assert "shared_workdir" not in data["config"]

    def test_idle_window_outlasts_the_default_poll_interval(self):
        """Autodown on kube: the window must exceed the 300s default poll."""
        with open(self.ENV_PATH) as f:
            data = yaml.safe_load(f)
        assert data["config"]["idle_minutes_to_autostop"] * 60 > 2 * 300


class TestSkyKubeLocalEnvironmentYaml:
    """Validate configurations/assets/environments/skypilot/kubernetes/local/environment.yaml."""

    ENV_PATH = (
        ASSETS_DIR
        / "environments"
        / "skypilot"
        / "kubernetes"
        / "local"
        / "environment.yaml"
    )

    def _load(self):
        with open(self.ENV_PATH) as f:
            return yaml.safe_load(f)

    def test_is_a_kubernetes_subtype(self):
        data = self._load()
        assert data["type"] == "Skypilot"
        assert data["subtype"] == "kubernetes"
        assert data["config"]["default_cloud"] == "kubernetes"

    def test_pins_the_kind_context(self):
        """allowed_contexts makes SkyPilot ignore the CURRENT kube context."""
        kube = self._load()["config"]["cloud_config"]["kubernetes"]
        assert kube["allowed_contexts"] == ["kind-skypilot"]

    def test_shared_volume_is_scoped_to_the_kind_context(self):
        """Per-context pod_config, so it can never replace a user's global one."""
        kube = self._load()["config"]["cloud_config"]["kubernetes"]
        assert "pod_config" not in kube
        spec = kube["context_configs"]["kind-skypilot"]["pod_config"]["spec"]
        claims = [v["persistentVolumeClaim"]["claimName"] for v in spec["volumes"]]
        mounts = [m["mountPath"] for m in spec["containers"][0]["volumeMounts"]]
        assert claims == ["gb-shared"]
        assert mounts == ["/shared"]

    def test_pods_use_images_preloaded_into_kind(self):
        """SkyPilot's own default is Always."""
        kube = self._load()["config"]["cloud_config"]["kubernetes"]
        spec = kube["context_configs"]["kind-skypilot"]["pod_config"]["spec"]
        (container,) = spec["containers"]
        assert container["name"] == "ray-node"
        assert container["imagePullPolicy"] == "IfNotPresent"

    @pytest.mark.parametrize("step", ["hfpull", "hfpush"])
    def test_hf_steps_request_less_than_their_10gi_default(self, step):
        """config.steps launcher resources must outrank the step.yaml 10Gi floor."""
        from gbserver.build.targetstep import _seed_step_config_with_env_defaults
        from gbserver.utils.filesystem import merge_dicts

        step_yaml = (
            REPO_ROOT / "src/gbserver/builtins/steps/skypilot" / step / "step.yaml"
        )
        step_config = yaml.safe_load(step_yaml.read_text())["config"]
        assert step_config["compute_config"]["total_memory_per_node"] == "10Gi"
        merged = merge_dicts(
            _seed_step_config_with_env_defaults(
                self._load()["config"], f"space://steps/{step}", {}
            ),
            step_config,
        )
        assert merged["launcher_config"]["resources"]["memory"] == "1+"

    def test_shared_workdir_and_hf_cache_live_on_the_volume(self):
        data = self._load()
        assert data["config"]["shared_workdir"] == "/shared"
        hf = [s for s in data["assetstores"] if s["store_uri"].endswith("/hf")]
        assert len(hf) == 1
        for pull in hf[0]["pull"]:
            assert not pull["config"].get("inline")
            assert pull["config"]["cache_path"].startswith("/shared/")


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

    def test_env_local_not_declared(self):
        """env:// is registered implicitly for every environment
        (Environment._register_default_envstore), so it must NOT be declared
        explicitly here — guards against re-adding the redundant entry."""
        with open(self.ENV_PATH) as f:
            data = yaml.safe_load(f)
        uris = {s["store_uri"] for s in (data.get("assetstores") or [])}
        assert "space://assetstores/env-local" not in uris


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

    def test_no_local_store(self):
        # The redundant `local` file store was removed; file: URIs are served by
        # the builtin `file` store, declared explicitly as space://assetstores/file
        # in each supporting environment.yaml.
        assert not (self.STORES_DIR / "local").exists()

    def test_builtin_file_store(self):
        # The builtin file store that replaces the removed `local` store.
        builtin = (
            REPO_ROOT
            / "src"
            / "gbserver"
            / "builtins"
            / "assetstores"
            / "file"
            / "store.yaml"
        )
        data = self._load(builtin)
        assert data["name"] == "file"
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

    @staticmethod
    def _store_entry(data, scheme):
        """Return the assetstores entry whose store_uri contains ``scheme``."""
        for s in data["assetstores"]:
            if scheme in s["store_uri"]:
                return s
        return None

    def test_bash_env_binds_hf_and_file(self):
        # The old `local` store was removed; file: is now the builtin `file`
        # store, declared explicitly. Bash implements pull+push, so pull+push.
        data = self._load(self.ENVS_DIR / "bash" / "environment.yaml")
        uris = {s["store_uri"] for s in data["assetstores"]}
        assert any("hf" in u for u in uris)
        assert not any("local" in u for u in uris)
        file_entry = self._store_entry(data, "assetstores/file")
        assert file_entry is not None, "bash must declare the file store"
        assert file_entry.get("pull"), "bash file store must declare pull"
        assert file_entry.get("push"), "bash file store must declare push"

    def test_docker_env_binds_hf_and_file_push_only(self):
        # Docker has pushasset_filestore but NO pullasset_filestore, so the file
        # store must declare push only — declaring pull would advertise a pull it
        # cannot service.
        data = self._load(self.ENVS_DIR / "docker" / "environment.yaml")
        uris = {s["store_uri"] for s in data["assetstores"]}
        assert any("hf" in u for u in uris)
        assert not any("local" in u for u in uris)
        file_entry = self._store_entry(data, "assetstores/file")
        assert file_entry is not None, "docker must declare the file store"
        assert file_entry.get("push"), "docker file store must declare push"
        assert not file_entry.get("pull"), "docker file store must NOT declare pull"

    def test_colocated_hello_steps_exist(self):
        # bash/docker/runpod get co-located hello steps; the single
        # Skypilot hello (under skypilot/aws) resolves for other Skypilot envs
        # via env-class match.
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
