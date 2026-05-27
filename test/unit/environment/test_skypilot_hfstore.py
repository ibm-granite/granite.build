"""Unit tests for Skypilot.pullasset_hfstore and Skypilot.pushasset_hfstore."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gbcommon.uri.hf import HfURI
from gbserver.types.buildconfig import BuildTargetStepConfig


@pytest.fixture
def skypilot_env():
    """Create a minimal Skypilot environment instance for testing asset methods."""
    from gbserver.environment.skypilot import Skypilot
    from gbserver.types.environmentconfig import EnvironmentConfig

    event_q = asyncio.Queue()
    config = EnvironmentConfig(
        name="test-skypilot",
        type="Skypilot",
        config={"default_cloud": "k8s", "idle_minutes_to_autostop": 0},
    )
    return Skypilot(event_q=event_q, environment_config=config)


@pytest.fixture
def mock_hfuri():
    """Return a mock HfURI that passes isinstance checks."""
    uri = MagicMock(spec=HfURI)
    uri.get_owner.return_value = "myorg"
    uri.get_repo.return_value = "myrepo"
    uri.get_revision.return_value = "main"
    uri.get_hf_type.return_value = "model"
    uri.__str__ = lambda self: "hf://models/myorg/myrepo"
    return uri


def _hfstore_mock(token: str = "tok-abc"):
    """Return a Hfstore mock that resolves a token and passes isinstance checks."""
    from gbserver.asset.hfstore import Hfstore

    store = MagicMock(spec=Hfstore)
    store._resolve_token.return_value = token
    return store


class TestPullassetHfstore:
    @pytest.mark.asyncio
    async def test_returns_binding_and_step_config(self, skypilot_env, mock_hfuri):
        """pullasset_hfstore returns (binding_dict, BuildTargetStepConfig) with cache path."""
        assetstore = _hfstore_mock()
        storeload_config = MagicMock()
        storeload_config.mode = "hf_pull"
        storeload_config.config = {"cache_path": "/data/cache"}

        binding_config, step_config = await skypilot_env.pullasset_hfstore(
            uri=mock_hfuri,
            assetstore=assetstore,
            storeload_config=storeload_config,
        )

        expected_path = str(Path("/data/cache/myorg/myrepo/main"))
        assert binding_config == {"binding": {"path": expected_path}}
        assert isinstance(step_config, BuildTargetStepConfig)
        hfpull_config = step_config.config["hfpull_config"]
        assert hfpull_config["owner"] == "myorg"
        assert hfpull_config["repo"] == "myrepo"
        assert hfpull_config["revision"] == "main"
        assert hfpull_config["path"] == expected_path

    @pytest.mark.asyncio
    async def test_injects_hf_token_into_launcher_envs(self, skypilot_env, mock_hfuri):
        """pullasset_hfstore puts HF_TOKEN under config.launcher_config.envs."""
        assetstore = _hfstore_mock(token="my-secret-token")
        storeload_config = MagicMock()
        storeload_config.mode = "hf_pull"
        storeload_config.config = {"cache_path": "/data/cache"}

        _, step_config = await skypilot_env.pullasset_hfstore(
            uri=mock_hfuri,
            assetstore=assetstore,
            storeload_config=storeload_config,
        )

        envs = step_config.config["launcher_config"]["envs"]
        assert envs["HF_TOKEN"] == "my-secret-token"

    @pytest.mark.asyncio
    async def test_falls_back_to_default_cache_dir_when_no_cache_path(
        self, skypilot_env, mock_hfuri
    ):
        """No cache_path -> uses get_hf_cache_dir default (~/.cache/gbserver/hf)."""
        assetstore = _hfstore_mock()
        storeload_config = MagicMock()
        storeload_config.mode = "hf_pull"
        storeload_config.config = {}

        binding_config, _ = await skypilot_env.pullasset_hfstore(
            uri=mock_hfuri,
            assetstore=assetstore,
            storeload_config=storeload_config,
        )

        path = binding_config["binding"]["path"]
        assert path.endswith(str(Path(".cache/gbserver/hf/myorg/myrepo/main")))

    @pytest.mark.asyncio
    async def test_step_uri_override(self, skypilot_env, mock_hfuri):
        """storeload_config.config.step_uri overrides the default builtin uri."""
        assetstore = _hfstore_mock()
        storeload_config = MagicMock()
        storeload_config.mode = "hf_pull"
        storeload_config.config = {
            "cache_path": "/data/cache",
            "step_uri": "file:///custom/hfpull",
        }

        _, step_config = await skypilot_env.pullasset_hfstore(
            uri=mock_hfuri,
            assetstore=assetstore,
            storeload_config=storeload_config,
        )

        assert step_config.step_uri == "file:///custom/hfpull"

    @pytest.mark.asyncio
    async def test_default_step_uri_points_to_builtin_hfpull(
        self, skypilot_env, mock_hfuri
    ):
        """Default step_uri resolves to the builtin hfpull dir."""
        from gbserver.types.constants import CODE_GBSERVER_BUILTINS_STEPS_HFPULL_URI

        assetstore = _hfstore_mock()
        storeload_config = MagicMock()
        storeload_config.mode = "hf_pull"
        storeload_config.config = {"cache_path": "/data/cache"}

        _, step_config = await skypilot_env.pullasset_hfstore(
            uri=mock_hfuri,
            assetstore=assetstore,
            storeload_config=storeload_config,
        )

        assert step_config.step_uri == CODE_GBSERVER_BUILTINS_STEPS_HFPULL_URI

    @pytest.mark.asyncio
    async def test_rejects_wrong_assetstore_type(self, skypilot_env, mock_hfuri):
        """pullasset_hfstore raises AssertionError if assetstore is not Hfstore."""
        storeload_config = MagicMock()
        storeload_config.mode = "hf_pull"
        storeload_config.config = {"cache_path": "/data/cache"}

        with pytest.raises(AssertionError, match="expected 'Hfstore'"):
            await skypilot_env.pullasset_hfstore(
                uri=mock_hfuri,
                assetstore=MagicMock(),
                storeload_config=storeload_config,
            )

    @pytest.mark.asyncio
    async def test_rejects_wrong_mode(self, skypilot_env, mock_hfuri):
        """pullasset_hfstore raises ValueError for non-hf_pull mode."""
        assetstore = _hfstore_mock()
        storeload_config = MagicMock()
        storeload_config.mode = "dmf_pull"
        storeload_config.config = {"cache_path": "/data/cache"}

        with pytest.raises(ValueError, match="unsupported storeload mode"):
            await skypilot_env.pullasset_hfstore(
                uri=mock_hfuri,
                assetstore=assetstore,
                storeload_config=storeload_config,
            )

    @pytest.mark.asyncio
    async def test_uses_env_shared_workdir_when_no_cache_path(self, mock_hfuri):
        """No cache_path on the store -> falls back to {shared_workdir}/hf_cache."""
        from gbserver.environment.skypilot import Skypilot
        from gbserver.types.environmentconfig import EnvironmentConfig

        env = Skypilot(
            event_q=asyncio.Queue(),
            environment_config=EnvironmentConfig(
                name="test-skypilot",
                type="Skypilot",
                config={
                    "default_cloud": "slurm",
                    "idle_minutes_to_autostop": 0,
                    "shared_workdir": "/shared",
                },
            ),
        )
        assetstore = _hfstore_mock()
        storeload_config = MagicMock()
        storeload_config.mode = "hf_pull"
        storeload_config.config = {}

        binding_config, _ = await env.pullasset_hfstore(
            uri=mock_hfuri,
            assetstore=assetstore,
            storeload_config=storeload_config,
        )

        assert binding_config["binding"]["path"] == str(
            Path("/shared/hf_cache/myorg/myrepo/main")
        )

    @pytest.mark.asyncio
    async def test_explicit_cache_path_overrides_shared_workdir(self, mock_hfuri):
        """Per-store cache_path wins over env shared_workdir when both are set."""
        from gbserver.environment.skypilot import Skypilot
        from gbserver.types.environmentconfig import EnvironmentConfig

        env = Skypilot(
            event_q=asyncio.Queue(),
            environment_config=EnvironmentConfig(
                name="test-skypilot",
                type="Skypilot",
                config={
                    "default_cloud": "slurm",
                    "idle_minutes_to_autostop": 0,
                    "shared_workdir": "/shared",
                },
            ),
        )
        assetstore = _hfstore_mock()
        storeload_config = MagicMock()
        storeload_config.mode = "hf_pull"
        storeload_config.config = {"cache_path": "/explicit/override"}

        binding_config, _ = await env.pullasset_hfstore(
            uri=mock_hfuri,
            assetstore=assetstore,
            storeload_config=storeload_config,
        )

        assert binding_config["binding"]["path"] == str(
            Path("/explicit/override/myorg/myrepo/main")
        )


class TestGetHfCacheDir:
    """Unit tests for the three-rung cache-path resolution chain."""

    def test_explicit_cache_path_wins(self):
        from gbserver.environment.local_assets import get_hf_cache_dir

        cfg = MagicMock()
        cfg.config = {"cache_path": "/explicit"}
        assert get_hf_cache_dir(cfg, default_workdir="/shared") == "/explicit"

    def test_default_workdir_used_when_no_cache_path(self):
        from gbserver.environment.local_assets import get_hf_cache_dir

        cfg = MagicMock()
        cfg.config = {}
        assert get_hf_cache_dir(cfg, default_workdir="/shared") == "/shared/hf_cache"

    def test_falls_back_to_home_cache_when_neither_set(self):
        from gbserver.environment.local_assets import get_hf_cache_dir

        cfg = MagicMock()
        cfg.config = {}
        result = get_hf_cache_dir(cfg)
        assert result.endswith(str(Path(".cache/gbserver/hf")))

    def test_none_storeload_config_uses_default_workdir(self):
        from gbserver.environment.local_assets import get_hf_cache_dir

        assert get_hf_cache_dir(None, default_workdir="/shared") == "/shared/hf_cache"


class TestPushassetHfstore:
    @pytest.mark.asyncio
    async def test_returns_build_target_step_config(self, skypilot_env, mock_hfuri):
        """pushasset_hfstore returns BuildTargetStepConfig with hfpush_config."""
        assetstore = _hfstore_mock()
        mock_hfuri.resolve_resource_group_id.return_value = None

        step_config = await skypilot_env.pushasset_hfstore(
            binding={"path": "/workspace/output/model"},
            binding_id="output_model",
            uri=mock_hfuri,
            assetstore=assetstore,
        )

        assert isinstance(step_config, BuildTargetStepConfig)
        hfpush_config = step_config.config["hfpush_config"]
        assert hfpush_config["path"] == "/workspace/output/model"
        assert hfpush_config["binding_id"] == "output_model"
        assert hfpush_config["owner"] == "myorg"
        assert hfpush_config["repo"] == "myrepo"
        assert hfpush_config["private"] is True

    @pytest.mark.asyncio
    async def test_injects_hf_token_into_launcher_envs(self, skypilot_env, mock_hfuri):
        """pushasset_hfstore puts HF_TOKEN under config.launcher_config.envs."""
        assetstore = _hfstore_mock(token="my-push-token")
        mock_hfuri.resolve_resource_group_id.return_value = None

        step_config = await skypilot_env.pushasset_hfstore(
            binding={"path": "/workspace/output"},
            binding_id="bid",
            uri=mock_hfuri,
            assetstore=assetstore,
        )

        envs = step_config.config["launcher_config"]["envs"]
        assert envs["HF_TOKEN"] == "my-push-token"

    @pytest.mark.asyncio
    async def test_raises_on_empty_uri(self, skypilot_env):
        """pushasset_hfstore raises ValueError for empty uri."""
        with pytest.raises(ValueError, match="Empty uri"):
            await skypilot_env.pushasset_hfstore(
                binding={"path": "/workspace/output"},
                uri=None,
            )

    @pytest.mark.asyncio
    async def test_raises_on_missing_path_in_binding(self, skypilot_env, mock_hfuri):
        """pushasset_hfstore raises AssertionError if binding lacks 'path'."""
        with pytest.raises(AssertionError, match="expected 'path'"):
            await skypilot_env.pushasset_hfstore(
                binding={},
                uri=mock_hfuri,
            )

    @pytest.mark.asyncio
    async def test_private_flag_from_output_config(self, skypilot_env, mock_hfuri):
        """pushasset_hfstore picks up private=False from output_config.store_push."""
        assetstore = _hfstore_mock()
        mock_hfuri.resolve_resource_group_id.return_value = None

        output_config = MagicMock()
        output_config.space_name = None
        output_config.store_push = MagicMock()
        output_config.store_push.config = {"hf": {"private": False}}

        step_config = await skypilot_env.pushasset_hfstore(
            binding={"path": "/workspace/output/model"},
            binding_id="bid",
            uri=mock_hfuri,
            assetstore=assetstore,
            output_config=output_config,
        )

        assert step_config.config["hfpush_config"]["private"] is False

    @pytest.mark.asyncio
    async def test_resource_group_id_from_output_config(self, skypilot_env, mock_hfuri):
        """Explicit resource_group_id from output_config skips hfuri.resolve_resource_group_id."""
        assetstore = _hfstore_mock()
        mock_hfuri.resolve_resource_group_id.side_effect = AssertionError(
            "should not be called"
        )

        output_config = MagicMock()
        output_config.space_name = None
        output_config.store_push = MagicMock()
        output_config.store_push.config = {
            "hf": {"resource_group_id": "rg-explicit-123"}
        }

        step_config = await skypilot_env.pushasset_hfstore(
            binding={"path": "/workspace/output/model"},
            binding_id="bid",
            uri=mock_hfuri,
            assetstore=assetstore,
            output_config=output_config,
        )

        hfpush_config = step_config.config["hfpush_config"]
        assert hfpush_config["hf"]["resource_group_id"] == "rg-explicit-123"

    @pytest.mark.asyncio
    async def test_step_uri_override(self, skypilot_env, mock_hfuri):
        """storepush_config.config.step_uri overrides the default builtin uri."""
        assetstore = _hfstore_mock()
        mock_hfuri.resolve_resource_group_id.return_value = None

        storepush_config = MagicMock()
        storepush_config.config = {"step_uri": "file:///custom/hfpush"}

        step_config = await skypilot_env.pushasset_hfstore(
            binding={"path": "/workspace/output/model"},
            binding_id="bid",
            uri=mock_hfuri,
            assetstore=assetstore,
            storepush_config=storepush_config,
        )

        assert step_config.step_uri == "file:///custom/hfpush"

    @pytest.mark.asyncio
    async def test_default_step_uri_points_to_builtin_hfpush(
        self, skypilot_env, mock_hfuri
    ):
        """Default step_uri resolves to the builtin hfpush dir."""
        from gbserver.types.constants import CODE_GBSERVER_BUILTINS_STEPS_HFPUSH_URI

        assetstore = _hfstore_mock()
        mock_hfuri.resolve_resource_group_id.return_value = None

        step_config = await skypilot_env.pushasset_hfstore(
            binding={"path": "/workspace/output/model"},
            binding_id="bid",
            uri=mock_hfuri,
            assetstore=assetstore,
        )

        assert step_config.step_uri == CODE_GBSERVER_BUILTINS_STEPS_HFPUSH_URI
