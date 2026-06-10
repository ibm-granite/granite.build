"""Unit tests for Lsf.pullasset_hfstore and Lsf.pushasset_hfstore."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gbcommon.uri.hf import HfURI
from gbserver.environment.environment import BINDING_KEY
from gbserver.types.buildconfig import BuildTargetStepConfig


@pytest.fixture
def lsf_env():
    """Create a minimal Lsf environment instance for testing asset methods."""
    from gbserver.environment.lsf import Lsf

    event_q = asyncio.Queue()
    env_config = MagicMock()
    env_config.config = {
        "workspace": {"local_dir": "/tmp/lsf_test", "remote_dir": "/remote/test"},
        "authentication": {"use_ssh": False, "login_nodes": []},
    }
    env_config.type = "Lsf"
    with patch(
        "gbserver.environment.environment.Environment.__init__", return_value=None
    ):
        lsf = Lsf(event_q=event_q, environment_config=env_config)
    return lsf


@pytest.fixture
def mock_hf_metadata():
    """Return metadata as Asset(uri=hfuri).get_metadata() would."""
    return {
        "uri": "hf://models/myorg/myrepo",
        "host": "huggingface.co",
        "owner": "myorg",
        "repo": "myrepo",
        "revision": "main",
        "hf_type": "model",
        "token_secretname": "HF_TOKEN",
    }


@pytest.fixture
def mock_hfuri():
    """Return a mock HfURI that passes isinstance checks."""
    uri = MagicMock(spec=HfURI)
    uri.get_owner.return_value = "myorg"
    uri.get_repo.return_value = "myrepo"
    uri.get_revision.return_value = "main"
    uri.hash.return_value = "abc123hash"
    uri.__str__ = lambda self: "hf://models/myorg/myrepo"
    return uri


class TestPullassetHfstore:
    @pytest.mark.asyncio
    async def test_returns_binding_config_with_path(
        self, lsf_env, mock_hfuri, mock_hf_metadata
    ):
        """pullasset_hfstore returns a binding_config with the expected cache path."""
        from gbserver.asset.hfstore import Hfstore

        assetstore = MagicMock(spec=Hfstore)
        storeload_config = MagicMock()
        storeload_config.mode = "hf_pull"
        storeload_config.config = {"cache_path": "/data/cache"}

        with (
            patch("gbserver.environment.lsf.Asset") as mock_asset_cls,
            patch.object(
                lsf_env, "_load_builtin_hf_lsf_section", return_value=({}, {})
            ),
        ):
            mock_asset_cls.return_value.get_metadata.return_value = mock_hf_metadata
            binding_config, step_config = await lsf_env.pullasset_hfstore(
                uri=mock_hfuri,
                assetstore=assetstore,
                storeload_config=storeload_config,
            )

        assert BINDING_KEY in binding_config
        expected_path = str(Path("/data/cache/myorg/myrepo/abc123hash"))
        assert binding_config[BINDING_KEY]["path"] == expected_path

    @pytest.mark.asyncio
    async def test_returns_build_target_step_config(
        self, lsf_env, mock_hfuri, mock_hf_metadata
    ):
        """pullasset_hfstore returns a BuildTargetStepConfig with hfpull_config."""
        from gbserver.asset.hfstore import Hfstore

        assetstore = MagicMock(spec=Hfstore)
        storeload_config = MagicMock()
        storeload_config.mode = "hf_pull"
        storeload_config.config = {"cache_path": "/data/cache"}

        with (
            patch("gbserver.environment.lsf.Asset") as mock_asset_cls,
            patch.object(
                lsf_env,
                "_load_builtin_hf_lsf_section",
                return_value=({"secrets": {}}, {"cwd": "."}),
            ),
        ):
            mock_asset_cls.return_value.get_metadata.return_value = mock_hf_metadata
            _, step_config = await lsf_env.pullasset_hfstore(
                uri=mock_hfuri,
                assetstore=assetstore,
                storeload_config=storeload_config,
            )

        assert isinstance(step_config, BuildTargetStepConfig)
        assert "hfpull_config" in step_config.config
        hfpull_config = step_config.config["hfpull_config"]
        assert hfpull_config["owner"] == "myorg"
        assert hfpull_config["repo"] == "myrepo"
        assert hfpull_config["revision"] == "main"
        assert "lsf" in step_config.config
        assert "workload" in step_config.config

    @pytest.mark.asyncio
    async def test_rejects_wrong_assetstore_type(self, lsf_env, mock_hfuri):
        """pullasset_hfstore raises AssertionError if assetstore is not Hfstore."""
        storeload_config = MagicMock()
        storeload_config.mode = "hf_pull"
        storeload_config.config = {"cache_path": "/data/cache"}

        with pytest.raises(AssertionError, match="expected 'Hfstore'"):
            await lsf_env.pullasset_hfstore(
                uri=mock_hfuri,
                assetstore=MagicMock(),
                storeload_config=storeload_config,
            )

    @pytest.mark.asyncio
    async def test_rejects_wrong_mode(self, lsf_env, mock_hfuri):
        """pullasset_hfstore raises AssertionError for non-hf_pull mode."""
        from gbserver.asset.hfstore import Hfstore

        assetstore = MagicMock(spec=Hfstore)
        storeload_config = MagicMock()
        storeload_config.mode = "dmf_pull"
        storeload_config.config = {"cache_path": "/data/cache"}

        with pytest.raises(AssertionError, match="Only 'hf_pull' mode"):
            await lsf_env.pullasset_hfstore(
                uri=mock_hfuri,
                assetstore=assetstore,
                storeload_config=storeload_config,
            )


class TestPushassetHfstore:
    @pytest.mark.asyncio
    async def test_returns_build_target_step_config(
        self, lsf_env, mock_hfuri, mock_hf_metadata
    ):
        """pushasset_hfstore returns BuildTargetStepConfig with hfpush_config."""
        from gbserver.asset.hfstore import Hfstore

        assetstore = MagicMock(spec=Hfstore)
        mock_hfuri.resolve_resource_group_id.return_value = None
        with (
            patch("gbserver.environment.lsf.Asset") as mock_asset_cls,
            patch.object(
                lsf_env,
                "_load_builtin_hf_lsf_section",
                return_value=({"secrets": {}}, {"cwd": "."}),
            ),
        ):
            mock_asset_cls.return_value.get_metadata.return_value = mock_hf_metadata
            step_config = await lsf_env.pushasset_hfstore(
                binding={"path": "/workspace/output/model"},
                binding_id="output_model",
                uri=mock_hfuri,
                assetstore=assetstore,
            )

        assert isinstance(step_config, BuildTargetStepConfig)
        assert "hfpush_config" in step_config.config
        hfpush_config = step_config.config["hfpush_config"]
        assert hfpush_config["path"] == "/workspace/output/model"
        assert hfpush_config["binding_id"] == "output_model"
        assert hfpush_config["owner"] == "myorg"
        assert hfpush_config["repo"] == "myrepo"
        assert hfpush_config["private"] is True
        assert "lsf" in step_config.config
        assert "workload" in step_config.config

    @pytest.mark.asyncio
    async def test_raises_on_empty_uri(self, lsf_env):
        """pushasset_hfstore raises ValueError for empty uri."""
        with pytest.raises(ValueError, match="Empty uri"):
            await lsf_env.pushasset_hfstore(
                binding={"path": "/workspace/output"},
                uri=None,
            )

    @pytest.mark.asyncio
    async def test_raises_on_missing_path_in_binding(
        self, lsf_env, mock_hfuri, mock_hf_metadata
    ):
        """pushasset_hfstore raises AssertionError if binding lacks 'path'."""
        with patch("gbserver.environment.lsf.Asset") as mock_asset_cls:
            mock_asset_cls.return_value.get_metadata.return_value = mock_hf_metadata
            with pytest.raises(AssertionError, match="expected 'path'"):
                await lsf_env.pushasset_hfstore(
                    binding={},
                    uri=mock_hfuri,
                )

    @pytest.mark.asyncio
    async def test_private_flag_from_storepush_config(
        self, lsf_env, mock_hfuri, mock_hf_metadata
    ):
        """pushasset_hfstore picks up private=False from storepush_config."""
        from gbserver.asset.hfstore import Hfstore

        assetstore = MagicMock(spec=Hfstore)
        mock_hfuri.resolve_resource_group_id.return_value = None
        storepush_config = MagicMock()
        storepush_config.config = {"hf": {"private": False}}

        with (
            patch("gbserver.environment.lsf.Asset") as mock_asset_cls,
            patch.object(
                lsf_env,
                "_load_builtin_hf_lsf_section",
                return_value=({"secrets": {}}, {"cwd": "."}),
            ),
        ):
            mock_asset_cls.return_value.get_metadata.return_value = mock_hf_metadata
            step_config = await lsf_env.pushasset_hfstore(
                binding={"path": "/workspace/output/model"},
                binding_id="output_model",
                uri=mock_hfuri,
                assetstore=assetstore,
                storepush_config=storepush_config,
            )

        assert step_config.config["hfpush_config"]["private"] is False


_HF_STEP_YAML = """\
name: hfpull
version: 1.0.0
type: upload
config:
  workload:
    cwd: "."
  compute_config:
    total_memory_per_node: 10Gi
environment_configs:
  Lsf:
    launchers:
      tuning:
        type: bsub
        monitors:
        - bsub_monitor
    monitors:
      bsub_monitor:
        type: bsub_monitor
"""


class TestLoadBuiltinHfLsfSection:
    def test_injects_hf_token_secret(self, lsf_env, tmp_path):
        """_load_builtin_hf_lsf_section injects HF_TOKEN into lsf_dict secrets."""
        step_file = tmp_path / "step.yaml"
        step_file.write_text(_HF_STEP_YAML)

        hf_metadata = {"token_secretname": "MY_HF_SECRET"}
        with patch.object(
            lsf_env, "_resolve_builtin_step_yaml", return_value=step_file
        ):
            lsf_dict, _ = lsf_env._load_builtin_hf_lsf_section("hfpull", hf_metadata)

        assert lsf_dict["skip_finding_output_artifacts"] is True
        secrets = lsf_dict["secrets"]["secret_names_to_use_as_env_variable"]
        assert len(secrets) == 1
        assert secrets[0]["env_name"] == "HF_TOKEN"
        assert secrets[0]["secret_name"] == "MY_HF_SECRET"

    def test_raises_on_missing_token_secretname(self, lsf_env, tmp_path):
        """_load_builtin_hf_lsf_section raises if token_secretname missing."""
        step_file = tmp_path / "step.yaml"
        step_file.write_text(_HF_STEP_YAML)

        with patch.object(
            lsf_env, "_resolve_builtin_step_yaml", return_value=step_file
        ):
            with pytest.raises(AssertionError, match="token_secretname is missing"):
                lsf_env._load_builtin_hf_lsf_section("hfpull", {})
