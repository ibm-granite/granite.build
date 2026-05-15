import asyncio
from unittest.mock import MagicMock, patch

import pytest

from gbserver.environment.skypilot import Skypilot
from gbserver.types.environmentconfig import EnvironmentConfig


@pytest.fixture
def slurm_env():
    event_q = asyncio.Queue()
    config = EnvironmentConfig(
        name="test-slurm",
        type="Skypilot",
        config={
            "default_cloud": "slurm",
            "idle_minutes_to_autostop": 0,
        },
    )
    return Skypilot(event_q=event_q, environment_config=config)


def _mock_sky():
    mock = MagicMock()
    mock.Resources = MagicMock(return_value=MagicMock())
    mock.Task = MagicMock(return_value=MagicMock())
    mock.launch = MagicMock(return_value="req-slurm")
    mock.stream_and_get = MagicMock(return_value=(1, MagicMock()))
    return mock


class TestSlurmInfraPath:
    @pytest.mark.asyncio
    async def test_infra_includes_cluster_and_partition(self, slurm_env):
        mock_sky = _mock_sky()

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            slurm_env._get_launch_ready_event("slurm-1")
            await slurm_env.launch_skypilot(
                launch_id="slurm-1",
                launcher_config={
                    "run": "hostname",
                    "resources": {
                        "cloud": "slurm",
                        "cluster": "slurm-docker",
                        "zone": "normal",
                        "accelerators": "GPU:1",
                    },
                },
                config={},
            )

        mock_sky.Resources.assert_called_once()
        call_kwargs = mock_sky.Resources.call_args[1]
        assert call_kwargs["infra"] == "slurm/slurm-docker/normal"
        assert call_kwargs["zone"] is None
        assert call_kwargs["accelerators"] == "GPU:1"

    @pytest.mark.asyncio
    async def test_infra_cluster_without_partition(self, slurm_env):
        mock_sky = _mock_sky()

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            slurm_env._get_launch_ready_event("slurm-2")
            await slurm_env.launch_skypilot(
                launch_id="slurm-2",
                launcher_config={
                    "run": "hostname",
                    "resources": {
                        "cloud": "slurm",
                        "cluster": "slurm-docker",
                    },
                },
                config={},
            )

        call_kwargs = mock_sky.Resources.call_args[1]
        assert call_kwargs["infra"] == "slurm/slurm-docker"
        assert call_kwargs["zone"] is None

    @pytest.mark.asyncio
    async def test_infra_bare_cloud_without_cluster(self, slurm_env):
        mock_sky = _mock_sky()

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            slurm_env._get_launch_ready_event("slurm-3")
            await slurm_env.launch_skypilot(
                launch_id="slurm-3",
                launcher_config={
                    "run": "hostname",
                    "resources": {"cloud": "slurm"},
                },
                config={},
            )

        call_kwargs = mock_sky.Resources.call_args[1]
        assert call_kwargs["infra"] == "slurm"
        assert call_kwargs["zone"] is None

    @pytest.mark.asyncio
    async def test_explicit_infra_string_takes_precedence(self, slurm_env):
        mock_sky = _mock_sky()

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            slurm_env._get_launch_ready_event("slurm-4")
            await slurm_env.launch_skypilot(
                launch_id="slurm-4",
                launcher_config={
                    "run": "hostname",
                    "resources": {
                        "infra": "slurm/my-cluster/gpu-partition",
                        "cloud": "slurm",
                        "cluster": "ignored",
                    },
                },
                config={},
            )

        call_kwargs = mock_sky.Resources.call_args[1]
        assert call_kwargs["infra"] == "slurm/my-cluster/gpu-partition"

    @pytest.mark.asyncio
    async def test_defaults_to_env_config_cloud(self, slurm_env):
        mock_sky = _mock_sky()

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            slurm_env._get_launch_ready_event("slurm-5")
            await slurm_env.launch_skypilot(
                launch_id="slurm-5",
                launcher_config={
                    "run": "hostname",
                    "resources": {},
                },
                config={},
            )

        call_kwargs = mock_sky.Resources.call_args[1]
        assert call_kwargs["infra"] == "slurm"
