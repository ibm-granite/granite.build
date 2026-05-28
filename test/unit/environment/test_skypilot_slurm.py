import asyncio
from unittest.mock import MagicMock, patch

import pytest

from gbserver.environment.skypilot import Skypilot
from gbserver.types.buildevent import EntityRunMetadata
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


class TestSharedWorkdirEnvVar:
    @pytest.mark.asyncio
    async def test_shared_workdir_exposed_as_env_var(self):
        """When env config sets shared_workdir, GB_SHARED_WORKDIR is exported to the task."""
        env = Skypilot(
            event_q=asyncio.Queue(),
            environment_config=EnvironmentConfig(
                name="test-slurm",
                type="Skypilot",
                config={
                    "default_cloud": "slurm",
                    "idle_minutes_to_autostop": 0,
                    "shared_workdir": "/shared",
                },
            ),
        )
        mock_sky = _mock_sky()

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            env._get_launch_ready_event("workdir-1")
            await env.launch_skypilot(
                launch_id="workdir-1",
                launcher_config={"run": "hostname", "resources": {}},
                config={},
            )

        envs = mock_sky.Task.call_args[1]["envs"]
        assert envs["GB_SHARED_WORKDIR"] == "/shared"

    @pytest.mark.asyncio
    async def test_shared_workdir_omitted_when_unset(self, slurm_env):
        """No shared_workdir on env config -> GB_SHARED_WORKDIR is not exported."""
        mock_sky = _mock_sky()

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            slurm_env._get_launch_ready_event("workdir-2")
            await slurm_env.launch_skypilot(
                launch_id="workdir-2",
                launcher_config={"run": "hostname", "resources": {}},
                config={},
            )

        envs = mock_sky.Task.call_args[1]["envs"] or {}
        assert "GB_SHARED_WORKDIR" not in envs


class TestBuildWorkdir:
    @pytest.mark.asyncio
    async def test_setup_skypilot_returns_workdir_and_stashes(self):
        """setup_skypilot returns the build_workdir path and stashes it."""
        env = Skypilot(
            event_q=asyncio.Queue(),
            environment_config=EnvironmentConfig(
                name="test-slurm",
                type="Skypilot",
                config={
                    "default_cloud": "slurm",
                    "shared_workdir": "/shared",
                },
            ),
        )
        runmetadata = EntityRunMetadata(
            build_id="b-123", targetrun_id="tr-456"
        )

        result = await env.setup_skypilot(
            setup_id="setup-1", runmetadata=runmetadata
        )

        expected = "/shared/builds/b-123/runs/tr-456"
        assert result == {"skypilot": {"build_workdir": expected}}
        assert env._setup_workdirs["setup-1"] == expected

    @pytest.mark.asyncio
    async def test_setup_skypilot_returns_empty_when_shared_workdir_unset(
        self, slurm_env
    ):
        """No shared_workdir -> setup_skypilot is a no-op returning {}."""
        runmetadata = EntityRunMetadata(
            build_id="b-1", targetrun_id="tr-1"
        )
        result = await slurm_env.setup_skypilot(
            setup_id="setup-2", runmetadata=runmetadata
        )
        assert result == {}
        assert "setup-2" not in slurm_env._setup_workdirs

    @pytest.mark.asyncio
    async def test_launch_skypilot_exports_build_workdir_and_prepends_cd(
        self, slurm_env
    ):
        """launch_skypilot reads setup_config.skypilot.build_workdir,
        exports GB_BUILD_WORKDIR, and prepends mkdir+cd to the run script."""
        mock_sky = _mock_sky()

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            slurm_env._get_launch_ready_event("bw-1")
            await slurm_env.launch_skypilot(
                launch_id="bw-1",
                launcher_config={"run": "hostname", "resources": {}},
                config={},
                setup_config={
                    "skypilot": {"build_workdir": "/shared/builds/b/runs/r"}
                },
            )

        task_kwargs = mock_sky.Task.call_args[1]
        assert (
            task_kwargs["envs"]["GB_BUILD_WORKDIR"]
            == "/shared/builds/b/runs/r"
        )
        run_script = task_kwargs["run"]
        assert run_script.startswith(
            'mkdir -p "$GB_BUILD_WORKDIR"\ncd "$GB_BUILD_WORKDIR"\n'
        )
        assert run_script.endswith("hostname")

    @pytest.mark.asyncio
    async def test_launch_skypilot_skips_workdir_wiring_when_unset(
        self, slurm_env
    ):
        """No build_workdir in setup_config -> run script is unchanged
        and GB_BUILD_WORKDIR is not exported."""
        mock_sky = _mock_sky()

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            slurm_env._get_launch_ready_event("bw-2")
            await slurm_env.launch_skypilot(
                launch_id="bw-2",
                launcher_config={"run": "hostname", "resources": {}},
                config={},
            )

        task_kwargs = mock_sky.Task.call_args[1]
        envs = task_kwargs["envs"] or {}
        assert "GB_BUILD_WORKDIR" not in envs
        assert task_kwargs["run"] == "hostname"

    @pytest.mark.asyncio
    async def test_teardown_skypilot_removes_stashed_workdir(self, slurm_env):
        """teardown_skypilot launches a sky task that rm -rf's the stashed
        path and pops it from _setup_workdirs."""
        slurm_env._setup_workdirs["setup-td"] = "/shared/builds/b/runs/r"
        mock_sky = _mock_sky()

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            await slurm_env.teardown_skypilot(setup_id="setup-td")

        mock_sky.Task.assert_called_once()
        task_kwargs = mock_sky.Task.call_args[1]
        assert task_kwargs["run"] == 'rm -rf "/shared/builds/b/runs/r"'
        mock_sky.launch.assert_called_once()
        assert "setup-td" not in slurm_env._setup_workdirs

    @pytest.mark.asyncio
    async def test_teardown_skypilot_noop_when_no_stashed_workdir(
        self, slurm_env
    ):
        """teardown_skypilot is a no-op when setup_id was not provisioned."""
        mock_sky = _mock_sky()

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            await slurm_env.teardown_skypilot(setup_id="never-set-up")

        mock_sky.Task.assert_not_called()
        mock_sky.launch.assert_not_called()
