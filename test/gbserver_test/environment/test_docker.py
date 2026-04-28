import asyncio
from unittest.mock import MagicMock, patch

import pytest

from gbserver.types.buildevent import BuildEventType, EntityRunMetadata

pytestmark = [pytest.mark.g4os, pytest.mark.unit]


class TestStepDockerConfig:
    def test_default_values(self):
        from gbserver.types.environment.docker import StepDockerConfig

        config = StepDockerConfig()
        assert config.image is None
        assert config.env == {}
        assert config.registry_auth is None
        assert config.pull_policy == "if-not-present"

    def test_from_dict(self):
        from gbserver.types.environment.docker import StepDockerConfig

        config = StepDockerConfig(
            image="pytorch/pytorch:2.0.0",
            env={"HF_HOME": {"value": "/cache"}},
            pull_policy="always",
            registry_auth={"username": "user", "password": "pass"},
        )
        assert config.image == "pytorch/pytorch:2.0.0"
        assert config.env["HF_HOME"]["value"] == "/cache"
        assert config.pull_policy == "always"
        assert config.registry_auth["username"] == "user"

    def test_invalid_pull_policy_allowed(self):
        """StepDockerConfig is a data class — validation happens at use time."""
        from gbserver.types.environment.docker import StepDockerConfig

        config = StepDockerConfig(pull_policy="bogus")
        assert config.pull_policy == "bogus"


from gbserver.environment.environment import Environment


class TestDockerDiscovery:
    def test_docker_registered(self):
        """Docker class is auto-discovered and registered."""
        assert "docker" in Environment.environment_types
        assert "Docker" in Environment.environment_types

    def test_docker_is_environment_subclass(self):
        from gbserver.environment.docker import Docker

        assert issubclass(Docker, Environment)


class TestDockerInit:
    def test_init_creates_instance(self):
        from gbserver.environment.docker import Docker

        event_q = asyncio.Queue()
        env = Docker(event_q=event_q)
        assert env.type == "Docker"
        assert env.launched_containers == {}

    def test_has_launch_types(self):
        from gbserver.environment.docker import Docker

        event_q = asyncio.Queue()
        env = Docker(event_q=event_q)
        assert "docker" in env.launch_types

    def test_has_cleanup_types(self):
        from gbserver.environment.docker import Docker

        event_q = asyncio.Queue()
        env = Docker(event_q=event_q)
        assert "docker" in env.cleanup_types

    def test_has_monitor_types(self):
        from gbserver.environment.docker import Docker

        event_q = asyncio.Queue()
        env = Docker(event_q=event_q)
        assert "docker_log" in env.monitor_types


class TestLaunchDocker:
    @pytest.fixture
    def docker_env(self):
        from gbserver.environment.docker import Docker
        from gbserver.types.environmentconfig import EnvironmentConfig

        event_q = asyncio.Queue()
        config = EnvironmentConfig(
            name="test-docker",
            type="Docker",
            config={"defaults": {"image": "python:3.12-slim"}},
        )
        return Docker(event_q=event_q, environment_config=config)

    @pytest.mark.asyncio
    async def test_launch_creates_container(self, docker_env):
        mock_docker = MagicMock()
        mock_container = MagicMock()
        mock_container.id = "container-abc123"
        mock_docker.from_env.return_value.containers.run.return_value = mock_container
        mock_docker.from_env.return_value.images.get.return_value = MagicMock()
        mock_docker.types.DeviceRequest = MagicMock

        with patch(
            "gbserver.environment.docker._import_docker", return_value=mock_docker
        ):
            launch_id = "test-launch-001"
            docker_env._get_launch_ready_event(launch_id)
            await docker_env.launch_docker(
                launch_id=launch_id,
                targetsteprun_asset_dir="/tmp/test-assets",
                launcher_config={
                    "image": "pytorch/pytorch:2.0.0",
                    "command": "python train.py",
                },
                config={
                    "compute_config": {
                        "num_gpus_per_node": 0,
                        "num_cpus_per_node": 4,
                        "total_memory_per_node": "8Gi",
                    }
                },
                run_metadata={"target_name": "training"},
                step={"name": "train"},
            )

        assert launch_id in docker_env.launched_containers
        assert docker_env.launched_containers[launch_id] == "container-abc123"
        assert docker_env._get_launch_ready_event(launch_id).is_set()
        mock_docker.from_env.return_value.containers.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_launch_with_gpu(self, docker_env):
        mock_docker = MagicMock()
        mock_container = MagicMock()
        mock_container.id = "container-gpu-001"
        mock_docker.from_env.return_value.containers.run.return_value = mock_container
        mock_docker.from_env.return_value.images.get.return_value = MagicMock()
        mock_docker.types.DeviceRequest = MagicMock

        with patch(
            "gbserver.environment.docker._import_docker", return_value=mock_docker
        ):
            launch_id = "test-launch-gpu"
            docker_env._get_launch_ready_event(launch_id)
            await docker_env.launch_docker(
                launch_id=launch_id,
                targetsteprun_asset_dir="/tmp/test-assets",
                launcher_config={"image": "nvcr.io/nvidia/pytorch:latest"},
                config={"compute_config": {"num_gpus_per_node": 2}},
                run_metadata={"target_name": "eval"},
                step={"name": "eval"},
            )

        call_kwargs = mock_docker.from_env.return_value.containers.run.call_args
        assert call_kwargs.kwargs.get("device_requests") is not None

    @pytest.mark.asyncio
    async def test_launch_sets_readiness_on_error(self, docker_env):
        mock_docker = MagicMock()
        mock_docker.errors.DockerException = type("DockerException", (Exception,), {})
        mock_docker.from_env.side_effect = Exception("Docker daemon not available")

        with patch(
            "gbserver.environment.docker._import_docker", return_value=mock_docker
        ):
            launch_id = "test-launch-err"
            docker_env._get_launch_ready_event(launch_id)
            with pytest.raises(Exception, match="Docker daemon not available"):
                await docker_env.launch_docker(
                    launch_id=launch_id,
                    targetsteprun_asset_dir="/tmp/test-assets",
                    launcher_config={"image": "test:latest"},
                    config={},
                    run_metadata={"target_name": "test"},
                    step={"name": "test"},
                )

        assert docker_env._get_launch_ready_event(launch_id).is_set()

    @pytest.mark.asyncio
    async def test_launch_pulls_image_if_not_present(self, docker_env):
        mock_docker = MagicMock()
        mock_client = mock_docker.from_env.return_value
        mock_docker.errors = MagicMock()
        mock_docker.errors.ImageNotFound = type("ImageNotFound", (Exception,), {})
        mock_client.images.get.side_effect = mock_docker.errors.ImageNotFound(
            "not found"
        )
        mock_container = MagicMock()
        mock_container.id = "container-pulled"
        mock_client.containers.run.return_value = mock_container
        mock_docker.types.DeviceRequest = MagicMock

        with patch(
            "gbserver.environment.docker._import_docker", return_value=mock_docker
        ):
            launch_id = "test-launch-pull"
            docker_env._get_launch_ready_event(launch_id)
            await docker_env.launch_docker(
                launch_id=launch_id,
                targetsteprun_asset_dir="/tmp/test-assets",
                launcher_config={"image": "myimage:latest"},
                config={"docker": {"pull_policy": "if-not-present"}},
                run_metadata={"target_name": "test"},
                step={"name": "test"},
            )

        mock_client.images.pull.assert_called_once()

    @pytest.mark.asyncio
    async def test_launch_uses_default_image_from_env_config(self, docker_env):
        mock_docker = MagicMock()
        mock_container = MagicMock()
        mock_container.id = "container-default"
        mock_docker.from_env.return_value.containers.run.return_value = mock_container
        mock_docker.from_env.return_value.images.get.return_value = MagicMock()
        mock_docker.types.DeviceRequest = MagicMock

        with patch(
            "gbserver.environment.docker._import_docker", return_value=mock_docker
        ):
            launch_id = "test-launch-default"
            docker_env._get_launch_ready_event(launch_id)
            await docker_env.launch_docker(
                launch_id=launch_id,
                targetsteprun_asset_dir="/tmp/test-assets",
                launcher_config={},  # no image
                config={},
                run_metadata={"target_name": "test"},
                step={"name": "test"},
            )

        call_kwargs = mock_docker.from_env.return_value.containers.run.call_args
        assert call_kwargs.kwargs.get("image") == "python:3.12-slim"


class TestMonitorDockerLog:
    @pytest.fixture
    def docker_env_with_container(self):
        from gbserver.environment.docker import Docker
        from gbserver.types.environmentconfig import EnvironmentConfig

        event_q = asyncio.Queue()
        config = EnvironmentConfig(
            name="test-docker",
            type="Docker",
            config={"defaults": {"image": "python:3.12-slim"}},
        )
        env = Docker(event_q=event_q, environment_config=config)
        launch_id = "monitor-test-001"
        env.launched_containers[launch_id] = "container-mon-123"
        env._release_monitors(launch_id)
        return env, launch_id, event_q

    @pytest.mark.asyncio
    async def test_monitor_emits_log_events(self, docker_env_with_container):
        env, launch_id, event_q = docker_env_with_container
        mock_docker = MagicMock()
        mock_container = MagicMock()
        mock_container.logs.return_value = [b"line 1\n", b"line 2\n"]
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.attrs = {"State": {"OOMKilled": False}}
        mock_docker.from_env.return_value.containers.get.return_value = mock_container

        with patch(
            "gbserver.environment.docker._import_docker", return_value=mock_docker
        ):
            await env.monitor_docker_log(
                launch_id=launch_id,
                event_q=event_q,
                entityrun_metadata=EntityRunMetadata(build_id="build-1"),
            )

        events = []
        while not event_q.empty():
            events.append(event_q.get_nowait())
        assert len(events) >= 2

    @pytest.mark.asyncio
    async def test_monitor_detects_nonzero_exit(self, docker_env_with_container):
        from gbserver.types.errors import LogMonitoringFailedException

        env, launch_id, event_q = docker_env_with_container
        mock_docker = MagicMock()
        mock_container = MagicMock()
        mock_container.logs.return_value = [b"error\n"]
        mock_container.wait.return_value = {"StatusCode": 1}
        mock_container.attrs = {"State": {"OOMKilled": False}}
        mock_docker.from_env.return_value.containers.get.return_value = mock_container

        with patch(
            "gbserver.environment.docker._import_docker", return_value=mock_docker
        ):
            with pytest.raises(
                LogMonitoringFailedException, match="exited with code 1"
            ):
                await env.monitor_docker_log(
                    launch_id=launch_id,
                    event_q=event_q,
                    entityrun_metadata=EntityRunMetadata(build_id="build-1"),
                    build_id="build-1",
                )

        events = []
        while not event_q.empty():
            events.append(event_q.get_nowait())
        message_events = [e for e in events if e.type == BuildEventType.MESSAGE_EVENT]
        assert len(message_events) >= 1

    @pytest.mark.asyncio
    async def test_monitor_no_container_returns(self, docker_env_with_container):
        env, launch_id, event_q = docker_env_with_container
        env.launched_containers.pop(launch_id)

        await env.monitor_docker_log(
            launch_id=launch_id,
            event_q=event_q,
            entityrun_metadata=EntityRunMetadata(build_id="build-1"),
        )


class TestCleanupDocker:
    @pytest.fixture
    def docker_env_with_container(self):
        from gbserver.environment.docker import Docker
        from gbserver.types.environmentconfig import EnvironmentConfig

        event_q = asyncio.Queue()
        config = EnvironmentConfig(name="test-docker", type="Docker", config={})
        env = Docker(event_q=event_q, environment_config=config)
        launch_id = "cleanup-test-001"
        env.launched_containers[launch_id] = "container-clean-456"
        return env, launch_id

    @pytest.mark.asyncio
    async def test_cleanup_stops_and_removes_container(self, docker_env_with_container):
        env, launch_id = docker_env_with_container
        mock_docker = MagicMock()
        mock_container = MagicMock()
        mock_docker.from_env.return_value.containers.get.return_value = mock_container

        with patch(
            "gbserver.environment.docker._import_docker", return_value=mock_docker
        ):
            await env.cleanup_docker(launch_id=launch_id)

        mock_container.stop.assert_called_once_with(timeout=30)
        mock_container.remove.assert_called_once_with(force=True)
        assert launch_id not in env.launched_containers

    @pytest.mark.asyncio
    async def test_cleanup_sets_stop_event(self, docker_env_with_container):
        env, launch_id = docker_env_with_container
        mock_docker = MagicMock()
        mock_container = MagicMock()
        mock_docker.from_env.return_value.containers.get.return_value = mock_container
        stop_event = env._get_launch_stopped_event(launch_id)

        with patch(
            "gbserver.environment.docker._import_docker", return_value=mock_docker
        ):
            await env.cleanup_docker(launch_id=launch_id)

        assert stop_event.is_set()

    @pytest.mark.asyncio
    async def test_cleanup_no_container_is_noop(self):
        from gbserver.environment.docker import Docker

        env = Docker(event_q=asyncio.Queue())
        await env.cleanup_docker(launch_id="nonexistent-launch")

    @pytest.mark.asyncio
    async def test_cleanup_handles_already_removed_container(
        self, docker_env_with_container
    ):
        env, launch_id = docker_env_with_container
        mock_docker = MagicMock()
        mock_docker.errors = MagicMock()
        mock_docker.errors.NotFound = type("NotFound", (Exception,), {})
        mock_docker.from_env.return_value.containers.get.side_effect = (
            mock_docker.errors.NotFound("gone")
        )

        with patch(
            "gbserver.environment.docker._import_docker", return_value=mock_docker
        ):
            await env.cleanup_docker(launch_id=launch_id)

        assert launch_id not in env.launched_containers


class TestDockerImportGuard:
    def test_import_guard_raises_when_docker_missing(self):
        from gbserver.environment.docker import _import_docker

        with patch.dict("sys.modules", {"docker": None}):
            with pytest.raises(ImportError, match="pip install.*gbserver.*docker"):
                _import_docker()
