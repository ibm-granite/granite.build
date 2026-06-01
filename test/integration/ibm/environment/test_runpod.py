import asyncio
from typing import Self
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gbserver.environment.environment import Environment


class TestRunpodDiscovery:
    def test_runpod_registered(self):
        """Runpod class is auto-discovered and registered."""
        assert "runpod" in Environment.environment_types
        assert "Runpod" in Environment.environment_types

    def test_runpod_is_environment_subclass(self):
        from gbserver.environment.runpod import Runpod

        assert issubclass(Runpod, Environment)


class TestRunpodInit:
    def test_init_creates_instance(self):
        from gbserver.environment.runpod import Runpod

        event_q = asyncio.Queue()
        env = Runpod(event_q=event_q)
        assert env.type == "Runpod"
        assert env._launched_pods == {}

    def test_has_launch_types(self):
        from gbserver.environment.runpod import Runpod

        event_q = asyncio.Queue()
        env = Runpod(event_q=event_q)
        assert "runpod" in env.launch_types

    def test_has_cleanup_types(self):
        from gbserver.environment.runpod import Runpod

        event_q = asyncio.Queue()
        env = Runpod(event_q=event_q)
        assert "runpod" in env.cleanup_types

    def test_has_monitor_types(self):
        from gbserver.environment.runpod import Runpod

        event_q = asyncio.Queue()
        env = Runpod(event_q=event_q)
        assert "pod_status_monitor" in env.monitor_types


class TestLaunchRunpod:
    @pytest.fixture
    def runpod_env(self):
        from gbserver.environment.runpod import Runpod
        from gbserver.types.environmentconfig import EnvironmentConfig

        event_q = asyncio.Queue()
        config = EnvironmentConfig(
            name="test-runpod",
            type="runpod",
            config={
                "authentication": {"api_key": "RUNPOD_API_KEY"},
                "defaults": {
                    "gpu_type": "NVIDIA A100 80GB PCIe",
                    "cloud_type": "SECURE",
                    "container_disk_gb": 50,
                    "volume_gb": 100,
                    "volume_mount_path": "/workspace",
                },
            },
        )
        env = Runpod(
            event_q=event_q,
            environment_config=config,
            secrets={"RUNPOD_API_KEY": "test-key-123"},
        )
        return env

    @pytest.mark.asyncio
    async def test_launch_creates_pod(self, runpod_env):
        mock_runpod = MagicMock()
        mock_runpod.create_pod.return_value = MagicMock(id="pod-abc123")
        mock_runpod.get_pod.return_value = {
            "id": "pod-abc123",
            "desiredStatus": "RUNNING",
            "runtime": {"uptimeInSeconds": 10},
        }

        with patch(
            "gbserver.environment.runpod._import_runpod", return_value=mock_runpod
        ):
            launch_id = "test-launch-001"
            runpod_env._get_launch_ready_event(launch_id)

            await runpod_env.launch_runpod(
                launch_id=launch_id,
                launcher_config={
                    "image": "pytorch/pytorch:2.0.0-cuda11.8-cudnn8-runtime",
                    "command": "python train.py",
                },
                config={
                    "compute_config": {"gpu_type": "A100-80GB", "num_gpus_per_node": 1}
                },
                run_metadata={"target_name": "training"},
                step={"name": "train"},
            )

        assert launch_id in runpod_env._launched_pods
        assert runpod_env._launched_pods[launch_id] == "pod-abc123"
        assert runpod_env._get_launch_ready_event(launch_id).is_set()
        mock_runpod.create_pod.assert_called_once()

    @pytest.mark.asyncio
    async def test_launch_uses_compute_config_gpu_type(self, runpod_env):
        mock_runpod = MagicMock()
        mock_runpod.create_pod.return_value = MagicMock(id="pod-xyz")
        mock_runpod.get_pod.return_value = {
            "id": "pod-xyz",
            "desiredStatus": "RUNNING",
            "runtime": {"uptimeInSeconds": 10},
        }

        with patch(
            "gbserver.environment.runpod._import_runpod", return_value=mock_runpod
        ):
            launch_id = "test-launch-002"
            runpod_env._get_launch_ready_event(launch_id)

            await runpod_env.launch_runpod(
                launch_id=launch_id,
                launcher_config={
                    "image": "nvcr.io/nvidia/pytorch:23.10-py3",
                },
                config={"compute_config": {"gpu_type": "H100-80GB"}},
                run_metadata={"target_name": "eval"},
                step={"name": "eval"},
            )

        call_kwargs = mock_runpod.create_pod.call_args
        assert (
            call_kwargs.kwargs.get("gpu_type_id") == "NVIDIA H100 80GB HBM3"
            or call_kwargs[1].get("gpu_type_id") == "NVIDIA H100 80GB HBM3"
        )

    @pytest.mark.asyncio
    async def test_launch_sets_readiness_on_error(self, runpod_env):
        """release_monitors must be called even if launch fails."""
        mock_runpod = MagicMock()
        mock_runpod.create_pod.side_effect = Exception("API error")

        with patch(
            "gbserver.environment.runpod._import_runpod", return_value=mock_runpod
        ):
            launch_id = "test-launch-err"
            runpod_env._get_launch_ready_event(launch_id)

            with pytest.raises(Exception, match="API error"):
                await runpod_env.launch_runpod(
                    launch_id=launch_id,
                    launcher_config={"image": "test:latest"},
                    config={},
                    run_metadata={"target_name": "test"},
                    step={"name": "test"},
                )

        assert runpod_env._get_launch_ready_event(launch_id).is_set()


from gbserver.types.buildevent import BuildEventType, EntityRunMetadata

pytestmark = pytest.mark.ibm


class TestMonitorPodStatus:
    @pytest.fixture
    def runpod_env_with_pod(self):
        from gbserver.environment.runpod import Runpod
        from gbserver.types.environmentconfig import EnvironmentConfig

        event_q = asyncio.Queue()
        config = EnvironmentConfig(
            name="test-runpod",
            type="runpod",
            config={
                "authentication": {"api_key": "RUNPOD_API_KEY"},
                "defaults": {"gpu_type": "NVIDIA A100 80GB PCIe"},
            },
        )
        env = Runpod(
            event_q=event_q,
            environment_config=config,
            secrets={"RUNPOD_API_KEY": "test-key"},
        )
        launch_id = "monitor-test-001"
        env._launched_pods[launch_id] = "pod-mon-123"
        env._release_monitors(launch_id)
        return env, launch_id, event_q

    @pytest.mark.asyncio
    async def test_monitor_detects_exited(self, runpod_env_with_pod):
        env, launch_id, event_q = runpod_env_with_pod
        mock_runpod = MagicMock()
        mock_runpod.get_pod.side_effect = [
            {
                "id": "pod-mon-123",
                "desiredStatus": "RUNNING",
                "runtime": {"uptimeInSeconds": 30},
            },
            {"id": "pod-mon-123", "desiredStatus": "EXITED", "runtime": None},
        ]

        with patch(
            "gbserver.environment.runpod._import_runpod", return_value=mock_runpod
        ):
            await env.monitor_pod_status_monitor(
                launch_id=launch_id,
                event_q=event_q,
                entityrun_metadata=EntityRunMetadata(build_id="build-1"),
                poll_interval=0.01,
            )

        assert mock_runpod.get_pod.call_count == 2

    @pytest.mark.asyncio
    async def test_monitor_respects_stop_event(self, runpod_env_with_pod):
        env, launch_id, event_q = runpod_env_with_pod
        mock_runpod = MagicMock()
        mock_runpod.get_pod.return_value = {
            "id": "pod-mon-123",
            "desiredStatus": "RUNNING",
            "runtime": {"uptimeInSeconds": 60},
        }

        stop_event = env._get_launch_stopped_event(launch_id)

        async def set_stop_after_delay():
            await asyncio.sleep(0.05)
            stop_event.set()

        with patch(
            "gbserver.environment.runpod._import_runpod", return_value=mock_runpod
        ):
            await asyncio.gather(
                env.monitor_pod_status_monitor(
                    launch_id=launch_id,
                    event_q=event_q,
                    entityrun_metadata=EntityRunMetadata(build_id="build-1"),
                    poll_interval=0.01,
                ),
                set_stop_after_delay(),
            )


class TestCleanupRunpod:
    @pytest.fixture
    def runpod_env_with_pod(self):
        from gbserver.environment.runpod import Runpod
        from gbserver.types.environmentconfig import EnvironmentConfig

        event_q = asyncio.Queue()
        config = EnvironmentConfig(
            name="test-runpod",
            type="runpod",
            config={
                "authentication": {"api_key": "RUNPOD_API_KEY"},
                "defaults": {"gpu_type": "NVIDIA A100 80GB PCIe"},
            },
        )
        env = Runpod(
            event_q=event_q,
            environment_config=config,
            secrets={"RUNPOD_API_KEY": "test-key"},
        )
        launch_id = "cleanup-test-001"
        env._launched_pods[launch_id] = "pod-clean-456"
        return env, launch_id

    @pytest.mark.asyncio
    async def test_cleanup_terminates_pod(self, runpod_env_with_pod):
        env, launch_id = runpod_env_with_pod
        mock_runpod = MagicMock()

        with patch(
            "gbserver.environment.runpod._import_runpod", return_value=mock_runpod
        ):
            await env.cleanup_runpod(launch_id=launch_id)

        mock_runpod.terminate_pod.assert_called_once_with("pod-clean-456")
        assert launch_id not in env._launched_pods

    @pytest.mark.asyncio
    async def test_cleanup_sets_stop_event(self, runpod_env_with_pod):
        env, launch_id = runpod_env_with_pod
        mock_runpod = MagicMock()
        stop_event = env._get_launch_stopped_event(launch_id)

        with patch(
            "gbserver.environment.runpod._import_runpod", return_value=mock_runpod
        ):
            await env.cleanup_runpod(launch_id=launch_id)

        assert stop_event.is_set()

    @pytest.mark.asyncio
    async def test_cleanup_no_pod_is_noop(self):
        from gbserver.environment.runpod import Runpod

        env = Runpod(event_q=asyncio.Queue())
        await env.cleanup_runpod(launch_id="nonexistent-launch")
