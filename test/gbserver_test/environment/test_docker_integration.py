"""Integration tests for Docker environment — require a Docker/Podman daemon."""

import asyncio
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.thirdparty


def _docker_available():
    """Check if Docker/Podman daemon is accessible."""
    try:
        import docker

        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


skipif_no_docker = pytest.mark.skipif(
    not _docker_available(),
    reason="Docker/Podman daemon not available",
)


@skipif_no_docker
class TestDockerIntegration:
    @pytest.mark.asyncio
    async def test_run_simple_container(self):
        """Launch alpine echo, verify logs and exit code."""
        from gbserver.environment.docker import Docker
        from gbserver.types.buildevent import EntityRunMetadata
        from gbserver.types.environmentconfig import EnvironmentConfig

        event_q = asyncio.Queue()
        config = EnvironmentConfig(
            name="integration-docker",
            type="Docker",
            config={"defaults": {"image": "alpine:latest"}},
        )
        env = Docker(event_q=event_q, environment_config=config)

        launch_id = "integration-001"

        await env.launch_docker(
            launch_id=launch_id,
            launcher_config={
                "image": "alpine:latest",
                "command": "echo hello-gbserver",
            },
            config={},
            run_metadata={"target_name": "test"},
            step={"name": "echo"},
        )

        assert launch_id in env.launched_containers

        await env.monitor_docker_log(
            launch_id=launch_id,
            event_q=event_q,
            entityrun_metadata=EntityRunMetadata(build_id="int-build-1"),
        )

        events = []
        while not event_q.empty():
            events.append(event_q.get_nowait())
        log_messages = [e.payload.msg for e in events if hasattr(e.payload, "msg")]
        assert any("hello-gbserver" in msg for msg in log_messages)

        await env.cleanup_docker(launch_id=launch_id)
        assert launch_id not in env.launched_containers

    @pytest.mark.asyncio
    async def test_bind_mount_data_flow(self):
        """Write a file in container via bind mount, read it on host."""
        from gbserver.environment.docker import Docker
        from gbserver.types.buildevent import EntityRunMetadata
        from gbserver.types.environmentconfig import EnvironmentConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            event_q = asyncio.Queue()
            config = EnvironmentConfig(
                name="integration-docker",
                type="Docker",
                config={},
            )
            env = Docker(event_q=event_q, environment_config=config)

            launch_id = "integration-bind-001"

            await env.launch_docker(
                launch_id=launch_id,
                targetsteprun_asset_dir=tmpdir,
                launcher_config={
                    "image": "alpine:latest",
                    "command": "sh -c 'echo test-output > /gb-workspace/result.txt'",
                },
                config={},
                run_metadata={"target_name": "test"},
                step={"name": "bind-test"},
            )

            await env.monitor_docker_log(
                launch_id=launch_id,
                event_q=event_q,
                entityrun_metadata=EntityRunMetadata(build_id="int-build-2"),
            )

            result_file = Path(tmpdir) / "result.txt"
            assert result_file.exists()
            assert result_file.read_text().strip() == "test-output"

            await env.cleanup_docker(launch_id=launch_id)
