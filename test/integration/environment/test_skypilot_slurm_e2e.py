"""End-to-end integration test for SLURM builds via SkyPilot.

Requires a running Docker SLURM cluster (see scripts/slurm/setup-slurm.sh).
Auto-skips if the cluster is not reachable via SSH.
"""

import asyncio
import subprocess
import uuid

import pytest

from gbserver.environment.skypilot import Skypilot
from gbserver.types.buildevent import BuildEventType, EntityRunMetadata
from gbserver.types.environmentconfig import EnvironmentConfig

pytestmark = [pytest.mark.skypilot_integration, pytest.mark.asyncio]


def _slurm_cluster_reachable() -> bool:
    """Check if the Docker SLURM cluster is reachable via SSH."""
    try:
        result = subprocess.run(
            [
                "ssh",
                "-F",
                "~/.slurm/config",
                "-o",
                "ConnectTimeout=3",
                "slurm-docker",
                "true",
            ],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


skipif_no_slurm = pytest.mark.skipif(
    not _slurm_cluster_reachable(),
    reason="Docker SLURM cluster not reachable (run: make slurm-setup)",
)


@skipif_no_slurm
class TestSlurmBuildLifecycle:
    """End-to-end lifecycle: launch -> monitor -> cleanup on Docker SLURM."""

    @pytest.fixture
    def slurm_env(self):
        event_q = asyncio.Queue()
        config = EnvironmentConfig(
            name="slurm-e2e",
            type="Skypilot",
            config={
                "default_cloud": "slurm",
                "idle_minutes_to_autostop": 0,
            },
        )
        return Skypilot(event_q=event_q, environment_config=config)

    @pytest.fixture
    def launch_id(self):
        return str(uuid.uuid4())[:12]

    @pytest.fixture
    def entityrun_metadata(self):
        return EntityRunMetadata(
            build_id="e2e-test-build",
            username="e2e-test",
            target_name="echo-test",
            targetrun_id="e2e-targetrun",
            targetsteprun_id="e2e-steprun",
        )

    async def test_launch_monitor_cleanup(
        self, slurm_env, launch_id, entityrun_metadata
    ):
        """Full lifecycle: launch a job, monitor to completion, then cleanup."""
        event_q = slurm_env.event_q
        launcher_config = {
            "resources": {
                "cloud": "slurm",
                "cluster": "slurm-docker",
                "zone": "normal",
            },
            "run": "echo hello && hostname",
        }

        # Launch
        await asyncio.wait_for(
            slurm_env.launch_skypilot(
                launch_id=launch_id,
                launcher_config=launcher_config,
            ),
            timeout=300,
        )

        assert launch_id in slurm_env._cluster_names
        assert launch_id in slurm_env._job_ids
        cluster_name = slurm_env._cluster_names[launch_id]
        assert cluster_name == f"gb-{launch_id[:12]}"

        # Monitor (poll fast for test speed)
        await asyncio.wait_for(
            slurm_env.monitor_skypilot_monitor(
                launch_id=launch_id,
                event_q=event_q,
                entityrun_metadata=entityrun_metadata,
                poll_interval=5,
            ),
            timeout=300,
        )

        # Verify events were emitted
        events = []
        while not event_q.empty():
            events.append(await event_q.get())

        assert len(events) > 0, "Expected at least one status event"
        for ev in events:
            assert ev.type == BuildEventType.MESSAGE_EVENT
        # Last event should mention terminal status
        last_msg = events[-1].payload.msg
        assert "SUCCEEDED" in last_msg or "FAILED" in last_msg

        # Cleanup
        await asyncio.wait_for(
            slurm_env.cleanup_skypilot(launch_id=launch_id),
            timeout=120,
        )

        assert launch_id not in slurm_env._cluster_names
        assert launch_id not in slurm_env._job_ids

    async def test_launch_succeeds(self, slurm_env, launch_id, entityrun_metadata):
        """Verify a simple echo job completes with SUCCEEDED status."""
        event_q = slurm_env.event_q
        launcher_config = {
            "resources": {
                "cloud": "slurm",
                "cluster": "slurm-docker",
                "zone": "normal",
            },
            "run": "echo 'integration test passed'",
        }

        try:
            await asyncio.wait_for(
                slurm_env.launch_skypilot(
                    launch_id=launch_id,
                    launcher_config=launcher_config,
                ),
                timeout=300,
            )

            await asyncio.wait_for(
                slurm_env.monitor_skypilot_monitor(
                    launch_id=launch_id,
                    event_q=event_q,
                    entityrun_metadata=entityrun_metadata,
                    poll_interval=5,
                ),
                timeout=300,
            )

            events = []
            while not event_q.empty():
                events.append(await event_q.get())

            succeeded = any("SUCCEEDED" in ev.payload.msg for ev in events)
            assert (
                succeeded
            ), f"Job did not succeed. Events: {[e.payload.msg for e in events]}"
        finally:
            await slurm_env.cleanup_skypilot(launch_id=launch_id)
