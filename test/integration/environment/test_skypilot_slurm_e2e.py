"""End-to-end integration test for SLURM builds via SkyPilot.

Requires a running Docker SLURM cluster (see scripts/slurm/setup-slurm.sh).
Auto-skips if the cluster is not reachable via SSH.
"""

import asyncio
import os
import subprocess
import uuid

import pytest
from libgbtest.constants import extended_testing_only

from gbserver.environment.skypilot import Skypilot
from gbserver.types.buildevent import BuildEventType, EntityRunMetadata
from gbserver.types.environmentconfig import EnvironmentConfig

# extended_testing_only: these launch real SLURM clusters and run for minutes, so they
# belong in `make extended-tests` (the `extended` marker), not the fast
# `make quick-tests` mock suite — where they'd otherwise run whenever slurm happens to
# be available locally and can intermittently stall the whole xdist run.
pytestmark = [
    pytest.mark.skypilot_integration,
    pytest.mark.asyncio,
    extended_testing_only,
]


def _slurm_cluster_reachable() -> bool:
    """Check if the Docker SLURM cluster is reachable via SSH.

    Connects directly with the key/port that ``setup-slurm.sh`` provisions
    rather than via ``~/.slurm/config`` — that file is no longer written by
    setup (the slurm SSH config is inlined in the fixture's environment.yaml
    and materialized by gbserver at launch time), so it would not exist at the
    test-collection time when this skip gate runs.
    """
    key = os.path.expanduser("~/.ssh/slurm_docker_key")
    port = os.environ.get("SLURM_SSH_PORT", "2222")
    host = os.environ.get("SLURM_SSH_HOST", "127.0.0.1")
    try:
        result = subprocess.run(
            [
                "ssh",
                "-i",
                key,
                "-p",
                port,
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "UserKnownHostsFile=/dev/null",
                "-o",
                "ConnectTimeout=3",
                f"root@{host}",
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
                # Inline the slurm-docker SSH config so gbserver materializes it
                # into ~/.slurm/config at launch time (setup-slurm.sh no longer
                # writes that file). Without this, SkyPilot reports
                # "Cluster slurm-docker not found ... Available clusters: []".
                # Host/Port mirror the SLURM_SSH_* vars used by the skip gate.
                "cluster_ssh_configs": {
                    "slurm": [
                        {
                            "Host": "slurm-docker",
                            "HostName": os.environ.get("SLURM_SSH_HOST", "127.0.0.1"),
                            "User": "root",
                            "Port": os.environ.get("SLURM_SSH_PORT", "2222"),
                            "IdentityFile": "~/.ssh/slurm_docker_key",
                            "StrictHostKeyChecking": "no",
                            "UserKnownHostsFile": "/dev/null",
                        }
                    ]
                },
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
            # Bound the teardown: a stalled `sky down` must fail the test, not hang
            # the worker (and the whole xdist run) forever. Mirrors the launch/monitor
            # timeouts above and the cleanup in test_launch_monitor_cleanup.
            await asyncio.wait_for(
                slurm_env.cleanup_skypilot(launch_id=launch_id),
                timeout=120,
            )
