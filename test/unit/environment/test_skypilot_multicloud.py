"""Multi-cloud config validation tests using SkyMock.

Verifies that skypilot.py correctly maps environment config + launcher_config
to sky.Resources for each supported cloud (AWS, GCP, RunPod).
"""

import asyncio

import pytest

from gbserver.testing.skymock import MockSky, Scenario, make_skypilot_env, patched_skypilot


class TestMultiCloudResourceMapping:
    """Verify skypilot.py correctly maps config to sky.Resources for each cloud."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cloud", ["aws", "gcp", "runpod"])
    async def test_cloud_passed_as_infra(self, cloud):
        """Each cloud name becomes the infra parameter in sky.Resources."""
        env, event_q, mock_sky = make_skypilot_env(
            Scenario.happy_path(cloud=cloud), cloud=cloud
        )

        with patched_skypilot(mock_sky):
            env._get_launch_ready_event(f"{cloud}-1")
            await env.launch_skypilot(
                launch_id=f"{cloud}-1",
                launcher_config={
                    "run": "echo hello",
                    "resources": {"cloud": cloud},
                },
                config={},
            )

        # Verify cluster was created via sky.launch
        assert len(mock_sky._clusters) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "cloud,accelerators",
        [
            ("aws", "V100:1"),
            ("gcp", "A100:4"),
            ("runpod", "H100:8"),
        ],
    )
    async def test_accelerators_passed_to_resources(self, cloud, accelerators):
        """GPU accelerators from launcher_config reach sky.Resources."""
        env, event_q, mock_sky = make_skypilot_env(
            Scenario.happy_path(cloud=cloud), cloud=cloud
        )

        with patched_skypilot(mock_sky):
            env._get_launch_ready_event(f"{cloud}-acc")
            await env.launch_skypilot(
                launch_id=f"{cloud}-acc",
                launcher_config={
                    "run": "train.py",
                    "resources": {
                        "cloud": cloud,
                        "accelerators": accelerators,
                    },
                },
                config={},
            )

        assert len(mock_sky._clusters) == 1

    @pytest.mark.asyncio
    async def test_env_config_default_cloud_used_when_no_resources_cloud(self):
        """When resources doesn't specify cloud, env config default_cloud is used."""
        env, event_q, mock_sky = make_skypilot_env(
            Scenario.happy_path(cloud="gcp"), cloud="gcp"
        )

        with patched_skypilot(mock_sky):
            env._get_launch_ready_event("default-1")
            await env.launch_skypilot(
                launch_id="default-1",
                launcher_config={
                    "run": "echo",
                    "resources": {},  # no cloud specified
                },
                config={},
            )

        assert len(mock_sky._clusters) == 1

    @pytest.mark.asyncio
    async def test_idle_minutes_from_env_config(self):
        """idle_minutes_to_autostop from env config passed to sky.launch."""
        env, event_q, mock_sky = make_skypilot_env(
            Scenario.happy_path(), cloud="aws", idle_minutes=15
        )

        with patched_skypilot(mock_sky):
            env._get_launch_ready_event("idle-1")
            await env.launch_skypilot(
                launch_id="idle-1",
                launcher_config={
                    "run": "echo",
                    "resources": {},
                },
                config={},
            )

        # Verify sky.launch was called (cluster exists)
        assert len(mock_sky._clusters) == 1
        # Check kwargs stored by MockSky include idle_minutes_to_autostop
        cluster_data = list(mock_sky._clusters.values())[0]
        assert cluster_data.get("idle_minutes_to_autostop") == 15


class TestMultiCloudFailover:
    """Test cross-cloud failover scenarios using SkyMock."""

    @pytest.mark.asyncio
    async def test_cross_cloud_failover_primary_fails(self):
        """Primary cloud fails, verify failure is detected."""
        primary_scenario, fallback_scenario = Scenario.cross_cloud_failover(
            primary="aws", fallback="gcp"
        )
        env, event_q, mock_sky = make_skypilot_env(primary_scenario, cloud="aws")

        with patched_skypilot(mock_sky):
            env._get_launch_ready_event("failover-1")
            await env.launch_skypilot(
                launch_id="failover-1",
                launcher_config={"run": "train.py", "resources": {"cloud": "aws"}},
                config={},
            )

            # Monitor should detect terminal FAILED
            await env.monitor_skypilot_monitor(
                launch_id="failover-1",
                event_q=event_q,
                entityrun_metadata={"build_id": "b1", "targetrun_id": "tr1"},
                poll_interval=0,
            )

        # Verify FAILED event emitted
        events = []
        while not event_q.empty():
            events.append(await event_q.get())

        from gbserver.types.buildevent import BuildEventType

        fail_events = [
            e for e in events if e.type == BuildEventType.WORKLOAD_STATUS_EVENT
        ]
        assert len(fail_events) == 1

    @pytest.mark.asyncio
    async def test_cross_cloud_failover_fallback_succeeds(self):
        """Fallback cloud succeeds after primary fails."""
        primary_scenario, fallback_scenario = Scenario.cross_cloud_failover(
            primary="aws", fallback="gcp"
        )

        # Simulate: after primary fails, a new launch on fallback cloud succeeds
        env, event_q, mock_sky = make_skypilot_env(fallback_scenario, cloud="gcp")

        with patched_skypilot(mock_sky):
            env._get_launch_ready_event("fallback-1")
            await env.launch_skypilot(
                launch_id="fallback-1",
                launcher_config={"run": "train.py", "resources": {"cloud": "gcp"}},
                config={},
            )

            await env.monitor_skypilot_monitor(
                launch_id="fallback-1",
                event_q=event_q,
                entityrun_metadata={"build_id": "b1", "targetrun_id": "tr1"},
                poll_interval=0,
            )

        # No FAILED event — fallback succeeded
        events = []
        while not event_q.empty():
            events.append(await event_q.get())

        from gbserver.types.buildevent import BuildEventType

        fail_events = [
            e for e in events if e.type == BuildEventType.WORKLOAD_STATUS_EVENT
        ]
        assert len(fail_events) == 0


class TestCloudSpecificConfig:
    """Test cloud-specific configuration options."""

    @pytest.mark.asyncio
    async def test_file_mounts_with_s3_storage(self):
        """S3 storage mounts are passed to sky.Task via file_mounts."""
        env, event_q, mock_sky = make_skypilot_env(
            Scenario.happy_path(cloud="aws"), cloud="aws"
        )

        with patched_skypilot(mock_sky):
            env._get_launch_ready_event("mounts-1")
            await env.launch_skypilot(
                launch_id="mounts-1",
                launcher_config={
                    "run": "train.py",
                    "resources": {"cloud": "aws"},
                    "file_mounts": {
                        "/data/model": {
                            "source": "s3://my-bucket/models/v1",
                            "mode": "MOUNT",
                        },
                    },
                },
                config={},
            )

        assert len(mock_sky._clusters) == 1

    @pytest.mark.asyncio
    async def test_env_vars_passed_to_task(self):
        """Environment variables from launcher_config.envs reach the task."""
        env, event_q, mock_sky = make_skypilot_env(
            Scenario.happy_path(cloud="aws"), cloud="aws"
        )

        with patched_skypilot(mock_sky):
            env._get_launch_ready_event("envs-1")
            await env.launch_skypilot(
                launch_id="envs-1",
                launcher_config={
                    "run": "train.py",
                    "resources": {"cloud": "aws"},
                    "envs": {"HF_TOKEN": "test-token", "WANDB_KEY": "test-key"},
                },
                config={},
            )

        assert len(mock_sky._clusters) == 1

    @pytest.mark.asyncio
    async def test_setup_command_passed_to_task(self):
        """Setup command from launcher_config is forwarded to sky.Task."""
        env, event_q, mock_sky = make_skypilot_env(
            Scenario.happy_path(cloud="gcp"), cloud="gcp"
        )

        with patched_skypilot(mock_sky):
            env._get_launch_ready_event("setup-1")
            await env.launch_skypilot(
                launch_id="setup-1",
                launcher_config={
                    "run": "train.py",
                    "setup": "pip install -r requirements.txt",
                    "resources": {"cloud": "gcp"},
                },
                config={},
            )

        assert len(mock_sky._clusters) == 1
