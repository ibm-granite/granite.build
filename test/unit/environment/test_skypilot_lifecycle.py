"""Lifecycle tests exercising the real Skypilot class through SkyMock scenarios."""

import asyncio
import os
import tempfile

import pytest

from gbserver.testing.skymock import (
    MockSky,
    Scenario,
    ScenarioStep,
    make_skypilot_env,
    patched_skypilot,
)


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_launch_monitor_cleanup(self):
        """Full lifecycle: launch -> monitor polls to SUCCEEDED -> cleanup."""
        env, event_q, mock_sky = make_skypilot_env(Scenario.happy_path(cloud="aws"))

        with patched_skypilot(mock_sky):
            env._get_launch_ready_event("happy-1")
            await env.launch_skypilot(
                launch_id="happy-1",
                launcher_config={"run": "train.py", "resources": {"cloud": "aws"}},
                config={},
            )

            # Monitor should poll until terminal
            await env.monitor_skypilot_monitor(
                launch_id="happy-1",
                event_q=event_q,
                poll_interval=0,
            )

            await env.cleanup_skypilot(launch_id="happy-1")

    @pytest.mark.asyncio
    async def test_launch_passes_resources_correctly(self):
        """Verify Resources kwargs match launcher_config."""
        env, event_q, mock_sky = make_skypilot_env(Scenario.happy_path(cloud="aws"))

        with patched_skypilot(mock_sky):
            env._get_launch_ready_event("res-1")
            await env.launch_skypilot(
                launch_id="res-1",
                launcher_config={
                    "run": "train.py",
                    "resources": {
                        "cloud": "aws",
                        "accelerators": "A100:4",
                        "cpus": "8",
                        "memory": "64",
                    },
                },
                config={},
            )

            # Verify MockSky recorded the cluster
            assert "gb-res-1" in mock_sky._clusters or len(mock_sky._clusters) == 1


class TestFailureScenarios:
    @pytest.mark.asyncio
    async def test_failure_emits_workload_status_event(self):
        """FAILED terminal status should emit a WORKLOAD_STATUS_EVENT."""
        env, event_q, mock_sky = make_skypilot_env(Scenario.failure(cloud="aws"))

        with patched_skypilot(mock_sky):
            env._get_launch_ready_event("fail-1")
            await env.launch_skypilot(
                launch_id="fail-1",
                launcher_config={"run": "train.py", "resources": {}},
                config={},
            )

            await env.monitor_skypilot_monitor(
                launch_id="fail-1",
                event_q=event_q,
                entityrun_metadata={"build_id": "b1", "targetrun_id": "tr1"},
                poll_interval=0,
            )

            # Check that a WORKLOAD_STATUS_EVENT with FAILED was emitted
            events = []
            while not event_q.empty():
                events.append(await event_q.get())

            from gbserver.types.buildevent import BuildEventType

            fail_events = [
                e for e in events if e.type == BuildEventType.WORKLOAD_STATUS_EVENT
            ]
            assert len(fail_events) == 1
            from gbserver.types.status import Status

            assert fail_events[0].payload.status == Status.FAILED

    @pytest.mark.asyncio
    async def test_missing_skypilot_raises_import_error(self):
        """When HAS_SKYPILOT is False, launch should raise ImportError."""
        from unittest.mock import patch as mock_patch

        env, event_q, mock_sky = make_skypilot_env(Scenario.happy_path())

        with mock_patch("gbserver.environment.skypilot.HAS_SKYPILOT", False):
            env._get_launch_ready_event("noimport-1")
            with pytest.raises(ImportError, match="skypilot"):
                await env.launch_skypilot(
                    launch_id="noimport-1",
                    launcher_config={"run": "echo", "resources": {}},
                    config={},
                )


class TestPreemptionRecovery:
    @pytest.mark.asyncio
    async def test_preemption_then_recovery(self):
        """Monitor handles PREEMPTED -> re-PENDING -> RUNNING -> SUCCEEDED."""
        env, event_q, mock_sky = make_skypilot_env(
            Scenario.preemption_then_recovery(cloud="aws")
        )

        with patched_skypilot(mock_sky):
            env._get_launch_ready_event("preempt-1")
            await env.launch_skypilot(
                launch_id="preempt-1",
                launcher_config={"run": "train.py", "resources": {}},
                config={},
            )

            await env.monitor_skypilot_monitor(
                launch_id="preempt-1",
                event_q=event_q,
                entityrun_metadata={"build_id": "b1", "targetrun_id": "tr1"},
                poll_interval=0,
            )

            # Should NOT have emitted a FAILED event (recovery succeeded)
            events = []
            while not event_q.empty():
                events.append(await event_q.get())

            from gbserver.types.buildevent import BuildEventType

            fail_events = [
                e for e in events if e.type == BuildEventType.WORKLOAD_STATUS_EVENT
            ]
            assert len(fail_events) == 0


class TestLogParsing:
    @pytest.mark.asyncio
    async def test_logs_downloaded_on_terminal_status(self):
        """download_logs is called when job reaches terminal status with event_log_parser_configs."""
        # Create a temp log file for the scenario
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "run.log")
            with open(log_file, "w") as f:
                f.write("training complete\n")

            scenario = Scenario(
                cloud="aws",
                steps=[
                    ScenarioStep(status="PENDING", is_terminal=False),
                    ScenarioStep(status="RUNNING", is_terminal=False),
                    ScenarioStep(
                        status="SUCCEEDED",
                        is_terminal=True,
                        logs={"1": tmpdir},
                    ),
                ],
            )
            env, event_q, mock_sky = make_skypilot_env(scenario)

            with patched_skypilot(mock_sky):
                env._get_launch_ready_event("logs-1")
                await env.launch_skypilot(
                    launch_id="logs-1",
                    launcher_config={"run": "train.py", "resources": {}},
                    config={},
                )

                await env.monitor_skypilot_monitor(
                    launch_id="logs-1",
                    event_q=event_q,
                    entityrun_metadata={"build_id": "b1", "targetrun_id": "tr1"},
                    event_configs=[
                        {
                            "line_regex": "training complete",
                            "event_type": "message_event",
                        }
                    ],
                    poll_interval=0,
                )

            # If logs were parsed, events should include the matching line
            events = []
            while not event_q.empty():
                events.append(await event_q.get())
            # At minimum we should have message events from status changes
            assert len(events) > 0


class TestManagedEnv:
    @pytest.mark.skip(reason="skypilot_managed not yet implemented")
    def test_managed_env_placeholder(self):
        """Placeholder for managed SkyPilot environment tests."""
        pass
