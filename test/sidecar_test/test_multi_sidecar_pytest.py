#!/usr/bin/env python3
"""
Test harness for SidecarOrchestrator with multiple user processes.

Tests the case where:
- Multiple user processes write to separate log files
- SidecarOrchestrator creates one Sidecar per log file
- Each Sidecar has its own LogFileMonitor and uses DummyMonitor for termination
- No actual child processes involved (logs are pre-populated)
"""

import asyncio
import os
from pathlib import Path

import pytest
import pytest_asyncio

# ---------- reuse the fixtures from test_sidecar_pytest ----------
from test_sidecar_pytest import fake_messaging

# --------------------- project imports ---------------------
from gbserver.monitoring.dummy_monitor import DummyMonitor
from gbserver.monitoring.sidecar import SidecarOrchestrator
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


# --------------------- pytest fixtures --------------------------
@pytest_asyncio.fixture
async def temp_log_files(tmp_path_factory):
    """Create two temporary log files for two user processes."""
    log_dir = tmp_path_factory.mktemp("logs")
    log_file_0 = log_dir / "output.log.0"
    log_file_1 = log_dir / "output.log.1"

    # Pre-populate log files with data from digit.txt
    fake_log_file = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../test-data/gbserver_test/monitoring/digit.txt",
        )
    )

    with open(fake_log_file, "r") as src:
        content = src.read()
        log_file_0.write_text(content)
        log_file_1.write_text(content)

    yield [log_file_0, log_file_1]


# --------------------- test cases -------------------------------
@pytest.mark.asyncio
async def test_multi_sidecar_happy_path(fake_messaging, temp_log_files):
    """
    Test SidecarOrchestrator with two user processes.
    - Pre-populated log files for two processes
    - SidecarOrchestrator creates two Sidecars
    - Each Sidecar monitors its own log file
    - DummyMonitor stops the sidecars after a delay
    """
    # 1) prepare config file
    events_config_file = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../test-data/gbserver_test/monitoring/sidecar_config.yaml",
        )
    )

    # 2) Set NUM_USER_PROCESSES environment variable
    os.environ["NUM_USER_PROCESSES"] = "2"

    # 3) Create a mock SidecarOrchestrator that uses fake messaging
    from unittest.mock import patch

    # We need to patch the RabbitMQBase.from_env_and_args to return our fake_messaging
    with patch(
        "gbserver.monitoring.sidecar.RabbitMQBase.from_env_and_args",
        return_value=fake_messaging,
    ):
        # Create orchestrator with test config
        log_paths_str = ",".join([str(log) for log in temp_log_files])
        orchestrator = SidecarOrchestrator(
            exchange_name="test",
            queue_name="test",
            routing_key="test.steprun.launch",
            config_file_path=events_config_file,
            custom_log_paths=log_paths_str,
        )
        # Override the messenger with our fake one
        orchestrator.msg = fake_messaging

    # 4) Create a dummy monitor to stop the orchestrator after a delay
    # We'll use the first sidecar's stop_event, but we need to create a shared one
    shared_stop_event = asyncio.Event()
    dummy = DummyMonitor(delay_sec=5, stop_event=shared_stop_event)

    # 5) Patch the sidecars to use the shared stop event
    async def run_orchestrator_with_dummy():
        """Run orchestrator but replace stop events with shared one."""
        await orchestrator.msg.setup()

        logger.info(
            "[Test] Reading configuration from file %s",
            orchestrator.config_file_path,
        )

        sidecar_configs = orchestrator._load_config()

        logger.info(
            "[Test] Starting %d sidecars",
            len(sidecar_configs),
        )

        # Create all sidecars with shared stop event
        for idx, (logfile_path, event_configs) in enumerate(sidecar_configs.items()):
            from gbserver.monitoring.sidecar import Sidecar

            sidecar = Sidecar(
                messenger=orchestrator.msg,
                log_path=logfile_path,
                event_configs=event_configs,
                idx=idx,
                targetsteprun_id=orchestrator.targetsteprun_id,
                launch_id=orchestrator.launch_id,
            )
            # Replace stop_event with shared one
            sidecar.stop_event = shared_stop_event
            orchestrator.sidecars.append(sidecar)

        # Run all sidecars as concurrent tasks
        sidecar_tasks = [
            asyncio.create_task(sidecar.run(), name=f"sidecar_{i}")
            for i, sidecar in enumerate(orchestrator.sidecars)
        ]

        # Wait for all sidecars to complete
        results = await asyncio.gather(*sidecar_tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("[Test] Sidecar %s exited with error: %s", i, result)
            else:
                logger.info("[Test] Sidecar %s exited cleanly.", i)

        logger.info("[Test] All sidecars have finished.")

    # 6) start orchestrator and dummy monitor concurrently
    orchestrator_task = asyncio.create_task(run_orchestrator_with_dummy())
    monitor_task = asyncio.create_task(dummy.monitor())

    # 7) small wait so sidecars parse log files and emit events
    await asyncio.sleep(1.0)

    # 8) verify events arrived (should have events from both log files)
    assert not fake_messaging._q.empty()
    new_artifact_events = []
    while not fake_messaging._q.empty():
        event = await fake_messaging._q.get()
        if (
            event.get("rk", "").lower()
            == f"{fake_messaging.addr.queue}.NEWARTIFACT_IN_ENVIRONMENT_EVENT".lower()
        ):
            new_artifact_events.append(event)
        logger.info(f"Received event: {event}")

    # Should have at least 2 events (one from each log file)
    assert len(new_artifact_events) >= 2
    logger.info(f"Received {len(new_artifact_events)} artifact events from 2 sidecars")

    # 9) wait for DummyMonitor to fire; graceful shutdown
    await asyncio.gather(orchestrator_task, monitor_task)
    assert orchestrator_task.done()
    assert monitor_task.done()
    logger.info("SidecarOrchestrator and DummyMonitor exited cleanly")

    # Cleanup
    del os.environ["NUM_USER_PROCESSES"]
