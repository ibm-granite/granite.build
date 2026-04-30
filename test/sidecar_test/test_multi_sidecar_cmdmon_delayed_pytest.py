#!/usr/bin/env python3
"""
Test harness for SidecarOrchestrator with multiple user processes and delayed start.

Tests the case where:
- SidecarOrchestrator starts first (with monitors waiting for processes)
- Then two child bash processes start after a delay
- Each Sidecar's internal CmdlineMonitor detects when its process exits
- Tests delayed/late-starting processes
"""

import asyncio
import os
import sys

import pytest
import pytest_asyncio


from test_multi_sidecar_cmdmon_pytest import child_scripts, temp_log_files

# ---------- reuse the fixtures from other tests ----------
from test_sidecar_pytest import fake_messaging

# --------------------- project imports ---------------------
from gbserver.monitoring.sidecar import SidecarOrchestrator
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


# --------------------- test cases -------------------------------
@pytest.mark.asyncio
async def test_multi_sidecar_with_cmdline_monitor_delayed(
    fake_messaging, temp_log_files, child_scripts
):
    """
    Test SidecarOrchestrator with delayed start of two user processes.
    - Start SidecarOrchestrator first (with monitors waiting for processes)
    - Then spawn two child bash processes after 10 seconds
    - Each Sidecar's internal CmdlineMonitor detects its process exit
    - Tests the CmdlineMonitor's ability to detect process startup and completion
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

    # 4) start orchestrator (which creates and runs sidecars)
    orchestrator_task = asyncio.create_task(orchestrator.run_sidecars())

    # 5) sleep to simulate a late startup of the monitored processes
    await asyncio.sleep(10)

    # 6) spawn both child processes that use tee
    children = []
    for idx, (script, log_file) in enumerate(zip(child_scripts, temp_log_files)):
        child = await asyncio.create_subprocess_exec(
            str(script),
            str(log_file),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        children.append(child)
        logger.info(f"Started child process {idx} (PID: {child.pid}) after delay")

    # 7) wait for both children to finish
    returncodes = await asyncio.gather(*[child.wait() for child in children])

    # 8) wait for orchestrator/sidecars to finish (should detect process exit via CmdlineMonitor)
    await orchestrator_task

    # 9) verify events arrived from both processes
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

    # Should have events from both log files (at least 3: 1 from digit.txt, 2 from tuning_1.txt)
    assert len(new_artifact_events) >= 3
    logger.info(f"Received {len(new_artifact_events)} artifact events from 2 sidecars")

    # 10) make sure both children exited
    for idx, returncode in enumerate(returncodes):
        assert returncode is not None
        logger.info(f"Child process {idx} exited with code {returncode}")

    logger.info("SidecarOrchestrator and children exited cleanly (delayed start test)")

    # Cleanup
    del os.environ["NUM_USER_PROCESSES"]
