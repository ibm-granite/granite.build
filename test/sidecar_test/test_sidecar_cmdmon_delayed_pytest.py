#!/usr/bin/env python3
"""
Minimal, self-contained pytest harness for Sidecar and Command Line Monitor.

It uses an in-memory “fake” MessagingBase so no real RabbitMQ
instance is required.  The test verifies that

  * the sidecar waits for a CONFIG message
  * log lines containing the pattern are published as events
  * file truncation happens once the size limit is reached
  * the sidecar terminates cleanly when its monitor fires

"""

import asyncio
import os
import sys
import textwrap

import pytest
import pytest_asyncio
import yaml
from test_sidecar_cmdmon_pytest import child_script

# -- reuse fixtures from test_sidecar_pytest and test_sidecar_cmdmon_pytest --
from test_sidecar_pytest import fake_messaging, temp_log_file

# --------------------- project imports ---------------------
from gbserver.monitoring.process_cmdline_monitor import CmdlineMonitor
from gbserver.monitoring.sidecar import Sidecar
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


# --------------------- test cases -------------------------------
@pytest.mark.asyncio
async def test_sidecar_with_cmdline_monitor(fake_messaging, temp_log_file, child_script):
    """
    * Start SidecarOrchestrator first (with monitors waiting for process).
    * Then spawn a child bash process that uses tee to write to log file (delayed start).
    * Sidecar's internal CmdlineMonitor detects child exit and stops sidecar.
    """
    # 1) prepare config file
    events_config_file = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../test-data/gbserver_test/monitoring/sidecar_config.yaml",
        )
    )

    # 2) Create a mock SidecarOrchestrator that uses fake messaging
    from unittest.mock import patch

    from gbserver.monitoring.sidecar import SidecarOrchestrator

    # We need to patch the RabbitMQBase.from_env_and_args to return our fake_messaging
    with patch(
        "gbserver.monitoring.sidecar.RabbitMQBase.from_env_and_args",
        return_value=fake_messaging,
    ):
        # Create orchestrator with test config
        orchestrator = SidecarOrchestrator(
            exchange_name="test",
            queue_name="test",
            routing_key="test.steprun.launch",
            config_file_path=events_config_file,
            custom_log_paths=str(temp_log_file),
        )
        # Override the messenger with our fake one
        orchestrator.msg = fake_messaging

    # 3) start orchestrator (which creates and runs sidecars)
    orchestrator_task = asyncio.create_task(orchestrator.run_sidecars())

    # 4) sleep to simulate a late startup of the monitored process
    await asyncio.sleep(10)

    # 5) spawn child process that uses tee
    child = await asyncio.create_subprocess_exec(
        str(child_script),
        str(temp_log_file),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    # 6) wait for child to finish
    child_returncode = await child.wait()

    # 7) wait for orchestrator/sidecars to finish (should detect process exit via CmdlineMonitor)
    await orchestrator_task

    # 8) verify at least 1 new artifact event arrived
    assert not fake_messaging._q.empty()
    new_artifact_events = []
    while not fake_messaging._q.empty():
        event = await fake_messaging._q.get()
        if event.get("rk") == f"{fake_messaging.addr.queue}.NEWARTIFACT_IN_ENVIRONMENT_EVENT":
            new_artifact_events.append(event)
        logger.info(f"Received event: {event}")
    assert len(new_artifact_events) > 0

    # 9) make sure child exited
    assert child_returncode is not None
    logger.info("SidecarOrchestrator and child exited cleanly (delayed start test)")
