#!/usr/bin/env python3
"""
Test harness for SidecarOrchestrator with multiple user processes and CmdlineMonitor.

Tests the case where:
- Two child bash processes use tee to write to separate log files
- SidecarOrchestrator creates two Sidecars, one per process
- Each Sidecar's internal CmdlineMonitor detects when its process exits
- Processes start before orchestrator begins monitoring
"""

import asyncio
import os
import sys
import textwrap

import pytest
import pytest_asyncio


# ---------- reuse the fixtures from test_sidecar_pytest ----------
from test_sidecar_pytest import fake_messaging

# --------------------- project imports ---------------------
from gbserver.monitoring.sidecar import SidecarOrchestrator
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


# --------------------- pytest fixtures --------------------------
@pytest_asyncio.fixture
async def child_scripts(tmp_path_factory):
    """
    Create two bash scripts that simulate two user processes.
    Each script uses tee to write to a log file.
    """
    script_dir = tmp_path_factory.mktemp("child")

    # Script 1: reads from digit.txt
    script_path_0 = script_dir / "reader_0.sh"
    fake_log_file_0 = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../test-data/gbserver_test/monitoring/digit.txt",
        )
    )
    script_path_0.write_text(textwrap.dedent(f"""#!/bin/bash
            cat {fake_log_file_0} | tee $1
            sleep 2
            """))
    script_path_0.chmod(0o755)

    # Script 2: reads from tuning_1.txt (different content)
    script_path_1 = script_dir / "reader_1.sh"
    fake_log_file_1 = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../test-data/gbserver_test/monitoring/tuning_1.txt",
        )
    )
    script_path_1.write_text(textwrap.dedent(f"""#!/bin/bash
            cat {fake_log_file_1} | tee $1
            sleep 2
            """))
    script_path_1.chmod(0o755)

    return [script_path_0, script_path_1]


@pytest_asyncio.fixture
async def temp_log_files(tmp_path_factory):
    """Create two temporary log files for two user processes."""
    log_dir = tmp_path_factory.mktemp("logs")
    log_file_0 = log_dir / "output.log.0"
    log_file_1 = log_dir / "output.log.1"
    log_file_0.touch()
    log_file_1.touch()
    yield [log_file_0, log_file_1]


# --------------------- test cases -------------------------------
@pytest.mark.asyncio
async def test_multi_sidecar_with_cmdline_monitor(fake_messaging, temp_log_files, child_scripts):
    """
    Test SidecarOrchestrator with two user processes and CmdlineMonitor.
    - Spawn two child bash processes that use tee to write to separate log files
    - SidecarOrchestrator creates two Sidecars
    - Each Sidecar's internal CmdlineMonitor detects its process exit
    - Tests the case where processes start before monitoring begins
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

    # 4) spawn both child processes that use tee (start immediately, before orchestrator)
    children = []
    for idx, (script, log_file) in enumerate(zip(child_scripts, temp_log_files)):
        child = await asyncio.create_subprocess_exec(
            str(script),
            str(log_file),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        children.append(child)
        logger.info(f"Started child process {idx} (PID: {child.pid})")

    # 5) start orchestrator (which creates and runs sidecars)
    # This tests the case where processes start before monitoring begins
    orchestrator_task = asyncio.create_task(orchestrator.run_sidecars())

    # 6) wait for both children to finish
    returncodes = await asyncio.gather(*[child.wait() for child in children])

    # 7) wait for orchestrator/sidecars to finish (should detect process exit via CmdlineMonitor)
    await orchestrator_task

    # 8) verify events arrived from both processes
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

    # 9) make sure both children exited
    for idx, returncode in enumerate(returncodes):
        assert returncode is not None
        logger.info(f"Child process {idx} exited with code {returncode}")

    logger.info("SidecarOrchestrator and children exited cleanly")

    # Cleanup
    del os.environ["NUM_USER_PROCESSES"]
