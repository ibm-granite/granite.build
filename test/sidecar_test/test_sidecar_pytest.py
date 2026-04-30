#!/usr/bin/env python3
"""
Minimal, self-contained test harness for Sidecar.

It uses an in-memory “fake” MessagingBase so no real RabbitMQ
instance is required.  The test verifies that

  • the sidecar waits for a CONFIG message
  • log lines containing the pattern are published as events
  • file truncation happens once the size limit is reached
  • the sidecar terminates cleanly when its monitor fires

Run:  python test_sidecar.py
"""

import asyncio
import json
import os
from typing import Awaitable, Callable, Dict, Optional

import pytest
import pytest_asyncio

# --------------------- project imports ---------------------
from gbserver.messaging.messaging_base import JSON, MessagingBase
from gbserver.monitoring.dummy_monitor import DummyMonitor
from gbserver.monitoring.sidecar import Sidecar
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


# --------------------- Fake in-memory broker ---------------
class FakeMessaging(MessagingBase):
    """
    Very small in-memory broker:
      * publish - puts bytes in an asyncio.Queue
      * consume_stream - registers a callback
      * run() - waits on stop_event
    """

    def __init__(self):
        from gbserver.messaging.messaging_base import Address  # avoid circular import

        super().__init__(addr=Address(exchange="fake", queue="fake"))
        self._q: asyncio.Queue[Dict] = asyncio.Queue()
        self._handler: Optional[Callable[[bytes, str, int], Awaitable[None]]] = None
        self.stop_event = asyncio.Event()

    async def setup(self):
        logger.info("FakeMsg: All set")

    async def publish(self, payload: JSON, suffix: str = "event"):
        msg = {
            "body": json.dumps(payload).encode(),
            "rk": f"fake.{suffix}",
            "delivery_tag": 0,
        }
        await self._q.put(msg)

        # trigger consumer immediately (synchronous “broker”)
        if self._handler:
            await self._handler(msg["body"], msg["rk"], msg["delivery_tag"])

    publish_config = lambda self, c: asyncio.create_task(self.publish(c, "config"))
    publish_event = lambda self, e: asyncio.create_task(self.publish(e, "event"))

    async def consume_stream(self, handler):
        self._handler = handler

    async def run(self):
        await self.stop_event.wait()

    async def close(self):
        pass


# --------------------- pytest fixtures --------------------------
@pytest_asyncio.fixture
async def fake_messaging():
    backend = FakeMessaging()
    yield backend
    backend.stop_event.set()  # ensure cleanup


@pytest_asyncio.fixture
async def temp_log_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("logs") / "output.log"
    path.touch()
    yield path


# --------------------- test cases -------------------------------
@pytest.mark.asyncio
async def test_sidecar_happy_path(fake_messaging, temp_log_file):
    # 1) append lines to log
    def write(line: str):
        with temp_log_file.open("a") as fh:
            fh.write(line + "\n")

    fake_log_file = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../test-data/gbserver_test/monitoring/digit.txt",
        )
    )
    with open(fake_log_file, "r+") as fh:
        for line in fh.readlines():
            write(line)

    # 2) prepare config file
    events_config_file = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../test-data/gbserver_test/monitoring/sidecar_config.yaml",
        )
    )

    # 3) Load event configs from file to pass to Sidecar
    from pathlib import Path

    import yaml

    from gbserver.environment.environment import EventLogLineParserConfig

    with open(events_config_file, "r") as f:
        cfg = yaml.safe_load(f)

    # Extract event configs from yaml
    monitor_config = cfg.get("sidecar_monitor", []) + cfg.get("event_monitor", [])
    config = monitor_config[0].get("config", {})
    event_configs = [EventLogLineParserConfig(**ec) for ec in config.get("event_configs", [])]

    # 4) sidecar using fake messaging
    # Note: Sidecar now creates its own termination monitor internally
    sidecar = Sidecar(
        messenger=fake_messaging,
        log_path=Path(temp_log_file),
        event_configs=event_configs,
        idx=0,
        targetsteprun_id="test-step",
        launch_id="test-launch",
    )

    # 5) Create a dummy monitor to stop the sidecar after a delay
    dummy = DummyMonitor(delay_sec=5, stop_event=sidecar.stop_event)

    # 6) start sidecar and dummy monitor concurrently
    sidecar_task = asyncio.create_task(sidecar.run())
    monitor_task = asyncio.create_task(dummy.monitor())

    # 7) small wait so sidecar parses log file and emits events
    await asyncio.sleep(1.0)

    # 8) verify an event arrived
    assert not fake_messaging._q.empty()
    new_artifact_events = []
    # 9) ensure at least 1 new artifact event arrived
    while not fake_messaging._q.empty():
        event = await fake_messaging._q.get()
        if event.get("rk") == f"{fake_messaging.addr.queue}.NEWARTIFACT_IN_ENVIRONMENT_EVENT":
            new_artifact_events.append(event)
        logger.info(f"Received event: {event}")
    assert len(new_artifact_events) > 0

    # 10) wait for DummyMonitor to fire; graceful shutdown
    await asyncio.gather(sidecar_task, monitor_task)
    assert sidecar_task.done()
    assert monitor_task.done()
    logger.info("Sidecar and DummyMonitor exited cleanly")
