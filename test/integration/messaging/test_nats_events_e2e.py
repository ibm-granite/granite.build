"""End-to-end test: publish and subscribe to build events via NATS.

Requires nats-server on PATH and nats-py installed. Skipped otherwise.
"""

import asyncio
import json
import shutil
import socket
import subprocess
import time

import pytest

HAS_NATS_SERVER = shutil.which("nats-server") is not None
HAS_NATS_PY = False
try:
    import nats  # noqa: F401

    HAS_NATS_PY = True
except ImportError:
    pass

pytestmark = pytest.mark.skipif(
    not (HAS_NATS_SERVER and HAS_NATS_PY),
    reason="nats-server not on PATH or nats-py not installed",
)


@pytest.fixture
def nats_server(tmp_path):
    """Start and stop a temporary nats-server with JetStream."""
    port = 14222  # non-default to avoid conflicts
    proc = subprocess.Popen(
        ["nats-server", "-js", "-p", str(port), "-sd", str(tmp_path / "nats-data")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait for server to be ready
    for _ in range(50):
        try:
            s = socket.create_connection(("localhost", port), timeout=0.2)
            s.close()
            break
        except (ConnectionRefusedError, OSError):
            time.sleep(0.1)
    yield f"nats://localhost:{port}"
    proc.terminate()
    proc.wait(timeout=5)


@pytest.mark.asyncio
async def test_publish_and_consume_via_nats(nats_server):
    """Publish an event and verify a subscriber receives it."""
    from gbserver.messaging.messaging_base import Address
    from gbserver.messaging.nats_messaging import NATSMessaging

    build_id = "test-build-e2e-001"
    subject_queue = f"build.{build_id}"

    # Set up a single NATSMessaging instance for publishing
    pub = NATSMessaging(
        addr=Address(exchange=None, queue=subject_queue, routing_key=None),
        nats_url=nats_server,
    )
    await pub.setup()

    # Set up subscriber on the same subject
    sub = NATSMessaging(
        addr=Address(exchange=None, queue=subject_queue, routing_key=None),
        nats_url=nats_server,
    )
    await sub.setup()

    received = []

    async def handler(data: bytes, routing_key: str):
        received.append(json.loads(data))

    # Start consuming in background
    consume_task = asyncio.create_task(sub.consume_stream(handler))

    # Give consumer time to subscribe
    await asyncio.sleep(0.5)

    # Publish an event
    event_payload = {
        "build_id": build_id,
        "event_type": "status_event",
        "timestamp": int(time.time()),
        "target_name": "training",
        "step_name": "space://steps/sft",
        "source": "test",
        "status": "running",
        "message": "Step started",
    }
    await pub.publish(payload=event_payload, suffix="status_event")

    # Wait for delivery
    await asyncio.sleep(1.0)

    # Cleanup
    consume_task.cancel()
    try:
        await consume_task
    except asyncio.CancelledError:
        pass
    await sub.close()
    await pub.close()

    assert len(received) == 1
    assert received[0]["build_id"] == build_id
    assert received[0]["status"] == "running"
    assert received[0]["event_type"] == "status_event"
