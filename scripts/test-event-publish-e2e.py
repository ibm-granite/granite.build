#!/usr/bin/env python3
"""
End-to-end test for build event notifications via RabbitMQ.

This script verifies that build events are published to the RabbitMQ
topic exchange and can be consumed by a scoped subscriber.

Flow:
1. Starts gbserver in standalone mode (SQLite + thread-based builds)
2. Provisions scoped RabbitMQ credentials via the subscribe endpoint
3. Connects a consumer to receive events for the build
4. Submits a build
5. Waits for the build to complete
6. Verifies events were received by the consumer

Usage:
    python scripts/test-event-publish-e2e.py [--build-dir PATH] [--timeout 120]

Requirements:
    - Activated venv with gbserver installed
    - Running RabbitMQ instance with management plugin enabled
    - Set RABBITMQ_HOST, RABBITMQ_PORT, GBSERVER_RABBITMQ_MGMT_URL env vars
"""

import os
import sys

# On macOS, the kqueue-based asyncio event loop in daemon threads can starve
# unless stderr is connected to a pipe.
if sys.platform == "darwin" and sys.stderr.isatty():
    _stderr_r, _stderr_w = os.pipe()
    _original_stderr_fd = os.dup(2)
    os.dup2(_stderr_w, 2)
    os.close(_stderr_w)

    def _stderr_pump():
        """Read from pipe and forward to original terminal stderr."""
        with os.fdopen(_stderr_r, "r", errors="replace") as pipe:
            with os.fdopen(_original_stderr_fd, "w") as tty:
                for line in pipe:
                    tty.write(line)
                    tty.flush()

    import threading as _th

    _th.Thread(target=_stderr_pump, daemon=True).start()

import argparse
import asyncio
import io
import json
import socket
import threading
import time
import zipfile
from base64 import b64encode

# Force standard asyncio event loop policy BEFORE any gbserver imports.
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# ─── Configuration ───────────────────────────────────────────────────────────

TERMINAL_STATUSES = {"success", "failed", "cancelled"}

# ─── Helpers ─────────────────────────────────────────────────────────────────


def get_free_port() -> int:
    """Find a free port on localhost."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ─── gbserver Standalone Runner ──────────────────────────────────────────────


def start_gbserver(port: int, build_dir: str) -> threading.Thread:
    """Start gbserver in standalone mode in a background thread."""
    os.environ["GBSERVER_METADATA_STORAGE"] = "sqlite"
    os.environ["GBSERVER_DEFAULT_BUILDRUNNER_TYPE"] = "thread"
    os.environ["GB_ENVIRONMENT"] = "STANDALONE"
    os.environ["GBSERVER_EVENT_PUBLISHING_ENABLED"] = "true"

    from gbserver.commands.command_standalone import _run_standalone

    started = threading.Event()

    def run():
        _run_standalone(
            port=port,
            space_dir=build_dir,
            on_started=started.set,
        )

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    if not started.wait(timeout=30):
        print("FAIL: gbserver failed to start within 30 seconds")
        sys.exit(1)

    # Wait for uvicorn to be fully ready
    for _ in range(40):
        try:
            import requests

            requests.get(f"http://127.0.0.1:{port}/api/v1", timeout=1)
            break
        except Exception:
            time.sleep(0.25)

    return thread


# ─── Build Submission ────────────────────────────────────────────────────────


def submit_build(server_port: int, build_dir: str) -> str:
    """Submit a build via REST API. Returns build_id."""
    import requests

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(build_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, build_dir)
                zf.write(file_path, arcname)
    zip_buffer.seek(0)
    build_archive = b64encode(zip_buffer.read()).decode("utf-8")

    base_url = f"http://127.0.0.1:{server_port}/api/v1"

    # Get space name
    spaces_resp = requests.get(f"{base_url}/spaces/spaces_for_user")
    spaces = spaces_resp.json().get("spaces", [])
    space_name = spaces[0]["name"] if spaces else "standalone"

    # Submit build
    resp = requests.post(
        f"{base_url}/builds/",
        json={
            "name": "event-publish-e2e-test",
            "build_archive": build_archive,
            "space_name": space_name,
            "username": "e2e-test-user",
        },
    )

    if resp.status_code != 200:
        print(f"FAIL: Build submission failed: {resp.status_code} {resp.text}")
        sys.exit(1)

    data = resp.json()
    build_id = data["build_id"]
    print(f"  Build ID: {build_id}")
    print(f"  Space:    {space_name}")
    return build_id


# ─── Event Subscription via RabbitMQ ────────────────────────────────────────


def subscribe_to_build(server_port: int, build_id: str) -> dict:
    """Subscribe to build events via the REST endpoint.

    Returns connection info + credentials for RabbitMQ consumer.
    """
    import requests

    resp = requests.post(
        f"http://127.0.0.1:{server_port}/api/v1/builds/{build_id}/events/subscribe",
        headers={"Authorization": "Bearer e2e-test-token"},
    )

    if resp.status_code != 200:
        print(f"FAIL: Subscribe failed: {resp.status_code} {resp.text}")
        sys.exit(1)

    return resp.json()


def consume_events(subscribe_info: dict, timeout: int) -> list:
    """Connect to RabbitMQ with scoped credentials and consume events.

    Returns list of received event dicts.
    """
    received = []

    async def _consume():
        import aio_pika

        consumer_conn = await aio_pika.connect(
            host=subscribe_info["host"],
            port=subscribe_info["port"],
            login=subscribe_info["username"],
            password=subscribe_info["password"],
        )
        consumer_chan = await consumer_conn.channel()
        queue = await consumer_chan.declare_queue(
            subscribe_info["queue"], exclusive=True
        )
        exchange = await consumer_chan.get_exchange(
            subscribe_info["exchange"], ensure=False
        )
        await queue.bind(exchange, routing_key=subscribe_info["routing_key"])

        async def on_message(msg):
            async with msg.process():
                event = json.loads(msg.body)
                received.append(event)

        await queue.consume(on_message)

        # Wait until timeout or until we see a terminal status event
        start = time.time()
        while time.time() - start < timeout:
            for evt in received:
                if evt.get("status") in TERMINAL_STATUSES:
                    await asyncio.sleep(2)  # Collect any trailing events
                    await consumer_conn.close()
                    return
            await asyncio.sleep(1)

        await consumer_conn.close()

    asyncio.run(_consume())
    return received


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end test for build event notifications"
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=None,
        help="Port for gbserver (default: auto)",
    )
    parser.add_argument(
        "--build-dir",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "test-data",
            "e2e",
            "standalone",
            "standalone-quickstart",
        ),
        help="Path to build directory (default: standalone-quickstart)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Max seconds to wait for build completion (default: 120)",
    )
    args = parser.parse_args()

    if not os.getenv("RABBITMQ_HOST"):
        print("ERROR: RABBITMQ_HOST not set. A running RabbitMQ is required.")
        print()
        print("Quick setup:")
        print(
            "  docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:management"
        )
        print("  export RABBITMQ_HOST=localhost RABBITMQ_PORT=5672")
        print("  export GBSERVER_RABBITMQ_MGMT_URL=http://localhost:15672")
        print(
            "  export GBSERVER_RABBITMQ_MGMT_USER=guest GBSERVER_RABBITMQ_MGMT_PASSWORD=guest"
        )
        sys.exit(1)

    # Clean stale SQLite DB to avoid leftover builds from previous runs
    db_path = os.path.join(os.path.expanduser("~"), ".llmb", "llmb-server.db")
    for path in (db_path, f"{db_path}.lck"):
        if os.path.exists(path):
            os.remove(path)

    server_port = args.server_port or get_free_port()
    build_dir = os.path.abspath(args.build_dir)

    print("=" * 60)
    print("  Build Event Notifications — End-to-End Test")
    print("=" * 60)
    print()
    print(f"  gbserver port:    {server_port}")
    print(f"  Build directory:  {build_dir}")
    print(f"  Timeout:          {args.timeout}s")
    print(
        f"  RabbitMQ:         {os.getenv('RABBITMQ_HOST')}:{os.getenv('RABBITMQ_PORT', '5672')}"
    )
    print()

    # Step 1: Start gbserver
    print("[1/4] Starting gbserver (standalone mode + event publishing)...")
    start_gbserver(server_port, build_dir)
    print(f"  Server ready at http://127.0.0.1:{server_port}")
    print()

    # Step 2: Submit build
    print("[2/4] Submitting build...")
    build_id = submit_build(server_port, build_dir)
    print()

    # Step 3: Subscribe to build events
    print("[3/4] Subscribing to build events via RabbitMQ...")
    subscribe_info = subscribe_to_build(server_port, build_id)
    print(f"  Username:     {subscribe_info['username']}")
    print(f"  Exchange:     {subscribe_info['exchange']}")
    print(f"  Routing key:  {subscribe_info['routing_key']}")
    print(f"  Queue:        {subscribe_info['queue']}")
    print(f"  Expires:      {subscribe_info['expires_at']}")
    print()

    # Step 4: Consume events until build completes
    print("[4/4] Consuming events (waiting for build to complete)...")
    events = consume_events(subscribe_info, timeout=args.timeout)

    print()
    print("=" * 60)
    print("  Results")
    print("=" * 60)
    print(f"  Events received: {len(events)}")
    print()

    if events:
        for evt in events:
            status = evt.get("status", "")
            target = evt.get("target_name", "")
            step = evt.get("step_name", "")
            scope = "build" if not target else ("step" if step else "target")
            print(f"    [{scope:6s}] {status:10s} {evt.get('message', '')}")

        print()
        if any(evt.get("status") in TERMINAL_STATUSES for evt in events):
            print("  PASS — Build events published and consumed end-to-end!")
        else:
            print("  WARNING — No terminal status event received.")
            sys.exit(1)
    else:
        print("  FAIL — No events were received!")
        print()
        print("  Possible causes:")
        print("  - GBSERVER_EVENT_PUBLISHING_ENABLED not set to true")
        print("  - RabbitMQ connection failed")
        print("  - Consumer credentials expired before connection")
        print("  - Build completed before consumer connected")
        sys.exit(1)

    # Cleanup: delete temp RabbitMQ user
    try:
        from gbserver.messaging.rabbitmq_admin import RabbitMQAdmin

        admin = RabbitMQAdmin(
            management_url=os.getenv(
                "GBSERVER_RABBITMQ_MGMT_URL", "http://localhost:15672"
            ),
            admin_user=os.getenv("GBSERVER_RABBITMQ_MGMT_USER", "guest"),
            admin_password=os.getenv("GBSERVER_RABBITMQ_MGMT_PASSWORD", "guest"),
        )
        asyncio.run(admin.delete_user(subscribe_info["username"]))
        print(f"  Cleaned up temp user: {subscribe_info['username']}")
    except Exception as e:
        print(f"  Warning: cleanup failed: {e}")


if __name__ == "__main__":
    main()
