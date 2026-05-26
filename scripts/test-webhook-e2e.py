#!/usr/bin/env python3
"""
End-to-end test for webhook event persistence.

This script verifies that build events are persisted to the webhook
event storage (write-ahead log) for later delivery by the Phase 2
delivery worker.

Flow:
1. Starts gbserver in standalone mode (SQLite + thread-based builds)
2. Creates an "active" webhook subscription via storage directly
3. Submits a build
4. Waits for the build to complete
5. Queries the webhook event storage for persisted events
6. Verifies events were written correctly

Usage:
    python scripts/test-webhook-e2e.py [--build-dir PATH] [--timeout 120]

Requirements:
    - Activated venv with gbserver installed
    - No external infrastructure needed (uses standalone mode with SQLite)
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
    os.environ["GBSERVER_WEBHOOKS_ENABLED"] = "true"

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


# ─── Subscription Creation ───────────────────────────────────────────────────


def create_active_subscription(space_name: str) -> str:
    """Create a space-wide webhook subscription directly via storage.

    Bypasses URL validation by going directly to the storage layer.
    Created with status="active" so the WebhookEventWriter picks it up.

    Returns the subscription UUID.
    """
    from gbserver.webhooks.models import StoredWebhookSubscription
    from gbserver.webhooks.sql_storage import create_webhook_storage

    storage = create_webhook_storage()
    subscription = StoredWebhookSubscription(
        space_name=space_name,
        build_id=None,
        build_filter=None,
        scope="space",
        status="active",
        active=True,
        webhook_url="https://example.com/e2e-test-endpoint",
        secret="e2e-test-secret",
        event_types=["*"],
        frequency=15,
        created_by="e2e-test-user",
    )
    storage.add(subscription)
    print(f"  Subscription ID: {subscription.uuid}")
    print(f"  Scope:           space-wide (all builds in '{space_name}')")
    print(f"  Status:          active")
    return subscription.uuid


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

    # Submit build (without webhook_url — subscription created directly)
    resp = requests.post(
        f"{base_url}/builds/",
        json={
            "name": "webhook-e2e-test",
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


# ─── Build Polling ───────────────────────────────────────────────────────────


def wait_for_build(server_port: int, build_id: str, timeout: int) -> str:
    """Poll build status until terminal. Returns final status."""
    import requests

    poll_interval = 3
    elapsed = 0

    while elapsed < timeout:
        try:
            r = requests.get(
                f"http://127.0.0.1:{server_port}/api/v1/builds/{build_id}/status",
                timeout=2,
            )
            data = r.json()
            build_status = data.get("build", {}).get("status", "unknown")
        except Exception as e:
            build_status = f"error: {e}"

        print(f"  [{elapsed:3d}s] Build status: {build_status}")

        if build_status in TERMINAL_STATUSES:
            return build_status

        time.sleep(poll_interval)
        elapsed += poll_interval

    return "timeout"


# ─── Event Query ─────────────────────────────────────────────────────────────


def query_persisted_events(subscription_id: str) -> list:
    """Query the webhook event storage for events matching a subscription."""
    from gbserver.webhooks.event_storage import create_webhook_event_storage

    storage = create_webhook_event_storage()
    events = storage.get_pending_for_subscription(subscription_id)
    return events


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end test for webhook event persistence"
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

    # Clean stale SQLite DB to avoid leftover builds from previous runs
    db_path = os.path.join(os.path.expanduser("~"), ".llmb", "llmb-server.db")
    for path in (db_path, f"{db_path}.lck"):
        if os.path.exists(path):
            os.remove(path)

    server_port = args.server_port or get_free_port()
    build_dir = os.path.abspath(args.build_dir)

    print("=" * 60)
    print("  Webhook Event Persistence — End-to-End Test")
    print("=" * 60)
    print()
    print(f"  gbserver port:    {server_port}")
    print(f"  Build directory:  {build_dir}")
    print(f"  Timeout:          {args.timeout}s")
    print()

    # Step 1: Start gbserver
    print("[1/5] Starting gbserver (standalone mode)...")
    start_gbserver(server_port, build_dir)
    print(f"  Server ready at http://127.0.0.1:{server_port}")
    print()

    # Step 2: Create active subscription BEFORE submitting build
    # (avoids race condition where events fire before subscription exists)
    print("[2/5] Creating active webhook subscription...")
    import requests

    spaces_resp = requests.get(
        f"http://127.0.0.1:{server_port}/api/v1/spaces/spaces_for_user"
    )
    spaces = spaces_resp.json().get("spaces", [])
    space_name = spaces[0]["name"] if spaces else "standalone"
    subscription_id = create_active_subscription(space_name)
    print()

    # Step 3: Submit build
    print("[3/5] Submitting build...")
    build_id = submit_build(server_port, build_dir)
    print()

    # Step 4: Wait for build to complete
    print("[4/5] Waiting for build to complete...")
    final_status = wait_for_build(server_port, build_id, args.timeout)
    print()

    # Step 5: Query persisted events
    print("[5/5] Querying persisted webhook events...")
    # Give a moment for any final events to be flushed
    time.sleep(2)
    events = query_persisted_events(subscription_id)

    print()
    print("=" * 60)
    print("  Results")
    print("=" * 60)
    print(f"  Build final status:   {final_status}")
    print(f"  Events persisted:     {len(events)}")

    if final_status == "timeout" and not events:
        print()
        print("  FAIL — Build timed out and no events were persisted!")
        sys.exit(1)

    if events:
        event_types = set()
        for evt in events:
            event_types.add(evt.event_type)
            print(
                f"    [{evt.event_type}] build={evt.build_id[:8]}... "
                f"delivered={evt.delivered}"
            )
        print()
        print(f"  Event types seen:     {sorted(event_types)}")
        print(f"  All undelivered:      {all(not e.delivered for e in events)}")

        event_types_upper = {t.upper() for t in event_types}
        if "STATUS_EVENT" in event_types_upper:
            print()
            print("  SUCCESS — Webhook events persisted end-to-end!")
            print("  (Ready for Phase 2 delivery worker to pick up)")
        else:
            print()
            print("  WARNING — No STATUS_EVENT found in persisted events.")
            sys.exit(1)
    else:
        print()
        print("  FAIL — No events were persisted!")
        print()
        print("  Possible causes:")
        print("  - Subscription was not 'active' when events were processed")
        print("  - Build completed before subscription was created")
        print("  - WebhookEventWriter did not find the subscription")
        sys.exit(1)


if __name__ == "__main__":
    main()
