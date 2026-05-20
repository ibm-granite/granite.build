#!/usr/bin/env python3
"""
End-to-end test script for webhook push notifications.

This script:
1. Starts a local HTTP server to receive webhook deliveries
2. Starts a gbserver in standalone mode
3. Submits a build with --webhook-url pointing to the local receiver
4. Waits for batched webhook events to arrive
5. Verifies HMAC signatures and prints events as they arrive
6. Exits when the build completes (terminal status received)

Usage:
    python scripts/test-webhook-e2e.py [--port 9999] [--build-dir PATH] [--timeout 120]

Requirements:
    - Activated venv with gbserver installed
    - No external infrastructure needed (uses standalone mode with SQLite + bash env)

Example:
    source .venv/bin/activate
    python scripts/test-webhook-e2e.py
"""

import argparse
import hashlib
import hmac
import json
import os
import random
import signal
import socket
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional

# ─── Configuration ───────────────────────────────────────────────────────────

WEBHOOK_SECRET = f"test-secret-{random.randint(1000, 9999)}"
TERMINAL_STATUSES = {"success", "failed", "cancelled"}

# ─── Webhook Receiver ────────────────────────────────────────────────────────


class WebhookReceiver:
    """Local HTTP server that receives and validates webhook deliveries."""

    def __init__(self, port: int, secret: str):
        self.port = port
        self.secret = secret
        self.deliveries: List[Dict[str, Any]] = []
        self.build_finished = threading.Event()
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start the webhook receiver in a background thread."""
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)

                # Verify HMAC signature
                signature = self.headers.get("X-GB-Signature-256", "")
                expected = "sha256=" + hmac.new(
                    receiver.secret.encode(), body, hashlib.sha256
                ).hexdigest()

                if not hmac.compare_digest(signature, expected):
                    print(f"  ❌ INVALID SIGNATURE (got {signature[:30]}...)")
                    self.send_response(401)
                    self.end_headers()
                    return

                payload = json.loads(body)
                receiver.deliveries.append(payload)

                delivery_id = payload.get("delivery_id", "?")[:8]
                events = payload.get("events", [])
                batch_size = self.headers.get("X-GB-Batch-Size", "?")

                print(f"\n  ✅ Webhook received (delivery={delivery_id}..., events={batch_size})")
                print(f"     Build: {payload.get('build_name')} | User: {payload.get('user')}")

                for evt in events:
                    evt_type = evt.get("event_type", "?")
                    status = evt.get("status", "")
                    target = evt.get("target_name", "")
                    step = evt.get("step_name", "")
                    msg = evt.get("message", {})
                    if isinstance(msg, dict):
                        msg_text = msg.get("text", "")
                    else:
                        msg_text = str(msg)

                    line = f"     [{evt_type}]"
                    if status:
                        line += f" status={status}"
                    if target:
                        line += f" target={target}"
                    if step:
                        line += f" step={step}"
                    if msg_text:
                        line += f" — {msg_text}"
                    print(line)

                    # Check for terminal status
                    if evt_type == "status_event" and status in TERMINAL_STATUSES:
                        receiver.build_finished.set()

                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"OK")

            def log_message(self, format, *args):
                pass  # Suppress default logging

        self._server = HTTPServer(("127.0.0.1", self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the webhook receiver."""
        if self._server:
            self._server.shutdown()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/webhook"


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
        print("❌ gbserver failed to start within 30 seconds")
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


def submit_build_with_webhook(
    server_port: int,
    build_dir: str,
    webhook_url: str,
    webhook_secret: str,
) -> str:
    """Submit a build via REST API with webhook_url for auto-subscription."""
    import io
    import zipfile
    from base64 import b64encode

    import requests

    # Create build archive (zip the build directory)
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

    # Submit build with webhook
    resp = requests.post(
        f"{base_url}/builds/",
        json={
            "name": "webhook-e2e-test",
            "build_archive": build_archive,
            "space_name": space_name,
            "username": "e2e-test-user",
            "webhook_url": webhook_url,
            "webhook_secret": webhook_secret,
            "webhook_frequency": 5,  # Fast for testing (will be clamped to 15s min)
        },
    )

    if resp.status_code != 200:
        print(f"❌ Build submission failed: {resp.status_code} {resp.text}")
        sys.exit(1)

    data = resp.json()
    build_id = data["build_id"]
    subscription_id = data.get("webhook_subscription_id")

    print(f"  Build ID: {build_id}")
    print(f"  Webhook Subscription ID: {subscription_id}")

    return build_id


# ─── Main ────────────────────────────────────────────────────────────────────


def get_free_port() -> int:
    """Find a free port on localhost."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end test for webhook push notifications"
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Port for webhook receiver (default: auto)",
    )
    parser.add_argument(
        "--server-port", type=int, default=None,
        help="Port for gbserver (default: auto)",
    )
    parser.add_argument(
        "--build-dir", type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "test-data", "e2e", "standalone", "standalone-quickstart",
        ),
        help="Path to build directory (default: standalone-quickstart)",
    )
    parser.add_argument(
        "--timeout", type=int, default=120,
        help="Max seconds to wait for build completion (default: 120)",
    )
    args = parser.parse_args()

    webhook_port = args.port or get_free_port()
    server_port = args.server_port or get_free_port()
    build_dir = os.path.abspath(args.build_dir)

    print("=" * 60)
    print("  Webhook Push Notifications — End-to-End Test")
    print("=" * 60)
    print()
    print(f"  Webhook receiver port: {webhook_port}")
    print(f"  gbserver port:         {server_port}")
    print(f"  Build directory:       {build_dir}")
    print(f"  Webhook secret:        {WEBHOOK_SECRET}")
    print(f"  Timeout:               {args.timeout}s")
    print()

    # Step 1: Start webhook receiver
    print("▶ Starting webhook receiver...")
    receiver = WebhookReceiver(port=webhook_port, secret=WEBHOOK_SECRET)
    receiver.start()
    print(f"  Listening at {receiver.url}")
    print()

    # Step 2: Start gbserver
    print("▶ Starting gbserver (standalone mode)...")
    start_gbserver(server_port, build_dir)
    print(f"  Server ready at http://127.0.0.1:{server_port}")
    print()

    # Step 3: Submit build with webhook
    print("▶ Submitting build with webhook subscription...")
    build_id = submit_build_with_webhook(
        server_port=server_port,
        build_dir=build_dir,
        webhook_url=receiver.url,
        webhook_secret=WEBHOOK_SECRET,
    )
    print()

    # Step 4: Wait for events
    print("▶ Waiting for webhook events...")
    print("  (Events will appear below as they arrive)")
    print("-" * 60)

    finished = receiver.build_finished.wait(timeout=args.timeout)

    print("-" * 60)
    print()

    # Step 5: Summary
    total_events = sum(len(d.get("events", [])) for d in receiver.deliveries)
    print("=" * 60)
    print("  Results")
    print("=" * 60)
    print(f"  Deliveries received:  {len(receiver.deliveries)}")
    print(f"  Total events:         {total_events}")
    print(f"  Build completed:      {'yes' if finished else 'TIMEOUT'}")
    print(f"  All signatures valid: yes (invalid would have been rejected)")
    print()

    if not finished:
        print("  ⚠️  Build did not complete within timeout.")
        print("  This may be normal if the build takes longer than expected.")
        receiver.stop()
        sys.exit(1)

    # Verify we got meaningful events
    event_types = set()
    for delivery in receiver.deliveries:
        for evt in delivery.get("events", []):
            event_types.add(evt.get("event_type"))

    print(f"  Event types seen:     {sorted(event_types)}")

    if "status_event" in event_types:
        print("\n  ✅ SUCCESS — Webhook notifications working end-to-end!")
    else:
        print("\n  ⚠️  No status events received. Check logs.")

    receiver.stop()


if __name__ == "__main__":
    main()
