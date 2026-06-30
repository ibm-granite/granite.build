#!/usr/bin/env python3
"""E2E test: subscribe to RabbitMQ and verify event delivery.

Usage:
    export GBSERVER_RABBITMQ_MGMT_URL='https://<host>:<mgmt-port>'
    export GBSERVER_RABBITMQ_MGMT_USER='admin'
    export GBSERVER_RABBITMQ_MGMT_PASSWORD='<password>'
    export RABBITMQ_HOST='<host>'  # optional; derived from mgmt URL if unset
    export GBSERVER_RABBITMQ_AMQP_PORT='<amqp-port>'
    export RABBITMQ_CA_CERT='/path/to/ca.pem'  # optional; enables TLS verification

    python scripts/e2e_subscribe_test.py --build-id <build-id> [--self-test] [--timeout 60]

If RABBITMQ_CA_CERT is set and points to a valid file, TLS verification is enabled.
Otherwise, TLS connections skip verification (suitable for testing only).

--self-test: publish simulated events to verify the pipeline without a live server.
"""

import argparse
import asyncio
import json
import os
import signal
import ssl
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


async def publish_simulated_events(
    build_id: str,
    exchange_name: str,
    host: str,
    port: int,
    tls: bool,
    ssl_ctx,
    mgmt_user: str,
    mgmt_password: str,
):
    """Publish simulated build lifecycle events (as the server would)."""
    import time

    import aio_pika

    # NOTE: Reuses management credentials for AMQP publish — valid for self-test
    # since the admin user has full permissions. In production, the server uses
    # separate RABBITMQ_USERNAME/PASSWORD for its publish connection.
    connect_kwargs = dict(host=host, port=port, login=mgmt_user, password=mgmt_password)
    if tls:
        connect_kwargs.update(ssl=True, ssl_context=ssl_ctx)

    conn = await aio_pika.connect(**connect_kwargs)
    chan = await conn.channel()
    exchange = await chan.declare_exchange(
        exchange_name, aio_pika.ExchangeType.TOPIC, durable=True
    )

    events = [
        {"event_type": "status_event", "status": "pending", "message": "Build queued"},
        {
            "event_type": "status_event",
            "status": "running",
            "message": "Build started",
            "target_name": "byoi",
        },
        {
            "event_type": "message_event",
            "message": "Pulling input artifacts...",
            "target_name": "byoi",
            "step_name": "lhpull",
        },
        {
            "event_type": "status_event",
            "status": "running",
            "message": "Step byoi running",
            "target_name": "byoi",
            "step_name": "byoi",
        },
        {
            "event_type": "status_event",
            "status": "success",
            "message": "Build completed successfully",
            "target_name": "byoi",
        },
    ]

    print(f"\n[publisher] Sending {len(events)} simulated events (1s apart)...")
    for i, evt in enumerate(events):
        payload = {
            "build_id": build_id,
            "timestamp": int(time.time()),
            "target_name": evt.get("target_name", ""),
            "step_name": evt.get("step_name", ""),
            "source": "e2e-self-test",
            **evt,
        }
        msg = aio_pika.Message(json.dumps(payload).encode())
        routing_key = f"build.{build_id}.{evt['event_type']}"
        await exchange.publish(msg, routing_key=routing_key)
        print(
            f"[publisher] [{i+1}/{len(events)}] "
            f"{evt['event_type']} -> {evt.get('status', evt.get('message', '')[:30])}"
        )
        await asyncio.sleep(1)

    await conn.close()
    print("[publisher] Done publishing.\n")


async def subscribe_and_listen(
    build_id: str, timeout_seconds: int = 300, self_test: bool = False
):
    """Provision scoped credentials and listen for build events."""
    import aio_pika

    from gbserver.messaging.rabbitmq_admin import RabbitMQAdmin

    mgmt_url = os.environ["GBSERVER_RABBITMQ_MGMT_URL"]
    mgmt_user = os.environ["GBSERVER_RABBITMQ_MGMT_USER"]
    mgmt_password = os.environ["GBSERVER_RABBITMQ_MGMT_PASSWORD"]
    host = os.environ["RABBITMQ_HOST"]
    port = int(os.environ.get("GBSERVER_RABBITMQ_AMQP_PORT", "5672"))
    tls = os.environ.get("RABBITMQ_TLS", "true").lower() in ("true", "1")
    ca_cert = os.environ.get("RABBITMQ_CA_CERT", "")
    exchange_name = os.environ.get("GBSERVER_BUILD_EVENTS_EXCHANGE", "build-events")

    ssl_ctx = None
    if tls:
        if ca_cert and os.path.isfile(ca_cert):
            ssl_ctx = ssl.create_default_context(cafile=ca_cert)
        else:
            ssl_ctx = ssl.create_default_context()
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE

    print(f"\n{'='*60}")
    print("  E2E RabbitMQ Subscription Test")
    print(f"{'='*60}")
    print(f"  Build ID:  {build_id}")
    print(f"  AMQP Host: {host}:{port} (TLS={tls})")
    print(f"  CA Cert:   {ca_cert or '(none — verification disabled)'}")
    print(f"  Exchange:  {exchange_name}")
    print(f"  Timeout:   {timeout_seconds}s")
    print(f"{'='*60}\n")

    print("[1/4] Provisioning scoped consumer credentials...")
    mgmt_tls_verify: bool | str = (
        ca_cert if (ca_cert and os.path.isfile(ca_cert)) else False
    )
    admin = RabbitMQAdmin(
        management_url=mgmt_url,
        admin_user=mgmt_user,
        admin_password=mgmt_password,
        tls_verify=mgmt_tls_verify,
    )
    creds = await admin.create_scoped_user(
        build_id=build_id,
        exchange=exchange_name,
        ttl_seconds=timeout_seconds + 60,
    )
    scoped_user = creds["username"]
    print(f"       User: {scoped_user}")
    print(
        f"       Expires: "
        f"{datetime.fromtimestamp(creds['expires_at'], tz=timezone.utc).isoformat()}"
    )

    print("\n[2/4] Connecting as scoped consumer...")
    consumer_conn = await aio_pika.connect(
        host=host,
        port=port,
        login=creds["username"],
        password=creds["password"],
        ssl=tls,
        ssl_context=ssl_ctx,
    )
    print("       Connected!")

    print(f"\n[3/4] Binding queue to routing key: build.{build_id}.#")
    consumer_chan = await consumer_conn.channel()
    queue = await consumer_chan.declare_queue(
        f"events.{build_id}.e2e-test",
        auto_delete=True,
    )
    consumer_exchange = await consumer_chan.get_exchange(exchange_name, ensure=False)
    await queue.bind(consumer_exchange, routing_key=f"build.{build_id}.#")
    print("       Queue bound, waiting for events...")

    print("\n[4/4] Listening for events (Ctrl+C to stop)...\n")

    event_count = 0
    stop_event = asyncio.Event()

    def handle_signal():
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal)

    async def on_message(message: aio_pika.abc.AbstractIncomingMessage):
        nonlocal event_count
        async with message.process():
            event_count += 1
            try:
                payload = json.loads(message.body)
            except json.JSONDecodeError:
                payload = {"raw": message.body.decode(errors="replace")}

            ts = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
            event_type = payload.get("event_type", "unknown")
            target = payload.get("target_name", "")
            step = payload.get("step_name", "")
            status = payload.get("status", "")
            msg = payload.get("message", "")

            location = f"{target}/{step}" if step else target
            print(
                f"  [{ts}] #{event_count:03d} {event_type:<25s} "
                f"| {location:<20s} | {status}"
            )
            if msg:
                print(f"           {msg[:100]}")
            if event_count <= 3:
                print(f"           {json.dumps(payload, indent=2)[:500]}")
            print()

            if event_type == "status_event" and status in (
                "success",
                "failed",
                "cancelled",
            ):
                print(f"  Build reached terminal state: {status}")
                stop_event.set()

    await queue.consume(on_message)

    if self_test:
        asyncio.create_task(
            publish_simulated_events(
                build_id,
                exchange_name,
                host,
                port,
                tls,
                ssl_ctx,
                mgmt_user,
                mgmt_password,
            )
        )

    try:
        await asyncio.wait_for(stop_event.wait(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        print(f"\n  Timeout reached ({timeout_seconds}s)")

    print(f"\n{'='*60}")
    print(f"  RESULT: Received {event_count} event(s) for build {build_id[:8]}...")
    print(f"{'='*60}\n")

    await consumer_conn.close()
    await admin.delete_user(scoped_user)
    print(f"[cleanup] Deleted scoped user: {scoped_user}")

    return event_count


def main():
    parser = argparse.ArgumentParser(
        description="E2E: subscribe to RabbitMQ and verify event delivery"
    )
    parser.add_argument("--build-id", required=True, help="Build ID to subscribe to")
    parser.add_argument(
        "--timeout", type=int, default=300, help="Max seconds to listen (default: 300)"
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Publish simulated events to verify the pipeline",
    )
    args = parser.parse_args()

    required_vars = [
        "GBSERVER_RABBITMQ_MGMT_URL",
        "GBSERVER_RABBITMQ_MGMT_USER",
        "GBSERVER_RABBITMQ_MGMT_PASSWORD",
        "RABBITMQ_HOST",
    ]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        print(
            f"[ERROR] Missing required env vars: {', '.join(missing)}", file=sys.stderr
        )
        sys.exit(1)

    count = asyncio.run(
        subscribe_and_listen(
            args.build_id, timeout_seconds=args.timeout, self_test=args.self_test
        )
    )
    sys.exit(0 if count > 0 else 1)


if __name__ == "__main__":
    main()
