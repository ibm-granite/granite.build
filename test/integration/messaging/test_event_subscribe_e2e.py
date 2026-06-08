"""End-to-end test for event subscription with local RabbitMQ.

Requires a running RabbitMQ instance with management plugin enabled.
Set RABBITMQ_HOST, RABBITMQ_PORT, GBSERVER_RABBITMQ_MGMT_URL env vars.

Run with: pytest test/integration/messaging/test_event_subscribe_e2e.py -v
"""

import asyncio
import json
import os

import pytest


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("RABBITMQ_HOST"),
    reason="RABBITMQ_HOST not set; skipping RabbitMQ integration test",
)
async def test_publish_and_consume_build_event():
    """Publish a build event and verify a scoped consumer receives it.

    Tests the full subscription flow:
    1. Publisher declares exchange and publishes to it
    2. RabbitMQAdmin provisions scoped credentials
    3. Scoped consumer connects, declares queue, binds, and receives the event
    """
    import aio_pika

    from gbserver.messaging.rabbitmq_admin import RabbitMQAdmin

    build_id = "test-build-e2e-001"
    exchange_name = "build-events-test"

    host = os.getenv("RABBITMQ_HOST", "localhost")
    port = int(os.getenv("RABBITMQ_PORT", "5672"))
    mgmt_url = os.getenv("GBSERVER_RABBITMQ_MGMT_URL", "http://localhost:15672")
    mgmt_user = os.getenv("GBSERVER_RABBITMQ_MGMT_USER", "guest")
    mgmt_password = os.getenv("GBSERVER_RABBITMQ_MGMT_PASSWORD", "guest")

    # 1. Publisher: declare exchange and publish an event
    pub_conn = await aio_pika.connect(
        host=host, port=port, login=mgmt_user, password=mgmt_password
    )
    pub_chan = await pub_conn.channel()
    exchange = await pub_chan.declare_exchange(
        exchange_name, aio_pika.ExchangeType.TOPIC, durable=True
    )

    # 2. Provision scoped consumer credentials
    admin = RabbitMQAdmin(
        management_url=mgmt_url,
        admin_user=mgmt_user,
        admin_password=mgmt_password,
    )
    creds = await admin.create_scoped_user(
        build_id=build_id,
        exchange=exchange_name,
        ttl_seconds=30,
    )

    # 3. Connect as the scoped consumer and bind to build events
    consumer_conn = await aio_pika.connect(
        host=host,
        port=port,
        login=creds["username"],
        password=creds["password"],
    )
    consumer_chan = await consumer_conn.channel()
    queue = await consumer_chan.declare_queue(f"events.{build_id}.test", exclusive=True)
    consumer_exchange = await consumer_chan.get_exchange(exchange_name, ensure=False)
    await queue.bind(consumer_exchange, routing_key=f"build.{build_id}.#")

    received_messages = []

    async def on_message(message: aio_pika.abc.AbstractIncomingMessage):
        async with message.process():
            received_messages.append(json.loads(message.body))

    await queue.consume(on_message)

    # 4. Publish an event (simulating what BuildEventPublisher would send)
    event_payload = {
        "build_id": build_id,
        "event_type": "status_event",
        "timestamp": 1780000000,
        "target_name": "train",
        "step_name": "",
        "source": "build-framework",
        "status": "running",
        "message": "training started",
    }
    message = aio_pika.Message(json.dumps(event_payload).encode())
    await exchange.publish(message, routing_key=f"build.{build_id}.status_event")

    # 5. Wait for delivery
    for _ in range(10):
        if received_messages:
            break
        await asyncio.sleep(0.5)

    # 6. Verify consumer received the event
    assert (
        len(received_messages) >= 1
    ), f"Expected at least 1 message, got {len(received_messages)}"

    msg = received_messages[0]
    assert msg["build_id"] == build_id
    assert msg["event_type"] == "status_event"
    assert msg["status"] == "running"
    assert msg["message"] == "training started"
    assert msg["target_name"] == "train"

    # Cleanup
    await consumer_conn.close()
    await pub_conn.close()
    await admin.delete_user(creds["username"])
