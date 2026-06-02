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
    """Publish a build event and verify a scoped consumer receives it."""
    from gbserver.messaging.build_event_publisher import BuildEventPublisher
    from gbserver.messaging.messaging_base import Address
    from gbserver.messaging.rabbitmq_admin import RabbitMQAdmin
    from gbserver.messaging.rabbitmq_base import RabbitMQBase, RabbitSettings
    from gbserver.types.buildevent import (
        BuildEvent,
        BuildEventStatusPayload,
        BuildEventType,
        EntityRunMetadata,
    )
    from gbserver.types.status import Status

    build_id = "test-build-e2e-001"
    exchange_name = "build-events-test"

    host = os.getenv("RABBITMQ_HOST", "localhost")
    port = int(os.getenv("RABBITMQ_PORT", "5672"))
    mgmt_url = os.getenv("GBSERVER_RABBITMQ_MGMT_URL", "http://localhost:15672")
    mgmt_user = os.getenv("GBSERVER_RABBITMQ_MGMT_USER", "guest")
    mgmt_password = os.getenv("GBSERVER_RABBITMQ_MGMT_PASSWORD", "guest")

    # 1. Set up publisher with test exchange
    publisher_settings = RabbitSettings(
        uri="amqp",
        host=host,
        port=port,
        user=mgmt_user,
        password=mgmt_password,
    )
    publisher_addr = Address(
        exchange=exchange_name,
        queue="build",
        routing_key=None,
    )
    rabbitmq_publisher = RabbitMQBase(
        addr=publisher_addr,
        settings=publisher_settings,
    )
    publisher = BuildEventPublisher(rabbitmq=rabbitmq_publisher)
    await publisher.setup()

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

    # 3. Connect as the scoped consumer
    consumer_settings = RabbitSettings(
        uri="amqp",
        host=host,
        port=port,
        user=creds["username"],
        password=creds["password"],
    )
    consumer_queue_name = f"build.{build_id}"
    consumer_addr = Address(
        exchange=exchange_name,
        queue=consumer_queue_name,
        routing_key=None,
    )
    consumer = RabbitMQBase(
        addr=consumer_addr,
        settings=consumer_settings,
    )
    await consumer.setup()

    received_messages = []

    async def message_handler(body: bytes, routing_key: str, delivery_tag: int):
        msg = json.loads(body)
        received_messages.append(msg)

    await consumer.consume_stream(handler=message_handler, stream_offset="first")

    # 4. Publish an event
    event = BuildEvent(
        run_metadata=EntityRunMetadata(build_id=build_id, target_name="train"),
        type=BuildEventType.STATUS_EVENT,
        payload=BuildEventStatusPayload(status=Status.RUNNING, msg="training started"),
    )
    await publisher.publish_event(event)

    # 5. Wait for delivery (allow time for broker routing)
    for _ in range(10):
        if received_messages:
            break
        await asyncio.sleep(0.5)

    # 6. Verify consumer received the event
    assert len(received_messages) >= 1, (
        f"Expected at least 1 message, got {len(received_messages)}"
    )

    msg = received_messages[0]
    assert msg["build_id"] == build_id
    assert msg["event_type"] == "status_event"
    assert msg["status"] == Status.RUNNING.value
    assert msg["message"] == "training started"
    assert msg["target_name"] == "train"

    # Cleanup
    await consumer.close()
    await publisher.close()
    await admin.delete_user(creds["username"])
