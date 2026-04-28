"""Integration tests for NATSMessaging with JetStream.

Requires a running nats-server with JetStream enabled:
    nats-server -js

These tests are skipped by default in CI. Run manually with:
    pytest -m nats_server test/gbserver_test/messaging/test_nats_jetstream_integration.py
"""

import asyncio
import json
import uuid

import pytest

from gbserver.messaging.messaging_base import Address

pytestmark = pytest.mark.nats_server


def _unique_queue():
    """Generate a unique queue name to avoid stream collisions between tests."""
    return f"inttest_{uuid.uuid4().hex[:8]}"


@pytest.mark.asyncio
class TestNATSJetStreamIntegration:
    """Integration tests against a real nats-server with JetStream."""

    async def test_publish_consume_round_trip(self):
        """Messages published via JetStream are consumed by durable consumer."""
        from gbserver.messaging.nats_messaging import NATSMessaging

        queue = _unique_queue()
        addr = Address(exchange=None, queue=queue)
        publisher = NATSMessaging(addr, nats_url="nats://localhost:4222")
        consumer = NATSMessaging(addr, nats_url="nats://localhost:4222")

        await publisher.setup()
        await consumer.setup()

        assert (
            publisher._jetstream_available
        ), "JetStream must be enabled on nats-server"

        # Publish a message
        payload = {"build_id": "test-123", "status": "started"}
        await publisher.publish(payload, suffix="status")

        # Consume it
        received = []

        async def handler(data, routing_key):
            received.append((json.loads(data), routing_key))

        consume_task = asyncio.create_task(consumer.consume_stream(handler))
        await asyncio.sleep(0.5)  # Give time for message delivery

        assert len(received) == 1
        assert received[0][0] == payload
        assert received[0][1] == "status"

        consume_task.cancel()
        await publisher.close()
        await consumer.close()

    async def test_jetstream_auto_detected(self):
        """JetStream is auto-detected when server supports it."""
        from gbserver.messaging.nats_messaging import NATSMessaging

        addr = Address(exchange=None, queue=_unique_queue())
        messaging = NATSMessaging(addr, nats_url="nats://localhost:4222")
        await messaging.setup()

        assert messaging._jetstream_available is True
        assert messaging._js is not None

        await messaging.close()

    async def test_durable_consumer_resumes(self):
        """Durable consumer resumes from last acked position after reconnect."""
        from gbserver.messaging.nats_messaging import NATSMessaging

        queue = _unique_queue()
        addr = Address(exchange=None, queue=queue)

        # First session: publish 2 messages, consume 1
        pub = NATSMessaging(addr, nats_url="nats://localhost:4222")
        await pub.setup()
        await pub.publish({"seq": 1}, suffix="event")
        await pub.publish({"seq": 2}, suffix="event")

        received_first = []

        async def handler_first(data, routing_key):
            received_first.append(json.loads(data))

        con1 = NATSMessaging(addr, nats_url="nats://localhost:4222")
        await con1.setup()
        task1 = asyncio.create_task(con1.consume_stream(handler_first))
        await asyncio.sleep(0.5)
        task1.cancel()
        await con1.close()

        # Both messages should have been delivered
        assert len(received_first) == 2

        # Second session: publish 1 more, consume — should only get new message
        await pub.publish({"seq": 3}, suffix="event")

        received_second = []

        async def handler_second(data, routing_key):
            received_second.append(json.loads(data))

        con2 = NATSMessaging(addr, nats_url="nats://localhost:4222")
        await con2.setup()
        task2 = asyncio.create_task(con2.consume_stream(handler_second))
        await asyncio.sleep(0.5)
        task2.cancel()
        await con2.close()
        await pub.close()

        assert len(received_second) == 1
        assert received_second[0]["seq"] == 3
