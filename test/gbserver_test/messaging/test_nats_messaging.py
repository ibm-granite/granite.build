"""Tests for NATSMessaging — NATS-based messaging backend."""

import asyncio
import json
import unittest.mock as mock
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gbserver.messaging.messaging_base import Address


class TestNATSMessagingDiscovery:
    """Tests that don't require a NATS server."""

    def test_discoverable_as_backend(self):
        """NATSMessaging is auto-discovered by the messaging plugin system."""
        from gbserver.messaging import discover_backends

        backends = discover_backends()
        assert "natsmessaging" in backends

    def test_import(self):
        """NATSMessaging can be imported."""
        from gbserver.messaging.nats_messaging import NATSMessaging

        assert NATSMessaging is not None


@pytest.mark.asyncio
class TestNATSMessagingUnit:
    """Unit tests using mocked NATS client."""

    async def test_setup_connects(self):
        """setup() connects to the NATS server."""
        from gbserver.messaging.nats_messaging import NATSMessaging

        addr = Address(exchange=None, queue="test_queue")
        messaging = NATSMessaging(addr, nats_url="nats://localhost:4222")

        mock_nc = AsyncMock()
        with patch("gbserver.messaging.nats_messaging.nats.connect", return_value=mock_nc):
            await messaging.setup()

        assert messaging._nc is mock_nc

    async def test_publish_sends_message(self):
        """publish() sends a JSON message to the correct subject."""
        from gbserver.messaging.nats_messaging import NATSMessaging

        addr = Address(exchange=None, queue="test_queue")
        messaging = NATSMessaging(addr, nats_url="nats://localhost:4222")

        mock_nc = AsyncMock()
        mock_nc.is_closed = False
        with patch("gbserver.messaging.nats_messaging.nats.connect", return_value=mock_nc):
            await messaging.setup()

        payload = {"event": "build_started", "build_id": "123"}
        await messaging.publish(payload, suffix="status")

        mock_nc.publish.assert_called_once()
        call_args = mock_nc.publish.call_args
        assert call_args[0][0] == "gbserver.test_queue.status"
        assert json.loads(call_args[0][1]) == payload

    async def test_publish_without_suffix(self):
        """publish() with empty suffix uses base subject."""
        from gbserver.messaging.nats_messaging import NATSMessaging

        addr = Address(exchange=None, queue="test_queue")
        messaging = NATSMessaging(addr, nats_url="nats://localhost:4222")

        mock_nc = AsyncMock()
        mock_nc.is_closed = False
        with patch("gbserver.messaging.nats_messaging.nats.connect", return_value=mock_nc):
            await messaging.setup()

        await messaging.publish({"data": "test"}, suffix="")

        call_args = mock_nc.publish.call_args
        assert call_args[0][0] == "gbserver.test_queue"

    async def test_publish_raises_if_not_connected(self):
        """publish() raises RuntimeError if setup() wasn't called."""
        from gbserver.messaging.nats_messaging import NATSMessaging

        addr = Address(exchange=None, queue="test_queue")
        messaging = NATSMessaging(addr, nats_url="nats://localhost:4222")

        with pytest.raises(RuntimeError, match="not connected"):
            await messaging.publish({"data": "test"}, suffix="")

    async def test_close_is_idempotent(self):
        """Calling close() multiple times doesn't raise."""
        from gbserver.messaging.nats_messaging import NATSMessaging

        addr = Address(exchange=None, queue="test_close")
        messaging = NATSMessaging(addr, nats_url="nats://localhost:4222")

        mock_nc = AsyncMock()
        mock_nc.is_closed = False
        with patch("gbserver.messaging.nats_messaging.nats.connect", return_value=mock_nc):
            await messaging.setup()

        await messaging.close()
        # After first close, _nc is None
        await messaging.close()  # Should not raise

    async def test_run_is_noop(self):
        """run() returns immediately (lightweight M1 version)."""
        from gbserver.messaging.nats_messaging import NATSMessaging

        addr = Address(exchange=None, queue="test_run")
        messaging = NATSMessaging(addr, nats_url="nats://localhost:4222")
        await messaging.run()  # Should not block or raise

    async def test_from_yaml(self):
        """NATSMessaging can be instantiated from YAML config."""
        from gbserver.messaging.messaging_base import MessagingBase

        yaml_config = """
type: natsmessaging
address:
  exchange: null
  queue: test_yaml
nats_url: nats://localhost:4222
"""
        messaging = MessagingBase.from_yaml(yaml_config)
        assert messaging.__class__.__name__ == "NATSMessaging"
        assert messaging.addr.queue == "test_yaml"


@pytest.mark.asyncio
class TestNATSJetStreamUnit:
    """Unit tests for JetStream functionality using mocked NATS client."""

    async def test_setup_detects_jetstream_available(self):
        """setup() enables JetStream when server supports it."""
        from gbserver.messaging.nats_messaging import NATSMessaging

        addr = Address(exchange=None, queue="test_js")
        messaging = NATSMessaging(addr, nats_url="nats://localhost:4222")

        mock_nc = AsyncMock()
        mock_js = AsyncMock()
        mock_nc.jetstream = MagicMock(return_value=mock_js)

        with patch("gbserver.messaging.nats_messaging.nats.connect", return_value=mock_nc):
            await messaging.setup()

        assert messaging._jetstream_available is True
        assert messaging._js is mock_js
        mock_js.account_info.assert_awaited_once()

    async def test_setup_falls_back_when_jetstream_unavailable(self):
        """setup() falls back to lightweight mode when JetStream is not enabled."""
        from gbserver.messaging.nats_messaging import NATSMessaging

        addr = Address(exchange=None, queue="test_no_js")
        messaging = NATSMessaging(addr, nats_url="nats://localhost:4222")

        mock_nc = AsyncMock()
        mock_js = AsyncMock()
        mock_js.account_info.side_effect = Exception("JetStream not enabled")
        mock_nc.jetstream = MagicMock(return_value=mock_js)

        with patch("gbserver.messaging.nats_messaging.nats.connect", return_value=mock_nc):
            await messaging.setup()

        assert messaging._jetstream_available is False
        assert messaging._js is None

    async def test_setup_creates_stream_when_jetstream_available(self):
        """setup() creates a JetStream stream for the configured queue."""
        from gbserver.messaging.nats_messaging import NATSMessaging

        addr = Address(exchange=None, queue="build123")
        messaging = NATSMessaging(addr, nats_url="nats://localhost:4222")

        mock_nc = AsyncMock()
        mock_js = AsyncMock()
        mock_nc.jetstream = MagicMock(return_value=mock_js)

        with patch("gbserver.messaging.nats_messaging.nats.connect", return_value=mock_nc):
            await messaging.setup()

        mock_js.add_stream.assert_awaited_once()
        call_kwargs = mock_js.add_stream.call_args
        config = (
            call_kwargs.kwargs.get("config") or call_kwargs[1].get("config") or call_kwargs[0][0]
        )
        assert config.name == "GBSERVER_BUILD123"
        assert "gbserver.build123.>" in config.subjects

    async def test_publish_uses_jetstream_when_available(self):
        """publish() uses js.publish() when JetStream is available."""
        from gbserver.messaging.nats_messaging import NATSMessaging

        addr = Address(exchange=None, queue="test_js_pub")
        messaging = NATSMessaging(addr, nats_url="nats://localhost:4222")

        mock_nc = AsyncMock()
        mock_nc.is_closed = False
        mock_js = AsyncMock()
        mock_nc.jetstream = MagicMock(return_value=mock_js)

        with patch("gbserver.messaging.nats_messaging.nats.connect", return_value=mock_nc):
            await messaging.setup()

        payload = {"event": "build_started", "build_id": "456"}
        await messaging.publish(payload, suffix="status")

        mock_js.publish.assert_awaited_once()
        call_args = mock_js.publish.call_args
        assert call_args[0][0] == "gbserver.test_js_pub.status"
        assert json.loads(call_args[0][1]) == payload
        # nc.publish should NOT have been called
        mock_nc.publish.assert_not_awaited()

    async def test_publish_uses_core_nats_when_jetstream_unavailable(self):
        """publish() uses nc.publish() when JetStream is not available."""
        from gbserver.messaging.nats_messaging import NATSMessaging

        addr = Address(exchange=None, queue="test_core_pub")
        messaging = NATSMessaging(addr, nats_url="nats://localhost:4222")

        mock_nc = AsyncMock()
        mock_nc.is_closed = False
        mock_js = AsyncMock()
        mock_js.account_info.side_effect = Exception("no JS")
        mock_nc.jetstream = MagicMock(return_value=mock_js)

        with patch("gbserver.messaging.nats_messaging.nats.connect", return_value=mock_nc):
            await messaging.setup()

        payload = {"event": "build_started"}
        await messaging.publish(payload, suffix="status")

        mock_nc.publish.assert_awaited_once()
        mock_js.publish.assert_not_awaited()

    async def test_consume_stream_creates_durable_consumer(self):
        """consume_stream() creates a JetStream durable push consumer."""
        from gbserver.messaging.nats_messaging import NATSMessaging

        addr = Address(exchange=None, queue="test_js_consume")
        messaging = NATSMessaging(addr, nats_url="nats://localhost:4222")

        mock_nc = AsyncMock()
        mock_nc.is_closed = False
        mock_js = AsyncMock()
        mock_nc.jetstream = MagicMock(return_value=mock_js)

        # Mock the subscription to yield one message then stop
        mock_msg = AsyncMock()
        mock_msg.subject = "gbserver.test_js_consume.status"
        mock_msg.data = b'{"event": "done"}'

        mock_sub = AsyncMock()

        async def _messages():
            yield mock_msg

        mock_sub.messages = _messages()
        mock_js.subscribe.return_value = mock_sub

        with patch("gbserver.messaging.nats_messaging.nats.connect", return_value=mock_nc):
            await messaging.setup()

        received = []

        async def handler(data, routing_key):
            received.append((data, routing_key))

        await messaging.consume_stream(handler)

        # Verify JetStream subscribe was called with correct params
        mock_js.subscribe.assert_awaited_once()
        call_args = mock_js.subscribe.call_args
        assert call_args[0][0] == "gbserver.test_js_consume.>"

        # Verify message was delivered and acked
        assert len(received) == 1
        assert received[0][1] == "status"
        mock_msg.ack.assert_awaited_once()

    async def test_consume_stream_nacks_on_handler_error(self):
        """consume_stream() naks the message when handler raises an exception."""
        from gbserver.messaging.nats_messaging import NATSMessaging

        addr = Address(exchange=None, queue="test_js_nack")
        messaging = NATSMessaging(addr, nats_url="nats://localhost:4222")

        mock_nc = AsyncMock()
        mock_nc.is_closed = False
        mock_js = AsyncMock()
        mock_nc.jetstream = MagicMock(return_value=mock_js)

        mock_msg = AsyncMock()
        mock_msg.subject = "gbserver.test_js_nack.status"
        mock_msg.data = b'{"event": "fail"}'

        mock_sub = AsyncMock()

        async def _messages():
            yield mock_msg

        mock_sub.messages = _messages()
        mock_js.subscribe.return_value = mock_sub

        with patch("gbserver.messaging.nats_messaging.nats.connect", return_value=mock_nc):
            await messaging.setup()

        async def bad_handler(data, routing_key):
            raise ValueError("handler failed")

        await messaging.consume_stream(bad_handler)

        # Message should be nak'd, not ack'd
        mock_msg.nak.assert_awaited_once()
        mock_msg.ack.assert_not_awaited()

    async def test_run_blocks_until_stop_event_with_jetstream(self):
        """run() blocks until stop_event is set when JetStream is available."""
        from gbserver.messaging.nats_messaging import NATSMessaging

        addr = Address(exchange=None, queue="test_js_run")
        messaging = NATSMessaging(addr, nats_url="nats://localhost:4222")

        mock_nc = AsyncMock()
        mock_js = AsyncMock()
        mock_nc.jetstream = MagicMock(return_value=mock_js)

        with patch("gbserver.messaging.nats_messaging.nats.connect", return_value=mock_nc):
            await messaging.setup()

        assert messaging._jetstream_available is True

        # run() should block; set stop_event after a short delay
        async def _set_stop():
            await asyncio.sleep(0.1)
            messaging._stop_event.set()

        task = asyncio.create_task(_set_stop())
        await asyncio.wait_for(messaging.run(), timeout=2.0)
        await task  # cleanup

    async def test_run_is_noop_without_jetstream(self):
        """run() returns immediately when JetStream is not available."""
        from gbserver.messaging.nats_messaging import NATSMessaging

        addr = Address(exchange=None, queue="test_no_js_run")
        messaging = NATSMessaging(addr, nats_url="nats://localhost:4222")

        mock_nc = AsyncMock()
        mock_js = AsyncMock()
        mock_js.account_info.side_effect = Exception("no JS")
        mock_nc.jetstream = MagicMock(return_value=mock_js)

        with patch("gbserver.messaging.nats_messaging.nats.connect", return_value=mock_nc):
            await messaging.setup()

        # Should return immediately (no timeout needed)
        await asyncio.wait_for(messaging.run(), timeout=1.0)

    async def test_from_yaml_with_jetstream_config(self):
        """NATSMessaging can be instantiated from YAML with JetStream config."""
        from gbserver.messaging.messaging_base import MessagingBase

        yaml_config = """
type: natsmessaging
address:
  exchange: null
  queue: test_yaml_js
nats_url: nats://localhost:4222
stream_max_age: 3600
max_deliver: 10
ack_wait: 60
"""
        messaging = MessagingBase.from_yaml(yaml_config)
        assert messaging.__class__.__name__ == "NATSMessaging"
        assert messaging.addr.queue == "test_yaml_js"
        assert messaging._stream_max_age == 3600
        assert messaging._max_deliver == 10
        assert messaging._ack_wait == 60
