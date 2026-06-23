"""Test that messaging discovery works without aio_pika installed."""

import sys
import unittest.mock as mock

import pytest

_RABBITMQ_MODULES = {
    "aio_pika": None,
    "aio_pika.abc": None,
    "aio_pika.exceptions": None,
    "aiormq": None,
    "aiormq.exceptions": None,
}


class TestOptionalRabbitMQ:
    def test_discover_backends_without_aio_pika(self):
        """Backend discovery should skip RabbitMQ when aio_pika is not installed.

        RabbitMQBase.is_available() reports HAS_RABBITMQ, and discover_backends()
        skips unavailable backends. Simulate the missing dependency by pinning
        HAS_RABBITMQ False — no sys.modules surgery needed (which would risk
        corrupting C-extension imports such as numpy for sibling tests).
        """
        import gbserver.messaging as messaging_pkg

        with mock.patch("gbserver.messaging.rabbitmq_base.HAS_RABBITMQ", False):
            backends = messaging_pkg.discover_backends()
            # RabbitMQ backend should not be offered.
            assert "rabbitmqbase" not in backends
            # ... but other backends (e.g. NATS) still are.
            assert "natsmessaging" in backends

    def test_messaging_base_importable_without_aio_pika(self):
        """MessagingBase should always be importable."""
        with mock.patch.dict(sys.modules, _RABBITMQ_MODULES):
            from gbserver.messaging.messaging_base import Address, MessagingBase

            addr = Address(exchange=None, queue="test")
            assert addr.queue == "test"
