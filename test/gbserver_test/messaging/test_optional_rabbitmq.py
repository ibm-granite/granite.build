"""Test that messaging discovery works without aio_pika installed."""

import importlib
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
        """Backend discovery should succeed even if aio_pika is not installed."""
        # Also remove the already-cached rabbitmq_base module so discover_backends()
        # tries to re-import it (and fails gracefully due to mocked aio_pika=None)
        modules_to_mock = {
            **_RABBITMQ_MODULES,
            "gbserver.messaging.rabbitmq_base": None,
        }
        with mock.patch.dict(sys.modules, modules_to_mock):
            messaging_init = importlib.import_module("gbserver.messaging")
            importlib.reload(messaging_init)
            backends = messaging_init.discover_backends()
            # RabbitMQ backend should not be in the list
            assert "rabbitmqbase" not in backends

    def test_messaging_base_importable_without_aio_pika(self):
        """MessagingBase should always be importable."""
        with mock.patch.dict(sys.modules, _RABBITMQ_MODULES):
            from gbserver.messaging.messaging_base import Address, MessagingBase

            addr = Address(exchange=None, queue="test")
            assert addr.queue == "test"
