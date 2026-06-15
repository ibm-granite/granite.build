"""Test that messaging discovery works without aio_pika installed."""

import importlib
import sys
import unittest.mock as mock

import pytest

pytestmark = pytest.mark.ibm

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
        # rabbitmq_base's top-level `import aio_pika` is gated on
        # optional_imports.HAS_RABBITMQ, which is computed once at import and
        # cached. Whether a prior test left it True or False is non-deterministic
        # under xdist, so pin it to False here to faithfully simulate "aio_pika is
        # not installed" — otherwise discovery may still import rabbitmq_base and
        # register the backend regardless of the mocked sys.modules.
        with (
            mock.patch.dict(sys.modules, modules_to_mock),
            mock.patch("gbserver.utils.optional_imports.HAS_RABBITMQ", False),
        ):
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
