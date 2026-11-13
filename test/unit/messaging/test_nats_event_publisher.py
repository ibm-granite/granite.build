# Copyright Granite.Build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for NATS-based event publishing in standalone mode."""

import os
from unittest.mock import MagicMock, patch

import pytest

from gbserver.messaging.build_event_publisher import BuildEventPublisher


class TestBuildEventPublisherBackendSelection:
    """Tests for from_env() backend selection logic."""

    @patch.dict(os.environ, {"GB_ENVIRONMENT": "STANDALONE"}, clear=False)
    @patch("gbserver.messaging.build_event_publisher.HAS_NATS", True)
    @patch(
        "gbserver.messaging.build_event_publisher._is_nats_reachable", return_value=True
    )
    @patch("gbserver.messaging.nats_messaging.NATSMessaging")
    def test_from_env_uses_nats_in_standalone(self, mock_nats_cls, _mock_reachable):
        """In standalone without RABBITMQ_HOST, from_env() creates NATSMessaging."""
        os.environ.pop("RABBITMQ_HOST", None)
        mock_nats_instance = MagicMock()
        mock_nats_cls.return_value = mock_nats_instance

        publisher = BuildEventPublisher.from_env()

        mock_nats_cls.assert_called_once()
        assert publisher._backend is mock_nats_instance

    @patch.dict(os.environ, {"RABBITMQ_HOST": "rmq.example.com"}, clear=False)
    @patch("gbserver.messaging.rabbitmq_base.RabbitMQBase.from_env_and_args")
    def test_from_env_uses_rabbitmq_when_host_set(self, mock_from_env):
        """When RABBITMQ_HOST is set, from_env() creates RabbitMQBase."""
        mock_rmq_instance = MagicMock()
        mock_from_env.return_value = mock_rmq_instance

        publisher = BuildEventPublisher.from_env()

        mock_from_env.assert_called_once()
        assert publisher._backend is mock_rmq_instance

    @patch.dict(os.environ, {"GB_ENVIRONMENT": "STANDALONE"}, clear=False)
    @patch("gbserver.messaging.build_event_publisher.HAS_NATS", True)
    @patch(
        "gbserver.messaging.build_event_publisher._is_nats_reachable", return_value=True
    )
    def test_is_configured_true_in_standalone_with_nats(self, _mock_reachable):
        """is_configured() returns True in standalone when nats-py is available and reachable."""
        os.environ.pop("RABBITMQ_HOST", None)
        assert BuildEventPublisher.is_configured() is True

    @patch.dict(os.environ, {"GB_ENVIRONMENT": "PROD"}, clear=False)
    @patch("gbserver.messaging.build_event_publisher.HAS_NATS", False)
    def test_is_configured_false_without_nats_or_rabbitmq(self):
        """is_configured() returns False when neither backend is available."""
        os.environ.pop("RABBITMQ_HOST", None)
        assert BuildEventPublisher.is_configured() is False
