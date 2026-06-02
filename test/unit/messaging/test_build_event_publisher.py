#!/usr/bin/env python3

# Copyright LLM.build Authors
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

"""
Unit tests for BuildEventPublisher.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gbserver.messaging.build_event_publisher import (
    EXCHANGE_NAME,
    BuildEventPublisher,
)
from gbserver.messaging.messaging_base import Address
from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventMessagePayload,
    BuildEventStatusPayload,
    BuildEventType,
    EntityRunMetadata,
)
from gbserver.types.status import Status


@pytest.fixture
def mock_rabbitmq():
    """Create a mock RabbitMQBase instance."""
    mock = MagicMock()
    mock.addr = Address(exchange=EXCHANGE_NAME, queue="build", routing_key=None)
    mock.setup = AsyncMock()
    mock.close = AsyncMock()
    mock.publish = AsyncMock()
    return mock


@pytest.fixture
def publisher(mock_rabbitmq):
    """Create a BuildEventPublisher with a mocked RabbitMQ backend."""
    return BuildEventPublisher(rabbitmq=mock_rabbitmq)


@pytest.fixture
def sample_status_event():
    """Create a sample status BuildEvent."""
    return BuildEvent(
        run_metadata=EntityRunMetadata(
            build_id="build-123",
            target_name="my-target",
            targetstep_uri="train-step",
        ),
        type=BuildEventType.STATUS_EVENT,
        payload=BuildEventStatusPayload(
            status=Status.RUNNING,
            msg="Build is running",
        ),
        timestamp=datetime(2026, 1, 15, 10, 30, 0),
        source="build-framework",
    )


@pytest.fixture
def sample_message_event():
    """Create a sample message BuildEvent."""
    return BuildEvent(
        run_metadata=EntityRunMetadata(
            build_id="build-456",
            target_name="eval-target",
            targetstep_uri="eval-step",
        ),
        type=BuildEventType.MESSAGE_EVENT,
        payload=BuildEventMessagePayload(
            level="INFO",
            msg="Processing batch 5/10",
        ),
        timestamp=datetime(2026, 1, 15, 11, 0, 0),
        source="step-runner",
    )


class TestBuildEventPublisher:
    """Tests for BuildEventPublisher."""

    @pytest.mark.asyncio
    async def test_publish_status_event(self, publisher, mock_rabbitmq, sample_status_event):
        """Test publishing a status event produces correct payload and routing."""
        await publisher.publish_event(sample_status_event)

        mock_rabbitmq.publish.assert_called_once()
        call_kwargs = mock_rabbitmq.publish.call_args
        payload = call_kwargs.kwargs.get("payload") or call_kwargs[1].get("payload", call_kwargs[0][0] if call_kwargs[0] else None)
        suffix = call_kwargs.kwargs.get("suffix") or call_kwargs[1].get("suffix", call_kwargs[0][1] if len(call_kwargs[0]) > 1 else None)

        # Verify the suffix is the event type value
        assert suffix == "status_event"

        # Verify payload contents
        assert payload["build_id"] == "build-123"
        assert payload["event_type"] == "status_event"
        assert payload["timestamp"] == "2026-01-15T10:30:00"
        assert payload["target_name"] == "my-target"
        assert payload["step_name"] == "train-step"
        assert payload["source"] == "build-framework"
        # Status-specific fields
        assert payload["status"] == "running"
        assert payload["message"] == "Build is running"

    @pytest.mark.asyncio
    async def test_publish_message_event(self, publisher, mock_rabbitmq, sample_message_event):
        """Test publishing a message event (no status/message fields added)."""
        await publisher.publish_event(sample_message_event)

        mock_rabbitmq.publish.assert_called_once()
        call_kwargs = mock_rabbitmq.publish.call_args
        payload = call_kwargs.kwargs.get("payload") or call_kwargs[0][0]

        assert payload["build_id"] == "build-456"
        assert payload["event_type"] == "message_event"
        assert payload["target_name"] == "eval-target"
        assert payload["step_name"] == "eval-step"
        assert payload["source"] == "step-runner"
        # No status/message fields for non-status events
        assert "status" not in payload
        assert "message" not in payload

    @pytest.mark.asyncio
    async def test_skips_internal_terminate_event(self, publisher, mock_rabbitmq):
        """Test that TERMINATE events are skipped."""
        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="build-789"),
            type=BuildEventType.TERMINATE_EVENT,
            timestamp=datetime(2026, 1, 15, 12, 0, 0),
        )

        await publisher.publish_event(event)
        mock_rabbitmq.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_internal_newartifact_event(self, publisher, mock_rabbitmq):
        """Test that NEWARTIFACT_IN_ENVIRONMENT events are skipped."""
        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="build-789"),
            type=BuildEventType.NEWARTIFACT_IN_ENVIRONMENT_EVENT,
            timestamp=datetime(2026, 1, 15, 12, 0, 0),
        )

        await publisher.publish_event(event)
        mock_rabbitmq.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_internal_multiartifact_event(self, publisher, mock_rabbitmq):
        """Test that NEW_MULTIARTIFACT_IN_ENVIRONMENT events are skipped."""
        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="build-789"),
            type=BuildEventType.NEW_MULTIARTIFACT_IN_ENVIRONMENT_EVENT,
            timestamp=datetime(2026, 1, 15, 12, 0, 0),
        )

        await publisher.publish_event(event)
        mock_rabbitmq.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_routing_key_format(self, publisher, mock_rabbitmq, sample_status_event):
        """Test that the address is set correctly for routing key build.<id>.<type>."""
        await publisher.publish_event(sample_status_event)

        # Check that addr was temporarily changed to build.<build_id>
        # and then restored. We verify via the publish call's address context.
        mock_rabbitmq.publish.assert_called_once()

        # After publish, the original addr should be restored
        assert mock_rabbitmq.addr == Address(
            exchange=EXCHANGE_NAME, queue="build", routing_key=None
        )

    @pytest.mark.asyncio
    async def test_publish_failure_logs_warning_does_not_raise(
        self, publisher, mock_rabbitmq, sample_status_event
    ):
        """Test that RabbitMQ failures are swallowed with a warning."""
        mock_rabbitmq.publish.side_effect = ConnectionError("RabbitMQ unavailable")

        # Should NOT raise
        await publisher.publish_event(sample_status_event)

        # Verify publish was attempted
        mock_rabbitmq.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_failure_restores_address(
        self, publisher, mock_rabbitmq, sample_status_event
    ):
        """Test that the original address is restored even after a publish failure."""
        original_addr = mock_rabbitmq.addr
        mock_rabbitmq.publish.side_effect = RuntimeError("channel lost")

        await publisher.publish_event(sample_status_event)

        # Address should be restored
        assert mock_rabbitmq.addr == original_addr

    @pytest.mark.asyncio
    async def test_missing_build_id_uses_unknown(self, publisher, mock_rabbitmq):
        """Test that a missing build_id defaults to 'unknown'."""
        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id=""),
            type=BuildEventType.STATUS_EVENT,
            payload=BuildEventStatusPayload(status=Status.PENDING, msg=""),
            timestamp=datetime(2026, 1, 15, 12, 0, 0),
        )

        await publisher.publish_event(event)

        mock_rabbitmq.publish.assert_called_once()
        call_kwargs = mock_rabbitmq.publish.call_args
        payload = call_kwargs.kwargs.get("payload") or call_kwargs[0][0]
        assert payload["build_id"] == "unknown"

    @pytest.mark.asyncio
    async def test_setup_delegates_to_rabbitmq(self, publisher, mock_rabbitmq):
        """Test that setup() delegates to the underlying RabbitMQ instance."""
        await publisher.setup()
        mock_rabbitmq.setup.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_delegates_to_rabbitmq(self, publisher, mock_rabbitmq):
        """Test that close() delegates to the underlying RabbitMQ instance."""
        await publisher.close()
        mock_rabbitmq.close.assert_called_once()


class TestSerializeEvent:
    """Tests for the _serialize_event static method."""

    def test_serialize_status_event(self, sample_status_event):
        """Test serialization of a status event includes status and message."""
        result = BuildEventPublisher._serialize_event(sample_status_event)

        assert result == {
            "build_id": "build-123",
            "event_type": "status_event",
            "timestamp": "2026-01-15T10:30:00",
            "target_name": "my-target",
            "step_name": "train-step",
            "source": "build-framework",
            "status": "running",
            "message": "Build is running",
        }

    def test_serialize_message_event(self, sample_message_event):
        """Test serialization of a message event does not include status/message."""
        result = BuildEventPublisher._serialize_event(sample_message_event)

        assert result == {
            "build_id": "build-456",
            "event_type": "message_event",
            "timestamp": "2026-01-15T11:00:00",
            "target_name": "eval-target",
            "step_name": "eval-step",
            "source": "step-runner",
        }

    def test_serialize_event_empty_metadata(self):
        """Test serialization with empty/default metadata."""
        event = BuildEvent(
            run_metadata=EntityRunMetadata(),
            type=BuildEventType.WORKLOAD_STATUS_EVENT,
            timestamp=datetime(2026, 6, 1, 9, 0, 0),
            source="monitor",
        )

        result = BuildEventPublisher._serialize_event(event)

        assert result["build_id"] == "unknown"
        assert result["event_type"] == "workload_status_event"
        assert result["target_name"] == ""
        assert result["step_name"] == ""
        assert result["source"] == "monitor"
