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
Tests for alert handlers.
"""

import os
from datetime import datetime, timezone
from typing import Self
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gbserver.resilience.alert_handlers import (
    CompositeAlertHandler,
    LoggingAlertHandler,
    NodeHealthAlert,
    RetryableAlertHandler,
    SlackAlertHandler,
    WebhookAlertHandler,
    create_alert_handler_from_env,
)


class TestNodeHealthAlert:
    """Tests for NodeHealthAlert class."""

    def test_create_alert(self: Self) -> None:
        """Test creating a node health alert."""
        alert = NodeHealthAlert(
            node_name="worker-node-1",
            failure_count=5,
            threshold=3,
            window_minutes=30,
            failures=[{"failure_type": "FailedMount", "build_id": "build-1"}],
        )

        assert alert.node_name == "worker-node-1"
        assert alert.failure_count == 5
        assert alert.threshold == 3
        assert alert.window_minutes == 30
        assert len(alert.failures) == 1
        assert isinstance(alert.timestamp, datetime)

    def test_alert_to_dict(self: Self) -> None:
        """Test converting alert to dictionary."""
        alert = NodeHealthAlert(
            node_name="worker-node-1",
            failure_count=5,
            threshold=3,
            window_minutes=30,
            failures=[],
        )

        alert_dict = alert.to_dict()

        assert alert_dict["alert_type"] == "node_health_threshold_exceeded"
        assert alert_dict["node_name"] == "worker-node-1"
        assert alert_dict["failure_count"] == 5
        assert alert_dict["recommended_action"] == "kubectl cordon worker-node-1"
        assert alert_dict["severity"] == "warning"


class TestLoggingAlertHandler:
    """Tests for LoggingAlertHandler."""

    @pytest.mark.asyncio
    async def test_send_alert_logs_error(self: Self) -> None:
        """Test that logging handler logs at ERROR level."""
        handler = LoggingAlertHandler()
        alert = NodeHealthAlert(
            node_name="worker-node-1",
            failure_count=5,
            threshold=3,
            window_minutes=30,
            failures=[],
        )

        with patch("gbserver.resilience.alert_handlers.logger") as mock_logger:
            result = await handler.send_alert(alert)

            assert result is True
            mock_logger.error.assert_called_once()
            # Check that worker-node-1 is in the call arguments (format string uses %s)
            call_args = mock_logger.error.call_args[0]
            assert "worker-node-1" in call_args  # node_name is one of the args


class TestWebhookAlertHandler:
    """Tests for WebhookAlertHandler."""

    @pytest.mark.asyncio
    async def test_send_alert_success(self: Self) -> None:
        """Test successful webhook alert."""
        handler = WebhookAlertHandler(
            webhook_url="https://example.com/webhook",
            headers={"Authorization": "Bearer token"},
        )
        alert = NodeHealthAlert(
            node_name="worker-node-1",
            failure_count=5,
            threshold=3,
            window_minutes=30,
            failures=[],
        )

        # Mock response object with status attribute
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        # Mock post method that returns the mock response
        mock_post = MagicMock(return_value=mock_response)

        # Mock session instance
        mock_session_instance = MagicMock()
        mock_session_instance.post = mock_post
        mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session_instance.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session_instance):
            result = await handler.send_alert(alert)

            assert result is True

    @pytest.mark.asyncio
    async def test_send_alert_failure(self: Self) -> None:
        """Test webhook alert failure."""
        handler = WebhookAlertHandler(webhook_url="https://example.com/webhook")
        alert = NodeHealthAlert(
            node_name="worker-node-1",
            failure_count=5,
            threshold=3,
            window_minutes=30,
            failures=[],
        )

        # Mock response object with status attribute and text method
        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        # Mock post method that returns the mock response
        mock_post = MagicMock(return_value=mock_response)

        # Mock session instance
        mock_session_instance = MagicMock()
        mock_session_instance.post = mock_post
        mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session_instance.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session_instance):
            result = await handler.send_alert(alert)

            assert result is False


class TestSlackAlertHandler:
    """Tests for SlackAlertHandler."""

    def test_format_slack_message(self: Self) -> None:
        """Test Slack message formatting."""
        handler = SlackAlertHandler(
            webhook_url="https://hooks.slack.com/services/xxx",
            channel="#test-channel",
            mention_users=["U1234567890"],
        )
        alert = NodeHealthAlert(
            node_name="worker-node-1",
            failure_count=5,
            threshold=3,
            window_minutes=30,
            failures=[
                {"failure_type": "FailedMount"},
                {"failure_type": "FailedMount"},
                {"failure_type": "FailedAttachVolume"},
            ],
        )

        message = handler._format_slack_message(alert)

        assert "blocks" in message
        assert message["channel"] == "#test-channel"
        # Check that blocks contain expected content
        blocks_text = str(message["blocks"])
        assert "worker-node-1" in blocks_text
        assert "<@U1234567890>" in blocks_text

    @pytest.mark.asyncio
    async def test_send_alert_success(self: Self) -> None:
        """Test successful Slack alert."""
        handler = SlackAlertHandler(
            webhook_url="https://hooks.slack.com/services/xxx",
        )
        alert = NodeHealthAlert(
            node_name="worker-node-1",
            failure_count=5,
            threshold=3,
            window_minutes=30,
            failures=[],
        )

        # Mock response object with status attribute
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        # Mock post method that returns the mock response
        mock_post = MagicMock(return_value=mock_response)

        # Mock session instance
        mock_session_instance = MagicMock()
        mock_session_instance.post = mock_post
        mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session_instance.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session_instance):
            result = await handler.send_alert(alert)

            assert result is True


class TestCompositeAlertHandler:
    """Tests for CompositeAlertHandler."""

    @pytest.mark.asyncio
    async def test_send_to_all_handlers(self: Self) -> None:
        """Test that alert is sent to all handlers."""
        handler1 = AsyncMock(spec=LoggingAlertHandler)
        handler1.send_alert = AsyncMock(return_value=True)

        handler2 = AsyncMock(spec=LoggingAlertHandler)
        handler2.send_alert = AsyncMock(return_value=True)

        composite = CompositeAlertHandler(handlers=[handler1, handler2])
        alert = NodeHealthAlert(
            node_name="worker-node-1",
            failure_count=5,
            threshold=3,
            window_minutes=30,
            failures=[],
        )

        result = await composite.send_alert(alert)

        assert result is True
        handler1.send_alert.assert_called_once_with(alert)
        handler2.send_alert.assert_called_once_with(alert)

    @pytest.mark.asyncio
    async def test_require_all_true(self: Self) -> None:
        """Test require_all=True returns False if any handler fails."""
        handler1 = AsyncMock()
        handler1.send_alert = AsyncMock(return_value=True)

        handler2 = AsyncMock()
        handler2.send_alert = AsyncMock(return_value=False)

        composite = CompositeAlertHandler(
            handlers=[handler1, handler2],
            require_all=True,
        )
        alert = NodeHealthAlert(
            node_name="worker-node-1",
            failure_count=5,
            threshold=3,
            window_minutes=30,
            failures=[],
        )

        result = await composite.send_alert(alert)

        assert result is False

    @pytest.mark.asyncio
    async def test_require_all_false(self: Self) -> None:
        """Test require_all=False returns True if any handler succeeds."""
        handler1 = AsyncMock()
        handler1.send_alert = AsyncMock(return_value=True)

        handler2 = AsyncMock()
        handler2.send_alert = AsyncMock(return_value=False)

        composite = CompositeAlertHandler(
            handlers=[handler1, handler2],
            require_all=False,
        )
        alert = NodeHealthAlert(
            node_name="worker-node-1",
            failure_count=5,
            threshold=3,
            window_minutes=30,
            failures=[],
        )

        result = await composite.send_alert(alert)

        assert result is True


class TestRetryableAlertHandler:
    """Tests for RetryableAlertHandler."""

    @pytest.mark.asyncio
    async def test_successful_send(self: Self) -> None:
        """Test successful alert send without retries."""
        mock_handler = AsyncMock()
        mock_handler.send_alert = AsyncMock(return_value=True)

        retryable = RetryableAlertHandler(
            handler=mock_handler,
            max_retries=3,
            initial_backoff=0.01,
        )

        alert = NodeHealthAlert(
            node_name="node-1",
            failure_count=5,
            threshold=3,
            window_minutes=30,
            failures=[],
        )

        result = await retryable.send_alert(alert)

        assert result is True
        assert mock_handler.send_alert.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_failure(self: Self) -> None:
        """Test that handler retries on failure."""
        mock_handler = AsyncMock()
        # Fail twice, succeed on third attempt
        mock_handler.send_alert = AsyncMock(side_effect=[False, False, True])

        retryable = RetryableAlertHandler(
            handler=mock_handler,
            max_retries=3,
            initial_backoff=0.01,
        )

        alert = NodeHealthAlert(
            node_name="node-1",
            failure_count=5,
            threshold=3,
            window_minutes=30,
            failures=[],
        )

        result = await retryable.send_alert(alert)

        assert result is True
        assert mock_handler.send_alert.call_count == 3

    @pytest.mark.asyncio
    async def test_dlq_on_max_retries(self: Self) -> None:
        """Test that failed alerts go to DLQ after max retries."""
        mock_handler = AsyncMock()
        mock_handler.send_alert = AsyncMock(return_value=False)  # Always fail

        retryable = RetryableAlertHandler(
            handler=mock_handler,
            max_retries=2,
            initial_backoff=0.01,
        )

        alert = NodeHealthAlert(
            node_name="node-1",
            failure_count=5,
            threshold=3,
            window_minutes=30,
            failures=[],
        )

        result = await retryable.send_alert(alert)

        # Should have failed
        assert result is False

        # Should have tried 3 times (initial + 2 retries)
        assert mock_handler.send_alert.call_count == 3

        # Should be in DLQ
        dlq_entries = retryable.get_dlq_entries()
        assert len(dlq_entries) == 1
        assert dlq_entries[0]["reason"] == "max_retries_exceeded"

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens(self: Self) -> None:
        """Test that circuit breaker opens after repeated failures."""
        mock_handler = AsyncMock()
        mock_handler.send_alert = AsyncMock(side_effect=Exception("Connection refused"))

        retryable = RetryableAlertHandler(
            handler=mock_handler,
            max_retries=2,
            initial_backoff=0.01,
        )

        # Trigger 5 failures to open circuit
        for i in range(5):
            alert = NodeHealthAlert(
                node_name=f"node-{i}",
                failure_count=5,
                threshold=3,
                window_minutes=30,
                failures=[],
            )
            await retryable.send_alert(alert)

        # Circuit should be open now
        assert retryable._circuit_open is True

        # Next alert should go straight to DLQ
        alert = NodeHealthAlert(
            node_name="node-6",
            failure_count=5,
            threshold=3,
            window_minutes=30,
            failures=[],
        )
        result = await retryable.send_alert(alert)

        assert result is False

        # Check DLQ has entry with circuit_breaker_open reason
        dlq_entries = retryable.get_dlq_entries()
        circuit_breaker_entries = [
            e for e in dlq_entries if e["reason"] == "circuit_breaker_open"
        ]
        assert len(circuit_breaker_entries) >= 1

    @pytest.mark.asyncio
    async def test_clear_dlq(self: Self) -> None:
        """Test clearing the dead letter queue."""
        mock_handler = AsyncMock()
        mock_handler.send_alert = AsyncMock(return_value=False)

        retryable = RetryableAlertHandler(
            handler=mock_handler,
            max_retries=0,  # Fail immediately
            initial_backoff=0.01,
        )

        # Add some entries to DLQ
        for i in range(3):
            alert = NodeHealthAlert(
                node_name=f"node-{i}",
                failure_count=5,
                threshold=3,
                window_minutes=30,
                failures=[],
            )
            await retryable.send_alert(alert)

        assert len(retryable.get_dlq_entries()) == 3

        # Clear DLQ
        cleared = retryable.clear_dlq()

        assert cleared == 3
        assert len(retryable.get_dlq_entries()) == 0


class TestCreateAlertHandlerFromEnv:
    """Tests for create_alert_handler_from_env."""

    def test_no_env_vars_returns_logging_handler(self: Self) -> None:
        """Test that with no env vars, only logging handler is returned."""
        with patch.dict(os.environ, {}, clear=True):
            handler = create_alert_handler_from_env()

            assert isinstance(handler, LoggingAlertHandler)

    def test_webhook_url_creates_composite(self: Self) -> None:
        """Test that webhook URL creates composite handler."""
        with patch.dict(
            os.environ,
            {"GBSERVER_NODE_HEALTH_ALERT_WEBHOOK_URL": "https://example.com/webhook"},
            clear=True,
        ):
            handler = create_alert_handler_from_env()

            assert isinstance(handler, CompositeAlertHandler)
            assert len(handler.handlers) == 2
            assert isinstance(handler.handlers[0], LoggingAlertHandler)
            assert isinstance(handler.handlers[1], WebhookAlertHandler)

    def test_slack_webhook_creates_composite(self: Self) -> None:
        """Test that Slack webhook creates composite handler."""
        with patch.dict(
            os.environ,
            {
                "GBSERVER_NODE_HEALTH_ALERT_SLACK_WEBHOOK_URL": "https://hooks.slack.com/xxx",
                "GBSERVER_NODE_HEALTH_ALERT_SLACK_CHANNEL": "#alerts",
                "GBSERVER_NODE_HEALTH_ALERT_SLACK_MENTION_USERS": "U123,U456",
            },
            clear=True,
        ):
            handler = create_alert_handler_from_env()

            assert isinstance(handler, CompositeAlertHandler)
            assert len(handler.handlers) == 2
            assert isinstance(handler.handlers[0], LoggingAlertHandler)
            assert isinstance(handler.handlers[1], SlackAlertHandler)
            assert handler.handlers[1].channel == "#alerts"
            assert handler.handlers[1].mention_users == ["U123", "U456"]

    def test_graceful_degradation_on_handler_failure(self: Self) -> None:
        """Test that handler failures fall back to logging."""
        # Mock WebhookAlertHandler to raise during initialization
        with patch(
            "gbserver.resilience.alert_handlers.WebhookAlertHandler",
            side_effect=Exception("Network error"),
        ):
            with patch.dict(
                os.environ,
                {
                    "GBSERVER_NODE_HEALTH_ALERT_WEBHOOK_URL": "https://example.com/webhook"
                },
                clear=True,
            ):
                handler = create_alert_handler_from_env()

                # Should fall back to logging handler only
                assert isinstance(handler, LoggingAlertHandler)
