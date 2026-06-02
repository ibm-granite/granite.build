"""Unit tests for Telegram notification adapter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gbserver.notifications.telegram_adapter import TelegramAdapter
from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventMessagePayload,
    BuildEventStatusPayload,
    BuildEventType,
    EntityRunMetadata,
)
from gbserver.types.status import Status


def _make_status_event() -> BuildEvent:
    """Create a BuildEvent with a status payload for testing."""
    return BuildEvent(
        run_metadata=EntityRunMetadata(
            build_id="abcdef1234567890",
            target_name="train-granite-7b",
        ),
        type=BuildEventType.STATUS_EVENT,
        payload=BuildEventStatusPayload(
            status=Status.RUNNING,
            msg="Training started successfully",
        ),
    )


def _make_message_event() -> BuildEvent:
    """Create a BuildEvent with a message payload (non-status) for testing."""
    return BuildEvent(
        run_metadata=EntityRunMetadata(
            build_id="deadbeef12345678",
            target_name="eval-granite-7b",
        ),
        type=BuildEventType.MESSAGE_EVENT,
        payload=BuildEventMessagePayload(
            level="INFO",
            msg="Step completed",
        ),
    )


class TestTelegramAdapter:
    """Tests for TelegramAdapter."""

    @pytest.mark.asyncio
    async def test_successful_delivery(self):
        """Telegram API returns 200 -> deliver returns True."""
        adapter = TelegramAdapter(bot_token="fake-token", chat_id="12345")
        event = _make_status_event()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await adapter.deliver(event)

        assert result is True
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert "botfake-token" in call_args[0][0]
        assert call_args[1]["json"]["chat_id"] == "12345"
        assert call_args[1]["json"]["parse_mode"] == "Markdown"

    @pytest.mark.asyncio
    async def test_failed_delivery(self):
        """Telegram API returns 400 -> deliver returns False."""
        adapter = TelegramAdapter(bot_token="fake-token", chat_id="12345")
        event = _make_status_event()

        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await adapter.deliver(event)

        assert result is False

    @pytest.mark.asyncio
    async def test_exception_handling(self):
        """Network exception -> deliver returns False without raising."""
        adapter = TelegramAdapter(bot_token="fake-token", chat_id="12345")
        event = _make_status_event()

        mock_session = AsyncMock()
        mock_session.post = MagicMock(side_effect=ConnectionError("Network down"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await adapter.deliver(event)

        assert result is False

    @pytest.mark.asyncio
    async def test_message_format_status_event(self):
        """Status events include status and message lines."""
        adapter = TelegramAdapter(bot_token="fake-token", chat_id="12345")
        event = _make_status_event()

        message = adapter._format_message(event)

        assert "*Build Event* `abcdef12`" in message
        assert "Target: `train-granite-7b`" in message
        assert "Event: `status_event`" in message
        assert "Status: *running*" in message
        assert "Message: Training started successfully" in message

    @pytest.mark.asyncio
    async def test_message_format_non_status_event(self):
        """Non-status events do not include status/message lines."""
        adapter = TelegramAdapter(bot_token="fake-token", chat_id="12345")
        event = _make_message_event()

        message = adapter._format_message(event)

        assert "*Build Event* `deadbeef`" in message
        assert "Target: `eval-granite-7b`" in message
        assert "Event: `message_event`" in message
        assert "Status:" not in message
        assert "Message:" not in message
