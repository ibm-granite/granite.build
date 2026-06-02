"""Unit tests for ntfy push notification adapter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gbserver.notifications.ntfy_adapter import NtfyAdapter
from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventMessagePayload,
    BuildEventStatusPayload,
    BuildEventType,
    EntityRunMetadata,
)
from gbserver.types.status import Status


def _make_status_event(status: Status = Status.RUNNING, msg: str = "Training started") -> BuildEvent:
    """Create a BuildEvent with a status payload for testing."""
    return BuildEvent(
        run_metadata=EntityRunMetadata(
            build_id="abcdef1234567890",
            target_name="train-granite-7b",
        ),
        type=BuildEventType.STATUS_EVENT,
        payload=BuildEventStatusPayload(
            status=status,
            msg=msg,
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


class TestNtfyAdapter:
    """Tests for NtfyAdapter."""

    @pytest.mark.asyncio
    async def test_successful_delivery(self):
        """ntfy returns 200 -> deliver returns True."""
        adapter = NtfyAdapter(topic="my-builds")
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
        assert call_args[0][0] == "https://ntfy.sh/my-builds"

    @pytest.mark.asyncio
    async def test_failed_delivery(self):
        """ntfy returns 500 -> deliver returns False."""
        adapter = NtfyAdapter(topic="my-builds")
        event = _make_status_event()

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
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
        adapter = NtfyAdapter(topic="my-builds")
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
        adapter = NtfyAdapter(topic="my-builds")
        event = _make_status_event()

        message = adapter._format_message(event)

        assert "Build: abcdef1234567890" in message
        assert "Target: train-granite-7b" in message
        assert "Event: status_event" in message
        assert "Status: running" in message
        assert "Message: Training started" in message

    @pytest.mark.asyncio
    async def test_priority_high_for_failed_builds(self):
        """Failed build status sets priority to high."""
        adapter = NtfyAdapter(topic="my-builds")
        event = _make_status_event(status=Status.FAILED, msg="OOM killed")

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await adapter.deliver(event)

        call_args = mock_session.post.call_args
        headers = call_args[1]["headers"]
        assert headers["Priority"] == "high"
        assert headers["Tags"] == "x"

    @pytest.mark.asyncio
    async def test_custom_server_url(self):
        """Custom server URL is used in the request."""
        adapter = NtfyAdapter(topic="builds", server="https://ntfy.example.com/")
        event = _make_message_event()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await adapter.deliver(event)

        call_args = mock_session.post.call_args
        assert call_args[0][0] == "https://ntfy.example.com/builds"

    @pytest.mark.asyncio
    async def test_title_contains_build_id_and_status(self):
        """Title header contains the build_id prefix and status for status events."""
        adapter = NtfyAdapter(topic="my-builds")
        event = _make_status_event(status=Status.SUCCESS, msg="Done")

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await adapter.deliver(event)

        call_args = mock_session.post.call_args
        title = call_args[1]["headers"]["Title"]
        assert "abcdef12" in title
        assert "success" in title
