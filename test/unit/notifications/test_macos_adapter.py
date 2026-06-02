"""Unit tests for macOS notification adapter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gbserver.notifications.macos_adapter import MacOSAdapter
from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventMessagePayload,
    BuildEventStatusPayload,
    BuildEventType,
    EntityRunMetadata,
)
from gbserver.types.status import Status


def _make_status_event(
    status: Status = Status.RUNNING, msg: str = "Training started"
) -> BuildEvent:
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


class TestMacOSAdapter:
    """Tests for MacOSAdapter."""

    @pytest.mark.asyncio
    async def test_successful_delivery(self):
        """osascript exits 0 -> deliver returns True."""
        adapter = MacOSAdapter()
        event = _make_status_event()

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        mock_subprocess = AsyncMock(return_value=mock_proc)
        with patch("platform.system", return_value="Darwin"):
            with patch("asyncio.create_subprocess_exec", mock_subprocess):
                result = await adapter.deliver(event)

        assert result is True

    @pytest.mark.asyncio
    async def test_osascript_failure_returns_false(self):
        """osascript exits non-zero -> deliver returns False."""
        adapter = MacOSAdapter()
        event = _make_status_event()

        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"some error"))

        mock_subprocess = AsyncMock(return_value=mock_proc)
        with patch("platform.system", return_value="Darwin"):
            with patch("asyncio.create_subprocess_exec", mock_subprocess):
                result = await adapter.deliver(event)

        assert result is False

    @pytest.mark.asyncio
    async def test_non_macos_platform_returns_false(self):
        """Non-macOS platform -> deliver returns False without calling osascript."""
        adapter = MacOSAdapter()
        event = _make_status_event()

        with patch("platform.system", return_value="Linux"):
            result = await adapter.deliver(event)

        assert result is False

    @pytest.mark.asyncio
    async def test_title_formatting_status_event(self):
        """Title includes build_id prefix and status for status events."""
        adapter = MacOSAdapter()
        event = _make_status_event(status=Status.SUCCESS, msg="Done")

        title = adapter._build_title(event)

        assert "abcdef12" in title
        assert "success" in title

    @pytest.mark.asyncio
    async def test_message_formatting_includes_target(self):
        """Message includes target name."""
        adapter = MacOSAdapter()
        event = _make_status_event()

        message = adapter._format_message(event)

        assert "train-granite-7b" in message
