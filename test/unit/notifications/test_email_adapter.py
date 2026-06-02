"""Unit tests for email notification adapter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gbserver.notifications.email_adapter import EmailAdapter
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


class TestEmailAdapter:
    """Tests for EmailAdapter."""

    @pytest.mark.asyncio
    async def test_successful_delivery(self):
        """SMTP send succeeds -> deliver returns True."""
        adapter = EmailAdapter(
            to="user@example.com",
            smtp_host="smtp.example.com",
            smtp_port=587,
            smtp_user="user",
            smtp_password="pass",
        )
        event = _make_status_event()

        mock_thread = AsyncMock(return_value=None)
        with patch("asyncio.to_thread", mock_thread):
            result = await adapter.deliver(event)

        assert result is True
        mock_thread.assert_called_once()

    @pytest.mark.asyncio
    async def test_smtp_failure_returns_false(self):
        """SMTP raises exception -> deliver returns False."""
        adapter = EmailAdapter(to="user@example.com", smtp_host="smtp.example.com")
        event = _make_status_event()

        mock_thread = AsyncMock(side_effect=ConnectionRefusedError("Connection refused"))
        with patch("asyncio.to_thread", mock_thread):
            result = await adapter.deliver(event)

        assert result is False

    @pytest.mark.asyncio
    async def test_subject_formatting_status_event(self):
        """Subject line includes build_id prefix and status for status events."""
        adapter = EmailAdapter(to="user@example.com")
        event = _make_status_event(status=Status.FAILED, msg="OOM killed")

        subject = adapter._build_subject(event)

        assert "[gbserver]" in subject
        assert "abcdef12" in subject
        assert "failed" in subject

    @pytest.mark.asyncio
    async def test_subject_formatting_non_status_event(self):
        """Subject line includes event type for non-status events."""
        adapter = EmailAdapter(to="user@example.com")
        event = _make_message_event()

        subject = adapter._build_subject(event)

        assert "[gbserver]" in subject
        assert "deadbeef" in subject
        assert "message_event" in subject

    @pytest.mark.asyncio
    async def test_message_body_contains_build_id_and_target(self):
        """Message body includes build_id, target, and status info."""
        adapter = EmailAdapter(to="user@example.com")
        event = _make_status_event()

        body = adapter._format_message(event)

        assert "abcdef1234567890" in body
        assert "train-granite-7b" in body
        assert "Status: running" in body
        assert "Training started" in body
        assert "gbserver standalone notification" in body
