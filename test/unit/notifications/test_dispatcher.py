"""Unit tests for standalone notification dispatcher."""

from unittest.mock import AsyncMock, patch

import pytest

from gbserver.notifications.adapter import NotificationAdapter
from gbserver.notifications.dispatcher import StandaloneDispatcher
from gbserver.types.buildevent import (
    BuildEvent,
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
            msg="Training started",
        ),
    )


def _make_message_event() -> BuildEvent:
    """Create a BuildEvent with message type for testing."""
    return BuildEvent(
        run_metadata=EntityRunMetadata(
            build_id="deadbeef12345678",
            target_name="eval-granite-7b",
        ),
        type=BuildEventType.MESSAGE_EVENT,
        payload=None,
    )


class TestStandaloneDispatcher:
    """Tests for StandaloneDispatcher."""

    @pytest.mark.asyncio
    async def test_dispatches_to_matching_adapter(self, tmp_path):
        """Event matching the adapter's filter is delivered."""
        config_file = tmp_path / "notifications.yaml"
        config_file.write_text(
            "notifications:\n"
            "  - type: telegram\n"
            "    bot_token: fake-token\n"
            "    chat_id: '12345'\n"
            "    events: [status_event]\n"
        )

        dispatcher = StandaloneDispatcher(config_path=str(config_file))
        assert len(dispatcher._adapters) == 1

        mock_deliver = AsyncMock(return_value=True)
        dispatcher._adapters[0].adapter.deliver = mock_deliver

        event = _make_status_event()
        await dispatcher.dispatch(event)

        mock_deliver.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_skips_non_matching_events(self, tmp_path):
        """Event not in the adapter's filter is not delivered."""
        config_file = tmp_path / "notifications.yaml"
        config_file.write_text(
            "notifications:\n"
            "  - type: telegram\n"
            "    bot_token: fake-token\n"
            "    chat_id: '12345'\n"
            "    events: [status_event]\n"
        )

        dispatcher = StandaloneDispatcher(config_path=str(config_file))
        mock_deliver = AsyncMock(return_value=True)
        dispatcher._adapters[0].adapter.deliver = mock_deliver

        event = _make_message_event()
        await dispatcher.dispatch(event)

        mock_deliver.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_adapter_failure_gracefully(self, tmp_path):
        """Adapter exception is caught; dispatch does not raise."""
        config_file = tmp_path / "notifications.yaml"
        config_file.write_text(
            "notifications:\n"
            "  - type: telegram\n"
            "    bot_token: fake-token\n"
            "    chat_id: '12345'\n"
            "    events: [status_event]\n"
        )

        dispatcher = StandaloneDispatcher(config_path=str(config_file))
        mock_deliver = AsyncMock(side_effect=RuntimeError("Network failure"))
        dispatcher._adapters[0].adapter.deliver = mock_deliver

        event = _make_status_event()
        # Should not raise
        await dispatcher.dispatch(event)

        mock_deliver.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_wildcard_matches_all_events(self, tmp_path):
        """Wildcard '*' in events list matches any event type."""
        config_file = tmp_path / "notifications.yaml"
        config_file.write_text(
            "notifications:\n"
            "  - type: telegram\n"
            "    bot_token: fake-token\n"
            "    chat_id: '12345'\n"
            "    events: ['*']\n"
        )

        dispatcher = StandaloneDispatcher(config_path=str(config_file))
        mock_deliver = AsyncMock(return_value=True)
        dispatcher._adapters[0].adapter.deliver = mock_deliver

        # Dispatch a status event
        event1 = _make_status_event()
        await dispatcher.dispatch(event1)
        assert mock_deliver.call_count == 1

        # Dispatch a message event - should also match wildcard
        event2 = _make_message_event()
        await dispatcher.dispatch(event2)
        assert mock_deliver.call_count == 2

    @pytest.mark.asyncio
    async def test_empty_config_produces_no_adapters(self):
        """With no config file, dispatcher has no adapters and dispatch is a no-op."""
        dispatcher = StandaloneDispatcher(
            config_path="/nonexistent/notifications.yaml"
        )
        assert len(dispatcher._adapters) == 0

        event = _make_status_event()
        # Should not raise
        await dispatcher.dispatch(event)

    @pytest.mark.asyncio
    async def test_unknown_adapter_type_is_skipped(self, tmp_path):
        """Unknown adapter type is logged and skipped."""
        config_file = tmp_path / "notifications.yaml"
        config_file.write_text(
            "notifications:\n"
            "  - type: unknown_service\n"
            "    api_key: some-key\n"
            "    events: ['*']\n"
        )

        dispatcher = StandaloneDispatcher(config_path=str(config_file))
        assert len(dispatcher._adapters) == 0
