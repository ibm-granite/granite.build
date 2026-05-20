"""Unit tests for WebhookDispatcher."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventStatusPayload,
    BuildEventType,
    EntityRunMetadata,
)
from gbserver.types.status import Status
from gbserver.webhooks.models import StoredWebhookSubscription


class TestWebhookDispatcher:
    """Tests for the batch dispatcher that orchestrates buffering, log scanning, and delivery."""

    def setup_method(self):
        """Set up a dispatcher with mock storage for each test."""
        from gbserver.webhooks.dispatcher import WebhookDispatcher

        self.mock_storage = MagicMock()
        self.dispatcher = WebhookDispatcher(
            webhook_storage=self.mock_storage,
            build_id="build-1",
            space_name="test-space",
            build_name="test-build",
            username="testuser",
            build_start_time="2026-05-20T11:00:00Z",
        )

    def _make_subscription(self, **kwargs):
        """Create a StoredWebhookSubscription with sensible defaults.

        Args:
            **kwargs: Overrides for subscription fields.

        Returns:
            A StoredWebhookSubscription instance.
        """
        defaults = {
            "space_name": "test-space",
            "build_id": "build-1",
            "webhook_url": "https://example.com/hook",
            "secret": "secret",
            "event_types": ["*"],
            "created_by": "user",
            "frequency": 30,
        }
        defaults.update(kwargs)
        return StoredWebhookSubscription(**defaults)

    def _make_status_event(self, status=Status.SUCCESS):
        """Create a STATUS_EVENT BuildEvent.

        Args:
            status: The status to include in the payload.

        Returns:
            A BuildEvent of type STATUS_EVENT.
        """
        return BuildEvent(
            type=BuildEventType.STATUS_EVENT,
            run_metadata=EntityRunMetadata(
                build_id="build-1",
                username="user",
                target_name="target-1",
                targetrun_id="tr-1",
                targetstep_uri="step://test",
                targetsteprun_id="tsr-1",
            ),
            payload=BuildEventStatusPayload(status=status, msg="Done"),
        )

    def test_accept_event_buffers_matching(self):
        """Events matching a wildcard subscription are buffered."""
        sub = self._make_subscription()
        self.dispatcher.start([sub])
        self.dispatcher.accept_event(self._make_status_event())
        assert self.dispatcher.buffer.pending_count(sub.uuid) == 1

    def test_accept_event_skips_excluded(self):
        """Events in excluded_types are not buffered even if event_types is wildcard."""
        sub = self._make_subscription(
            event_types=["*"], excluded_types=["status_event"]
        )
        self.dispatcher.start([sub])
        self.dispatcher.accept_event(self._make_status_event())
        assert self.dispatcher.buffer.pending_count(sub.uuid) == 0

    def test_accept_event_skips_internal(self):
        """Internal events (TERMINATE_EVENT) are never buffered."""
        sub = self._make_subscription()
        self.dispatcher.start([sub])
        event = BuildEvent(
            type=BuildEventType.TERMINATE_EVENT,
            run_metadata=EntityRunMetadata(
                build_id="build-1",
                username="u",
                target_name="t",
                targetrun_id="tr",
                targetstep_uri="s",
                targetsteprun_id="tsr",
            ),
            payload=MagicMock(),
        )
        self.dispatcher.accept_event(event)
        assert self.dispatcher.buffer.pending_count(sub.uuid) == 0

    def test_accept_event_multiple_subscriptions(self):
        """Events are buffered to all matching subscriptions."""
        sub1 = self._make_subscription()
        sub2 = self._make_subscription(event_types=["status_event"])
        self.dispatcher.start([sub1, sub2])
        self.dispatcher.accept_event(self._make_status_event())
        assert self.dispatcher.buffer.pending_count(sub1.uuid) == 1
        assert self.dispatcher.buffer.pending_count(sub2.uuid) == 1

    def test_accept_event_non_matching_type(self):
        """Events are not buffered to subscriptions that don't match the type."""
        sub = self._make_subscription(event_types=["message_event"])
        self.dispatcher.start([sub])
        self.dispatcher.accept_event(self._make_status_event())
        assert self.dispatcher.buffer.pending_count(sub.uuid) == 0

    @pytest.mark.asyncio
    async def test_flush_subscription_delivers_batch(self):
        """Flushing a subscription with buffered events delivers a payload."""
        sub = self._make_subscription()
        self.dispatcher.start([sub])
        self.dispatcher.accept_event(self._make_status_event())

        with patch("gbserver.webhooks.dispatcher.WebhookDelivery") as mock_cls:
            mock_delivery = AsyncMock()
            mock_delivery.deliver = AsyncMock(return_value=True)
            mock_cls.return_value = mock_delivery

            await self.dispatcher.flush_subscription(sub)

            mock_delivery.deliver.assert_called_once()
            payload = mock_delivery.deliver.call_args[0][0]
            assert payload["build_id"] == "build-1"
            assert payload["user"] == "testuser"
            assert payload["space_name"] == "test-space"
            assert payload["build_name"] == "test-build"
            assert len(payload["events"]) == 1
            assert payload["events"][0]["event_type"] == "status_event"

    @pytest.mark.asyncio
    async def test_flush_subscription_with_log_pattern(self):
        """Log lines matching a subscription's pattern generate LOG_EVENT entries."""
        sub = self._make_subscription(log_pattern=r"(?i)error")
        self.dispatcher.start([sub])
        # No regular events, but log lines match
        with patch("gbserver.webhooks.dispatcher.WebhookDelivery") as mock_cls:
            mock_delivery = AsyncMock()
            mock_delivery.deliver = AsyncMock(return_value=True)
            mock_cls.return_value = mock_delivery

            await self.dispatcher.flush_subscription(
                sub, log_lines=["INFO ok", "ERROR bad"]
            )

            mock_delivery.deliver.assert_called_once()
            payload = mock_delivery.deliver.call_args[0][0]
            assert len(payload["events"]) == 1
            assert payload["events"][0]["event_type"] == "LOG_EVENT"

    @pytest.mark.asyncio
    async def test_flush_subscription_empty_does_nothing(self):
        """Flushing with no buffered events and no log matches does not deliver."""
        sub = self._make_subscription()
        self.dispatcher.start([sub])
        with patch("gbserver.webhooks.dispatcher.WebhookDelivery") as mock_cls:
            await self.dispatcher.flush_subscription(sub)
            mock_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_flush_all_ready_flushes_ready_subscriptions(self):
        """flush_all_ready delivers only to subscriptions whose interval elapsed."""
        import time

        sub = self._make_subscription()
        self.dispatcher.start([sub])
        self.dispatcher.accept_event(self._make_status_event())

        # Backdate flush timer to simulate elapsed interval
        with self.dispatcher.buffer._lock:
            self.dispatcher.buffer._last_flush[sub.uuid] = time.time() - 60.0

        with patch("gbserver.webhooks.dispatcher.WebhookDelivery") as mock_cls:
            mock_delivery = AsyncMock()
            mock_delivery.deliver = AsyncMock(return_value=True)
            mock_cls.return_value = mock_delivery

            await self.dispatcher.flush_all_ready()

            mock_delivery.deliver.assert_called_once()

    @pytest.mark.asyncio
    async def test_flush_final_flushes_all(self):
        """flush_final forces a flush of all subscriptions regardless of timer."""
        sub = self._make_subscription()
        self.dispatcher.start([sub])
        self.dispatcher.accept_event(self._make_status_event())

        with patch("gbserver.webhooks.dispatcher.WebhookDelivery") as mock_cls:
            mock_delivery = AsyncMock()
            mock_delivery.deliver = AsyncMock(return_value=True)
            mock_cls.return_value = mock_delivery

            await self.dispatcher.flush_final()

            mock_delivery.deliver.assert_called_once()

    @pytest.mark.asyncio
    async def test_flush_final_with_log_lines(self):
        """flush_final passes log_lines to the flush process."""
        sub = self._make_subscription(log_pattern=r"WARN")
        self.dispatcher.start([sub])

        with patch("gbserver.webhooks.dispatcher.WebhookDelivery") as mock_cls:
            mock_delivery = AsyncMock()
            mock_delivery.deliver = AsyncMock(return_value=True)
            mock_cls.return_value = mock_delivery

            await self.dispatcher.flush_final(log_lines=["WARN something happened"])

            mock_delivery.deliver.assert_called_once()
            payload = mock_delivery.deliver.call_args[0][0]
            assert payload["events"][0]["event_type"] == "LOG_EVENT"

    @pytest.mark.asyncio
    async def test_flush_subscription_error_does_not_propagate_in_flush_all(self):
        """Errors in flush_all_ready are logged but do not stop other flushes."""
        import time

        sub1 = self._make_subscription()
        sub2 = self._make_subscription()
        self.dispatcher.start([sub1, sub2])
        self.dispatcher.accept_event(self._make_status_event())

        # Backdate both flush timers
        with self.dispatcher.buffer._lock:
            self.dispatcher.buffer._last_flush[sub1.uuid] = time.time() - 60.0
            self.dispatcher.buffer._last_flush[sub2.uuid] = time.time() - 60.0

        with patch("gbserver.webhooks.dispatcher.WebhookDelivery") as mock_cls:
            mock_delivery = AsyncMock()
            mock_delivery.deliver = AsyncMock(side_effect=Exception("network error"))
            mock_cls.return_value = mock_delivery

            # Should not raise
            await self.dispatcher.flush_all_ready()

    def test_stop_clears_state(self):
        """After stop(), all subscriptions and buffers are cleared."""
        sub = self._make_subscription()
        self.dispatcher.start([sub])
        self.dispatcher.accept_event(self._make_status_event())
        self.dispatcher.stop()
        assert self.dispatcher.buffer.pending_count(sub.uuid) == 0
        assert len(self.dispatcher._subscriptions) == 0

    def test_start_registers_multiple_subscriptions(self):
        """start() registers all provided subscriptions."""
        sub1 = self._make_subscription()
        sub2 = self._make_subscription(
            webhook_url="https://other.com/hook", secret="other"
        )
        self.dispatcher.start([sub1, sub2])
        assert sub1.uuid in self.dispatcher._subscriptions
        assert sub2.uuid in self.dispatcher._subscriptions

    def test_serialize_event_status_payload(self):
        """_serialize_event correctly serializes a status event."""
        event = self._make_status_event(status=Status.RUNNING)
        data = self.dispatcher._serialize_event(event)
        assert data["event_type"] == "status_event"
        assert data["status"] == "running"
        assert data["target_name"] == "target-1"
        assert data["step_name"] == "step://test"
        assert data["message"] == {"text": "Done"}
        assert "event_id" in data
        assert "timestamp" in data
