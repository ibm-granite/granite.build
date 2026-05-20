"""Unit tests for WebhookBatchBuffer."""

import time

from gbserver.webhooks.batch_buffer import WebhookBatchBuffer
from gbserver.webhooks.models import StoredWebhookSubscription


class TestWebhookBatchBuffer:
    """Tests for the per-subscription event batch buffer."""

    def _make_subscription(self, sub_id="sub-1", frequency=30):
        """Create a StoredWebhookSubscription with a predictable UUID.

        Args:
            sub_id: UUID override for deterministic testing.
            frequency: Batch flush interval in seconds.

        Returns:
            A StoredWebhookSubscription instance with the given UUID.
        """
        sub = StoredWebhookSubscription(
            space_name="s",
            build_id="b1",
            webhook_url="https://example.com",
            secret="s",
            event_types=["*"],
            created_by="u",
            frequency=frequency,
        )
        sub.uuid = sub_id
        return sub

    def test_add_event_to_buffer(self):
        """Register a subscription, add one event, verify pending_count is 1."""
        buffer = WebhookBatchBuffer()
        sub = self._make_subscription()
        buffer.register_subscription(sub)

        buffer.add_event("sub-1", {"type": "build.started", "build_id": "b1"})

        assert buffer.pending_count("sub-1") == 1

    def test_flush_returns_accumulated_events(self):
        """Add two events, flush returns both, then pending_count is 0."""
        buffer = WebhookBatchBuffer()
        sub = self._make_subscription()
        buffer.register_subscription(sub)

        buffer.add_event("sub-1", {"type": "build.started", "build_id": "b1"})
        buffer.add_event("sub-1", {"type": "build.completed", "build_id": "b1"})

        events = buffer.flush("sub-1")

        assert len(events) == 2
        assert events[0]["type"] == "build.started"
        assert events[1]["type"] == "build.completed"
        assert buffer.pending_count("sub-1") == 0

    def test_flush_empty_buffer_returns_empty_list(self):
        """Flush immediately after registration returns an empty list."""
        buffer = WebhookBatchBuffer()
        sub = self._make_subscription()
        buffer.register_subscription(sub)

        events = buffer.flush("sub-1")

        assert events == []

    def test_is_ready_to_flush(self):
        """is_ready_to_flush respects the frequency interval."""
        buffer = WebhookBatchBuffer()
        sub = self._make_subscription(frequency=1)
        buffer.register_subscription(sub)

        buffer.add_event("sub-1", {"type": "build.started", "build_id": "b1"})

        # Immediately after registration, not ready yet
        assert buffer.is_ready_to_flush("sub-1") is False

        # Backdate the last flush time by 2 seconds to simulate elapsed time
        with buffer._lock:
            buffer._last_flush["sub-1"] = time.time() - 2.0

        assert buffer.is_ready_to_flush("sub-1") is True

    def test_is_ready_to_flush_empty_buffer_not_ready(self):
        """Even if frequency elapsed, an empty buffer is not ready."""
        buffer = WebhookBatchBuffer()
        sub = self._make_subscription(frequency=1)
        buffer.register_subscription(sub)

        # Backdate flush time
        with buffer._lock:
            buffer._last_flush["sub-1"] = time.time() - 2.0

        assert buffer.is_ready_to_flush("sub-1") is False

    def test_unregister_subscription(self):
        """Unregister removes the subscription and discards events."""
        buffer = WebhookBatchBuffer()
        sub = self._make_subscription()
        buffer.register_subscription(sub)

        buffer.add_event("sub-1", {"type": "build.started", "build_id": "b1"})
        buffer.unregister_subscription("sub-1")

        assert buffer.pending_count("sub-1") == 0

    def test_get_ready_subscriptions(self):
        """get_ready_subscriptions returns only subs whose interval elapsed."""
        buffer = WebhookBatchBuffer()

        sub1 = self._make_subscription(sub_id="sub-1", frequency=1)
        sub2 = self._make_subscription(sub_id="sub-2", frequency=1)
        buffer.register_subscription(sub1)
        buffer.register_subscription(sub2)

        buffer.add_event("sub-1", {"type": "build.started", "build_id": "b1"})
        buffer.add_event("sub-2", {"type": "build.started", "build_id": "b2"})

        # Backdate only sub-1
        with buffer._lock:
            buffer._last_flush["sub-1"] = time.time() - 2.0

        ready = buffer.get_ready_subscriptions()

        assert "sub-1" in ready
        assert "sub-2" not in ready

    def test_add_event_unregistered_is_noop(self):
        """Adding an event to an unregistered subscription does nothing."""
        buffer = WebhookBatchBuffer()

        # Should not raise
        buffer.add_event("nonexistent", {"type": "build.started"})

        assert buffer.pending_count("nonexistent") == 0

    def test_register_idempotent(self):
        """Registering an already-registered subscription is a no-op."""
        buffer = WebhookBatchBuffer()
        sub = self._make_subscription()
        buffer.register_subscription(sub)

        buffer.add_event("sub-1", {"type": "build.started", "build_id": "b1"})

        # Re-register should not clear existing events
        buffer.register_subscription(sub)

        assert buffer.pending_count("sub-1") == 1

    def test_flush_resets_timer(self):
        """After flush, is_ready_to_flush returns False again."""
        buffer = WebhookBatchBuffer()
        sub = self._make_subscription(frequency=1)
        buffer.register_subscription(sub)

        buffer.add_event("sub-1", {"type": "build.started", "build_id": "b1"})

        # Backdate to make ready
        with buffer._lock:
            buffer._last_flush["sub-1"] = time.time() - 2.0

        assert buffer.is_ready_to_flush("sub-1") is True

        buffer.flush("sub-1")

        # After flush, timer is reset — not ready anymore (even if we add event)
        buffer.add_event("sub-1", {"type": "build.completed", "build_id": "b1"})
        assert buffer.is_ready_to_flush("sub-1") is False

    def test_flush_unregistered_returns_empty(self):
        """Flushing an unregistered subscription returns empty list."""
        buffer = WebhookBatchBuffer()

        assert buffer.flush("nonexistent") == []
