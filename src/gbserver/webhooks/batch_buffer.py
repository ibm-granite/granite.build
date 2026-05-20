"""In-memory event accumulator with per-subscription flush timers.

Provides a thread-safe buffer that collects webhook events per subscription
and tracks when each subscription's batch interval has elapsed, signalling
readiness for delivery.
"""

import threading
import time
from typing import Any, Dict, List

from gbserver.utils.logger import get_logger
from gbserver.webhooks.models import StoredWebhookSubscription

logger = get_logger(__name__)


class WebhookBatchBuffer:
    """Accumulates events per subscription with time-based flush readiness.

    Thread-safe. Holds events until a subscription's frequency interval has
    elapsed, at which point the buffer can be flushed.

    Internal state:
        _buffers: Maps subscription UUID to its pending event list.
        _frequencies: Maps subscription UUID to its flush interval (seconds).
        _last_flush: Maps subscription UUID to the timestamp of its last flush.
        _lock: Threading lock protecting all internal state.
    """

    def __init__(self) -> None:
        self._buffers: Dict[str, List[Dict[str, Any]]] = {}
        self._frequencies: Dict[str, int] = {}
        self._last_flush: Dict[str, float] = {}
        self._lock = threading.Lock()

    def register_subscription(self, subscription: StoredWebhookSubscription) -> None:
        """Register a subscription for buffering. No-op if already registered.

        Sets up the internal buffer, frequency, and flush timer for the
        subscription. If the subscription is already registered, this method
        does nothing (preserving any existing buffered events).

        Args:
            subscription: The webhook subscription to register.
        """
        with self._lock:
            sub_id = subscription.uuid
            if sub_id in self._buffers:
                logger.debug("Subscription %s already registered, skipping", sub_id)
                return
            self._buffers[sub_id] = []
            self._frequencies[sub_id] = subscription.frequency
            self._last_flush[sub_id] = time.time()
            logger.info(
                "Registered subscription %s with frequency %ds",
                sub_id,
                self._frequencies[sub_id],
            )

    def unregister_subscription(self, subscription_id: str) -> None:
        """Remove subscription, discarding pending events.

        Cleans up all internal state associated with the subscription.
        No-op if the subscription is not registered.

        Args:
            subscription_id: The UUID of the subscription to remove.
        """
        with self._lock:
            self._buffers.pop(subscription_id, None)
            self._frequencies.pop(subscription_id, None)
            self._last_flush.pop(subscription_id, None)
            logger.info("Unregistered subscription %s", subscription_id)

    def add_event(self, subscription_id: str, event_data: Dict[str, Any]) -> None:
        """Add event to subscription's buffer. No-op if not registered.

        Args:
            subscription_id: The UUID of the target subscription.
            event_data: The event payload dict to buffer.
        """
        with self._lock:
            if subscription_id not in self._buffers:
                logger.debug(
                    "Ignoring event for unregistered subscription %s",
                    subscription_id,
                )
                return
            self._buffers[subscription_id].append(event_data)

    def pending_count(self, subscription_id: str) -> int:
        """Return number of pending events. 0 if not registered.

        Args:
            subscription_id: The UUID of the subscription to check.

        Returns:
            The number of buffered events awaiting flush.
        """
        with self._lock:
            buf = self._buffers.get(subscription_id)
            if buf is None:
                return 0
            return len(buf)

    def is_ready_to_flush(self, subscription_id: str) -> bool:
        """True if frequency seconds elapsed since last flush AND buffer non-empty.

        Args:
            subscription_id: The UUID of the subscription to check.

        Returns:
            True if the subscription has pending events and its batch interval
            has elapsed since the last flush, False otherwise.
        """
        with self._lock:
            if subscription_id not in self._buffers:
                return False
            if not self._buffers[subscription_id]:
                return False
            elapsed = time.time() - self._last_flush[subscription_id]
            return elapsed >= self._frequencies[subscription_id]

    def flush(self, subscription_id: str) -> List[Dict[str, Any]]:
        """Return and clear all buffered events. Resets flush timer.

        Args:
            subscription_id: The UUID of the subscription to flush.

        Returns:
            List of all buffered event dicts. Empty list if subscription is
            not registered or has no pending events.
        """
        with self._lock:
            if subscription_id not in self._buffers:
                return []
            events = self._buffers[subscription_id]
            self._buffers[subscription_id] = []
            self._last_flush[subscription_id] = time.time()
            if events:
                logger.info(
                    "Flushed %d events for subscription %s",
                    len(events),
                    subscription_id,
                )
            return events

    def get_ready_subscriptions(self) -> List[str]:
        """Return subscription IDs whose batch interval has elapsed and have events.

        Returns:
            List of subscription UUIDs that are ready for flush.
        """
        now = time.time()
        ready = []
        with self._lock:
            for sub_id, buf in self._buffers.items():
                if not buf:
                    continue
                elapsed = now - self._last_flush[sub_id]
                if elapsed >= self._frequencies[sub_id]:
                    ready.append(sub_id)
        return ready
