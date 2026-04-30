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
Node health monitoring and alerting for tracking node failure patterns.

This module is responsible for recording node failures to persistent storage
and firing alerts when failure thresholds are exceeded. All data access and
querying is handled by the storage layer (INodeFailureStorage).
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Self, Set

from gbserver.metrics.metrics_client import MetricsClient
from gbserver.resilience.alert_handlers import AlertHandler, NodeHealthAlert
from gbserver.storage.node_failure_storage import INodeFailureStorage
from gbserver.storage.stored_node_failure import StoredNodeFailure
from gbserver.utils.atomic import AtomicInteger
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """
    Simple rate limiter using sliding window algorithm.

    Parameters
    ----------
    max_events : int
        Maximum number of events allowed
    window_seconds : int
        Time window in seconds
    """

    def __init__(self: Self, max_events: int, window_seconds: int) -> None:
        self.max_events = max_events
        self.window_seconds = window_seconds
        self._events: List[datetime] = []
        self._lock = asyncio.Lock()

    async def allow(self: Self) -> bool:
        """Check if an event is allowed under the rate limit."""
        async with self._lock:
            now = datetime.now(timezone.utc)
            cutoff = now - timedelta(seconds=self.window_seconds)

            # Remove old events
            self._events = [ts for ts in self._events if ts > cutoff]

            # Check if we're under the limit
            if len(self._events) < self.max_events:
                self._events.append(now)
                return True

            return False

    async def get_remaining(self: Self) -> int:
        """Get number of remaining events allowed."""
        async with self._lock:
            now = datetime.now(timezone.utc)
            cutoff = now - timedelta(seconds=self.window_seconds)
            self._events = [ts for ts in self._events if ts > cutoff]
            return max(0, self.max_events - len(self._events))


class NodeHealthTracker:
    """
    Monitors node health and fires alerts when failure thresholds are exceeded.

    This class persists failure events to storage and checks thresholds to
    trigger alerts. All data querying is delegated to INodeFailureStorage.

    Parameters
    ----------
    node_failure_storage : INodeFailureStorage
        Persistent storage for node failure data (required)
    metrics_client : Optional[MetricsClient]
        Client for sending metrics to monitoring backend
    alert_threshold : int
        Number of failures within alert_window to trigger an alert
    alert_window_minutes : int
        Time window in minutes for counting failures
    alert_handler : Optional[AlertHandler]
        Handler for sending alerts when threshold is exceeded
    """

    def __init__(
        self: Self,
        node_failure_storage: INodeFailureStorage,
        metrics_client: Optional[MetricsClient] = None,
        alert_threshold: int = 5,
        alert_window_minutes: int = 30,
        alert_handler: Optional[AlertHandler] = None,
    ) -> None:
        if alert_threshold < 1:
            raise ValueError(f"alert_threshold must be >= 1, got {alert_threshold}")
        if alert_window_minutes <= 0:
            raise ValueError(f"alert_window_minutes must be > 0, got {alert_window_minutes}")

        self.node_failure_storage = node_failure_storage
        self.metrics_client = metrics_client
        self.alert_threshold = alert_threshold
        self.alert_window_minutes = alert_window_minutes
        self.alert_handler = alert_handler

        # Track which nodes have already triggered alerts (per-process, ephemeral)
        self._alerted_nodes: Set[str] = set()

        self._running = False

        # Self-monitoring metrics (thread-safe)
        self._failures_recorded_count = AtomicInteger()
        self._alerts_sent_count = AtomicInteger()
        self._alerts_failed_count = AtomicInteger()

        # Rate limiter for alerts (max 10 alerts per minute across all nodes)
        self._alert_rate_limiter = RateLimiter(max_events=10, window_seconds=60)

        # Track pending alert tasks for proper cleanup and testability
        self._pending_alert_tasks: Set[asyncio.Task] = set()

    @property
    def is_running(self: Self) -> bool:
        """Check if the tracker is currently running."""
        return self._running

    def get_metrics(self: Self) -> Dict[str, int]:
        """
        Get self-monitoring metrics for the tracker.

        Returns:
            Dict with metrics:
            - failures_recorded: Total number of failures recorded
            - alerts_sent: Total number of alerts successfully sent
            - alerts_failed: Total number of alerts that failed to send
        """
        return {
            "failures_recorded": self._failures_recorded_count.get(),
            "alerts_sent": self._alerts_sent_count.get(),
            "alerts_failed": self._alerts_failed_count.get(),
        }

    async def start(self: Self) -> None:
        """Mark the tracker as running."""
        if self._running:
            return
        self._running = True
        logger.info("[NodeHealthTracker] Started")

    async def stop(self: Self) -> None:
        """Stop the tracker and wait for pending alerts."""
        self._running = False
        await self.flush_pending_alerts()
        logger.info("[NodeHealthTracker] Stopped")

    async def flush_pending_alerts(self: Self) -> None:
        """
        Wait for all pending alert tasks to complete.

        Useful for testing to ensure alerts have been sent before making
        assertions, and for graceful shutdown.
        """
        if not self._pending_alert_tasks:
            return

        pending_count = len(self._pending_alert_tasks)
        logger.debug(
            "[NodeHealthTracker] Flushing %d pending alert tasks",
            pending_count,
        )

        await asyncio.gather(*self._pending_alert_tasks, return_exceptions=True)

        logger.debug(
            "[NodeHealthTracker] All %d alert tasks completed",
            pending_count,
        )

    async def _send_alert_with_rate_limit(self: Self, alert: NodeHealthAlert) -> None:
        """Send alert with rate limiting."""
        if not await self._alert_rate_limiter.allow():
            remaining = await self._alert_rate_limiter.get_remaining()
            logger.warning(
                "[NodeHealthTracker] Alert rate limit exceeded for node %s. "
                "Remaining capacity: %d. Alert will be dropped.",
                alert.node_name,
                remaining,
            )
            return

        await self._send_alert_async(alert)

    async def _send_alert_async(self: Self, alert: NodeHealthAlert) -> None:
        """Send alert asynchronously via configured handler."""
        if not self.alert_handler:
            return

        try:
            success = await self.alert_handler.send_alert(alert)
            if success:
                logger.info(
                    "[NodeHealthTracker] Alert sent for node %s",
                    alert.node_name,
                )
                self._alerts_sent_count.fetch_and_add()
            else:
                logger.warning(
                    "[NodeHealthTracker] Alert handler returned failure for node %s",
                    alert.node_name,
                )
                self._alerts_failed_count.fetch_and_add()
        except Exception as e:
            logger.error(
                "[NodeHealthTracker] Failed to send alert for node %s: %s",
                alert.node_name,
                e,
            )
            self._alerts_failed_count.fetch_and_add()

    @staticmethod
    def _stored_failure_to_dict(stored: StoredNodeFailure) -> Dict:
        """Convert a StoredNodeFailure to a dict for alert payloads."""
        meta = stored.metadata or {}
        return {
            "node_name": stored.node_name,
            "build_id": stored.build_id,
            "launch_id": stored.launch_id,
            "failure_type": stored.failure_type,
            "timestamp": stored.created_time.isoformat(),
            "retry_count": stored.retry_count,
            "metadata": meta,
            "namespace": meta.get("namespace", ""),
            "cluster": meta.get("cluster", ""),
        }

    async def record_failure(
        self: Self,
        node_name: str,
        build_id: str,
        launch_id: str,
        failure_type: str,
        retry_count: int = 0,
        metadata: Optional[Dict] = None,
        namespace: str = "",
        cluster: str = "",
    ) -> bool:
        """
        Record a node failure event.

        Persists the failure to storage, checks the alert threshold, and
        fires an alert if the threshold is exceeded.

        Args:
            node_name: Name of the Kubernetes node that failed
            build_id: ID of the build
            launch_id: ID of the launch/step
            failure_type: Type of failure (e.g., "FailedMount")
            retry_count: Which retry attempt this was
            metadata: Additional context
            namespace: Kubernetes namespace where the failure occurred
            cluster: Kubernetes cluster context where the failure occurred

        Returns:
            bool: True if this failure triggered an alert threshold
        """
        # Include namespace/cluster in metadata for storage
        storage_metadata = dict(metadata or {})
        if namespace:
            storage_metadata["namespace"] = namespace
        if cluster:
            storage_metadata["cluster"] = cluster

        stored_failure = StoredNodeFailure(
            node_name=node_name,
            build_id=build_id,
            launch_id=launch_id,
            failure_type=failure_type,
            retry_count=retry_count,
            metadata=storage_metadata,
        )

        # Persist to storage
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.node_failure_storage.add, stored_failure)
        except Exception as e:
            logger.error("[NodeHealthTracker] Failed to persist failure to storage: %s", e)
            return False

        self._failures_recorded_count.fetch_and_add()

        # Query storage for threshold evaluation
        since = datetime.now(timezone.utc) - timedelta(minutes=self.alert_window_minutes)
        try:
            recent_failures = await loop.run_in_executor(
                None,
                self.node_failure_storage.get_unresolved_failures_for_node_since,
                node_name,
                since,
            )
        except Exception as e:
            logger.error("[NodeHealthTracker] Failed to query failures for threshold: %s", e)
            return False

        failure_count = len(recent_failures)
        should_alert = (
            failure_count >= self.alert_threshold and node_name not in self._alerted_nodes
        )

        if should_alert:
            self._alerted_nodes.add(node_name)
            logger.warning(
                "[NodeHealthTracker] Alert threshold reached for node %s: "
                "%d failures in %d minutes",
                node_name,
                failure_count,
                self.alert_window_minutes,
            )

        logger.info(
            "[NodeHealthTracker] Recorded failure for node %s "
            "(build_id=%s, launch_id=%s, type=%s, retry=%d, "
            "recent_failures=%d/%d)",
            node_name,
            build_id,
            launch_id,
            failure_type,
            retry_count,
            failure_count,
            self.alert_threshold,
        )

        # Send alert outside of any lock
        if should_alert and self.alert_handler:
            try:
                alert = NodeHealthAlert(
                    node_name=node_name,
                    failure_count=failure_count,
                    threshold=self.alert_threshold,
                    window_minutes=self.alert_window_minutes,
                    failures=[self._stored_failure_to_dict(f) for f in recent_failures[-5:]],
                    namespace=namespace,
                    cluster=cluster,
                )
                task = asyncio.create_task(self._send_alert_with_rate_limit(alert))
                self._pending_alert_tasks.add(task)
                task.add_done_callback(self._pending_alert_tasks.discard)
            except Exception as e:
                logger.error("[NodeHealthTracker] Failed to create alert: %s", e)

        return should_alert

    async def reset_alert_status(self: Self, node_name: str) -> None:
        """
        Reset alert status for a node.

        Allows re-alerting if failures continue after acknowledgment.
        This is per-process ephemeral state.

        Args:
            node_name: Name of the node
        """
        if node_name in self._alerted_nodes:
            self._alerted_nodes.remove(node_name)
            logger.info(
                "[NodeHealthTracker] Reset alert status for node %s",
                node_name,
            )
