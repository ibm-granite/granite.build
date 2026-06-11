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
Tests for NodeHealthTracker.

The tracker is responsible for recording failures to storage and firing
alerts when thresholds are exceeded. Query methods are tested separately
in test_node_failure_storage.py.
"""

import asyncio
import json
from typing import Self
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from libgbtest.utils import AbstractSingletonStorageUsingTest

from gbserver.resilience.node_health_tracker import (
    NodeHealthTracker,
)
from gbserver.types.constants import is_standalone


class _NodeHealthTrackerTestBase(AbstractSingletonStorageUsingTest):
    """Common base for the NodeHealthTracker tests.

    Storage selection is left to the environment: when GB_ENVIRONMENT=STANDALONE
    the default factory is SQLite and no cloud credentials are needed, so the
    cloud-config gate is disabled here; in DEV/STAGING the SQL backend is used
    and the usual cloud config is required.  Only ``_is_cloud_config_required``
    is overridden — the default ``_get_storage_factory`` already picks SQLite vs
    SQL from GBSERVER_METADATA_STORAGE.  The leading underscore / non-``Test``
    name keeps pytest from collecting this base directly.
    """

    @classmethod
    def _is_cloud_config_required(cls) -> bool:
        return not is_standalone()


class TestNodeHealthTracker(_NodeHealthTrackerTestBase):
    """Tests for NodeHealthTracker class."""

    @pytest.fixture
    def mock_metrics_client(self: Self) -> MagicMock:
        """Create a mock metrics client."""
        client = MagicMock()
        client.increment = AsyncMock()
        return client

    @pytest_asyncio.fixture
    async def tracker(self: Self, mock_metrics_client: MagicMock) -> NodeHealthTracker:
        """Create a node health tracker instance with storage."""
        tracker = NodeHealthTracker(
            node_failure_storage=self.storage.node_failure_storage,
            metrics_client=mock_metrics_client,
            alert_threshold=3,
            alert_window_minutes=10,
        )
        await tracker.start()
        yield tracker
        await tracker.stop()

    @pytest.mark.asyncio
    async def test_record_single_failure(
        self: Self,
        tracker: NodeHealthTracker,
    ) -> None:
        """Test recording a single node failure."""
        should_alert = await tracker.record_failure(
            node_name="worker-node-1",
            build_id="build-123",
            launch_id="launch-456",
            failure_type="FailedMount",
        )

        assert not should_alert  # Below threshold
        metrics = tracker.get_metrics()
        assert metrics["failures_recorded"] == 1

    @pytest.mark.asyncio
    async def test_record_persists_to_storage(
        self: Self,
        tracker: NodeHealthTracker,
    ) -> None:
        """Test that record_failure persists to storage."""
        await tracker.record_failure(
            node_name="worker-node-1",
            build_id="build-123",
            launch_id="launch-456",
            failure_type="FailedMount",
            namespace="granite-build",
            cluster="test-cluster",
        )

        # Verify persisted to storage
        failures = self.storage.node_failure_storage.get_recent_failures(
            "worker-node-1", minutes=10
        )
        assert len(failures) == 1
        assert failures[0].node_name == "worker-node-1"
        assert failures[0].build_id == "build-123"
        assert failures[0].failure_type == "FailedMount"
        assert failures[0].metadata.get("namespace") == "granite-build"
        assert failures[0].metadata.get("cluster") == "test-cluster"

    @pytest.mark.asyncio
    async def test_alert_threshold(
        self: Self,
        tracker: NodeHealthTracker,
    ) -> None:
        """Test alert when threshold is reached."""
        # Record failures below threshold
        for i in range(2):
            should_alert = await tracker.record_failure(
                node_name="worker-node-1",
                build_id=f"build-{i}",
                launch_id=f"launch-{i}",
                failure_type="FailedMount",
            )
            assert not should_alert

        # Third failure should trigger alert
        should_alert = await tracker.record_failure(
            node_name="worker-node-1",
            build_id="build-3",
            launch_id="launch-3",
            failure_type="FailedMount",
        )
        assert should_alert

        # Fourth failure should NOT trigger alert (already alerted)
        should_alert = await tracker.record_failure(
            node_name="worker-node-1",
            build_id="build-4",
            launch_id="launch-4",
            failure_type="FailedMount",
        )
        assert not should_alert

    @pytest.mark.asyncio
    async def test_reset_alert_status(
        self: Self,
        tracker: NodeHealthTracker,
    ) -> None:
        """Test resetting alert status for a node."""
        # Trigger alert
        for i in range(3):
            await tracker.record_failure(
                node_name="worker-node-1",
                build_id=f"build-{i}",
                launch_id=f"launch-{i}",
                failure_type="FailedMount",
            )

        # Should not alert again (already alerted)
        should_alert = await tracker.record_failure(
            node_name="worker-node-1",
            build_id="build-4",
            launch_id="launch-4",
            failure_type="FailedMount",
        )
        assert not should_alert

        # Reset alert status
        await tracker.reset_alert_status("worker-node-1")

        # Should alert again after reset
        should_alert = await tracker.record_failure(
            node_name="worker-node-1",
            build_id="build-5",
            launch_id="launch-5",
            failure_type="FailedMount",
        )
        assert should_alert

    @pytest.mark.asyncio
    async def test_is_running_property(
        self: Self,
    ) -> None:
        """Test is_running property reflects tracker state."""
        tracker = NodeHealthTracker(
            node_failure_storage=self.storage.node_failure_storage,
        )

        assert tracker.is_running is False

        await tracker.start()
        assert tracker.is_running is True

        await tracker.stop()
        assert tracker.is_running is False

    def test_configuration_validation(self: Self) -> None:
        """Test that invalid configuration raises ValueError."""
        storage = self.storage.node_failure_storage

        with pytest.raises(ValueError, match="alert_threshold must be >= 1"):
            NodeHealthTracker(node_failure_storage=storage, alert_threshold=0)

        with pytest.raises(ValueError, match="alert_threshold must be >= 1"):
            NodeHealthTracker(node_failure_storage=storage, alert_threshold=-1)

        with pytest.raises(ValueError, match="alert_window_minutes must be > 0"):
            NodeHealthTracker(node_failure_storage=storage, alert_window_minutes=0)

        with pytest.raises(ValueError, match="alert_window_minutes must be > 0"):
            NodeHealthTracker(node_failure_storage=storage, alert_window_minutes=-5)

    @pytest.mark.asyncio
    async def test_alert_handler_integration(
        self: Self,
    ) -> None:
        """Test that record_failure invokes alert handler when threshold is reached."""
        mock_handler = AsyncMock()
        mock_handler.send_alert = AsyncMock(return_value=True)

        tracker = NodeHealthTracker(
            node_failure_storage=self.storage.node_failure_storage,
            alert_threshold=3,
            alert_window_minutes=10,
            alert_handler=mock_handler,
        )
        await tracker.start()

        try:
            # Record failures below threshold
            for i in range(2):
                await tracker.record_failure(
                    node_name="worker-node-1",
                    build_id=f"build-{i}",
                    launch_id=f"launch-{i}",
                    failure_type="UnhealthyInsufficientPods",
                )

            await tracker.flush_pending_alerts()
            assert mock_handler.send_alert.call_count == 0

            # Record third failure - should trigger alert
            await tracker.record_failure(
                node_name="worker-node-1",
                build_id="build-3",
                launch_id="launch-3",
                failure_type="UnhealthyInsufficientPods",
                namespace="granite-build",
                cluster="test-cluster",
            )

            await tracker.flush_pending_alerts()
            assert mock_handler.send_alert.call_count == 1

            # Verify alert data
            call_args = mock_handler.send_alert.call_args
            alert = call_args[0][0]
            assert alert.node_name == "worker-node-1"
            assert alert.failure_count == 3
            assert alert.threshold == 3
            assert alert.window_minutes == 10
            assert len(alert.failures) == 3
            assert alert.namespace == "granite-build"
            assert alert.cluster == "test-cluster"

            # Fourth failure should NOT trigger another alert
            await tracker.record_failure(
                node_name="worker-node-1",
                build_id="build-4",
                launch_id="launch-4",
                failure_type="UnhealthyInsufficientPods",
            )

            await tracker.flush_pending_alerts()
            assert mock_handler.send_alert.call_count == 1

        finally:
            await tracker.stop()

    @pytest.mark.asyncio
    async def test_rate_limiting(
        self: Self,
    ) -> None:
        """Test that alerts are rate limited."""
        mock_handler = AsyncMock()
        mock_handler.send_alert = AsyncMock(return_value=True)

        tracker = NodeHealthTracker(
            node_failure_storage=self.storage.node_failure_storage,
            alert_threshold=1,  # Alert on first failure
            alert_handler=mock_handler,
        )
        await tracker.start()

        try:
            # Try to trigger 15 alerts (rate limit is 10/minute)
            for i in range(15):
                node_name = f"worker-node-{i}"
                await tracker.record_failure(
                    node_name=node_name,
                    build_id=f"build-{i}",
                    launch_id=f"launch-{i}",
                    failure_type="FailedMount",
                )

            await tracker.flush_pending_alerts()

            # Should have sent at most 10 alerts (rate limit)
            assert mock_handler.send_alert.call_count <= 10

        finally:
            await tracker.stop()

    @pytest.mark.asyncio
    async def test_self_monitoring_metrics(
        self: Self,
    ) -> None:
        """Test that tracker reports self-monitoring metrics."""
        tracker = NodeHealthTracker(
            node_failure_storage=self.storage.node_failure_storage,
        )
        await tracker.start()

        try:
            await tracker.record_failure("node-1", "build-1", "launch-1", "FailedMount")
            await tracker.record_failure(
                "node-2", "build-2", "launch-2", "FailedAttachVolume"
            )

            metrics = tracker.get_metrics()
            assert metrics["failures_recorded"] == 2

        finally:
            await tracker.stop()

    @pytest.mark.asyncio
    async def test_retry_handler_integration(
        self: Self,
    ) -> None:
        """Test integration between RetryHandler and NodeHealthTracker."""
        from gbserver.resilience.retry_handler import RetryHandler
        from gbserver.resilience.strategies.unhealthy_insufficient_pods import (
            UnhealthyInsufficientPodsRetryStrategy,
        )
        from gbserver.types.buildevent import (
            BuildEvent,
            BuildEventMessagePayload,
            BuildEventType,
            EntityRunMetadata,
        )

        tracker = NodeHealthTracker(
            node_failure_storage=self.storage.node_failure_storage,
            alert_threshold=5,
            alert_window_minutes=10,
        )
        await tracker.start()

        # Create mock environment
        mock_env = MagicMock()
        mock_env.retry_with_node_anti_affinity = AsyncMock(return_value=True)
        mock_env.namespace = "granite-build"
        mock_env.kube_context = "test-cluster/api-server:6443/user"

        downstream_queue = asyncio.Queue()
        handler = RetryHandler(
            launch_id="integration-test-launch",
            build_id="integration-test-build",
            downstream_queue=downstream_queue,
            environment=mock_env,
            max_retries=3,
            strategies=[
                UnhealthyInsufficientPodsRetryStrategy(object_types=["AppWrapper"])
            ],
            node_health_tracker=tracker,
        )

        processor_task = asyncio.create_task(handler.process_events())

        try:
            events = [
                {
                    "object_type": "AppWrapper",
                    "object_name": "test-appwrapper",
                    "reason": "Unhealthy",
                    "message": "InsufficientPodsReady: 0/1 pods are ready",
                },
                {
                    "object_type": "Pod",
                    "object_name": "test-pod-1",
                    "reason": "FailedMount",
                    "message": "Unable to attach or mount volumes on node worker-node-xyz",
                },
            ]

            pod_placement = {"test-pod-1": "worker-node-xyz"}
            data = {
                "events": events,
                "state": "Unhealthy",
                "pod_placement": pod_placement,
            }
            msg = f"```json\n{json.dumps(data, indent=2)}\n```"

            event = BuildEvent(
                run_metadata=EntityRunMetadata(build_id="integration-test-build"),
                type=BuildEventType.MESSAGE_EVENT,
                payload=BuildEventMessagePayload(msg=msg),
            )

            wrapper_queue = handler.get_wrapper_queue()
            await wrapper_queue.put(event)

            # Wait for event to be processed
            for _ in range(100):
                stored = self.storage.node_failure_storage.get_recent_failures(
                    "worker-node-xyz", minutes=10
                )
                if len(stored) > 0:
                    break
                await asyncio.sleep(0.01)

            stored = self.storage.node_failure_storage.get_recent_failures(
                "worker-node-xyz", minutes=10
            )
            assert len(stored) == 1, "Node failure should be recorded in storage"

            failure = stored[0]
            assert failure.node_name == "worker-node-xyz"
            assert failure.build_id == "integration-test-build"
            assert failure.launch_id == "integration-test-launch"
            assert failure.failure_type == "UnhealthyInsufficientPodsRetryStrategy"
            assert failure.metadata.get("namespace") == "granite-build"
            assert (
                failure.metadata.get("cluster") == "test-cluster/api-server:6443/user"
            )

        finally:
            handler.stop()
            await processor_task
            await tracker.stop()


class TestNodeHealthTrackerSingleton(_NodeHealthTrackerTestBase):
    """Tests for the lazy singleton accessor in resilience/__init__.py."""

    @pytest.fixture(autouse=True)
    def _reset_singleton(self: Self):
        """Reset singleton state before and after each test."""
        from gbserver.resilience import reset_node_health_tracker

        reset_node_health_tracker()
        yield
        reset_node_health_tracker()

    def test_set_overrides_singleton(self: Self) -> None:
        """set_node_health_tracker() overrides the lazy singleton."""
        from gbserver.resilience import (
            get_node_health_tracker,
            set_node_health_tracker,
        )

        mock_tracker = MagicMock(spec=NodeHealthTracker)
        set_node_health_tracker(mock_tracker)
        assert get_node_health_tracker() is mock_tracker

    def test_get_returns_same_instance(self: Self) -> None:
        """get_node_health_tracker() returns the same instance on repeated calls."""
        from gbserver.resilience import (
            get_node_health_tracker,
            set_node_health_tracker,
        )

        mock_tracker = MagicMock(spec=NodeHealthTracker)
        set_node_health_tracker(mock_tracker)

        t1 = get_node_health_tracker()
        t2 = get_node_health_tracker()
        assert t1 is t2

    def test_reset_clears_singleton(self: Self) -> None:
        """reset_node_health_tracker() clears state so next get re-initializes."""
        from gbserver.resilience import (
            get_node_health_tracker,
            reset_node_health_tracker,
            set_node_health_tracker,
        )

        mock_tracker = MagicMock(spec=NodeHealthTracker)
        set_node_health_tracker(mock_tracker)
        assert get_node_health_tracker() is mock_tracker

        reset_node_health_tracker()
        assert get_node_health_tracker() is not mock_tracker

    def test_graceful_degradation_on_storage_failure(self: Self) -> None:
        """If storage is unavailable, get_node_health_tracker() returns None."""
        from gbserver.resilience import get_node_health_tracker

        with patch(
            "gbserver.storage.singleton_storage.get_admin_storage",
            side_effect=Exception("no storage"),
        ):
            result = get_node_health_tracker()
            assert result is None

    def test_does_not_retry_after_failure(self: Self) -> None:
        """After a failed init, subsequent calls return None without retrying."""
        from gbserver.resilience import get_node_health_tracker

        mock_get_storage = MagicMock(side_effect=Exception("no storage"))
        with patch(
            "gbserver.storage.singleton_storage.get_admin_storage",
            mock_get_storage,
        ):
            assert get_node_health_tracker() is None

        # Second call should not invoke get_admin_storage again
        mock_get_storage.reset_mock()
        assert get_node_health_tracker() is None
        mock_get_storage.assert_not_called()
