"""
Mock tests for node failure event storage in the buildwatcher pipeline.

Simulates K8s failure events (NCCL errors, pod evictions, volume mount failures)
and verifies they flow correctly through the retry pipeline to persistent storage.

Tests three layers:
1. Strategy detection — each strategy correctly identifies failures and nodes
2. Tracker → Storage — record_failure() persists to PostgreSQL storage
3. RetryHandler → Storage — full pipeline from simulated event to persistent storage
"""

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from gbserver_test.test_utils import AbstractSingletonStorageUsingTest

from gbserver.resilience.node_health_tracker import NodeHealthTracker
from gbserver.resilience.retry_handler import RetryHandler
from gbserver.resilience.strategies import (
    NCCLErrorRetryStrategy,
    PodEvictionRetryStrategy,
    UnhealthyInsufficientPodsRetryStrategy,
)
from gbserver.storage.sql.storage_factory import SQLStorageFactory
from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventMessagePayload,
    BuildEventType,
    EntityRunMetadata,
)

# ── BuildEvent factory helpers ──────────────────────────────────────


def _make_run_metadata(build_id: str = "test-build-001") -> EntityRunMetadata:
    return EntityRunMetadata(build_id=build_id, target_name="test-target")


def _wrap_json_in_markdown(data: dict) -> str:
    """Wrap a dict as a ```json markdown code block, matching monitor output format."""
    return f"```json\n{json.dumps(data)}\n```"


def make_unhealthy_mount_event(
    node_name: str = "worker-node-1",
    pod_name: str = "test-pod-1",
    build_id: str = "test-build-001",
) -> BuildEvent:
    """Create a BuildEvent matching UnhealthyInsufficientPodsRetryStrategy."""
    msg_data = {
        "state": "Unhealthy",
        "previous_state": "Running",
        "events": [
            {
                "object_type": "AppWrapper",
                "reason": "Unhealthy",
                "message": "InsufficientPodsReady: 0/1 pods are ready",
                "object_name": "test-appwrapper",
            },
            {
                "object_type": "Pod",
                "reason": "FailedMount",
                "object_name": pod_name,
                "message": "Unable to attach or mount volumes",
            },
        ],
        "pod_placement": {pod_name: node_name},
    }
    return BuildEvent(
        run_metadata=_make_run_metadata(build_id),
        type=BuildEventType.MESSAGE_EVENT,
        payload=BuildEventMessagePayload(msg=_wrap_json_in_markdown(msg_data)),
    )


def make_pod_eviction_event(
    node_name: str = "worker-node-2",
    pod_name: str = "test-pod-2",
    build_id: str = "test-build-002",
) -> BuildEvent:
    """Create a BuildEvent matching PodEvictionRetryStrategy."""
    msg_data = {
        "state": "Failed",
        "previous_state": "Running",
        "events": [
            {
                "object_type": "AppWrapper",
                "reason": "Unhealthy",
                "message": "Pod evicted due to resource pressure",
                "object_name": "test-appwrapper",
            },
            {
                "object_type": "Pod",
                "reason": "Evicted",
                "object_name": pod_name,
                "message": "The node was low on resource: memory",
            },
        ],
        "pod_placement": {pod_name: node_name},
    }
    return BuildEvent(
        run_metadata=_make_run_metadata(build_id),
        type=BuildEventType.MESSAGE_EVENT,
        payload=BuildEventMessagePayload(msg=_wrap_json_in_markdown(msg_data)),
    )


def make_nccl_error_event(
    node_name: str = "gpu-node-1",
    build_id: str = "test-build-003",
) -> BuildEvent:
    """Create a BuildEvent matching NCCLErrorRetryStrategy with node in JSON payload."""
    msg_data = {
        "node_name": node_name,
        "error": "RuntimeError: NCCL Error 3: internal error",
    }
    # NCCL strategy checks msg directly for patterns, then parses JSON for node extraction.
    msg = f"RuntimeError: NCCL Error 3: internal error\n{_wrap_json_in_markdown(msg_data)}"
    return BuildEvent(
        run_metadata=_make_run_metadata(build_id),
        type=BuildEventType.MESSAGE_EVENT,
        payload=BuildEventMessagePayload(msg=msg),
    )


def make_cuda_error_event(
    node_name: str = "gpu-node-2",
    build_id: str = "test-build-004",
) -> BuildEvent:
    """Create a BuildEvent for CUDA illegal memory access."""
    msg_data = {"node_name": node_name}
    msg = f"RuntimeError: CUDA error: an illegal memory access was encountered\n{_wrap_json_in_markdown(msg_data)}"
    return BuildEvent(
        run_metadata=_make_run_metadata(build_id),
        type=BuildEventType.MESSAGE_EVENT,
        payload=BuildEventMessagePayload(msg=msg),
    )


def make_non_retryable_event(build_id: str = "test-build-000") -> BuildEvent:
    """Create a BuildEvent that should NOT trigger any retry strategy."""
    msg_data = {
        "state": "Running",
        "events": [
            {
                "object_type": "AppWrapper",
                "reason": "Running",
                "message": "All pods are ready",
            }
        ],
    }
    return BuildEvent(
        run_metadata=_make_run_metadata(build_id),
        type=BuildEventType.MESSAGE_EVENT,
        payload=BuildEventMessagePayload(msg=_wrap_json_in_markdown(msg_data)),
    )


# ── Layer 1: Strategy Detection Tests ───────────────────────────────


class TestUnhealthyInsufficientPodsStrategy:
    """Tests for UnhealthyInsufficientPodsRetryStrategy detection and node extraction."""

    def test_should_retry_on_failed_mount(self):
        strategy = UnhealthyInsufficientPodsRetryStrategy(object_types=["AppWrapper"])
        event = make_unhealthy_mount_event()
        assert strategy.should_retry(event) is True

    def test_extract_nodes_on_failed_mount(self):
        strategy = UnhealthyInsufficientPodsRetryStrategy(object_types=["AppWrapper"])
        event = make_unhealthy_mount_event(node_name="bad-node-1", pod_name="pod-abc")
        nodes = strategy.extract_nodes_to_avoid(event)
        assert nodes == {"bad-node-1"}

    def test_should_not_retry_on_non_matching_event(self):
        strategy = UnhealthyInsufficientPodsRetryStrategy(object_types=["AppWrapper"])
        event = make_non_retryable_event()
        assert strategy.should_retry(event) is False


class TestPodEvictionStrategy:
    """Tests for PodEvictionRetryStrategy detection and node extraction."""

    def test_should_retry_on_eviction(self):
        strategy = PodEvictionRetryStrategy(
            object_types=["AppWrapper"], avoid_eviction_nodes=True
        )
        event = make_pod_eviction_event()
        assert strategy.should_retry(event) is True

    def test_extract_nodes_when_avoid_enabled(self):
        strategy = PodEvictionRetryStrategy(
            object_types=["AppWrapper"], avoid_eviction_nodes=True
        )
        event = make_pod_eviction_event(node_name="evict-node-1", pod_name="pod-xyz")
        nodes = strategy.extract_nodes_to_avoid(event)
        assert nodes == {"evict-node-1"}

    def test_extract_no_nodes_when_avoid_disabled(self):
        strategy = PodEvictionRetryStrategy(
            object_types=["AppWrapper"], avoid_eviction_nodes=False
        )
        event = make_pod_eviction_event(node_name="evict-node-1")
        nodes = strategy.extract_nodes_to_avoid(event)
        assert nodes == set()

    def test_should_not_retry_on_non_matching_event(self):
        strategy = PodEvictionRetryStrategy(object_types=["AppWrapper"])
        event = make_non_retryable_event()
        assert strategy.should_retry(event) is False


class TestNCCLErrorStrategy:
    """Tests for NCCLErrorRetryStrategy detection and node extraction."""

    def test_should_retry_on_nccl_error(self):
        strategy = NCCLErrorRetryStrategy()
        event = make_nccl_error_event()
        assert strategy.should_retry(event) is True

    def test_should_retry_on_cuda_error(self):
        strategy = NCCLErrorRetryStrategy()
        event = make_cuda_error_event()
        assert strategy.should_retry(event) is True

    def test_extract_node_from_json_payload(self):
        strategy = NCCLErrorRetryStrategy()
        event = make_nccl_error_event(node_name="gpu-bad-1")
        nodes = strategy.extract_nodes_to_avoid(event)
        assert nodes == {"gpu-bad-1"}

    def test_should_not_retry_on_non_matching_event(self):
        strategy = NCCLErrorRetryStrategy()
        event = make_non_retryable_event()
        assert strategy.should_retry(event) is False


# ── Layer 2: Tracker → Storage Tests ────────────────────────────────


@pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Requires SQL database access",
)
@pytest.mark.asyncio
class TestTrackerStorage(AbstractSingletonStorageUsingTest):
    """Tests that NodeHealthTracker.record_failure() persists correctly to PostgreSQL storage."""

    @classmethod
    def _get_storage_factory(cls):
        return SQLStorageFactory()

    @pytest_asyncio.fixture(autouse=True)
    async def _tracker(self):
        """Create a NodeHealthTracker wired to PostgreSQL storage."""
        storage = self.storage.node_failure_storage
        self._tracker_instance = NodeHealthTracker(
            metrics_client=MagicMock(),
            alert_threshold=5,
            alert_window_minutes=30,
            node_failure_storage=storage,
        )
        await self._tracker_instance.start()
        yield
        await self._tracker_instance.stop()

    async def test_record_unhealthy_mount_failure(self):
        await self._tracker_instance.record_failure(
            node_name="worker-node-1",
            build_id="build-100",
            launch_id="launch-100",
            failure_type="UnhealthyInsufficientPodsRetryStrategy",
            retry_count=0,
            metadata={
                "strategy": "UnhealthyInsufficientPodsRetryStrategy",
                "event_type": "message_event",
            },
            namespace="test-ns",
            cluster="test-cluster",
        )

        results = self.storage.node_failure_storage.get_by_where(
            {"node_name": "worker-node-1"}
        )
        assert len(results) == 1
        stored = results[0]
        assert stored.node_name == "worker-node-1"
        assert stored.build_id == "build-100"
        assert stored.launch_id == "launch-100"
        assert stored.failure_type == "UnhealthyInsufficientPodsRetryStrategy"
        assert stored.retry_count == 0
        assert stored.metadata["strategy"] == "UnhealthyInsufficientPodsRetryStrategy"
        assert stored.metadata["namespace"] == "test-ns"
        assert stored.metadata["cluster"] == "test-cluster"
        assert stored.resolved is False

    async def test_record_pod_eviction_failure(self):
        await self._tracker_instance.record_failure(
            node_name="worker-node-2",
            build_id="build-200",
            launch_id="launch-200",
            failure_type="PodEvictionRetryStrategy",
            retry_count=1,
            metadata={
                "strategy": "PodEvictionRetryStrategy",
                "event_type": "message_event",
            },
        )

        results = self.storage.node_failure_storage.get_by_where(
            {"node_name": "worker-node-2"}
        )
        assert len(results) == 1
        stored = results[0]
        assert stored.failure_type == "PodEvictionRetryStrategy"
        assert stored.retry_count == 1

    async def test_record_nccl_error_failure(self):
        await self._tracker_instance.record_failure(
            node_name="gpu-node-1",
            build_id="build-300",
            launch_id="launch-300",
            failure_type="NCCLErrorRetryStrategy",
            retry_count=0,
            metadata={
                "strategy": "NCCLErrorRetryStrategy",
                "event_type": "message_event",
            },
        )

        results = self.storage.node_failure_storage.get_by_where(
            {"node_name": "gpu-node-1"}
        )
        assert len(results) == 1
        assert results[0].failure_type == "NCCLErrorRetryStrategy"

    async def test_multi_failure_same_node(self):
        for i in range(3):
            await self._tracker_instance.record_failure(
                node_name="flaky-node",
                build_id=f"build-{400 + i}",
                launch_id=f"launch-{400 + i}",
                failure_type="UnhealthyInsufficientPodsRetryStrategy",
                retry_count=i,
            )

        results = self.storage.node_failure_storage.get_by_where(
            {"node_name": "flaky-node"}
        )
        assert len(results) == 3
        build_ids = {r.build_id for r in results}
        assert build_ids == {"build-400", "build-401", "build-402"}


# ── Layer 3: RetryHandler → Storage Integration Tests ───────────────


@pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Requires SQL database access",
)
@pytest.mark.asyncio
class TestRetryHandlerIntegration(AbstractSingletonStorageUsingTest):
    """Tests that RetryHandler correctly flows simulated K8s failure events to storage."""

    @classmethod
    def _get_storage_factory(cls):
        return SQLStorageFactory()

    @pytest_asyncio.fixture(autouse=True)
    async def _tracker(self):
        """Create a NodeHealthTracker wired to PostgreSQL storage."""
        storage = self.storage.node_failure_storage
        self._tracker_instance = NodeHealthTracker(
            metrics_client=MagicMock(),
            alert_threshold=5,
            alert_window_minutes=30,
            node_failure_storage=storage,
        )
        await self._tracker_instance.start()
        yield
        await self._tracker_instance.stop()

    def _make_mock_env(self, namespace="prod-ns", cluster="prod-cluster"):
        mock_env = MagicMock()
        mock_env.retry_workload = AsyncMock()
        mock_env.namespace = namespace
        mock_env.kube_context = cluster
        return mock_env

    async def test_unhealthy_mount_event_persisted(self):
        mock_env = self._make_mock_env(namespace="prod-ns", cluster="prod-cluster")
        handler = RetryHandler(
            launch_id="launch-rh-1",
            downstream_queue=asyncio.Queue(),
            environment=mock_env,
            max_retries=3,
            strategies=[
                UnhealthyInsufficientPodsRetryStrategy(object_types=["AppWrapper"])
            ],
            node_health_tracker=self._tracker_instance,
            build_id="build-rh-1",
        )

        event = make_unhealthy_mount_event(
            node_name="bad-mount-node", pod_name="pod-rh-1", build_id="build-rh-1"
        )
        retry_triggered = await handler._evaluate_and_retry(event)

        assert retry_triggered is True
        mock_env.retry_workload.assert_awaited_once()

        results = self.storage.node_failure_storage.get_by_where(
            {"node_name": "bad-mount-node"}
        )
        assert len(results) == 1
        stored = results[0]
        assert stored.build_id == "build-rh-1"
        assert stored.launch_id == "launch-rh-1"
        assert stored.failure_type == "UnhealthyInsufficientPodsRetryStrategy"
        assert stored.metadata["namespace"] == "prod-ns"
        assert stored.metadata["cluster"] == "prod-cluster"

    async def test_pod_eviction_event_persisted(self):
        mock_env = self._make_mock_env(
            namespace="staging-ns", cluster="staging-cluster"
        )
        handler = RetryHandler(
            launch_id="launch-rh-2",
            downstream_queue=asyncio.Queue(),
            environment=mock_env,
            max_retries=3,
            strategies=[
                PodEvictionRetryStrategy(
                    object_types=["AppWrapper"], avoid_eviction_nodes=True
                )
            ],
            node_health_tracker=self._tracker_instance,
            build_id="build-rh-2",
        )

        event = make_pod_eviction_event(
            node_name="evict-node", pod_name="pod-rh-2", build_id="build-rh-2"
        )
        retry_triggered = await handler._evaluate_and_retry(event)

        assert retry_triggered is True

        results = self.storage.node_failure_storage.get_by_where(
            {"node_name": "evict-node"}
        )
        assert len(results) == 1
        assert results[0].failure_type == "PodEvictionRetryStrategy"

    async def test_nccl_error_event_persisted(self):
        mock_env = self._make_mock_env(namespace="gpu-ns", cluster="gpu-cluster")
        handler = RetryHandler(
            launch_id="launch-rh-3",
            downstream_queue=asyncio.Queue(),
            environment=mock_env,
            max_retries=3,
            strategies=[NCCLErrorRetryStrategy()],
            node_health_tracker=self._tracker_instance,
            build_id="build-rh-3",
        )

        event = make_nccl_error_event(node_name="gpu-bad-node", build_id="build-rh-3")
        retry_triggered = await handler._evaluate_and_retry(event)

        assert retry_triggered is True

        results = self.storage.node_failure_storage.get_by_where(
            {"node_name": "gpu-bad-node"}
        )
        assert len(results) == 1
        stored = results[0]
        assert stored.failure_type == "NCCLErrorRetryStrategy"
        assert stored.metadata["namespace"] == "gpu-ns"

    async def test_non_retryable_event_not_persisted(self):
        mock_env = self._make_mock_env(namespace="test-ns", cluster="test-cluster")
        handler = RetryHandler(
            launch_id="launch-rh-4",
            downstream_queue=asyncio.Queue(),
            environment=mock_env,
            max_retries=3,
            node_health_tracker=self._tracker_instance,
            build_id="build-rh-4",
        )

        event = make_non_retryable_event(build_id="build-rh-4")
        retry_triggered = await handler._evaluate_and_retry(event)

        assert retry_triggered is False
        mock_env.retry_workload.assert_not_awaited()

        # Nothing should be persisted
        results = self.storage.node_failure_storage.get_by_where({})
        assert len(results) == 0
