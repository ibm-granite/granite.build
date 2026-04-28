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
Unit tests for UnhealthyInsufficientPodsRetryStrategy.
"""

import json
from typing import Self

import pytest

from gbserver.resilience.strategies import UnhealthyInsufficientPodsRetryStrategy
from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventType,
    EntityRunMetadata,
    EventPayload,
)


def create_unhealthy_event(
    object_type: str = "AppWrapper",
    include_failed_mount: bool = True,
    node_name: str = "worker-node-1",
    pod_name: str = "test-pod-1",
) -> BuildEvent:
    """
    Create a BuildEvent that simulates Unhealthy + InsufficientPodsReady.

    Args:
        object_type: The K8s object type (AppWrapper, Job, etc.)
        include_failed_mount: Whether to include FailedMount pod events
        node_name: The node where the pod failed
        pod_name: The name of the failed pod
    """
    events = [
        {
            "object_type": object_type,
            "object_name": "test-appwrapper",
            "reason": "Unhealthy",
            "message": "InsufficientPodsReady: 0/1 pods are ready",
            "timestamp": "2026-01-22T10:00:00Z",
        }
    ]

    pod_placement = {}

    if include_failed_mount:
        events.append(
            {
                "object_type": "Pod",
                "object_name": pod_name,
                "reason": "FailedMount",
                "message": "Unable to attach or mount volumes",
                "timestamp": "2026-01-22T10:00:01Z",
            }
        )
        pod_placement[pod_name] = node_name

    data = {
        "events": events,
        "state": "Unhealthy",
        "pod_placement": pod_placement,
    }

    msg = f"```json\n{json.dumps(data, indent=2)}\n```"

    payload = EventPayload.payload_parser(
        event_type=BuildEventType.MESSAGE_EVENT,
        data={"msg": msg},
    )
    return BuildEvent(
        run_metadata=EntityRunMetadata(build_id="test-build-id"),
        type=BuildEventType.MESSAGE_EVENT,
        payload=payload,
    )


def create_normal_event() -> BuildEvent:
    """Create a normal BuildEvent that should not trigger retry."""
    payload = EventPayload.payload_parser(
        event_type=BuildEventType.MESSAGE_EVENT,
        data={"msg": "Normal status update"},
    )
    return BuildEvent(
        run_metadata=EntityRunMetadata(build_id="test-build-id"),
        type=BuildEventType.MESSAGE_EVENT,
        payload=payload,
    )


class TestUnhealthyInsufficientPodsRetryStrategy:
    """Tests for UnhealthyInsufficientPodsRetryStrategy."""

    def test_should_retry_with_matching_conditions(self: Self) -> None:
        """Test that strategy detects matching failure conditions for AppWrapper."""
        strategy = UnhealthyInsufficientPodsRetryStrategy(object_types=["AppWrapper"])
        event = create_unhealthy_event(object_type="AppWrapper")

        result = strategy.should_retry(event=event)

        assert result is True

    def test_should_not_retry_wrong_object_type(self: Self) -> None:
        """Test that strategy ignores events from non-monitored object types."""
        strategy = UnhealthyInsufficientPodsRetryStrategy(object_types=["AppWrapper"])
        event = create_unhealthy_event(object_type="Job")

        result = strategy.should_retry(event=event)

        assert result is False

    def test_should_retry_with_multiple_object_types(self: Self) -> None:
        """Test that strategy works when configured for multiple object types."""
        strategy = UnhealthyInsufficientPodsRetryStrategy(
            object_types=["AppWrapper", "Job", "Deployment"]
        )

        # Test with Job
        event_job = create_unhealthy_event(object_type="Job")
        result_job = strategy.should_retry(event=event_job)
        assert result_job is True

        # Test with Deployment
        event_deployment = create_unhealthy_event(object_type="Deployment")
        result_deployment = strategy.should_retry(event=event_deployment)
        assert result_deployment is True

    def test_should_not_retry_normal_event(self: Self) -> None:
        """Test that strategy ignores normal status events."""
        strategy = UnhealthyInsufficientPodsRetryStrategy(object_types=["AppWrapper"])
        event = create_normal_event()

        result = strategy.should_retry(event=event)

        assert result is False

    def test_should_not_retry_without_insufficient_pods_ready(self: Self) -> None:
        """Test that strategy requires InsufficientPodsReady message."""
        events = [
            {
                "object_type": "AppWrapper",
                "object_name": "test-appwrapper",
                "reason": "Unhealthy",
                "message": "Some other error message",  # No InsufficientPodsReady
            }
        ]

        data = {"events": events, "state": "Unhealthy", "pod_placement": {}}
        msg = f"```json\n{json.dumps(data, indent=2)}\n```"
        payload = EventPayload.payload_parser(
            event_type=BuildEventType.MESSAGE_EVENT,
            data={"msg": msg},
        )
        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=payload,
        )

        strategy = UnhealthyInsufficientPodsRetryStrategy(object_types=["AppWrapper"])
        result = strategy.should_retry(event=event)

        assert result is False

    def test_extract_nodes_to_avoid_failed_mount(self: Self) -> None:
        """Test that strategy extracts nodes with FailedMount errors."""
        strategy = UnhealthyInsufficientPodsRetryStrategy(object_types=["AppWrapper"])
        event = create_unhealthy_event(
            object_type="AppWrapper",
            include_failed_mount=True,
            node_name="worker-node-1",
            pod_name="test-pod-1",
        )

        nodes = strategy.extract_nodes_to_avoid(event=event)

        assert "worker-node-1" in nodes
        assert len(nodes) == 1

    def test_extract_nodes_to_avoid_failed_attach_volume(self: Self) -> None:
        """Test that strategy extracts nodes with FailedAttachVolume errors."""
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
                "reason": "FailedAttachVolume",
                "message": "Multi-Attach error for volume",
            },
        ]

        pod_placement = {"test-pod-1": "worker-node-2"}

        data = {"events": events, "state": "Unhealthy", "pod_placement": pod_placement}
        msg = f"```json\n{json.dumps(data, indent=2)}\n```"
        payload = EventPayload.payload_parser(
            event_type=BuildEventType.MESSAGE_EVENT,
            data={"msg": msg},
        )
        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=payload,
        )

        strategy = UnhealthyInsufficientPodsRetryStrategy(object_types=["AppWrapper"])
        nodes = strategy.extract_nodes_to_avoid(event=event)

        assert "worker-node-2" in nodes
        assert len(nodes) == 1

    def test_extract_nodes_multiple_pod_failures(self: Self) -> None:
        """Test extracting multiple failed nodes from multiple pod failures."""
        events = [
            {
                "object_type": "AppWrapper",
                "object_name": "test-appwrapper",
                "reason": "Unhealthy",
                "message": "InsufficientPodsReady: 0/3 pods are ready",
            },
            {
                "object_type": "Pod",
                "object_name": "test-pod-1",
                "reason": "FailedMount",
                "message": "Unable to attach or mount volumes",
            },
            {
                "object_type": "Pod",
                "object_name": "test-pod-2",
                "reason": "FailedAttachVolume",
                "message": "Volume attach failed",
            },
            {
                "object_type": "Pod",
                "object_name": "test-pod-3",
                "reason": "FailedMount",
                "message": "MountVolume.SetUp failed",
            },
        ]

        pod_placement = {
            "test-pod-1": "worker-node-1",
            "test-pod-2": "worker-node-2",
            "test-pod-3": "worker-node-1",  # Same node as pod-1
        }

        data = {"events": events, "state": "Unhealthy", "pod_placement": pod_placement}
        msg = f"```json\n{json.dumps(data, indent=2)}\n```"
        payload = EventPayload.payload_parser(
            event_type=BuildEventType.MESSAGE_EVENT,
            data={"msg": msg},
        )
        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=payload,
        )

        strategy = UnhealthyInsufficientPodsRetryStrategy(object_types=["AppWrapper"])
        nodes = strategy.extract_nodes_to_avoid(event=event)

        assert "worker-node-1" in nodes
        assert "worker-node-2" in nodes
        assert len(nodes) == 2  # Deduplicated

    def test_extract_nodes_without_pod_placement_info(self: Self) -> None:
        """Test that strategy handles missing pod placement information gracefully."""
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
                "message": "Unable to attach or mount volumes",
            },
        ]

        # Missing pod_placement data
        data = {"events": events, "state": "Unhealthy", "pod_placement": {}}
        msg = f"```json\n{json.dumps(data, indent=2)}\n```"
        payload = EventPayload.payload_parser(
            event_type=BuildEventType.MESSAGE_EVENT,
            data={"msg": msg},
        )
        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=payload,
        )

        strategy = UnhealthyInsufficientPodsRetryStrategy(object_types=["AppWrapper"])
        nodes = strategy.extract_nodes_to_avoid(event=event)

        # Should return empty set when pod placement info is missing
        assert len(nodes) == 0

    def test_default_object_types(self: Self) -> None:
        """Test that strategy defaults to AppWrapper when no object_types specified."""
        strategy = UnhealthyInsufficientPodsRetryStrategy()

        assert strategy.object_types == ["AppWrapper"]

        # Verify it works with AppWrapper
        event = create_unhealthy_event(object_type="AppWrapper")
        result = strategy.should_retry(event=event)
        assert result is True

    def test_ignores_non_mount_related_pod_failures(self: Self) -> None:
        """Test that strategy only tracks mount-related failures."""
        events = [
            {
                "object_type": "AppWrapper",
                "object_name": "test-appwrapper",
                "reason": "Unhealthy",
                "message": "InsufficientPodsReady: 0/2 pods are ready",
            },
            {
                "object_type": "Pod",
                "object_name": "test-pod-1",
                "reason": "FailedMount",
                "message": "Mount issue",
            },
            {
                "object_type": "Pod",
                "object_name": "test-pod-2",
                "reason": "CrashLoopBackOff",  # Not a mount issue
                "message": "Container crashed",
            },
        ]

        pod_placement = {
            "test-pod-1": "worker-node-1",
            "test-pod-2": "worker-node-2",
        }

        data = {"events": events, "state": "Unhealthy", "pod_placement": pod_placement}
        msg = f"```json\n{json.dumps(data, indent=2)}\n```"
        payload = EventPayload.payload_parser(
            event_type=BuildEventType.MESSAGE_EVENT,
            data={"msg": msg},
        )
        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=payload,
        )

        strategy = UnhealthyInsufficientPodsRetryStrategy(object_types=["AppWrapper"])
        nodes = strategy.extract_nodes_to_avoid(event=event)

        # Should only include worker-node-1 (FailedMount), not worker-node-2 (CrashLoopBackOff)
        assert "worker-node-1" in nodes
        assert "worker-node-2" not in nodes
        assert len(nodes) == 1


def create_quota_exhaustion_event(
    message: str = "0/172 nodes are available: 105 Insufficient memory, 143 Insufficient cpu, 151 Insufficient nvidia.com/gpu",
) -> BuildEvent:
    """Create a BuildEvent simulating FailedScheduling due to quota exhaustion."""
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
            "reason": "FailedScheduling",
            "message": message,
        },
    ]

    data = {"events": events, "state": "Unhealthy", "pod_placement": {}}
    msg = f"```json\n{json.dumps(data, indent=2)}\n```"

    payload = EventPayload.payload_parser(
        event_type=BuildEventType.MESSAGE_EVENT,
        data={"msg": msg},
    )
    return BuildEvent(
        run_metadata=EntityRunMetadata(build_id="test-build-id"),
        type=BuildEventType.MESSAGE_EVENT,
        payload=payload,
    )


class TestQuotaExhaustionRetry:
    """Tests for quota exhaustion detection in UnhealthyInsufficientPodsRetryStrategy."""

    def test_should_retry_quota_exhaustion_gpu(self: Self) -> None:
        """Test detection of FailedScheduling with Insufficient nvidia.com/gpu."""
        strategy = UnhealthyInsufficientPodsRetryStrategy(object_types=["AppWrapper"])
        event = create_quota_exhaustion_event(
            "0/172 nodes are available: 151 Insufficient nvidia.com/gpu"
        )

        result = strategy.should_retry(event=event)

        assert result is True
        assert strategy._is_quota_exhaustion is True

    def test_should_retry_quota_exhaustion_cpu_memory(self: Self) -> None:
        """Test detection of FailedScheduling with Insufficient cpu and memory."""
        strategy = UnhealthyInsufficientPodsRetryStrategy(object_types=["AppWrapper"])
        event = create_quota_exhaustion_event(
            "0/172 nodes are available: 105 Insufficient memory, 143 Insufficient cpu"
        )

        result = strategy.should_retry(event=event)

        assert result is True
        assert strategy._is_quota_exhaustion is True

    def test_should_retry_quota_nodes_available(self: Self) -> None:
        """Test detection of FailedScheduling with 'nodes are available' pattern."""
        strategy = UnhealthyInsufficientPodsRetryStrategy(object_types=["AppWrapper"])
        event = create_quota_exhaustion_event(
            "0/50 nodes are available: 50 node(s) had untolerated taint"
        )

        result = strategy.should_retry(event=event)

        assert result is True
        assert strategy._is_quota_exhaustion is True

    def test_quota_exhaustion_no_nodes_to_avoid(self: Self) -> None:
        """Test that quota exhaustion returns empty nodes_to_avoid set."""
        strategy = UnhealthyInsufficientPodsRetryStrategy(object_types=["AppWrapper"])
        event = create_quota_exhaustion_event()

        # Trigger should_retry to set _is_quota_exhaustion flag
        strategy.should_retry(event=event)
        nodes = strategy.extract_nodes_to_avoid(event=event)

        assert len(nodes) == 0

    def test_quota_exhaustion_backoff_delay(self: Self) -> None:
        """Test exponential backoff delay for quota exhaustion."""
        strategy = UnhealthyInsufficientPodsRetryStrategy(object_types=["AppWrapper"])
        event = create_quota_exhaustion_event()

        # Trigger should_retry to set _is_quota_exhaustion flag
        strategy.should_retry(event=event)

        assert strategy.get_retry_delay(retry_count=0) == 30.0  # 30 * 2^0
        assert strategy.get_retry_delay(retry_count=1) == 60.0  # 30 * 2^1
        assert strategy.get_retry_delay(retry_count=2) == 120.0  # 30 * 2^2
        assert strategy.get_retry_delay(retry_count=3) == 240.0  # 30 * 2^3
        assert strategy.get_retry_delay(retry_count=4) == 300.0  # capped at 300

    def test_mount_failure_no_backoff(self: Self) -> None:
        """Test that mount failures (non-quota) have zero delay."""
        strategy = UnhealthyInsufficientPodsRetryStrategy(object_types=["AppWrapper"])
        event = create_unhealthy_event(object_type="AppWrapper")

        # Trigger should_retry for mount failure path
        strategy.should_retry(event=event)

        assert strategy._is_quota_exhaustion is False
        assert strategy.get_retry_delay(retry_count=0) == 0.0
        assert strategy.get_retry_delay(retry_count=1) == 0.0

    def test_default_get_retry_delay(self: Self) -> None:
        """Test that base RetryStrategy.get_retry_delay returns 0.0."""
        from gbserver.resilience.retry_handler import RetryStrategy

        # Use a concrete subclass to test the base implementation
        strategy = UnhealthyInsufficientPodsRetryStrategy(object_types=["AppWrapper"])
        # Without triggering should_retry, _is_quota_exhaustion is False
        assert strategy.get_retry_delay(retry_count=0) == 0.0

    def test_quota_flag_resets_between_calls(self: Self) -> None:
        """Test that _is_quota_exhaustion flag resets on each should_retry call."""
        strategy = UnhealthyInsufficientPodsRetryStrategy(object_types=["AppWrapper"])

        # First call: quota exhaustion
        quota_event = create_quota_exhaustion_event()
        strategy.should_retry(event=quota_event)
        assert strategy._is_quota_exhaustion is True

        # Second call: normal unhealthy event (no quota)
        normal_event = create_unhealthy_event(object_type="AppWrapper")
        strategy.should_retry(event=normal_event)
        assert strategy._is_quota_exhaustion is False

    def test_failed_scheduling_without_quota_patterns_no_retry(self: Self) -> None:
        """Test that FailedScheduling without quota patterns doesn't trigger retry."""
        events = [
            {
                "object_type": "Pod",
                "object_name": "test-pod-1",
                "reason": "FailedScheduling",
                "message": "pod has unbound immediate PersistentVolumeClaims",
            },
        ]

        data = {"events": events, "state": "Pending", "pod_placement": {}}
        msg = f"```json\n{json.dumps(data, indent=2)}\n```"
        payload = EventPayload.payload_parser(
            event_type=BuildEventType.MESSAGE_EVENT,
            data={"msg": msg},
        )
        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=payload,
        )

        strategy = UnhealthyInsufficientPodsRetryStrategy(object_types=["AppWrapper"])
        result = strategy.should_retry(event=event)

        assert result is False
        assert strategy._is_quota_exhaustion is False
