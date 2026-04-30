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
Unit tests for PodEvictionRetryStrategy.
"""

import json
from typing import Self

import pytest

from gbserver.resilience.strategies import PodEvictionRetryStrategy
from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventType,
    EntityRunMetadata,
    EventPayload,
)


def create_eviction_event(
    object_type: str = "AppWrapper",
    eviction_reason: str = "Preempted",
    state: str = "Failed",
    previous_state: str = "Running",
    node_name: str = "dmf-nnnqh-gpu-worker-3-n6r6x",
    pod_name: str = "gb8zqlwumz-master-0",
) -> BuildEvent:
    """
    Create a BuildEvent that simulates a pod eviction.

    Args:
        object_type: The K8s object type (AppWrapper, Job, etc.)
        eviction_reason: The eviction reason (Preempted, Evicted)
        state: Current state of the workload
        previous_state: Previous state of the workload
        node_name: The node where the pod was running
        pod_name: The name of the evicted pod
    """
    events = [
        {
            "object_type": object_type,
            "object_name": "gb8zqlwumz",
            "reason": "Unhealthy",
            "message": "FailedComponent: Found 1 failed components",
            "type": "Normal",
            "time": "2026-01-28 06:25:40+00:00",
        },
        {
            "object_type": object_type,
            "object_name": "gb8zqlwumz",
            "reason": "FinishedWorkload",
            "message": "Workload 'granite-build/appwrapper-gb8zqlwumz-fa6dd' is declared finished",
            "type": "Normal",
            "time": "2026-01-28 06:25:40+00:00",
        },
        {
            "object_type": "Pod",
            "object_name": pod_name,
            "reason": eviction_reason,
            "message": f"{eviction_reason} by a pod on node {node_name}",
            "type": "Normal",
            "time": "2026-01-28 06:25:07.090166+00:00",
        },
        {
            "object_type": "Pod",
            "object_name": pod_name,
            "reason": "Killing",
            "message": "Stopping container pytorch",
            "type": "Normal",
            "time": "2026-01-28 06:25:07+00:00",
        },
    ]

    pod_placement = {pod_name: node_name}

    data = {
        "appwrapper": "gb8zqlwumz",
        "state": state,
        "previous_state": previous_state,
        "current_resets": 0,
        "max_retries": "unlimited",
        "pod_placement": pod_placement,
        "failed_pods": {},
        "events": events,
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


class TestPodEvictionRetryStrategy:
    """Tests for PodEvictionRetryStrategy."""

    def test_should_retry_with_preempted_pod(self: Self) -> None:
        """Test that strategy detects preempted pods."""
        strategy = PodEvictionRetryStrategy(object_types=["AppWrapper"])
        event = create_eviction_event(
            object_type="AppWrapper",
            eviction_reason="Preempted",
            state="Failed",
            previous_state="Running",
        )

        result = strategy.should_retry(event=event)

        assert result is True

    def test_should_retry_with_evicted_pod(self: Self) -> None:
        """Test that strategy detects evicted pods."""
        strategy = PodEvictionRetryStrategy(object_types=["AppWrapper"])
        event = create_eviction_event(
            object_type="AppWrapper",
            eviction_reason="Evicted",
            state="Failed",
            previous_state="Running",
        )

        result = strategy.should_retry(event=event)

        assert result is True

    def test_should_not_retry_wrong_object_type(self: Self) -> None:
        """Test that strategy ignores events from non-monitored object types."""
        strategy = PodEvictionRetryStrategy(object_types=["AppWrapper"])
        event = create_eviction_event(
            object_type="Job",
            eviction_reason="Preempted",
        )

        result = strategy.should_retry(event=event)

        assert result is False

    def test_should_retry_with_multiple_object_types(self: Self) -> None:
        """Test that strategy works when configured for multiple object types."""
        strategy = PodEvictionRetryStrategy(object_types=["AppWrapper", "Job", "Deployment"])

        # Test with Job
        event_job = create_eviction_event(
            object_type="Job",
            eviction_reason="Preempted",
        )
        result_job = strategy.should_retry(event=event_job)
        assert result_job is True

        # Test with Deployment
        event_deployment = create_eviction_event(
            object_type="Deployment",
            eviction_reason="Evicted",
        )
        result_deployment = strategy.should_retry(event=event_deployment)
        assert result_deployment is True

    def test_should_not_retry_normal_event(self: Self) -> None:
        """Test that strategy ignores normal status events."""
        strategy = PodEvictionRetryStrategy(object_types=["AppWrapper"])
        event = create_normal_event()

        result = strategy.should_retry(event=event)

        assert result is False

    def test_should_not_retry_without_eviction(self: Self) -> None:
        """Test that strategy requires eviction/preemption events."""
        events = [
            {
                "object_type": "AppWrapper",
                "object_name": "test-appwrapper",
                "reason": "Unhealthy",
                "message": "Some other error",
            },
            {
                "object_type": "Pod",
                "object_name": "test-pod-1",
                "reason": "Failed",  # Not Preempted or Evicted
                "message": "Container failed",
            },
        ]

        data = {
            "state": "Failed",
            "previous_state": "Running",
            "events": events,
            "pod_placement": {},
        }
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

        strategy = PodEvictionRetryStrategy(object_types=["AppWrapper"])
        result = strategy.should_retry(event=event)

        assert result is False

    def test_should_not_retry_if_not_previously_running(self: Self) -> None:
        """Test that strategy only retries workloads that were running."""
        strategy = PodEvictionRetryStrategy(object_types=["AppWrapper"])
        event = create_eviction_event(
            object_type="AppWrapper",
            eviction_reason="Preempted",
            state="Failed",
            previous_state="Pending",  # Was not running
        )

        result = strategy.should_retry(event=event)

        assert result is False

    def test_should_not_retry_if_not_failed(self: Self) -> None:
        """Test that strategy requires Failed state."""
        strategy = PodEvictionRetryStrategy(object_types=["AppWrapper"])
        event = create_eviction_event(
            object_type="AppWrapper",
            eviction_reason="Preempted",
            state="Running",  # Not Failed
            previous_state="Running",
        )

        result = strategy.should_retry(event=event)

        assert result is False

    def test_extract_nodes_default_no_avoidance(self: Self) -> None:
        """Test that by default, nodes are not avoided for evictions."""
        strategy = PodEvictionRetryStrategy(
            object_types=["AppWrapper"],
            avoid_eviction_nodes=False,  # Default
        )
        event = create_eviction_event(
            object_type="AppWrapper",
            eviction_reason="Preempted",
            node_name="worker-node-1",
        )

        nodes = strategy.extract_nodes_to_avoid(event=event)

        # Should not avoid any nodes by default
        assert len(nodes) == 0

    def test_extract_nodes_with_avoidance_enabled(self: Self) -> None:
        """Test that nodes can be extracted when avoidance is enabled."""
        strategy = PodEvictionRetryStrategy(
            object_types=["AppWrapper"],
            avoid_eviction_nodes=True,  # Enable node avoidance
        )
        event = create_eviction_event(
            object_type="AppWrapper",
            eviction_reason="Preempted",
            node_name="worker-node-1",
            pod_name="test-pod-1",
        )

        nodes = strategy.extract_nodes_to_avoid(event=event)

        assert "worker-node-1" in nodes
        assert len(nodes) == 1

    def test_extract_nodes_multiple_evictions(self: Self) -> None:
        """Test extracting nodes from multiple pod evictions."""
        events = [
            {
                "object_type": "AppWrapper",
                "object_name": "test-appwrapper",
                "reason": "Unhealthy",
                "message": "FailedComponent: Found 2 failed components",
            },
            {
                "object_type": "Pod",
                "object_name": "test-pod-1",
                "reason": "Preempted",
                "message": "Preempted by higher priority pod",
            },
            {
                "object_type": "Pod",
                "object_name": "test-pod-2",
                "reason": "Evicted",
                "message": "Node out of memory",
            },
            {
                "object_type": "Pod",
                "object_name": "test-pod-3",
                "reason": "Preempted",
                "message": "Preempted",
            },
        ]

        pod_placement = {
            "test-pod-1": "worker-node-1",
            "test-pod-2": "worker-node-2",
            "test-pod-3": "worker-node-1",  # Same node as pod-1
        }

        data = {
            "state": "Failed",
            "previous_state": "Running",
            "events": events,
            "pod_placement": pod_placement,
        }
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

        strategy = PodEvictionRetryStrategy(
            object_types=["AppWrapper"],
            avoid_eviction_nodes=True,
        )
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
                "message": "FailedComponent",
            },
            {
                "object_type": "Pod",
                "object_name": "test-pod-1",
                "reason": "Preempted",
                "message": "Preempted",
            },
        ]

        # Missing pod_placement data
        data = {
            "state": "Failed",
            "previous_state": "Running",
            "events": events,
            "pod_placement": {},
        }
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

        strategy = PodEvictionRetryStrategy(
            object_types=["AppWrapper"],
            avoid_eviction_nodes=True,
        )
        nodes = strategy.extract_nodes_to_avoid(event=event)

        # Should return empty set when pod placement info is missing
        assert len(nodes) == 0

    def test_default_object_types(self: Self) -> None:
        """Test that strategy defaults to AppWrapper when no object_types specified."""
        strategy = PodEvictionRetryStrategy()

        assert strategy.object_types == ["AppWrapper"]

        # Verify it works with AppWrapper
        event = create_eviction_event(object_type="AppWrapper")
        result = strategy.should_retry(event=event)
        assert result is True

    def test_ignores_non_eviction_pod_failures(self: Self) -> None:
        """Test that strategy only tracks eviction/preemption, not other failures."""
        events = [
            {
                "object_type": "AppWrapper",
                "object_name": "test-appwrapper",
                "reason": "Unhealthy",
                "message": "FailedComponent: Found 2 failed components",
            },
            {
                "object_type": "Pod",
                "object_name": "test-pod-1",
                "reason": "Preempted",  # This is an eviction
                "message": "Preempted",
            },
            {
                "object_type": "Pod",
                "object_name": "test-pod-2",
                "reason": "OOMKilled",  # Not an eviction
                "message": "Container killed",
            },
        ]

        pod_placement = {
            "test-pod-1": "worker-node-1",
            "test-pod-2": "worker-node-2",
        }

        data = {
            "state": "Failed",
            "previous_state": "Running",
            "events": events,
            "pod_placement": pod_placement,
        }
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

        strategy = PodEvictionRetryStrategy(
            object_types=["AppWrapper"],
            avoid_eviction_nodes=True,
        )

        # Should trigger retry (has Unhealthy + Preempted)
        result = strategy.should_retry(event=event)
        assert result is True

        # Should only include worker-node-1 (Preempted), not worker-node-2 (OOMKilled)
        nodes = strategy.extract_nodes_to_avoid(event=event)
        assert "worker-node-1" in nodes
        assert "worker-node-2" not in nodes
        assert len(nodes) == 1

    def test_real_world_example(self: Self) -> None:
        """Test with the actual example from the user's message."""
        # This is the exact JSON from the user's example
        data = {
            "appwrapper": "gb8zqlwumz",
            "state": "Failed",
            "previous_state": "Running",
            "current_resets": 0,
            "max_retries": "unlimited",
            "workload_status": [
                {
                    "workload_name": "appwrapper-gb8zqlwumz-fa6dd",
                    "workload_status": {
                        "admission": {
                            "clusterQueue": "granite-build-cluster-queue",
                            "podSetAssignments": [
                                {
                                    "count": 1,
                                    "flavors": {
                                        "cpu": "default-flavor",
                                        "memory": "default-flavor",
                                        "nvidia.com/gpu": "default-flavor",
                                        "pods": "default-flavor",
                                    },
                                    "name": "gb8zqlwumz-0-0",
                                    "resourceUsage": {
                                        "cpu": "64",
                                        "memory": "512Gi",
                                        "nvidia.com/gpu": "8",
                                        "pods": "1",
                                    },
                                }
                            ],
                        },
                        "conditions": [
                            {
                                "lastTransitionTime": "2026-01-28T05:48:28Z",
                                "message": "Quota reserved in ClusterQueue granite-build-cluster-queue",
                                "observedGeneration": 1,
                                "reason": "QuotaReserved",
                                "status": "True",
                                "type": "QuotaReserved",
                            },
                            {
                                "lastTransitionTime": "2026-01-28T05:48:28Z",
                                "message": "The workload is admitted",
                                "observedGeneration": 1,
                                "reason": "Admitted",
                                "status": "True",
                                "type": "Admitted",
                            },
                            {
                                "lastTransitionTime": "2026-01-28T06:25:40Z",
                                "message": "AppWrapper failed",
                                "observedGeneration": 1,
                                "reason": "Failed",
                                "status": "True",
                                "type": "Finished",
                            },
                        ],
                    },
                }
            ],
            "pod_placement": {"gb8zqlwumz-master-0": "dmf-nnnqh-gpu-worker-3-n6r6x"},
            "failed_pods": {},
            "events": [
                {
                    "object_type": "AppWrapper",
                    "object_name": "gb8zqlwumz",
                    "reason": "Unhealthy",
                    "message": "FailedComponent: Found 1 failed components",
                    "type": "Normal",
                    "time": "2026-01-28 06:25:40+00:00",
                },
                {
                    "object_type": "AppWrapper",
                    "object_name": "gb8zqlwumz",
                    "reason": "FinishedWorkload",
                    "message": "Workload 'granite-build/appwrapper-gb8zqlwumz-fa6dd' is declared finished",
                    "type": "Normal",
                    "time": "2026-01-28 06:25:40+00:00",
                },
                {
                    "object_type": "Pod",
                    "object_name": "gb8zqlwumz-master-0",
                    "reason": "Preempted",
                    "message": "Preempted by a pod on node dmf-nnnqh-gpu-worker-3-n6r6x",
                    "type": "Normal",
                    "time": "2026-01-28 06:25:07.090166+00:00",
                },
                {
                    "object_type": "Pod",
                    "object_name": "gb8zqlwumz-master-0",
                    "reason": "Killing",
                    "message": "Stopping container pytorch",
                    "type": "Normal",
                    "time": "2026-01-28 06:25:07+00:00",
                },
                {
                    "object_type": "Pod",
                    "object_name": "gb8zqlwumz-master-0",
                    "reason": "Killing",
                    "message": "Stopping container sidecar",
                    "type": "Normal",
                    "time": "2026-01-28 06:25:07+00:00",
                },
            ],
        }

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

        strategy = PodEvictionRetryStrategy(object_types=["AppWrapper"])

        # Should detect this as a retryable eviction
        result = strategy.should_retry(event=event)
        assert result is True

        # With avoidance enabled, should extract the node
        strategy_with_avoidance = PodEvictionRetryStrategy(
            object_types=["AppWrapper"],
            avoid_eviction_nodes=True,
        )
        nodes = strategy_with_avoidance.extract_nodes_to_avoid(event=event)
        assert "dmf-nnnqh-gpu-worker-3-n6r6x" in nodes
