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
Retry strategy for Unhealthy + InsufficientPodsReady conditions and quota exhaustion.

This strategy is designed for Kubernetes environments where workloads fail due to
pod scheduling or infrastructure issues, particularly volume mount failures and
cluster-wide resource exhaustion (FailedScheduling with Insufficient cpu/memory/gpu).
"""

import json
from typing import List, Optional, Self, Set

from gbserver.resilience.retry_handler import RetryStrategy
from gbserver.types.buildevent import BuildEvent, BuildEventType
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class UnhealthyInsufficientPodsRetryStrategy(RetryStrategy):
    """
    Retry strategy for Unhealthy + InsufficientPodsReady conditions and quota exhaustion.

    This strategy triggers retries when:
    - A Kubernetes object (AppWrapper, Job, Deployment, etc.) is marked as Unhealthy
      and the message contains InsufficientPodsReady (node-specific failures)
    - Pods have FailedScheduling events with quota exhaustion messages like
      "Insufficient cpu", "Insufficient memory", "Insufficient nvidia.com/gpu"
      (cluster-wide resource exhaustion)

    For node-specific failures: extracts nodes to avoid, retries immediately.
    For quota exhaustion: no node avoidance (counterproductive), exponential backoff.

    Parameters
    ----------
    object_types : List[str]
        List of Kubernetes object types to monitor (e.g., ["AppWrapper"], ["Job", "Deployment"])
        If None, monitors all object types
    """

    # Patterns indicating cluster-wide resource exhaustion in FailedScheduling messages
    QUOTA_EXHAUSTION_PATTERNS = [
        "Insufficient cpu",
        "Insufficient memory",
        "Insufficient nvidia.com/gpu",
        "Insufficient ephemeral-storage",
        "nodes are available",
    ]

    def __init__(
        self: Self,
        object_types: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the retry strategy.

        Args:
            object_types: List of K8s object types to monitor. If None, monitors all types.
                         Default: ["AppWrapper"] for backward compatibility.
        """
        # Default to AppWrapper for backward compatibility
        self.object_types = object_types if object_types is not None else ["AppWrapper"]
        # Flag set by should_retry() to indicate quota exhaustion was detected
        # Used by get_retry_delay() and extract_nodes_to_avoid() to differentiate behavior
        self._is_quota_exhaustion = False

    def should_retry(
        self: Self,
        event: BuildEvent,
    ) -> bool:
        """
        Check for Unhealthy + InsufficientPodsReady or FailedScheduling quota exhaustion.

        Analyzes BuildEvents emitted by monitors (e.g., AppWrapperMonitor) which contain
        Kubernetes events from the K8s API server. The monitor embeds K8s event data
        in the BuildEvent payload.

        Detects two failure modes:
        1. Unhealthy + InsufficientPodsReady (node-specific, e.g. mount failures)
        2. FailedScheduling with quota exhaustion messages (cluster-wide resource shortage)
        """
        # Reset per-call state
        self._is_quota_exhaustion = False

        # Only process MESSAGE_EVENT types
        if event.type != BuildEventType.MESSAGE_EVENT:
            return False

        # Extract the message payload from the BuildEvent
        # The monitor embeds K8s events and state info in this payload
        try:
            msg = event.payload.msg
            # The message contains JSON with K8s object state info from the monitor
            # Try to extract it from markdown code block
            if "```json" in msg:
                json_start = msg.find("```json") + 7
                json_end = msg.find("```", json_start)
                json_str = msg[json_start:json_end].strip()
                data = json.loads(json_str)
            else:
                # Try to parse the whole message as JSON
                data = json.loads(msg)

            events = data.get("events", [])
            state = data.get("state", "")

            # Check for Unhealthy + InsufficientPodsReady
            has_unhealthy_event = False
            has_insufficient_pods_ready = False

            for ev in events:
                object_type = ev.get("object_type", "")

                # Check if this event is for one of our monitored object types
                if object_type in self.object_types:
                    reason = ev.get("reason", "")
                    message = ev.get("message", "")

                    if reason == "Unhealthy":
                        has_unhealthy_event = True
                        if "InsufficientPodsReady" in message:
                            has_insufficient_pods_ready = True
                            logger.info(
                                "Detected Unhealthy event with InsufficientPodsReady for %s: %s",
                                object_type,
                                message,
                            )

                # Check for FailedScheduling with quota exhaustion (Pod events)
                if object_type == "Pod" and ev.get("reason") == "FailedScheduling":
                    message = ev.get("message", "")
                    if any(pattern in message for pattern in self.QUOTA_EXHAUSTION_PATTERNS):
                        self._is_quota_exhaustion = True
                        logger.warning(
                            "Detected FailedScheduling with quota exhaustion: %s",
                            message,
                        )

            # Retry if either condition is met
            if self._is_quota_exhaustion:
                logger.info(
                    "Quota exhaustion detected, will retry with backoff. state=%s",
                    state,
                )
                return True

            should_retry = has_unhealthy_event and has_insufficient_pods_ready
            if should_retry:
                logger.info(
                    "Conditions met for retry: Unhealthy=%s, InsufficientPodsReady=%s, state=%s, object_types=%s",
                    has_unhealthy_event,
                    has_insufficient_pods_ready,
                    state,
                    self.object_types,
                )

            return should_retry

        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            logger.debug("Could not parse event for retry evaluation: %s", e)
            return False

    def extract_nodes_to_avoid(
        self: Self,
        event: BuildEvent,
    ) -> Set[str]:
        """
        Extract nodes where pods failed due to mount issues.

        Returns empty set for quota exhaustion since node avoidance is
        counterproductive for cluster-wide resource shortages.
        """
        # Quota exhaustion is cluster-wide — node avoidance makes it worse
        if self._is_quota_exhaustion:
            logger.info(
                "Quota exhaustion detected, skipping node avoidance "
                "(cluster-wide resource shortage)"
            )
            return set()

        failed_nodes = set()

        try:
            msg = event.payload.msg
            if "```json" in msg:
                json_start = msg.find("```json") + 7
                json_end = msg.find("```", json_start)
                json_str = msg[json_start:json_end].strip()
                data = json.loads(json_str)
            else:
                data = json.loads(msg)

            events = data.get("events", [])
            pod_placement = data.get("pod_placement", {})

            for ev in events:
                if ev.get("object_type") == "Pod":
                    reason = ev.get("reason", "")
                    # Look for FailedMount or other mount-related issues
                    if reason in ["FailedMount", "FailedAttachVolume"]:
                        pod_name = ev.get("object_name", "")
                        # Get the node where this pod was placed
                        node_name = pod_placement.get(pod_name)
                        if node_name:
                            failed_nodes.add(node_name)
                            logger.warning(
                                "Pod %s failed on node %s with reason %s, will avoid this node",
                                pod_name,
                                node_name,
                                reason,
                            )

        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            logger.debug("Could not extract nodes to avoid: %s", e)

        return failed_nodes

    def get_retry_delay(
        self: Self,
        retry_count: int,
    ) -> float:
        """
        Return delay before retrying.

        For quota exhaustion: exponential backoff (30s, 60s, 120s, 240s, capped at 300s).
        For node-specific failures (mount issues): immediate retry (0s).
        """
        if self._is_quota_exhaustion:
            delay = min(30 * (2**retry_count), 300)
            logger.info(
                "Quota exhaustion backoff: %.1f seconds (retry_count=%d)",
                delay,
                retry_count,
            )
            return delay
        return 0.0
