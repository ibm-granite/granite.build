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
Retry strategy for pod evictions and preemptions.

This strategy handles cases where pods are evicted or preempted by the Kubernetes
scheduler due to resource pressure, higher-priority workloads, or node maintenance.
"""

import json
from typing import List, Optional, Self, Set

from gbserver.resilience.retry_handler import RetryStrategy
from gbserver.types.buildevent import BuildEvent, BuildEventType
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class PodEvictionRetryStrategy(RetryStrategy):
    """
    Retry strategy for pod evictions and preemptions.

    This strategy triggers retries when:
    - A Kubernetes object (AppWrapper, Job, Deployment, etc.) enters Failed state
    - Pods were preempted or evicted (reasons: "Preempted", "Evicted")
    - The workload was running before failure (previous_state: "Running")

    Unlike mount failures, evictions typically don't require node avoidance since
    the eviction is usually due to resource pressure or higher-priority workloads,
    not node-specific issues. However, if the same node keeps evicting pods, we
    may want to avoid it.

    Parameters
    ----------
    object_types : List[str]
        List of Kubernetes object types to monitor (e.g., ["AppWrapper"], ["Job"])
        If None, monitors all object types
    avoid_eviction_nodes : bool
        If True, avoid nodes where evictions occurred. If False (default), don't
        avoid any nodes, as evictions are typically cluster-wide resource issues.
    """

    def __init__(
        self: Self,
        object_types: Optional[List[str]] = None,
        avoid_eviction_nodes: bool = False,
    ) -> None:
        """
        Initialize the retry strategy.

        Args:
            object_types: List of K8s object types to monitor. If None, monitors all types.
                         Default: ["AppWrapper"] for backward compatibility.
            avoid_eviction_nodes: Whether to avoid nodes where evictions occurred.
        """
        # Default to AppWrapper for backward compatibility
        self.object_types = object_types if object_types is not None else ["AppWrapper"]
        self.avoid_eviction_nodes = avoid_eviction_nodes

    def should_retry(
        self: Self,
        event: BuildEvent,
    ) -> bool:
        """
        Check for Failed state with pod eviction/preemption.

        Analyzes BuildEvents emitted by monitors (e.g., AppWrapperMonitor) which contain
        Kubernetes events from the K8s API server. The monitor embeds K8s event data
        in the BuildEvent payload.
        """
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
            previous_state = data.get("previous_state", "")

            # Check for Failed state transition from Running
            if state != "Failed":
                return False

            # If it wasn't running before, don't retry
            # (we only want to retry workloads that were interrupted)
            if previous_state != "Running":
                logger.debug(
                    "Workload failed but was not running (previous_state=%s), not retrying",
                    previous_state,
                )
                return False

            # Check for Unhealthy event (indicates something went wrong)
            has_unhealthy_event = False
            has_eviction_or_preemption = False

            for ev in events:
                object_type = ev.get("object_type", "")

                # Check if this is an event for one of our monitored object types
                if object_type in self.object_types:
                    reason = ev.get("reason", "")
                    message = ev.get("message", "")

                    if reason == "Unhealthy":
                        has_unhealthy_event = True
                        logger.info(
                            "Detected Unhealthy event for %s: %s",
                            object_type,
                            message,
                        )

                # Check for pod eviction/preemption events
                if object_type == "Pod":
                    reason = ev.get("reason", "")
                    if reason in ["Preempted", "Evicted"]:
                        has_eviction_or_preemption = True
                        pod_name = ev.get("object_name", "")
                        message = ev.get("message", "")
                        logger.info(
                            "Detected pod %s event for pod %s: %s",
                            reason,
                            pod_name,
                            message,
                        )

            should_retry = has_unhealthy_event and has_eviction_or_preemption

            if should_retry:
                logger.info(
                    "Conditions met for retry: state=%s, previous_state=%s, "
                    "unhealthy=%s, eviction/preemption=%s, object_types=%s",
                    state,
                    previous_state,
                    has_unhealthy_event,
                    has_eviction_or_preemption,
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
        Extract nodes where evictions occurred (optional).

        By default, we don't avoid nodes for evictions since they're usually
        cluster-wide resource issues, not node-specific problems. However, if
        avoid_eviction_nodes is True, we'll extract the nodes.
        """
        if not self.avoid_eviction_nodes:
            return set()

        evicted_nodes = set()

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
                    # Look for Preempted or Evicted events
                    if reason in ["Preempted", "Evicted"]:
                        pod_name = ev.get("object_name", "")
                        # Get the node where this pod was placed
                        node_name = pod_placement.get(pod_name)
                        if node_name:
                            evicted_nodes.add(node_name)
                            logger.info(
                                "Pod %s was %s on node %s%s",
                                pod_name,
                                reason.lower(),
                                node_name,
                                (
                                    " (will avoid this node)"
                                    if self.avoid_eviction_nodes
                                    else ""
                                ),
                            )

        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            logger.debug("Could not extract nodes to avoid: %s", e)

        return evicted_nodes
