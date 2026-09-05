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

from gbserver.resilience.appwrapper_classifier import (
    AppWrapperVerdict,
    classify_appwrapper_failure,
)
from gbserver.resilience.retry_handler import RetryStrategy
from gbserver.types.buildevent import BuildEvent, BuildEventType
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class PodEvictionRetryStrategy(RetryStrategy):
    """
    Retry strategy for pod evictions and preemptions.

    This strategy triggers a retry when the shared preemption classifier
    (:func:`classify_appwrapper_failure`) judges a ``Failed`` AppWrapper snapshot
    to be a transient preemption/eviction/requeue -- i.e. normal Kueue lifecycle
    churn rather than a workload failure. The classifier draws on durable signals
    (Kueue Workload ``Evicted``/``Preempted`` conditions and ``requeueState``, and
    a sticky "preemption observed this launch" flag), so detection no longer
    depends on the causal K8s events still being present in the terminal snapshot
    -- they age out of the ~1h event window well before the AppWrapper exhausts
    its own retryLimit and lands in ``Failed``.

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

        Note on ``object_types``: this method no longer filters the preemption
        decision by ``self.object_types``. The old implementation only counted an
        ``Unhealthy`` event when it was reported on one of the configured wrapper
        object types (e.g. ``AppWrapper``), which meant a preempted pod wrapped by
        a different object type was ignored. Preemption/requeue is transient
        regardless of what wraps the pod, and gb only launches AppWrappers today,
        so the gate added fragility without value and was dropped. ``object_types``
        is still honored by :meth:`extract_nodes_to_avoid` (node selection).

        If a future need arises to restrict retries to specific wrapper types
        (e.g. a mixed environment where a non-AppWrapper ``Failed`` must stay
        terminal), re-introduce the gate here rather than in the classifier: after
        parsing ``data``, return ``False`` unless at least one event with
        ``object_type in self.object_types`` is present (or thread an
        ``object_types`` argument through ``classify_appwrapper_failure`` so the
        Kueue-condition/reset signals are likewise scoped). Keeping it out of the
        pure classifier preserves the classifier's single-responsibility shape.
        """
        # Only process MESSAGE_EVENT types
        if event.type != BuildEventType.MESSAGE_EVENT:
            return False

        # Extract the message payload from the BuildEvent
        # The monitor embeds K8s events and state info in this payload
        try:
            msg = event.payload.msg  # type: ignore[union-attr]
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
        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            logger.debug("Could not parse event for retry evaluation: %s", e)
            return False

        if not isinstance(data, dict):
            return False

        # Only retry workloads that were interrupted mid-run (a workload that
        # never reached Running failed for a different reason).
        previous_state = data.get("previous_state", "")
        if previous_state != "Running":
            logger.debug(
                "Workload failed but was not running (previous_state=%s), not retrying",
                previous_state,
            )
            return False

        # Delegate the transient-vs-terminal decision to the shared classifier,
        # which reads durable preemption signals (Kueue Workload conditions,
        # requeueState, a sticky preemption flag) rather than requiring the causal
        # K8s events to still be present in this snapshot.
        should_retry = (
            classify_appwrapper_failure(data) == AppWrapperVerdict.TRANSIENT_PREEMPTION
        )

        if should_retry:
            logger.info(
                "Conditions met for preemption retry: state=%s, previous_state=%s, "
                "max_resets_seen=%s, object_types=%s",
                data.get("state"),
                previous_state,
                data.get("max_resets_seen"),
                self.object_types,
            )

        return should_retry

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
            msg = event.payload.msg  # type: ignore[union-attr]
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
