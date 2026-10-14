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
from gbserver.resilience.strategies.appwrapper_classifier import (
    AppWrapperVerdict,
    classify_appwrapper_failure,
)
from gbserver.types.buildevent import BuildEvent, BuildEventType
from gbserver.types.constants import GBSERVER_MAX_PREEMPTIONS
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

# AppWrapper phases a workload passes through once it has started running.
STARTED_STATES = {"Running", "Resetting", "Resuming"}


class PodEvictionRetryStrategy(RetryStrategy):
    """
    Retry strategy for pod evictions and preemptions.

    The classifier (:func:`classify_appwrapper_failure`) attributes a ``Failed``
    AppWrapper snapshot per pod: when every failed pod was disrupted by the cluster
    (scheduler preemption, drain, kubelet eviction, node taint), :meth:`should_retry`
    asks the engine to relaunch. :meth:`is_exhausted` budgets those relaunches on
    ``max_preemptions`` instead of the shared ``max_retries``, so a disruption is
    relaunched even with step retries disabled, and a workload disrupted more than
    ``max_preemptions`` times fails instead of relaunching forever.

    Evictions typically don't require node avoidance since they are cluster-wide
    resource pressure, not node-specific -- but the same node repeatedly evicting
    can be avoided via ``avoid_eviction_nodes``.

    Parameters
    ----------
    object_types : List[str]
        List of Kubernetes object types to monitor (e.g., ["AppWrapper"], ["Job"])
        If None, monitors all object types
    avoid_eviction_nodes : bool
        If True, avoid nodes where evictions occurred. If False (default), don't
        avoid any nodes, as evictions are typically cluster-wide resource issues.
    max_preemptions : int
        Relaunches allowed for disruptions before the workload is failed.
        Defaults to GBSERVER_MAX_PREEMPTIONS.
    """

    def __init__(
        self: Self,
        object_types: Optional[List[str]] = None,
        avoid_eviction_nodes: bool = False,
        max_preemptions: int = GBSERVER_MAX_PREEMPTIONS,
    ) -> None:
        """
        Initialize the retry strategy.

        Args:
            object_types: List of K8s object types to monitor. If None, monitors all types.
                         Default: ["AppWrapper"] for backward compatibility.
            avoid_eviction_nodes: Whether to avoid nodes where evictions occurred.
            max_preemptions: Disruption relaunch budget; see class docstring.
        """
        # Default to AppWrapper for backward compatibility
        self.object_types = object_types if object_types is not None else ["AppWrapper"]
        self.avoid_eviction_nodes = avoid_eviction_nodes
        # Coerce: strategy config comes from YAML, where the value may be a string.
        self.max_preemptions = int(max_preemptions)
        # Disruption relaunches this launch (spans relaunches); see is_exhausted.
        self.preemption_count = 0

    @staticmethod
    def _parse(event: BuildEvent) -> Optional[dict]:
        """Parse the ```json``` block the AppWrapper monitor embeds in the event
        msg. Returns the dict, or None when absent/unparseable/not an object."""
        if event.type != BuildEventType.MESSAGE_EVENT:
            return None
        try:
            msg = event.payload.msg  # type: ignore[union-attr]
            if "```json" in msg:
                json_start = msg.find("```json") + 7
                json_end = msg.find("```", json_start)
                data = json.loads(msg[json_start:json_end].strip())
            else:
                data = json.loads(msg)
        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            logger.debug("Could not parse event for retry evaluation: %s", e)
            return None
        return data if isinstance(data, dict) else None

    def _is_disruption(self: Self, event: BuildEvent) -> bool:
        """True if this event is a ``Failed`` snapshot of a started workload that
        the classifier attributes to pod disruption.

        Note on ``object_types``: this does not filter on ``self.object_types``. A
        disrupted pod is not a workload failure regardless of what wraps it, and gb
        only launches AppWrappers today. ``object_types`` is still honored by
        :meth:`extract_nodes_to_avoid`; to restrict by wrapper type later, gate here
        rather than in the pure classifier.
        """
        data = self._parse(event)
        if data is None:
            return False
        # Only a workload that started: one that never left NotStarted/Suspended
        # failed for a different reason. Resetting/Resuming are included because
        # with retryLimit > 0 the last phase seen before Failed may be either.
        if data.get("previous_state", "") not in STARTED_STATES:
            return False
        return classify_appwrapper_failure(data) == AppWrapperVerdict.DISRUPTION

    def should_retry(
        self: Self,
        event: BuildEvent,
    ) -> bool:
        """True for a ``Failed`` caused by pod disruption. The relaunch budget is
        enforced by :meth:`is_exhausted`, so exceeding it fails the step."""
        return self._is_disruption(event)

    def is_exhausted(
        self: Self,
        event: BuildEvent,
        _retry_count: int,
        _max_retries: int,
    ) -> bool:
        """Budget disruption relaunches on ``max_preemptions``, ignoring the shared
        retry budget, so a disruption is relaunched even with step retries disabled.

        The engine asks once per matching event, right before relaunching, so this
        doubles as the relaunch counter; the count spans relaunches of the launch.
        """
        self.preemption_count += 1
        exhausted = self.preemption_count > self.max_preemptions
        if exhausted:
            logger.error(
                "Workload disrupted %d times (> max_preemptions=%d); failing.",
                self.preemption_count,
                self.max_preemptions,
            )
        else:
            logger.info(
                "Relaunching after pod disruption (%d/%d).",
                self.preemption_count,
                self.max_preemptions,
            )
        return exhausted

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
