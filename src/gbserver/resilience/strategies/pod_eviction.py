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


class PodEvictionRetryStrategy(RetryStrategy):
    """
    Retry strategy for pod evictions and preemptions.

    The classifier (:func:`classify_appwrapper_failure`) judges a ``Failed``
    AppWrapper snapshot from durable signals -- Kueue Workload ``Evicted``/
    ``Preempted`` conditions and ``requeueState``, plus a sticky "preemption
    observed this launch" flag -- that survive the ~1h K8s event window.
    :meth:`should_retry` marks a transient preemption retriable (so the engine
    relaunches it while retries remain); :meth:`is_exhausted` tolerates it on a
    separate ``max_preemptions`` budget rather than the shared retry budget, so a
    preemption does not fail the build even with step retries disabled, until it
    has occurred more than ``max_preemptions`` times.

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
        Cumulative preemptions tolerated before a repeatedly-preempted workload is
        failed rather than vetoed as transient. Defaults to GBSERVER_MAX_PREEMPTIONS.
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
            max_preemptions: Cumulative preemption ceiling; see class docstring.
        """
        # Default to AppWrapper for backward compatibility
        self.object_types = object_types if object_types is not None else ["AppWrapper"]
        self.avoid_eviction_nodes = avoid_eviction_nodes
        # Coerce: strategy config comes from YAML, where the value may be a string.
        self.max_preemptions = int(max_preemptions)
        # Preemptions seen this launch (spans relaunches); see is_exhausted.
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

    def _is_transient_preemption(self: Self, event: BuildEvent) -> bool:
        """True if this event is a ``Failed`` snapshot of a running workload that
        the classifier judges a transient preemption/eviction/requeue.

        Note on ``object_types``: this does not filter on ``self.object_types``.
        The old implementation only counted an ``Unhealthy`` event reported on a
        configured wrapper type, so a preempted pod wrapped by a different type was
        ignored. Preemption is transient regardless of what wraps the pod, and gb
        only launches AppWrappers today, so the gate added fragility without value.
        ``object_types`` is still honored by :meth:`extract_nodes_to_avoid`. To
        restrict to specific wrapper types later, gate here (return False unless an
        event with ``object_type in self.object_types`` is present) rather than in
        the pure classifier.
        """
        data = self._parse(event)
        if data is None:
            return False
        # Only workloads interrupted mid-run: one that never reached Running
        # failed for a different reason.
        if data.get("previous_state", "") != "Running":
            return False
        return (
            classify_appwrapper_failure(data) == AppWrapperVerdict.TRANSIENT_PREEMPTION
        )

    def should_retry(
        self: Self,
        event: BuildEvent,
    ) -> bool:
        """True for a transient preemption/eviction/requeue. Pure: the classifier
        reads durable preemption signals rather than requiring the causal K8s
        events to still be present. The ceiling is enforced by :meth:`is_exhausted`,
        not here, so exceeding it makes the event terminal rather than merely
        un-retriable."""
        return self._is_transient_preemption(event)

    def is_exhausted(
        self: Self,
        event: BuildEvent,
        _retry_count: int,
        _max_retries: int,
    ) -> bool:
        """A preemption is tolerated -- ignoring the shared retry budget -- until
        it has occurred more than ``max_preemptions`` times, so the engine keeps
        the build alive across preemptions (even with step retries disabled) but
        still fails one preempted endlessly instead of hanging.

        The engine calls this once per retriable event, so it doubles as the
        occurrence counter; the count accumulates across relaunches and covers any
        preemption signal (Kueue requeue included), not just AppWrapper resets.
        """
        self.preemption_count += 1
        exhausted = self.preemption_count > self.max_preemptions
        if exhausted:
            logger.error(
                "Workload preempted %d times (> max_preemptions=%d); failing.",
                self.preemption_count,
                self.max_preemptions,
            )
        else:
            logger.info(
                "Tolerating transient preemption (%d/%d).",
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
