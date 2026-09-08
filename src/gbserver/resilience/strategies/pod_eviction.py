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

    This strategy owns the K8s preemption verdict end to end, so the transient-
    vs-terminal decision lives in one place rather than being split with the
    generic engine. It uses the classifier (:func:`classify_appwrapper_failure`)
    to judge a ``Failed`` AppWrapper snapshot, from durable signals (Kueue Workload
    ``Evicted``/``Preempted`` conditions and ``requeueState``, and a sticky
    "preemption observed this launch" flag) that survive the ~1h K8s event window:

    * :meth:`should_retry` -- relaunch a transient preemption when retries remain;
    * :meth:`veto_terminal` -- tell the engine NOT to fail a transient preemption
      even when retries are disabled/exhausted, until the workload has been
      preempted more than ``max_preemptions`` times (the ceiling that keeps
      endless preemption from hanging silently). The count is cumulative over the
      strategy's lifetime, which spans relaunches.

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
        # Cumulative preemption classifications over this strategy's lifetime,
        # which spans relaunches (the handler and its strategies outlive the
        # monitor's pause/unpause).
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
        """Retry a transient preemption/eviction/requeue when retries remain.

        Delegates the transient-vs-terminal decision to the shared classifier via
        :meth:`_is_transient_preemption` (which reads durable preemption signals
        rather than requiring the causal K8s events to still be present).
        """
        should_retry = self._is_transient_preemption(event)
        if should_retry:
            logger.info(
                "Conditions met for preemption retry (object_types=%s)",
                self.object_types,
            )
        return should_retry

    def veto_terminal(
        self: Self,
        event: BuildEvent,
    ) -> bool:
        """Veto the engine's terminal verdict for a transient preemption so it does
        not fail the build even when retries are disabled/exhausted -- until the
        cumulative preemption count exceeds ``max_preemptions``, at which point the
        workload is allowed to fail so endless preemption can't hang silently.

        Counting here (once per event, on the single terminal path) means the
        ceiling engages for any preemption signal -- Kueue requeue included -- not
        just those that bump the AppWrapper reset count, and accumulates across
        relaunches.
        """
        if not self._is_transient_preemption(event):
            return False
        self.preemption_count += 1
        if self.preemption_count > self.max_preemptions:
            logger.error(
                "Workload preempted %d times (> max_preemptions=%d); "
                "allowing terminal failure.",
                self.preemption_count,
                self.max_preemptions,
            )
            return False
        logger.info(
            "Vetoing terminal failure for transient preemption (%d/%d).",
            self.preemption_count,
            self.max_preemptions,
        )
        return True

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
