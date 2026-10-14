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
Classify a ``Failed`` AppWrapper snapshot as a pod disruption or a workload failure.

Background
----------
gb's step chart wraps a Job with ``backoffLimit: 0`` and sets the AppWrapper
``retryLimit`` to 0 by default, so nothing recreates a pod that dies: a pod
disrupted by the cluster -- scheduler preemption, a node drain (Eviction API),
kubelet node-pressure eviction, a node taint -- fails the Job, and the AppWrapper
goes ``Failed`` shortly after. That is not a workload failure and should be
relaunched rather than fail the step.

(Kueue preemption is a different path: Kueue suspends the AppWrapper and resumes
it later, it never reaches ``Failed``, so it is not classified here.)

Kubernetes records the cause on each pod, so the verdict is attributed per pod:
the ``AppWrapperMonitor`` collects ``disrupted_pods`` (pod name -> reason) from the
pod's ``DisruptionTarget`` condition, an ``Evicted`` pod status, or a
``Preempted``/``Evicted`` event on that pod, alongside ``failed_pods``. A snapshot
is a disruption only when every failed pod was disrupted; a failed pod with no
disruption record (a crash, OOM, image pull failure, ...) makes it terminal. A
disrupted pod's own non-zero exit (SIGKILL on eviction) is the side effect of the
disruption, not a separate failure.

This is a pure function over the JSON payload the monitor embeds in its
``MESSAGE_EVENT`` -- no I/O -- so it is unit-testable with fixture payloads.
"""

from enum import Enum
from typing import Any, Dict, Optional, Set

from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class AppWrapperVerdict(Enum):
    """Classification of a ``Failed`` AppWrapper snapshot."""

    #: Every failed pod was disrupted by the cluster -- relaunch, don't fail.
    DISRUPTION = "disruption"
    #: At least one pod failed on its own -- a genuine workload failure.
    TERMINAL_FAILURE = "terminal_failure"
    #: Not a ``Failed`` snapshot, or no disruption recorded. Callers keep the
    #: pre-existing behavior (terminal), so ``UNKNOWN`` never loosens a verdict.
    UNKNOWN = "unknown"


# Pod event reasons that name a disruption. Substrings cover reason drift across
# versions; only the controlled ``reason`` field is matched, never the free-form
# message ("... No preemption victims found" is the opposite of a preemption).
DISRUPTION_EVENT_REASONS = {"Preempted", "Evicted"}
DISRUPTION_REASON_SUBSTRINGS = ("preempt", "evict")


def event_disruption_reason(ev: Dict[str, Any]) -> Optional[str]:
    """Return the event's reason if it names a pod disruption, else None.

    Args:
        ev: a K8s event as the monitor serializes it (``reason``, ``object_type``...).

    Returns:
        The event ``reason`` for a Pod ``Preempted``/``Evicted``-style event, else None.
    """
    if ev.get("object_type", "Pod") != "Pod":
        return None
    reason = (ev.get("reason") or "").strip()
    if reason in DISRUPTION_EVENT_REASONS or any(
        sub in reason.lower() for sub in DISRUPTION_REASON_SUBSTRINGS
    ):
        return reason
    return None


def _disrupted_pod_names(data: Dict[str, Any]) -> Set[str]:
    """Pods the snapshot attributes to a cluster disruption: the monitor's
    ``disrupted_pods`` plus any pod named by a disruption event in ``events``."""
    names = set((data.get("disrupted_pods") or {}).keys())
    for ev in data.get("events", []) or []:
        if event_disruption_reason(ev) and ev.get("object_name"):
            names.add(ev["object_name"])
    return names


def classify_appwrapper_failure(
    data: Optional[Dict[str, Any]],
) -> AppWrapperVerdict:
    """Classify a parsed AppWrapper state-change payload.

    Args:
        data: the decoded object from the monitor's ```json``` block, or None.

    Returns:
        ``DISRUPTION`` when the snapshot is ``Failed``, a disruption is recorded,
        and every failed pod is among the disrupted ones; ``TERMINAL_FAILURE`` when
        some failed pod was not disrupted; ``UNKNOWN`` otherwise.
    """
    if not isinstance(data, dict) or data.get("state", "") != "Failed":
        # "Exception:" states are gb/infra errors, not disruptions.
        return AppWrapperVerdict.UNKNOWN

    aw = data.get("appwrapper", "<unknown>")
    disrupted = _disrupted_pod_names(data)
    undisrupted_failures = sorted(
        set((data.get("failed_pods") or {}).keys()) - disrupted
    )
    if undisrupted_failures:
        logger.info(
            "AppWrapper %s Failed; pods failed without disruption %s: TERMINAL",
            aw,
            undisrupted_failures,
        )
        return AppWrapperVerdict.TERMINAL_FAILURE
    if disrupted:
        logger.info(
            "AppWrapper %s Failed; pods disrupted %s: DISRUPTION",
            aw,
            sorted(disrupted),
        )
        return AppWrapperVerdict.DISRUPTION
    return AppWrapperVerdict.UNKNOWN
