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
Classify an AppWrapper failure snapshot as transient (preemption/requeue) or
genuinely terminal.

Background
----------
gb derives a K8s step's pass/fail from the AppWrapper's coarse ``.status.phase``
string. When pods are preempted/evicted, the AppWrapper controller does in-place
resets up to its own ``retryLimit``; once that is exhausted the phase becomes
``Failed`` -- even though nothing about the *workload* failed. Preemption and
requeue are a normal part of the Kueue/AppWrapper lifecycle (especially for
multi-GPU steps under GPU pressure) and must not fail the build.

This module is the single source of truth used by both the retry handler
(to decide "don't treat this Failed as terminal") and the pod-eviction retry
strategy (to decide "relaunch this"). It is a **pure function** over the JSON
payload the ``AppWrapperMonitor`` embeds in its ``MESSAGE_EVENT`` -- no I/O, no
Kubernetes calls -- so it is trivially unit-testable with fixture payloads.

Design principle: **preemption wins on a mixed snapshot.** When a workload is
preempted/evicted, its containers are frequently OOM-killed or otherwise hard-
killed as a *side effect* of the eviction, so those hard-failure reasons show up
in the very same snapshot as the preemption. Treating that as terminal would
re-introduce exactly the false-positive failure this module exists to prevent.
So the order is: if any preemption/eviction/requeue signal is present, classify
``TRANSIENT_PREEMPTION``; only when a hard-failure reason is present *and there
is no preemption signal at all* do we classify ``TERMINAL_FAILURE``. Anything
else is ``UNKNOWN``, which the callers treat exactly as today's behavior
(terminal), so this classifier can never make a currently-terminal snapshot
*more* lenient by accident.

This module only classifies a single snapshot; the ceiling that fails a workload
preempted endlessly (so it can't hang forever) lives in the RetryHandler.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class AppWrapperVerdict(Enum):
    """Classification of a terminal-looking AppWrapper snapshot."""

    #: Failure caused by pod preemption / eviction / Kueue requeue -- expected
    #: lifecycle churn, not a workload failure. Should not fail the build.
    TRANSIENT_PREEMPTION = "transient_preemption"
    #: A genuine, non-retriable workload failure (crash, non-preemption OOM,
    #: unresolvable image pull, ...). Should fail the build.
    TERMINAL_FAILURE = "terminal_failure"
    #: Not enough signal to tell. Callers preserve the pre-existing behavior
    #: (treat as terminal), so ``UNKNOWN`` never loosens today's verdict.
    UNKNOWN = "unknown"


# Pod-event reasons that indicate a preemption/eviction interruption.
PREEMPTION_EVENT_REASONS = {"Preempted", "Evicted"}

# Substrings (lower-cased) that indicate preemption/eviction when the reason
# string drifts (Kueue/scheduler wording varies across versions). Matched against
# the controlled ``reason`` field, not the free-form message: the standard
# FailedScheduling message "... No preemption victims found for incoming pod"
# contains "preempt" but is the opposite of a preemption.
PREEMPTION_REASON_SUBSTRINGS = ("preempt", "evict")

# Kueue Workload condition types that indicate an interruption/requeue.
PREEMPTION_WORKLOAD_CONDITIONS = {"Evicted", "Preempted"}

# Reason tokens (lower-cased) that indicate a genuine, non-preemption terminal
# failure the controller won't usefully retry. Matched as substrings because the
# monitor records composite reasons like "container failed with exit code 137;
# reason: OOMKilled".
HARD_FAILURE_REASON_TOKENS = (
    "crashloopbackoff",
    "errimagepull",
    "imagepullbackoff",
    "invalidimagename",
    "oomkill",  # covers OOMKilled / OOMKilling
)


def _reason_str_is_preemption(reason: str) -> bool:
    reason = reason.strip()
    return reason in PREEMPTION_EVENT_REASONS or any(
        sub in reason.lower() for sub in PREEMPTION_REASON_SUBSTRINGS
    )


def _event_indicates_preemption(ev: Dict[str, Any]) -> bool:
    """True if a single K8s event dict describes a preemption/eviction, judged
    from the controlled ``reason`` field only (not the free-form message)."""
    return _reason_str_is_preemption(ev.get("reason") or "")


def _events_have_preemption(events: List[Dict[str, Any]]) -> bool:
    return any(_event_indicates_preemption(ev) for ev in events)


def _workload_conditions_have_preemption(data: Dict[str, Any]) -> bool:
    """Scan the Kueue Workload conditions for an Evicted/Preempted condition or a
    requeue marker. These live on the Workload object and, unlike K8s Events, do
    not age out -- so they survive the ~1h event window that made the old
    event-only detection fragile.

    Prefers the monitor's distilled ``workload_conditions`` -- a list of
    ``{workload_name, conditions: [{type, status, reason}], requeued}`` -- and
    falls back to the raw ``workload_status`` (the Kueue Workload ``.status``
    dicts) when the distilled field is absent.
    """
    distilled = data.get("workload_conditions")
    if distilled:
        for entry in distilled:
            if entry.get("requeued"):
                return True
            for cond in entry.get("conditions", []) or []:
                if _condition_is_preemption(cond):
                    return True
        return False

    for entry in data.get("workload_status", []) or []:
        status = entry.get("workload_status", {}) or {}
        for cond in status.get("conditions", []) or []:
            if _condition_is_preemption(cond):
                return True
        if status.get("requeueState"):
            return True
    return False


def _condition_is_preemption(cond: Dict[str, Any]) -> bool:
    ctype = cond.get("type", "")
    cstatus = str(cond.get("status", "")).lower()
    if ctype in PREEMPTION_WORKLOAD_CONDITIONS and cstatus == "true":
        return True
    return _reason_str_is_preemption(cond.get("reason") or "")


def _reason_is_hard_failure(reason: str) -> bool:
    return any(tok in reason.lower() for tok in HARD_FAILURE_REASON_TOKENS)


def _has_hard_terminal_failure(data: Dict[str, Any]) -> bool:
    """True if the snapshot carries evidence of a genuine, non-preemption failure.

    Two shapes are checked:

    * a ``failed_pods`` entry whose composite ``failure-reason`` contains a hard,
      non-retriable token (image pull, OOM, crash-loop) and no preemption token; or
    * an event with such a reason that is not a preemption event.
    """
    for pod_status in (data.get("failed_pods") or {}).values():
        reason = (pod_status.get("failure-reason") or "").strip()
        if (
            reason
            and _reason_is_hard_failure(reason)
            and not _reason_str_is_preemption(reason)
        ):
            return True

    for ev in data.get("events", []) or []:
        reason = (ev.get("reason") or "").strip()
        if (
            reason
            and _reason_is_hard_failure(reason)
            and not _event_indicates_preemption(ev)
        ):
            return True
    return False


def _has_fresh_preemption_signal(data: Dict[str, Any]) -> bool:
    """A preemption signal present in THIS snapshot (not the sticky flag).

    Only preemption-specific signals count; the AppWrapper ``resettingCount`` is
    deliberately excluded because the controller resets pods in place on *any*
    failure. These are a Kueue Workload ``Evicted``/``Preempted`` condition or
    ``requeueState``, or a Preempted/Evicted K8s event in the current snapshot.
    """
    return _workload_conditions_have_preemption(data) or _events_have_preemption(
        data.get("events", []) or []
    )


def classify_appwrapper_failure(
    data: Optional[Dict[str, Any]],
) -> AppWrapperVerdict:
    """Classify a parsed AppWrapper state-change payload.

    Args:
        data: the decoded object from the monitor's ```json``` block, or None.

    Returns:
        AppWrapperVerdict. ``UNKNOWN`` when ``data`` is None, is not a terminal
        ``Failed`` snapshot, or carries no preemption signal -- callers treat
        ``UNKNOWN`` as terminal, preserving today's behavior.
    """
    if not isinstance(data, dict):
        return AppWrapperVerdict.UNKNOWN

    state = data.get("state", "")
    # Only classify a terminal "Failed" snapshot. "Exception:" states are
    # gb/infra errors (e.g. the AppWrapper vanished), not workload preemption,
    # and stay terminal.
    if state != "Failed":
        return AppWrapperVerdict.UNKNOWN

    aw = data.get("appwrapper", "<unknown>")
    fresh_preemption = _has_fresh_preemption_signal(data)
    hard_failure = _has_hard_terminal_failure(data)

    # A preemption signal in THIS snapshot wins over a co-occurring hard failure:
    # a preempted pod is often OOM/hard-killed as a side effect of the eviction.
    if fresh_preemption:
        logger.info("AppWrapper %s Failed with a preemption signal; TRANSIENT", aw)
        return AppWrapperVerdict.TRANSIENT_PREEMPTION

    # A genuine hard failure in this snapshot beats the *sticky* flag alone -- an
    # earlier preemption must not mask a later real crash into a non-terminal
    # verdict (which, with no further events, would hang instead of failing).
    if hard_failure:
        logger.info("AppWrapper %s Failed with a hard terminal reason; TERMINAL", aw)
        return AppWrapperVerdict.TERMINAL_FAILURE

    # Otherwise the sticky "preemption seen this launch" flag makes it transient:
    # the causal events have aged out but the failure is still preemption-shaped.
    if data.get("preemption_observed"):
        logger.info("AppWrapper %s Failed after a prior preemption; TRANSIENT", aw)
        return AppWrapperVerdict.TRANSIENT_PREEMPTION

    return AppWrapperVerdict.UNKNOWN
