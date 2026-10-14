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
Table-driven unit tests for the AppWrapper disruption classifier.

The classifier is a pure function over the JSON payload the AppWrapperMonitor
embeds in its MESSAGE_EVENT, so these tests exercise it directly with fixture
dicts -- no Kubernetes, no monitor.
"""

from typing import Any, Dict

import pytest

from gbserver.resilience.strategies.appwrapper_classifier import (
    AppWrapperVerdict,
    classify_appwrapper_failure,
    event_disruption_reason,
)

OOM = {"failure-reason": "main failed with exit code 137; reason: OOMKilled"}
SIGKILL = {"failure-reason": "main failed with exit code 137; reason: Error"}


def _failed(**overrides: Any) -> Dict[str, Any]:
    """A minimal 'Failed' snapshot with no failed or disrupted pods."""
    base: Dict[str, Any] = {
        "appwrapper": "gbtest",
        "state": "Failed",
        "previous_state": "Running",
        "events": [],
        "failed_pods": {},
        "disrupted_pods": {},
    }
    base.update(overrides)
    return base


def _pod_event(pod: str, reason: str) -> Dict[str, Any]:
    return {"object_type": "Pod", "object_name": pod, "reason": reason, "message": ""}


class TestClassifyAppWrapperFailure:
    """Verdict table for classify_appwrapper_failure."""

    @pytest.mark.parametrize("data", [None, "nope", _failed(state="Running")])
    def test_not_a_failed_snapshot_is_unknown(self, data: Any) -> None:
        assert classify_appwrapper_failure(data) == AppWrapperVerdict.UNKNOWN

    def test_exception_state_is_unknown(self) -> None:
        data = _failed(state="Exception: AppWrapper vanished")
        assert classify_appwrapper_failure(data) == AppWrapperVerdict.UNKNOWN

    def test_failed_with_nothing_recorded_is_unknown(self) -> None:
        assert classify_appwrapper_failure(_failed()) == AppWrapperVerdict.UNKNOWN

    def test_failed_pod_without_disruption_is_terminal(self) -> None:
        data = _failed(failed_pods={"pod-1": OOM})
        assert classify_appwrapper_failure(data) == AppWrapperVerdict.TERMINAL_FAILURE

    @pytest.mark.parametrize(
        "reason",
        [
            "PreemptionByScheduler",
            "EvictionByEvictionAPI",
            "TerminationByKubelet",
            "DeletionByTaintManager",
            "Evicted",
        ],
    )
    def test_disrupted_failed_pod_is_disruption(self, reason: str) -> None:
        # The disrupted pod's own non-zero exit is the side effect of the kill.
        data = _failed(failed_pods={"pod-1": SIGKILL}, disrupted_pods={"pod-1": reason})
        assert classify_appwrapper_failure(data) == AppWrapperVerdict.DISRUPTION

    def test_disrupted_pod_oom_side_effect_is_disruption(self) -> None:
        data = _failed(failed_pods={"pod-1": OOM}, disrupted_pods={"pod-1": "Evicted"})
        assert classify_appwrapper_failure(data) == AppWrapperVerdict.DISRUPTION

    def test_disrupted_pod_deleted_before_seen_failed_is_disruption(self) -> None:
        # Scheduler preemption deletes the pod; only the disruption record remains.
        data = _failed(disrupted_pods={"pod-1": "Preempted"})
        assert classify_appwrapper_failure(data) == AppWrapperVerdict.DISRUPTION

    def test_disruption_from_pod_event_in_snapshot(self) -> None:
        data = _failed(
            failed_pods={"pod-1": SIGKILL},
            disrupted_pods={},
            events=[_pod_event("pod-1", "Preempted")],
        )
        assert classify_appwrapper_failure(data) == AppWrapperVerdict.DISRUPTION

    def test_other_pod_failing_on_its_own_is_terminal(self) -> None:
        # One pod was disrupted, another crashed on its own: a real failure, so an
        # earlier disruption cannot mask it.
        data = _failed(
            failed_pods={"pod-1": SIGKILL, "pod-2": OOM},
            disrupted_pods={"pod-1": "PreemptionByScheduler"},
        )
        assert classify_appwrapper_failure(data) == AppWrapperVerdict.TERMINAL_FAILURE

    def test_failed_scheduling_no_preemption_victims_is_not_disruption(self) -> None:
        # "preempt" in the free-form message must not read as a preemption.
        ev = _pod_event("pod-1", "FailedScheduling")
        ev["message"] = "0/8 nodes are available: No preemption victims found"
        data = _failed(failed_pods={"pod-1": OOM}, events=[ev])
        assert classify_appwrapper_failure(data) == AppWrapperVerdict.TERMINAL_FAILURE


class TestEventDisruptionReason:
    """event_disruption_reason matches pod disruption events by reason only."""

    @pytest.mark.parametrize("reason", ["Preempted", "Evicted", "PreemptedByKueue"])
    def test_disruption_reasons(self, reason: str) -> None:
        assert event_disruption_reason(_pod_event("p", reason)) == reason

    @pytest.mark.parametrize("reason", ["FailedScheduling", "OOMKilled", "", None])
    def test_other_reasons(self, reason: Any) -> None:
        assert event_disruption_reason(_pod_event("p", reason)) is None

    def test_non_pod_object_is_ignored(self) -> None:
        ev = {"object_type": "AppWrapper", "object_name": "aw", "reason": "Evicted"}
        assert event_disruption_reason(ev) is None
