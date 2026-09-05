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
Table-driven unit tests for the AppWrapper preemption classifier.

The classifier is a pure function over the JSON payload the AppWrapperMonitor
embeds in its MESSAGE_EVENT, so these tests exercise it directly with fixture
dicts -- no Kubernetes, no monitor.
"""

from typing import Any, Dict

import pytest

from gbserver.resilience.appwrapper_classifier import (
    AppWrapperVerdict,
    classify_appwrapper_failure,
)


def _failed(**overrides: Any) -> Dict[str, Any]:
    """A minimal terminal 'Failed' snapshot with no preemption signal."""
    base: Dict[str, Any] = {
        "appwrapper": "gbtest",
        "state": "Failed",
        "previous_state": "Running",
        "current_resets": 0,
        "max_resets_seen": 0,
        "preemption_observed": False,
        "workload_status": [],
        "events": [],
        "failed_pods": {},
    }
    base.update(overrides)
    return base


class TestClassifyAppWrapperFailure:
    """Verdict table for classify_appwrapper_failure."""

    def test_none_is_unknown(self) -> None:
        assert classify_appwrapper_failure(None) == AppWrapperVerdict.UNKNOWN

    def test_non_dict_is_unknown(self) -> None:
        assert classify_appwrapper_failure("nope") == AppWrapperVerdict.UNKNOWN  # type: ignore[arg-type]

    def test_non_failed_state_is_unknown(self) -> None:
        # Running / Succeeded / Exception are not this classifier's concern.
        assert (
            classify_appwrapper_failure(_failed(state="Running"))
            == AppWrapperVerdict.UNKNOWN
        )
        assert (
            classify_appwrapper_failure(_failed(state="Exception: gone"))
            == AppWrapperVerdict.UNKNOWN
        )

    def test_failed_with_no_signal_is_unknown(self) -> None:
        # A bare Failed with nothing to go on preserves today's terminal default.
        assert classify_appwrapper_failure(_failed()) == AppWrapperVerdict.UNKNOWN

    # ---- preemption signals (each independently sufficient) ----

    def test_sticky_preemption_observed(self) -> None:
        assert (
            classify_appwrapper_failure(_failed(preemption_observed=True))
            == AppWrapperVerdict.TRANSIENT_PREEMPTION
        )

    def test_reset_count_alone_is_not_preemption(self) -> None:
        # resettingCount bumps on any failure (crashes included), so it is not a
        # preemption signal on its own: a bare Failed with only resets is UNKNOWN.
        assert (
            classify_appwrapper_failure(_failed(max_resets_seen=2, current_resets=2))
            == AppWrapperVerdict.UNKNOWN
        )

    def test_preemption_via_pod_event(self) -> None:
        data = _failed(
            events=[{"object_type": "Pod", "reason": "Preempted", "message": "x"}]
        )
        assert (
            classify_appwrapper_failure(data) == AppWrapperVerdict.TRANSIENT_PREEMPTION
        )

    def test_preemption_via_event_message_substring(self) -> None:
        # Reason string drift: reason unknown but message says evicted.
        data = _failed(
            events=[
                {"object_type": "Pod", "reason": "Weird", "message": "pod was evicted"}
            ]
        )
        assert (
            classify_appwrapper_failure(data) == AppWrapperVerdict.TRANSIENT_PREEMPTION
        )

    def test_preemption_via_kueue_workload_condition(self) -> None:
        # Events aged out (empty) but the Kueue Workload condition persists.
        data = _failed(
            workload_status=[
                {
                    "workload_name": "wl-1",
                    "workload_status": {
                        "conditions": [{"type": "Evicted", "status": "True"}]
                    },
                }
            ]
        )
        assert (
            classify_appwrapper_failure(data) == AppWrapperVerdict.TRANSIENT_PREEMPTION
        )

    def test_preemption_via_kueue_requeue_state(self) -> None:
        data = _failed(
            workload_status=[
                {
                    "workload_name": "wl-1",
                    "workload_status": {"requeueState": {"count": 3}},
                }
            ]
        )
        assert (
            classify_appwrapper_failure(data) == AppWrapperVerdict.TRANSIENT_PREEMPTION
        )

    # ---- genuine terminal failures (only when NO preemption signal) ----

    def test_hard_failure_via_failed_pod_reason(self) -> None:
        data = _failed(
            failed_pods={"pod-1": {"failure-reason": "OOMKilled", "logs": {}}}
        )
        assert classify_appwrapper_failure(data) == AppWrapperVerdict.TERMINAL_FAILURE

    def test_hard_failure_via_event_reason(self) -> None:
        data = _failed(
            events=[{"object_type": "Pod", "reason": "ImagePullBackOff", "message": ""}]
        )
        assert classify_appwrapper_failure(data) == AppWrapperVerdict.TERMINAL_FAILURE

    # ---- mixed snapshot: preemption wins ----

    def test_mixed_preemption_and_oom_is_transient(self) -> None:
        data = _failed(
            events=[
                {"object_type": "Pod", "reason": "Preempted", "message": "Preempted"},
                {"object_type": "Pod", "reason": "OOMKilled", "message": "killed"},
            ]
        )
        assert (
            classify_appwrapper_failure(data) == AppWrapperVerdict.TRANSIENT_PREEMPTION
        )

    def test_reset_count_with_hard_pod_is_terminal(self) -> None:
        # Resets are not a preemption signal, so a hard pod failure with only
        # resets (no preemption event/condition) is a genuine crash.
        data = _failed(
            current_resets=2,
            failed_pods={"pod-1": {"failure-reason": "OOMKilled", "logs": {}}},
        )
        assert classify_appwrapper_failure(data) == AppWrapperVerdict.TERMINAL_FAILURE

    def test_composite_failure_reason_is_terminal(self) -> None:
        # The monitor records a composite reason; substring matching must catch it.
        data = _failed(
            failed_pods={
                "pod-1": {
                    "failure-reason": "trainer failed with exit code 137; reason: OOMKilled",
                    "logs": {},
                }
            }
        )
        assert classify_appwrapper_failure(data) == AppWrapperVerdict.TERMINAL_FAILURE

    def test_preemption_failure_reason_on_pod_is_not_hard_terminal(self) -> None:
        # A failed pod whose reason is itself a preemption must not count as a
        # hard terminal failure; combined with the event it stays transient.
        data = _failed(
            failed_pods={"pod-1": {"failure-reason": "Evicted", "logs": {}}},
            events=[{"object_type": "Pod", "reason": "Evicted", "message": "evicted"}],
        )
        assert (
            classify_appwrapper_failure(data) == AppWrapperVerdict.TRANSIENT_PREEMPTION
        )
