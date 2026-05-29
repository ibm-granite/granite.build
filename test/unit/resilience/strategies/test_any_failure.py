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

"""Tests for AnyFailureRetryStrategy."""

import json

from gbserver.resilience.strategies.any_failure import AnyFailureRetryStrategy
from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventMessagePayload,
    BuildEventType,
    BuildEventWorkloadStatusPayload,
    EntityRunMetadata,
)
from gbserver.types.status import Status


def _msg_event(msg: str) -> BuildEvent:
    return BuildEvent(
        run_metadata=EntityRunMetadata(build_id="b-1"),
        type=BuildEventType.MESSAGE_EVENT,
        payload=BuildEventMessagePayload(msg=msg),
    )


def _status_event(status: Status) -> BuildEvent:
    return BuildEvent(
        run_metadata=EntityRunMetadata(build_id="b-1"),
        type=BuildEventType.WORKLOAD_STATUS_EVENT,
        payload=BuildEventWorkloadStatusPayload(status=status),
    )


class TestAnyFailureRetryStrategy:
    def test_workload_status_failed_triggers_retry(self):
        strategy = AnyFailureRetryStrategy()
        assert strategy.should_retry(_status_event(Status.FAILED))

    def test_workload_status_success_does_not_trigger(self):
        strategy = AnyFailureRetryStrategy()
        assert not strategy.should_retry(_status_event(Status.SUCCESS))

    def test_workload_status_running_does_not_trigger(self):
        strategy = AnyFailureRetryStrategy()
        assert not strategy.should_retry(_status_event(Status.RUNNING))

    def test_message_event_with_state_failed_triggers(self):
        """Bare-JSON MESSAGE_EVENT body with state=Failed → retry."""
        strategy = AnyFailureRetryStrategy()
        msg = json.dumps(
            {
                "state": "Failed",
                "previous_state": "Running",
                "events": [{"reason": "Whatever"}],
            }
        )
        assert strategy.should_retry(_msg_event(msg))

    def test_message_event_with_state_failed_inside_markdown_fence(self):
        """state=Failed wrapped in ```json ... ``` markdown fence → retry."""
        strategy = AnyFailureRetryStrategy()
        body = json.dumps({"state": "Failed"})
        wrapped = f"Some preamble.\n```json\n{body}\n```\nMore prose."
        assert strategy.should_retry(_msg_event(wrapped))

    def test_message_event_with_state_running_does_not_trigger(self):
        strategy = AnyFailureRetryStrategy()
        msg = json.dumps({"state": "Running"})
        assert not strategy.should_retry(_msg_event(msg))

    def test_message_event_non_json_does_not_trigger(self):
        strategy = AnyFailureRetryStrategy()
        assert not strategy.should_retry(_msg_event("just a status update"))

    def test_message_event_empty_msg_does_not_trigger(self):
        strategy = AnyFailureRetryStrategy()
        assert not strategy.should_retry(_msg_event(""))

    def test_extract_nodes_to_avoid_returns_empty(self):
        """AnyFailureRetryStrategy does not pin failures to nodes."""
        strategy = AnyFailureRetryStrategy()
        nodes = strategy.extract_nodes_to_avoid(_status_event(Status.FAILED))
        assert nodes == set()

    def test_accepts_object_types_is_false(self):
        """The strategy is K8s-agnostic so it does not filter object_types."""
        assert AnyFailureRetryStrategy.accepts_object_types is False
