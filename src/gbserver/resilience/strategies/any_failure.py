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

"""Retry strategy that fires on any failure event regardless of root cause."""

import json
import re
from typing import Self, Set

from gbserver.resilience.retry_handler import RetryStrategy
from gbserver.types.buildevent import BuildEvent, BuildEventType
from gbserver.types.status import Status
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class AnyFailureRetryStrategy(RetryStrategy):
    """Retry on any failure event regardless of root cause.

    Designed for environments (e.g. SkyPilot) where finer-grained
    cause-specific strategies don't fit and the user wants the
    framework to retry on any failure up to ``max_retries``. Fires on:

    1. ``BuildEventType.WORKLOAD_STATUS_EVENT`` whose payload reports
       ``status == Status.FAILED``.
    2. ``BuildEventType.MESSAGE_EVENT`` whose payload ``msg`` contains
       a JSON object with ``state == "Failed"`` — either as the whole
       body or wrapped in a ```json ... ``` markdown fence. Matches
       the simulator output and the K8s sidecar monitor format.

    Does not extract nodes to avoid; cloud allocators decide
    placement and Skypilot has no portable node-exclusion knob.
    """

    # No K8s object filtering.
    accepts_object_types = False

    def should_retry(self: Self, event: BuildEvent) -> bool:
        """Return True if the event represents any kind of failure."""
        if self._is_workload_status_failed(event):
            logger.warning(
                "[AnyFailureRetryStrategy] Detected WORKLOAD_STATUS_EVENT FAILED"
            )
            return True
        if self._is_message_state_failed(event):
            logger.warning(
                "[AnyFailureRetryStrategy] Detected MESSAGE_EVENT with state=Failed"
            )
            return True
        return False

    def extract_nodes_to_avoid(self: Self, _event: BuildEvent) -> Set[str]:
        """Return an empty set — this strategy does not pin failures to nodes."""
        return set()

    @staticmethod
    def _is_workload_status_failed(event: BuildEvent) -> bool:
        """True iff event is a WORKLOAD_STATUS_EVENT with status == FAILED."""
        if event.type != BuildEventType.WORKLOAD_STATUS_EVENT:
            return False
        payload = event.payload
        status = getattr(payload, "status", None) if payload else None
        return status == Status.FAILED

    @staticmethod
    def _is_message_state_failed(event: BuildEvent) -> bool:
        """True iff event is a MESSAGE_EVENT whose msg JSON reports state=Failed."""
        if event.type != BuildEventType.MESSAGE_EVENT:
            return False
        payload = event.payload
        if payload is None or not hasattr(payload, "msg"):
            return False
        msg = payload.msg
        if not msg:
            return False

        # Try the whole body as bare JSON first.
        try:
            data = json.loads(msg)
            if isinstance(data, dict) and data.get("state") == "Failed":
                return True
        except (json.JSONDecodeError, TypeError):
            pass

        # Fall back to JSON inside a ```json ... ``` markdown fence.
        match = re.search(r"```json\s*\n(.*?)\n```", msg, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                if isinstance(data, dict) and data.get("state") == "Failed":
                    return True
            except json.JSONDecodeError:
                pass
        return False
