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

"""Status module."""

from enum import StrEnum, auto


class Status(StrEnum):
    """Status implementation."""

    SUBMITTED = auto()
    PENDING = auto()
    RUNNING = auto()
    SUCCESS = auto()
    FAILED = auto()
    INVALID = auto()
    CANCELLED = auto()
    CANCEL_REQUESTED = auto()

    def is_finished(self) -> bool:
        """Determine of the status (of a build) indicates the job is no longer running.
        This includes SUCCESS, FAILED, INVALID or CANCELLED states.
        If status PENDING, RUNNING, or CANCEL_REQUESTED, then the build is assumed to either not have started or currently is running.
        """
        return self in (Status.SUCCESS, Status.FAILED, Status.INVALID, Status.CANCELLED)

    def is_cancellable(self) -> bool:
        """Determine if the build is cancellable based on this status.
        This generally means we can set the CANCEL_REQUESTED status on the build.
        """
        return not self.is_finished()


STATUS_TO_ICON = {
    Status.SUBMITTED: "◌",
    Status.PENDING: "🔵",
    Status.RUNNING: "⚡",
    Status.SUCCESS: "✅",
    Status.FAILED: "❌",
    Status.INVALID: "❌",
    Status.CANCELLED: "⚠️",
    Status.CANCEL_REQUESTED: "⚠️",
}
