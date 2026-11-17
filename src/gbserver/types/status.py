#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum, auto


class Status(StrEnum):

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
