# SPDX-License-Identifier: Apache-2.0

"""Waiting out a build that may retry in place (``retries.max_retries``).

A build test that drives a ``BuildRunner`` itself, and expects attempts to fail
before the build succeeds, can't wait for "SUCCESS or FAILED": FAILED is passed
through between attempts. :func:`wait_for_last_attempt` waits for the build's last
attempt instead.
"""

import time
from typing import Any, Callable, Optional

from gbserver.types.status import Status


def wait_for_last_attempt(
    read_build: Callable[[], Any],
    runner_alive: Callable[[], bool],
    max_retries: int,
    timeout_s: float,
    poll_s: float = 10,
) -> Optional[Any]:
    """Wait until the build succeeds or its last attempt has ended.

    FAILED counts as the end only once ``retry_count`` has reached ``max_retries``:
    the buildrunner bumps ``retry_count`` and moves the build to RUNNING in one
    guarded update (``BuildRunner.__prepare_retry``), so FAILED with retries left
    is the gap between attempts. A runner that has stopped ends the wait too.

    The runner is checked BEFORE the build is read: if it stops between the two,
    the record read after it is already its last one, never a stale RUNNING.

    :param read_build: Reads the build's current record (``status``, ``retry_count``).
    :param runner_alive: Whether the runner driving the build is still running.
    :param max_retries: The build's ``retries.max_retries``.
    :param timeout_s: How long to wait.
    :param poll_s: How often to read.
    :returns: The last record read, or None if ``timeout_s`` passed first.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        alive = runner_alive()
        build = read_build()
        if build.status in (Status.SUCCESS, Status.CANCELLED, Status.INVALID):
            return build
        if build.status == Status.FAILED and build.retry_count >= max_retries:
            return build
        if not alive:
            return build
        time.sleep(poll_s)
    return None
