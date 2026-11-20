# SPDX-License-Identifier: Apache-2.0

"""Unit tests for libgbtest.buildrunner.retry_wait.wait_for_last_attempt."""

from types import SimpleNamespace

import pytest
from libgbtest.buildrunner.retry_wait import wait_for_last_attempt

from gbserver.types.status import Status


def _builds(*records):
    """A read_build that returns ``records`` in turn, then the last one forever."""
    it = iter(records)
    last = []

    def read():
        try:
            last[:] = [next(it)]
        except StopIteration:
            pass
        status, retry_count = last[0]
        return SimpleNamespace(status=status, retry_count=retry_count)

    return read


def _wait(read, alive=lambda: True, max_retries=1, timeout_s=5):
    return wait_for_last_attempt(read, alive, max_retries, timeout_s, poll_s=0)


def test_failed_between_attempts_is_waited_through():
    """FAILED with retries left is the gap before the retry, not the end."""
    read = _builds(
        (Status.RUNNING, 0),
        (Status.FAILED, 0),
        (Status.RUNNING, 1),
        (Status.SUCCESS, 1),
    )
    assert _wait(read).status == Status.SUCCESS


def test_failed_at_the_last_attempt_ends_the_wait():
    """A real failure is reported as soon as it's final, not as a timeout."""
    read = _builds(
        (Status.RUNNING, 0), (Status.FAILED, 0), (Status.RUNNING, 1), (Status.FAILED, 1)
    )
    build = _wait(read)
    assert (build.status, build.retry_count) == (Status.FAILED, 1)


@pytest.mark.parametrize("status", [Status.CANCELLED, Status.INVALID])
def test_cancelled_or_invalid_ends_the_wait(status):
    assert _wait(_builds((status, 0))).status == status


def test_never_returns_a_record_read_before_the_runner_stopped():
    """The runner stops, and the build turns SUCCESS, at the second of the loop's
    reads, whichever that is: reading the build first would return RUNNING."""
    reads = []

    def tick():
        reads.append(1)
        return len(reads) >= 2

    def alive():
        return not tick()

    def read():
        stopped = tick()
        return SimpleNamespace(
            status=Status.SUCCESS if stopped else Status.RUNNING, retry_count=1
        )

    assert _wait(read, alive).status == Status.SUCCESS


def test_a_stopped_runner_ends_the_wait():
    build = _wait(_builds((Status.RUNNING, 0)), alive=lambda: False)
    assert build.status == Status.RUNNING


def test_timeout_returns_none():
    assert _wait(_builds((Status.RUNNING, 0)), timeout_s=0.05) is None
