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

"""Unit tests for AbstractBuildTest's abort handling: waits, interrupts, cleanup.

Uses a bare AbstractBuildTest (no storage, no build) with fakes standing in for
the runner, storage and URIs.
"""

import threading
import time
from types import SimpleNamespace
from typing import Any, cast

import pytest
from libgbtest.abort_cleanup import SerializedCleanup
from libgbtest.buildrunner import buildtest
from libgbtest.buildrunner.buildtest import AbstractBuildTest
from libgbtest.buildrunner.utils import ExceptionRaisingThread

from gbserver.types.status import Status


def _bare_test() -> AbstractBuildTest:
    """Make an AbstractBuildTest without running setup_method.

    :returns: A test object holding only the state setup_method gives the abort
        and cleanup code (no storage, cluster login or build).
    """
    t = object.__new__(cast(Any, AbstractBuildTest))
    t._output_cleanup = SerializedCleanup(t._delete_build_outputs)
    t._deleted_artifact_uris = set()
    t._deleted_pod_build_ids = set()
    t._abort_waits = threading.Event()
    t._active_runner = None
    t._wait_deadline = None
    t.class_tested = None
    return t


class _RunningItemStorage:
    """Item storage whose one item stays RUNNING forever."""

    def get_by_uuid(self, uuid):
        """Look up items the way IItemStorage.get_by_uuid does.

        :param uuid: An item id, or None for all items.
        :returns: The RUNNING item, or a list holding it when ``uuid`` is None.
        """
        item = SimpleNamespace(uuid="b1", status=Status.RUNNING)
        return item if uuid is not None else [item]


def _wait_for_success(t: AbstractBuildTest, timeout: float) -> None:
    """Wait for the always-RUNNING build to reach SUCCESS, which it never does.

    :param t: The test object whose _wait_for_status is exercised.
    :param timeout: The wait's own timeout, in seconds.
    :raises AssertionError: When the wait is aborted or times out.
    """
    t._wait_for_status(
        "b1", "build", "b1", cast(Any, _RunningItemStorage()), [Status.SUCCESS], timeout
    )


# --- status waits -------------------------------------------------------------


def test_wait_returns_at_once_when_aborted():
    t = _bare_test()
    threading.Timer(0.2, t._abort_waits.set).start()
    start = time.monotonic()
    with pytest.raises(AssertionError, match="aborted"):
        _wait_for_success(t, timeout=600)
    assert time.monotonic() - start < 2


def test_wait_is_bounded_by_the_test_deadline():
    t = _bare_test()
    t._wait_deadline = time.monotonic() + 0.5
    start = time.monotonic()
    with pytest.raises(AssertionError, match="exceeded the timeout"):
        _wait_for_success(t, timeout=600)
    assert time.monotonic() - start < 2


# --- interrupt while the build runs -------------------------------------------


class _InterruptedRunner:
    """Runner whose start_and_wait() is interrupted, as by a SIGTERM."""

    def __init__(self):
        """Set up the event that records a stop() call."""
        self.stopped = threading.Event()

    def start_and_wait(self):
        """Run briefly, then be interrupted.

        :raises KeyboardInterrupt: Always, as the mapped SIGTERM would.
        """
        time.sleep(0.2)
        raise KeyboardInterrupt("pytest received SIGTERM")

    def stop(self):
        """Record that the build was stopped."""
        self.stopped.set()


def test_interrupt_stops_the_build_and_the_waiter():
    t = _bare_test()
    runner = _InterruptedRunner()
    waiter = ExceptionRaisingThread(
        target=_wait_for_success, args=(t, 600), daemon=True
    )
    with pytest.raises(KeyboardInterrupt):
        t._run_with_waiter(runner, waiter)
    assert runner.stopped.is_set()
    assert not waiter.is_alive()
    assert t._active_runner is None


def test_abort_does_not_block_on_a_hung_stop(monkeypatch):
    monkeypatch.setattr(buildtest, "ABORT_STOP_TIMEOUT_SECONDS", 0.2)
    t = _bare_test()
    t._active_runner = SimpleNamespace(stop=lambda: time.sleep(5))
    start = time.monotonic()
    t._abort_build("a test")
    assert time.monotonic() - start < 1
    assert t._abort_waits.is_set()


# --- incremental output cleanup -----------------------------------------------


class _FakeUri:
    """Stand-in for URI.get_uri()'s result that records deletions."""

    deleted: list[str] = []

    def __init__(self, uri):
        """:param uri: The artifact URI string."""
        self.uri = uri

    def delete(self):
        """Record the deletion instead of deleting anything.

        :returns: True, as a successful URI.delete() does.
        """
        _FakeUri.deleted.append(self.uri)
        return True


def test_cleanup_pass_deletes_only_new_artifacts(monkeypatch):
    _FakeUri.deleted = []
    monkeypatch.setattr(buildtest.URI, "get_uri", staticmethod(_FakeUri))
    artifacts = [SimpleNamespace(uri="hf://a")]
    t = _bare_test()
    t.storage = SimpleNamespace(
        build_storage=SimpleNamespace(
            get_by_uuid=lambda _: [SimpleNamespace(uuid="b1")]
        ),
        artifact_registry=SimpleNamespace(get_by_where=lambda _: list(artifacts)),
    )
    t._output_cleanup.run("watchdog")
    artifacts.append(SimpleNamespace(uri="hf://b"))  # registered after that pass
    t._output_cleanup.run("teardown", final=True)
    assert _FakeUri.deleted == ["hf://a", "hf://b"]
    assert t._output_cleanup.run("watchdog") is False
