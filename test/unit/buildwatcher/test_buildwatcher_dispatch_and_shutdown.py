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

"""BuildWatcher dispatch idempotence and bounded shutdown.

Two defects, both observed as the same CI flake (`test_cancel_reaps_workload_child`
failing on Python 3.11 while 3.12 passed the same commit):

* A build already running under this watcher could be dispatched a *second* time.
  The "already seen" lists only suppress a re-dispatch while the build still holds
  the dispatched status; a build that leaves PENDING and returns to it is dropped by
  the lists' garbage collection and looks new. The second runner relaunched the
  workload that the first one's cancellation had already reaped, so the test saw a
  live PID and blamed reaping.
* Shutdown and cancel joined threads with no timeout while holding
  ``_builds_lock`` -- the same lock the build threads need -- so a wedged thread
  hung the process with no diagnostic. In CI that is a job killed mid-run with no
  pytest summary.

These tests drive the private methods directly with stub threads/runners: the real
race needs 3.11 scheduling to lose, but the *guards* are deterministic.
"""

import threading
import time

import pytest

from gbserver.buildwatcher import buildwatcher as bw_mod
from gbserver.buildwatcher.buildwatcher import BuildWatcher

pytestmark = pytest.mark.standalone

BUILD_ID = "11111111-2222-3333-4444-555555555555"


class _StubRunner:
    """Minimal AbstractBuildRunner stand-in."""

    def __init__(self, fail_stop: bool = False):
        self.stopped = False
        self._fail_stop = fail_stop

    def stop(self):
        self.stopped = True
        if self._fail_stop:
            raise RuntimeError("stop() blew up")

    def start_and_wait(self):
        pass


def _watcher(monkeypatch) -> BuildWatcher:
    """A BuildWatcher with no storage, config file, or workspace side effects."""
    monkeypatch.setattr(
        BuildWatcher, "_BuildWatcher__reload_config_file", lambda *a, **k: None
    )
    monkeypatch.setattr(bw_mod, "get_admin_storage", lambda: object())
    monkeypatch.setattr(bw_mod, "create_temp_subdir", lambda p: bw_mod.Path("."))
    return BuildWatcher(gh_token="", all_build_space_uri=None)


class _Build:
    """Stand-in for StoredBuild (only uuid/name are read on this path)."""

    def __init__(self, uuid: str, name: str = "test"):
        self.uuid = uuid
        self.name = name


class TestDispatchIsIdempotent:
    def test_second_dispatch_is_refused_while_first_is_alive(self, monkeypatch):
        """The regression: one build id must never get two runners."""
        w = _watcher(monkeypatch)
        created = []

        def _create(build):
            created.append(build.uuid)
            return _StubRunner()

        monkeypatch.setattr(
            BuildWatcher,
            "_BuildWatcher__create_build_runner",
            lambda self, b: _create(b),
        )
        release = threading.Event()
        # A live thread standing in for an in-flight build.
        alive = threading.Thread(target=release.wait, daemon=True)
        alive.start()
        w.build_threads[BUILD_ID] = alive
        w.build_runners[BUILD_ID] = _StubRunner()

        try:
            w._BuildWatcher__start_build(_Build(BUILD_ID))
            assert created == [], "a second runner was created for a live build"
            assert w.build_threads[BUILD_ID] is alive, "the live thread was replaced"
        finally:
            release.set()
            alive.join(timeout=5)

    def test_dispatch_proceeds_when_previous_thread_is_dead(self, monkeypatch):
        """A finished build may legitimately be dispatched again (e.g. a restart)."""
        w = _watcher(monkeypatch)
        monkeypatch.setattr(
            BuildWatcher,
            "_BuildWatcher__create_build_runner",
            lambda self, b: _StubRunner(),
        )
        dead = threading.Thread(target=lambda: None)
        dead.start()
        dead.join()
        w.build_threads[BUILD_ID] = dead

        w._BuildWatcher__start_build(_Build(BUILD_ID))
        assert w.build_threads[BUILD_ID] is not dead
        w.build_threads[BUILD_ID].join(timeout=5)

    def test_dispatch_of_unknown_build_proceeds(self, monkeypatch):
        w = _watcher(monkeypatch)
        monkeypatch.setattr(
            BuildWatcher,
            "_BuildWatcher__create_build_runner",
            lambda self, b: _StubRunner(),
        )
        w._BuildWatcher__start_build(_Build(BUILD_ID))
        assert BUILD_ID in w.build_threads
        w.build_threads[BUILD_ID].join(timeout=5)


class TestShutdownIsBounded:
    def test_wedged_build_thread_does_not_hang_shutdown(self, monkeypatch):
        """A thread that never exits must not block shutdown forever."""
        monkeypatch.setattr(bw_mod, "_SHUTDOWN_JOIN_TIMEOUT_S", 0.2)
        w = _watcher(monkeypatch)
        release = threading.Event()
        wedged = threading.Thread(target=release.wait, daemon=True)
        wedged.start()
        runner = _StubRunner()
        w.build_threads[BUILD_ID] = wedged
        w.build_runners[BUILD_ID] = runner
        w.worker_thread = threading.Thread(target=lambda: None)
        w.worker_thread.start()

        try:
            started = time.monotonic()
            w._BuildWatcher__wait_for_completion()
            elapsed = time.monotonic() - started
            assert elapsed < 30, f"shutdown took {elapsed:.1f}s despite the bound"
            assert runner.stopped, "the runner was never asked to stop"
        finally:
            release.set()
            wedged.join(timeout=5)

    def test_one_failing_runner_does_not_strand_the_others(self, monkeypatch):
        monkeypatch.setattr(bw_mod, "_SHUTDOWN_JOIN_TIMEOUT_S", 0.2)
        w = _watcher(monkeypatch)
        bad, good = _StubRunner(fail_stop=True), _StubRunner()
        w.build_runners["bad"] = bad
        w.build_runners["good"] = good
        w.worker_thread = threading.Thread(target=lambda: None)
        w.worker_thread.start()

        w._BuildWatcher__wait_for_completion()
        assert bad.stopped and good.stopped, "a raising stop() stranded the rest"

    def test_join_does_not_hold_the_builds_lock(self, monkeypatch):
        """Joining under _builds_lock deadlocks the threads that need that lock."""
        monkeypatch.setattr(bw_mod, "_SHUTDOWN_JOIN_TIMEOUT_S", 5)
        w = _watcher(monkeypatch)
        acquired = threading.Event()
        release = threading.Event()

        def _needs_the_lock():
            # A build thread reporting status takes _builds_lock; if shutdown holds
            # it across the join, this never runs and both sides wait forever.
            with w._builds_lock:
                acquired.set()
            release.wait()

        t = threading.Thread(target=_needs_the_lock, daemon=True)
        w.build_threads[BUILD_ID] = t
        w.build_runners[BUILD_ID] = _StubRunner()
        w.worker_thread = threading.Thread(target=lambda: None)
        w.worker_thread.start()
        t.start()

        try:
            w._BuildWatcher__wait_for_completion()
            assert acquired.wait(
                timeout=5
            ), "shutdown held _builds_lock across the join"
        finally:
            release.set()
            t.join(timeout=5)


class TestCancelIsBounded:
    def test_cancel_does_not_hang_on_a_wedged_thread(self, monkeypatch):
        monkeypatch.setattr(bw_mod, "_SHUTDOWN_JOIN_TIMEOUT_S", 0.2)
        w = _watcher(monkeypatch)
        release = threading.Event()
        wedged = threading.Thread(target=release.wait, daemon=True)
        wedged.start()
        runner = _StubRunner()
        w.build_threads[BUILD_ID] = wedged
        w.build_runners[BUILD_ID] = runner

        try:
            started = time.monotonic()
            w._BuildWatcher__process_cancel_requested_build(_Build(BUILD_ID))
            elapsed = time.monotonic() - started
            assert elapsed < 30, f"cancel took {elapsed:.1f}s despite the bound"
            assert runner.stopped
            # Tracking is dropped even though the thread never finished, so the
            # build cannot be treated as live forever.
            assert BUILD_ID not in w.build_runners
            assert BUILD_ID not in w.build_threads
        finally:
            release.set()
            wedged.join(timeout=5)

    def test_thread_stays_visible_while_joining(self, monkeypatch):
        """Tracking must not vanish mid-join, or a re-dispatch slips through.

        __start_build treats a live build_threads entry as "already running". If
        cancel drops that entry before the join finishes, a poll in that window sees
        a build whose CANCELLED status is not yet persisted and starts a second
        runner -- the very bug this cancel path is meant to avoid.
        """
        monkeypatch.setattr(bw_mod, "_SHUTDOWN_JOIN_TIMEOUT_S", 5)
        w = _watcher(monkeypatch)
        release = threading.Event()
        visible_during_join = []

        def _workload():
            # Runs while cancel is joining us; record what a concurrent dispatch
            # check would see at this moment.
            visible_during_join.append(BUILD_ID in w.build_threads)
            release.wait(timeout=5)

        t = threading.Thread(target=_workload, daemon=True)
        w.build_threads[BUILD_ID] = t
        w.build_runners[BUILD_ID] = _StubRunner()
        t.start()
        time.sleep(0.2)  # let the workload record its observation
        release.set()

        w._BuildWatcher__process_cancel_requested_build(_Build(BUILD_ID))
        assert visible_during_join == [
            True
        ], "build_threads entry disappeared while the cancel was still joining"
        # ...and is cleaned up once the join completes.
        assert BUILD_ID not in w.build_threads
        t.join(timeout=5)
