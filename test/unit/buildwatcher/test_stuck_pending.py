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

"""BuildWatcher stuck-PENDING re-arm (__redispatch_stuck_pending_builds).

A build whose runner never started, or died before reporting RUNNING, sits PENDING
with no live runner thread. Because dispatch is gated on the seen-list
(active_pending_builds), once recorded there it is never retried. The watchdog drops
such a build from the seen-list so the next poll re-dispatches it; it never fails the
build, and it leaves a build with a live runner alone.
"""

import datetime

import pytest

from gbserver.buildwatcher import buildwatcher as bw_mod
from gbserver.buildwatcher.buildwatcher import BuildWatcher
from gbserver.types.status import Status

pytestmark = pytest.mark.standalone

BUILD_ID = "11111111-2222-3333-4444-555555555555"


def _watcher(monkeypatch, stuck_timeout: int = 900) -> BuildWatcher:
    monkeypatch.setattr(
        BuildWatcher, "_BuildWatcher__reload_config_file", lambda *a, **k: None
    )
    monkeypatch.setattr(bw_mod, "get_admin_storage", lambda: object())
    monkeypatch.setattr(bw_mod, "create_temp_subdir", lambda p: bw_mod.Path("."))
    w = BuildWatcher(gh_token="", all_build_space_uri=None)
    w.config.stuck_build_timeout_seconds = stuck_timeout
    return w


class _Build:
    def __init__(self, uuid: str = BUILD_ID, age_seconds: float = 0.0):
        self.uuid = uuid
        self.updated_time = datetime.datetime.now(
            datetime.timezone.utc
        ) - datetime.timedelta(seconds=age_seconds)


class _Thread:
    """Stand-in whose is_alive() is fixed (no real thread)."""

    def __init__(self, alive: bool):
        self._alive = alive

    def is_alive(self) -> bool:
        return self._alive


def _patch_pending(monkeypatch, builds):
    monkeypatch.setattr(
        BuildWatcher,
        "_BuildWatcher__get_builds_matching_status",
        lambda self, s: list(builds) if s == Status.PENDING else [],
    )


class TestRedispatchStuckPending:
    def test_stale_no_runner_is_rearmed(self, monkeypatch):
        w = _watcher(monkeypatch, stuck_timeout=900)
        w.active_pending_builds = [BUILD_ID]  # already "seen" -> normally never retried
        _patch_pending(
            monkeypatch, [_Build(age_seconds=1000)]
        )  # past timeout, no thread
        w._BuildWatcher__redispatch_stuck_pending_builds()
        # dropped from the seen-list so the next poll re-dispatches it
        assert BUILD_ID not in w.active_pending_builds

    def test_fresh_no_runner_is_left_alone(self, monkeypatch):
        w = _watcher(monkeypatch, stuck_timeout=900)
        w.active_pending_builds = [BUILD_ID]
        _patch_pending(monkeypatch, [_Build(age_seconds=0)])  # within timeout
        w._BuildWatcher__redispatch_stuck_pending_builds()
        assert BUILD_ID in w.active_pending_builds, "a fresh build was re-armed"

    def test_live_runner_is_left_alone(self, monkeypatch):
        w = _watcher(monkeypatch, stuck_timeout=900)
        w.active_pending_builds = [BUILD_ID]
        w.build_threads[BUILD_ID] = _Thread(alive=True)
        _patch_pending(monkeypatch, [_Build(age_seconds=1000)])  # old but has a runner
        w._BuildWatcher__redispatch_stuck_pending_builds()
        assert BUILD_ID in w.active_pending_builds, "a live-runner build was re-armed"

    def test_dead_thread_stale_is_rearmed(self, monkeypatch):
        # A dead-but-uncollected thread entry reads as no live runner.
        w = _watcher(monkeypatch, stuck_timeout=900)
        w.active_pending_builds = [BUILD_ID]
        w.build_threads[BUILD_ID] = _Thread(alive=False)
        _patch_pending(monkeypatch, [_Build(age_seconds=1000)])
        w._BuildWatcher__redispatch_stuck_pending_builds()
        assert BUILD_ID not in w.active_pending_builds

    def test_not_in_seen_list_is_noop(self, monkeypatch):
        # Post-restart the seen-list is empty; the next poll dispatches normally, and the
        # watchdog must not error when the id isn't present.
        w = _watcher(monkeypatch, stuck_timeout=900)
        assert w.active_pending_builds == []
        _patch_pending(monkeypatch, [_Build(age_seconds=1000)])
        w._BuildWatcher__redispatch_stuck_pending_builds()  # must not raise
        assert w.active_pending_builds == []
