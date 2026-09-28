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

"""Last-resort reaping of leaked workload process groups.

Workloads run with ``start_new_session=True``, so they lead their own process group
and are not reached by signalling the server's group. The owning environment reaps
them on the normal paths; this registry is the fallback for when the owning *thread*
is what failed, so shutdown can still attempt a kill instead of leaking the tree.

These tests spawn real process groups -- the point is the signalling behaviour, which
a mock cannot demonstrate.
"""

import os
import signal
import subprocess
import sys
import time

import pytest

from gbserver.utils import spawned_groups

pytestmark = pytest.mark.standalone


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _spawn_group(seconds: int = 300) -> subprocess.Popen:
    """A session leader with a child, mirroring bash.launch_nohup's topology."""
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"import subprocess,sys;subprocess.run([sys.executable,'-c','import time;time.sleep({seconds})'])",
        ],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


@pytest.fixture(autouse=True)
def _clean_registry():
    """Keep the module-level registry from leaking between tests."""
    for pid in spawned_groups.tracked_pids():
        spawned_groups.unregister(pid)
    yield
    for pid in spawned_groups.tracked_pids():
        spawned_groups.unregister(pid)


class TestRegistry:
    def test_register_and_unregister(self):
        proc = _spawn_group()
        try:
            spawned_groups.register(proc.pid, "test")
            assert proc.pid in spawned_groups.tracked_pids()
            spawned_groups.unregister(proc.pid)
            assert proc.pid not in spawned_groups.tracked_pids()
        finally:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=10)

    def test_register_ignores_dead_pid(self):
        proc = _spawn_group(1)
        pid = proc.pid
        os.killpg(os.getpgid(pid), signal.SIGKILL)
        proc.wait(timeout=10)
        spawned_groups.register(pid, "already dead")
        assert pid not in spawned_groups.tracked_pids()

    def test_unregister_unknown_pid_is_a_noop(self):
        spawned_groups.unregister(999999)  # must not raise


class TestReapAll:
    def test_reaps_a_registered_group(self):
        """The whole point: a leaked group is actually killed."""
        proc = _spawn_group()
        spawned_groups.register(proc.pid, "leaked workload")
        try:
            survivors = spawned_groups.reap_all(grace_seconds=3)
            assert survivors == [], f"reap_all left survivors: {survivors}"
            deadline = time.monotonic() + 10
            while _alive(proc.pid) and time.monotonic() < deadline:
                time.sleep(0.1)
            assert not _alive(proc.pid), "session leader survived reaping"
        finally:
            if _alive(proc.pid):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass

    def test_registry_is_emptied_after_reaping(self):
        proc = _spawn_group()
        spawned_groups.register(proc.pid, "leaked")
        try:
            spawned_groups.reap_all(grace_seconds=2)
            assert spawned_groups.tracked_pids() == []
        finally:
            if _alive(proc.pid):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass

    def test_empty_registry_is_a_noop(self):
        assert spawned_groups.reap_all() == []

    def test_sigterm_is_tried_before_sigkill(self, monkeypatch):
        """A cooperative workload must get a chance to exit cleanly.

        Without this the SIGKILL escalation hides a missing SIGTERM: the group dies
        either way, so only the signal *order* distinguishes them.
        """
        proc = _spawn_group()
        real_pgid = os.getpgid(proc.pid)
        spawned_groups.register(proc.pid, "graceful")
        sent = []
        real_signal = spawned_groups._signal
        monkeypatch.setattr(
            spawned_groups,
            "_signal",
            lambda pgid, sig: (sent.append(sig), real_signal(pgid, sig))[1],
        )
        try:
            spawned_groups.reap_all(grace_seconds=5)
            assert signal.SIGTERM in sent, f"SIGTERM was never sent (sent={sent})"
            assert sent[0] == signal.SIGTERM, f"first signal was not SIGTERM: {sent}"
            # A process that dies on SIGTERM must not also be SIGKILLed.
            assert (
                signal.SIGKILL not in sent
            ), f"escalated to SIGKILL despite a clean SIGTERM exit: {sent}"
        finally:
            monkeypatch.undo()
            if _alive(proc.pid):
                os.killpg(real_pgid, signal.SIGKILL)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass

    def test_sigkill_escalation_when_sigterm_is_ignored(self, monkeypatch):
        """A workload that ignores SIGTERM is still killed."""
        # The child announces on stdout once SIGTERM is ignored, so the test waits on
        # that instead of a sleep: under parallel load a fixed sleep can race the
        # handler install, SIGTERM then kills it, and no escalation is needed.
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import signal,sys,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                "sys.stdout.write(chr(10).join(['ready','']));sys.stdout.flush();time.sleep(300)",
            ],
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        real_pgid = os.getpgid(proc.pid)
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "ready", "child never armed SIG_IGN"
        spawned_groups.register(proc.pid, "ignores sigterm")
        sent = []
        real_signal = spawned_groups._signal
        monkeypatch.setattr(
            spawned_groups,
            "_signal",
            lambda pgid, sig: (sent.append(sig), real_signal(pgid, sig))[1],
        )
        try:
            survivors = spawned_groups.reap_all(grace_seconds=2)
            assert signal.SIGKILL in sent, f"never escalated to SIGKILL: {sent}"
            assert survivors == [], f"SIGKILL failed to reap: {survivors}"
        finally:
            monkeypatch.undo()
            if _alive(proc.pid):
                os.killpg(real_pgid, signal.SIGKILL)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass

    def test_unregistered_group_is_left_alone(self):
        """Only what we registered may be signalled."""
        proc = _spawn_group()
        try:
            assert spawned_groups.reap_all(grace_seconds=1) == []
            assert _alive(proc.pid), "reap_all killed a group it never registered"
        finally:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=10)

    def test_skips_pid_that_is_no_longer_its_own_group_leader(self, monkeypatch):
        """Guards against killing a recycled pid's unrelated group."""
        proc = _spawn_group()
        real_pgid = os.getpgid(proc.pid)
        spawned_groups.register(proc.pid, "will look recycled")
        killed = []
        monkeypatch.setattr(
            spawned_groups, "_signal", lambda pgid, sig: killed.append((pgid, sig))
        )
        # Report a different group for this pid, as a recycled pid would. Patched on
        # the module's os reference only for the duration of reap_all.
        monkeypatch.setattr(spawned_groups.os, "getpgid", lambda pid: pid + 1)
        try:
            assert spawned_groups.reap_all(grace_seconds=1) == []
            assert killed == [], "signalled a pid that was no longer its group leader"
        finally:
            monkeypatch.undo()
            os.killpg(real_pgid, signal.SIGKILL)
            proc.wait(timeout=10)

    def test_reports_survivors_rather_than_blocking(self, monkeypatch):
        """An unkillable group is given up on, not waited for indefinitely."""
        proc = _spawn_group()
        real_pgid = os.getpgid(proc.pid)
        spawned_groups.register(proc.pid, "pretend-unkillable")
        # Swallow every signal so the group cannot die, and skip the zombie collection
        # that would otherwise report it gone -- this is the "survives SIGKILL" path.
        monkeypatch.setattr(spawned_groups, "_signal", lambda pgid, sig: None)
        monkeypatch.setattr(spawned_groups, "_collect_zombie", lambda pid: False)
        try:
            started = time.monotonic()
            survivors = spawned_groups.reap_all(grace_seconds=1)
            elapsed = time.monotonic() - started
            assert survivors == [proc.pid], f"expected a reported survivor: {survivors}"
            assert (
                elapsed < 10
            ), f"reap_all blocked for {elapsed:.1f}s instead of giving up"
        finally:
            monkeypatch.undo()
            os.killpg(real_pgid, signal.SIGKILL)
            proc.wait(timeout=10)


class TestBashIntegration:
    def test_launch_registers_and_cleanup_unregisters(self):
        """bash.py must register on launch and unregister once it reaps itself."""
        import inspect

        from gbserver.environment import bash

        launch_src = inspect.getsource(bash.Bash.launch_nohup)
        cleanup_src = inspect.getsource(bash.Bash.cleanup_nohup)
        assert "spawned_groups.register" in launch_src
        assert "spawned_groups.unregister" in cleanup_src


class TestShutdownCallsReaper:
    def test_watcher_shutdown_reaps(self):
        import inspect

        from gbserver.buildwatcher import buildwatcher

        src = inspect.getsource(buildwatcher.BuildWatcher)
        assert "spawned_groups.reap_all" in src

    def test_cli_termination_reaps(self):
        import inspect

        from gbserver.commands import command_build_runner

        src = inspect.getsource(command_build_runner.run_build_handling_signals)
        assert "spawned_groups.reap_all" in src
