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

"""BuildRunnerProcess cancels its build-runner child instead of leaking it.

The child cannot outlive the watcher, so a stop() (including the default
``on_build_orchestrator_shutdown()``) must cancel it: SIGINT, which the build-runner
CLI turns into CANCELLED, escalating to SIGKILL after a grace period. The runner must
also return when the child exits on its own. ``subprocess.Popen`` is faked.
"""

import io
import signal
import subprocess
import threading
from types import SimpleNamespace

import pytest

from gbserver.buildrunner import buildrunnerprocess as brp_mod
from gbserver.buildrunner.buildrunnerprocess import BuildRunnerProcess
from gbserver.types.status import Status

pytestmark = pytest.mark.standalone

BUILD_ID = "11111111-2222-3333-4444-555555555555"


class _FakeChild:
    """Stand-in for the build-runner child process.

    Args:
        exits_on_sigint: whether the child exits when sent SIGINT (else it must be
            killed).
        exit_after_polls: if set, the child exits on its own after this many polls.
    """

    def __init__(self, exits_on_sigint: bool = True, exit_after_polls=None):
        self.pid = 4242
        self.returncode = None
        self.signals: list[int] = []
        self.killed = False
        self._exits_on_sigint = exits_on_sigint
        self._exit_after_polls = exit_after_polls
        self._polls = 0
        self.stdout = io.StringIO("")
        self.stderr = io.StringIO("")

    def poll(self):
        self._polls += 1
        if self._exit_after_polls is not None and self._polls >= self._exit_after_polls:
            self.returncode = 0
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None and timeout is not None:
            raise subprocess.TimeoutExpired("gbserver", timeout)
        return self.returncode

    def send_signal(self, sig):
        self.signals.append(sig)
        if self._exits_on_sigint:
            self.returncode = 0

    def kill(self):
        self.killed = True
        self.returncode = -9


def _install_fake_popen(monkeypatch, **child_kwargs) -> _FakeChild:
    """Make ``subprocess.Popen`` return one prepared fake child.

    The runner asserts ``isinstance(process, subprocess.Popen)``, so the patched
    ``Popen`` is a class whose constructor hands back that child.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        child_kwargs: passed to ``_FakeChild``.

    Returns:
        The fake child the runner will launch.
    """

    class _FakePopen(_FakeChild):
        child = None

        def __new__(cls, *args, **kwargs):
            return cls.child

        def __init__(self, *args, **kwargs):
            pass  # Already initialized below; Popen-style args are ignored.

    child = object.__new__(_FakePopen)
    _FakeChild.__init__(child, **child_kwargs)
    _FakePopen.child = child
    monkeypatch.setattr(brp_mod.subprocess, "Popen", _FakePopen)
    return child


def _process_runner() -> BuildRunnerProcess:
    """A BuildRunnerProcess with fake storage, bypassing its __init__ side effects."""
    runner = BuildRunnerProcess.__new__(BuildRunnerProcess)
    runner.stored_build = SimpleNamespace(uuid=BUILD_ID)
    runner.storage = SimpleNamespace(
        table_name_prefix="",
        build_storage=SimpleNamespace(get_by_uuid=lambda uuid: object()),
    )
    runner.gh_token = ""
    runner.workspace_dir = "/tmp"
    runner.monitoring_interval = 0.01
    runner.gh_api_endpoint = ""
    runner.is_running = False
    runner.is_stop_requested = False
    return runner


def test_cancel_child_sends_sigint_and_waits():
    """A child that honors SIGINT is waited on, not killed."""
    child = _FakeChild(exits_on_sigint=True)
    BuildRunnerProcess._cancel_child(child)  # type: ignore[arg-type]
    assert child.signals == [signal.SIGINT]
    assert not child.killed


def test_cancel_child_kills_after_grace(monkeypatch):
    """A child that ignores SIGINT is killed once the grace period runs out."""
    monkeypatch.setattr(brp_mod, "_CHILD_CANCEL_GRACE_S", 0.01)
    child = _FakeChild(exits_on_sigint=False)
    BuildRunnerProcess._cancel_child(child)  # type: ignore[arg-type]
    assert child.signals == [signal.SIGINT]
    assert child.killed


@pytest.mark.timeout(30)
def test_returns_when_child_exits_on_its_own(monkeypatch):
    """The runner returns when its build finishes; nothing is signalled."""
    child = _install_fake_popen(monkeypatch, exit_after_polls=3)
    runner = _process_runner()

    runner.start_and_wait()

    assert child.signals == [] and not child.killed
    assert not runner.is_running


@pytest.mark.timeout(30)
def test_stop_cancels_running_child(monkeypatch):
    """stop() from another thread SIGINTs the child and start_and_wait returns."""
    child = _install_fake_popen(monkeypatch, exits_on_sigint=True)
    runner = _process_runner()
    thread = threading.Thread(target=runner.start_and_wait, daemon=True)
    thread.start()
    while not runner.is_running and thread.is_alive():
        thread.join(timeout=0.01)

    runner.on_build_orchestrator_shutdown()  # default: stop()
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert child.signals == [signal.SIGINT]


def test_stop_before_start_marks_cancelled(monkeypatch):
    """A stop that wins the race with start never launches the child."""
    finalized = []
    monkeypatch.setattr(
        brp_mod, "finalize_build_status", lambda uuid, status: finalized.append(status)
    )
    monkeypatch.setattr(
        brp_mod.subprocess,
        "Popen",
        lambda **_kw: pytest.fail("child launched after stop"),
    )
    runner = _process_runner()
    runner.is_stop_requested = True

    runner.start_and_wait()

    assert finalized == [Status.CANCELLED]
