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

"""Signal handling for the ``gbserver build-runner`` CLI.

``run_build_handling_signals`` runs ``BuildRunner.start_and_wait`` on a background
thread and, from the main thread, reacts to termination signals:

* SIGINT (Ctrl+C) -> ``build_runner.stop()``      -> build marked CANCELLED
* SIGTERM          -> ``build_runner.stop_and_fail()`` -> build marked FAILED

These tests drive the glue with a fake runner (no real build) that delivers the
signal to this process from inside ``start_and_wait`` — so the handler installed by
``run_build_handling_signals`` is guaranteed to be active when the signal arrives.
"""

import os
import signal
import threading

import pytest

from gbserver.commands.command_build_runner import run_build_handling_signals


class _FakeBuildRunner:
    """Minimal stand-in for BuildRunner that self-delivers a signal.

    ``start_and_wait`` (invoked on the background thread by
    ``run_build_handling_signals``) sends ``signum`` to this process, then blocks
    until ``stop``/``stop_and_fail`` releases it. This mimics a build running until
    the CLI cancels/fails it in response to the signal.
    """

    def __init__(self, signum: int) -> None:
        self._signum = signum
        self._released = threading.Event()
        self.stop_called = False
        self.stop_and_fail_called = False
        self.failure_reason: str | None = None

    def start_and_wait(self) -> None:
        # Handlers are installed before this thread starts, so the signal is
        # delivered to the main thread while it waits in the join loop.
        os.kill(os.getpid(), self._signum)
        # Bound the wait so a regression can't hang the suite indefinitely.
        assert self._released.wait(timeout=10), "runner was never stopped"

    def stop(self) -> None:
        self.stop_called = True
        self._released.set()

    def stop_and_fail(self, failure_reason: str) -> None:
        self.stop_and_fail_called = True
        self.failure_reason = failure_reason
        self._released.set()


@pytest.mark.timeout(30)
def test_sigint_cancels_build() -> None:
    """Ctrl+C (SIGINT) drives BuildRunner.stop() and not stop_and_fail()."""
    runner = _FakeBuildRunner(signal.SIGINT)

    run_build_handling_signals(runner)  # type: ignore[arg-type]

    assert runner.stop_called is True
    assert runner.stop_and_fail_called is False


@pytest.mark.timeout(30)
def test_sigterm_fails_build() -> None:
    """SIGTERM drives BuildRunner.stop_and_fail() with a reason, not stop()."""
    runner = _FakeBuildRunner(signal.SIGTERM)

    run_build_handling_signals(runner)  # type: ignore[arg-type]

    assert runner.stop_and_fail_called is True
    assert runner.stop_called is False
    assert runner.failure_reason == "Build runner received SIGTERM"


@pytest.mark.timeout(30)
def test_default_signal_handlers_restored() -> None:
    """Original SIGINT/SIGTERM handlers are restored after the helper returns."""
    before_int = signal.getsignal(signal.SIGINT)
    before_term = signal.getsignal(signal.SIGTERM)

    run_build_handling_signals(_FakeBuildRunner(signal.SIGINT))  # type: ignore[arg-type]

    assert signal.getsignal(signal.SIGINT) is before_int
    assert signal.getsignal(signal.SIGTERM) is before_term


class _RaisingBuildRunner:
    """Stand-in whose ``start_and_wait`` fails (e.g. a storage error)."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def start_and_wait(self) -> None:
        raise self._exc


@pytest.mark.timeout(30)
def test_build_failure_propagates() -> None:
    """A failure in start_and_wait is re-raised on the main thread.

    The build runs on a background thread; without re-raising, the exception would
    hit threading.excepthook, join() would return normally, and the CLI would exit
    0 on a build that never ran.
    """
    boom = ValueError("Storage of build failed without error.")
    before_int = signal.getsignal(signal.SIGINT)
    before_term = signal.getsignal(signal.SIGTERM)

    with pytest.raises(ValueError) as excinfo:
        run_build_handling_signals(_RaisingBuildRunner(boom))  # type: ignore[arg-type]

    assert excinfo.value is boom  # re-raised verbatim
    # Handlers are still restored even on the failure path.
    assert signal.getsignal(signal.SIGINT) is before_int
    assert signal.getsignal(signal.SIGTERM) is before_term


class _HangingBuildRunner:
    """Stand-in whose graceful shutdown hangs, and which sends a second signal.

    ``start_and_wait`` delivers the first signal then blocks (never released by
    stop/stop_and_fail), simulating a shutdown that does not complete. The stop
    handlers deliver a *second* signal so the test can exercise the repeat-signal
    force-exit path.
    """

    def __init__(self, signum: int) -> None:
        self._signum = signum
        self.released = threading.Event()

    def start_and_wait(self) -> None:
        os.kill(os.getpid(), self._signum)  # first signal
        self.released.wait(timeout=10)  # blocks; graceful shutdown "hangs"

    def _resignal(self) -> None:
        os.kill(os.getpid(), self._signum)  # second signal -> force exit

    def stop(self) -> None:
        self._resignal()

    def stop_and_fail(self, failure_reason: str) -> None:
        self._resignal()


@pytest.mark.timeout(30)
@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_repeat_signal_force_exits(monkeypatch, signum) -> None:
    """A second termination signal hard-exits instead of being swallowed.

    Graceful shutdown can hang; without this, repeat Ctrl+C would be ignored and
    the process left unkillable. os._exit is patched to observe the call (a real
    one would kill the test process).
    """
    exits: list[int] = []

    class _ForceExit(BaseException):
        pass

    def _fake_exit(code: int):
        exits.append(code)
        raise _ForceExit

    monkeypatch.setattr(os, "_exit", _fake_exit)
    runner = _HangingBuildRunner(signum)
    try:
        with pytest.raises(_ForceExit):
            run_build_handling_signals(runner)  # type: ignore[arg-type]
    finally:
        runner.released.set()  # let the daemon thread finish promptly

    assert exits == [128 + signum]
