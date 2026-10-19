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

"""Unit tests for libgbtest.abort_cleanup: cleanup on abnormal test aborts."""

import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import pytest
from libgbtest import abort_cleanup
from libgbtest.abort_cleanup import (
    CleanupWatchdog,
    SerializedCleanup,
    TimeoutDeadline,
    bounded_timeout,
    wait_deadline,
)

# These tests time real waits and deadlines, so opt out of conftest's mock-mode
# time.sleep, which shortens any sub-second sleep to 0.01s.
pytestmark = pytest.mark.live("time")

# --- SerializedCleanup ------------------------------------------------------


def test_serialized_cleanup_runs_until_closed():
    calls = []
    cleanup = SerializedCleanup(lambda: calls.append(1))
    assert cleanup.run("watchdog") is True
    assert cleanup.run("teardown", final=True) is True
    assert cleanup.run("watchdog") is False
    assert calls == [1, 1]


def test_serialized_cleanup_final_run_closes_even_when_fn_raises():
    def boom():
        raise RuntimeError("boom")

    cleanup = SerializedCleanup(boom)
    with pytest.raises(RuntimeError):
        cleanup.run("teardown", final=True)
    assert cleanup.run("watchdog") is False


def test_serialized_cleanup_waits_for_a_run_in_progress():
    # Teardown must not proceed (and release storage) while the watchdog's pass
    # is still running: its own run blocks until that pass has finished.
    in_watchdog_pass = threading.Event()
    release = threading.Event()
    order = []

    def fn():
        if not in_watchdog_pass.is_set():
            in_watchdog_pass.set()
            release.wait(2)
            order.append("watchdog done")
        else:
            order.append("teardown pass")

    cleanup = SerializedCleanup(fn)
    watchdog = threading.Thread(target=cleanup.run, args=("watchdog",))
    watchdog.start()
    assert in_watchdog_pass.wait(2)
    teardown = threading.Thread(target=cleanup.run, args=("teardown", True))
    teardown.start()
    time.sleep(0.2)
    assert order == []  # teardown is waiting on the lock
    release.set()
    watchdog.join(2)
    teardown.join(2)
    assert order == ["watchdog done", "teardown pass"]


# --- wait_deadline / bounded_timeout ----------------------------------------


def _deadline(remaining: float, method: str = "thread") -> TimeoutDeadline:
    """A pytest-timeout deadline that fires ``remaining`` seconds from now.

    :param remaining: Seconds until the simulated timeout fires.
    :param method: The pytest-timeout method it simulates.
    :returns: The deadline.
    """
    return TimeoutDeadline(
        kill_at=time.monotonic() + remaining,
        method=method,
        disable_debugger_detection=False,
    )


def test_wait_deadline_untimed_is_none():
    assert wait_deadline(None, margin_seconds=120) is None
    assert bounded_timeout(1800, None) == 1800


def test_bounded_timeout_leaves_short_timeout_alone():
    deadline = wait_deadline(_deadline(1200), margin_seconds=120)
    assert bounded_timeout(600, deadline) == 600


def test_bounded_timeout_ends_before_the_kill():
    deadline = wait_deadline(_deadline(1200), margin_seconds=120)
    assert 1070 <= bounded_timeout(1800, deadline) <= 1080


def test_bounded_timeout_shares_one_deadline_across_waits():
    # Two waits in a row cannot add up past the deadline: the second only gets
    # what the first left.
    deadline = time.monotonic() + 0.3
    first = bounded_timeout(1800, deadline)
    time.sleep(first)
    assert bounded_timeout(1800, deadline) == pytest.approx(0, abs=0.05)


def test_bounded_timeout_is_never_negative():
    assert bounded_timeout(60, time.monotonic() - 5) == 0


# --- CleanupWatchdog --------------------------------------------------------


def _watchdog(deadline, abort, cleanup):
    """A watchdog that aborts 0.4s and cleans up 0.2s before ``deadline``.

    :param deadline: The simulated pytest-timeout deadline, or None.
    :param abort: The abort stage's callable.
    :param cleanup: The cleanup stage's callable.
    :returns: The (unstarted) watchdog.
    """
    return CleanupWatchdog(
        deadline,
        abort=abort,
        cleanup=cleanup,
        abort_lead_seconds=0.4,
        cleanup_lead_seconds=0.2,
    )


def test_watchdog_aborts_then_cleans_up_before_the_kill():
    order = []
    done = threading.Event()
    watchdog = _watchdog(
        _deadline(0.6),
        abort=lambda: order.append("abort"),
        cleanup=lambda: (order.append("cleanup"), done.set()),
    )
    assert watchdog.start() is True
    assert done.wait(2)
    assert order == ["abort", "cleanup"]


def test_watchdog_blocked_abort_does_not_delay_cleanup():
    unblock = threading.Event()
    cleaned = threading.Event()
    watchdog = _watchdog(
        _deadline(0.6), abort=lambda: unblock.wait(5), cleanup=cleaned.set
    )
    watchdog.start()
    try:
        assert cleaned.wait(2)
    finally:
        unblock.set()


def test_watchdog_cancel_prevents_firing():
    fired = threading.Event()
    watchdog = _watchdog(_deadline(0.6), abort=fired.set, cleanup=fired.set)
    watchdog.start()
    watchdog.cancel()
    assert not fired.wait(0.8)


def test_watchdog_not_armed_for_signal_method():
    watchdog = _watchdog(_deadline(10, "signal"), lambda: None, lambda: None)
    assert watchdog.start() is False


def test_watchdog_not_armed_without_deadline():
    assert _watchdog(None, lambda: None, lambda: None).start() is False


def test_watchdog_does_nothing_under_debugger(monkeypatch):
    # pytest-timeout does not kill a test being debugged, so leave it alone.
    monkeypatch.setattr(abort_cleanup, "_is_debugging", lambda: True)
    fired = threading.Event()
    watchdog = _watchdog(_deadline(0.5), abort=fired.set, cleanup=fired.set)
    watchdog.start()
    assert not fired.wait(0.8)


def test_is_debugging_is_false_without_pytest_timeout(monkeypatch):
    # A None entry in sys.modules makes `import pytest_timeout` raise ImportError.
    monkeypatch.setitem(sys.modules, "pytest_timeout", None)
    assert abort_cleanup._is_debugging() is False


def test_watchdog_runs_without_pytest_timeout(monkeypatch):
    # The stages must still run when pytest-timeout is not installed.
    monkeypatch.setitem(sys.modules, "pytest_timeout", None)
    done = threading.Event()
    watchdog = _watchdog(_deadline(0.6), abort=lambda: None, cleanup=done.set)
    watchdog.start()
    assert done.wait(2)


# --- termination signals ----------------------------------------------------


@pytest.fixture
def default_termination_handlers():
    """Put SIGTERM / SIGHUP at their defaults for a test, then restore them."""
    saved = {sig: signal.getsignal(sig) for sig in abort_cleanup.TERMINATION_SIGNALS}
    for sig in saved:
        signal.signal(sig, signal.SIG_DFL)
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


def test_map_and_restore_signal_handlers(default_termination_handlers):
    previous = abort_cleanup.map_termination_signals_to_sigint()
    assert set(previous) == set(abort_cleanup.TERMINATION_SIGNALS)
    for sig in abort_cleanup.TERMINATION_SIGNALS:
        assert signal.getsignal(sig) is signal.default_int_handler
    abort_cleanup.restore_signal_handlers(previous)
    for sig in abort_cleanup.TERMINATION_SIGNALS:
        assert signal.getsignal(sig) is signal.SIG_DFL


def test_map_leaves_ignored_signal_alone(default_termination_handlers):
    # nohup ignores SIGHUP; mapping it would abort the run when the terminal closes.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    previous = abort_cleanup.map_termination_signals_to_sigint()
    try:
        assert signal.SIGHUP not in previous
        assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
        assert signal.getsignal(signal.SIGTERM) is signal.default_int_handler
    finally:
        abort_cleanup.restore_signal_handlers(previous)


_SIGNAL_CONFTEST = """\
from libgbtest import abort_cleanup

def pytest_configure(config):
    config._prev_handlers = abort_cleanup.map_termination_signals_to_sigint()

def pytest_unconfigure(config):
    abort_cleanup.restore_signal_handlers(config._prev_handlers)
"""

_SIGNAL_TEST = """\
import os, time
from pathlib import Path

MARKER = Path(os.environ["MARKER_DIR"])

class TestSleeps:
    def teardown_method(self, method):
        (MARKER / "teardown").write_text("ran")
        if os.environ.get("HANG_TEARDOWN"):
            time.sleep(60)

    def test_sleep(self):
        (MARKER / "started").write_text("yes")
        time.sleep(60)
"""


def _start_pytest(
    tmp_path: Path,
    conftest: str,
    test_source: str,
    extra_args: Sequence[str] = (),
    extra_env: Optional[dict[str, str]] = None,
    **popen_kwargs: Any,
) -> subprocess.Popen:
    """Start a bare pytest run of a one-file test session in ``tmp_path``.

    The session sees only libgbtest (not the repo's conftest), and the test can
    write marker files to ``$MARKER_DIR`` (``tmp_path``) for the caller to check.

    :param tmp_path: Directory for the session's files and markers.
    :param conftest: Source of the session's conftest.py.
    :param test_source: Source of its single test file, test_session.py.
    :param extra_args: More pytest arguments.
    :param extra_env: More environment variables for the run.
    :param popen_kwargs: Overrides for ``subprocess.Popen`` (output is discarded
        by default).
    :returns: The running pytest process.
    """
    (tmp_path / "conftest.py").write_text(conftest)
    test_file = tmp_path / "test_session.py"
    test_file.write_text(test_source)
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(Path(abort_cleanup.__file__).resolve().parents[1]),
        "MARKER_DIR": str(tmp_path),
        **(extra_env or {}),
    }
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    cmd += ["-p", "no:randomly", "--rootdir", str(tmp_path), *extra_args]
    popen_kwargs.setdefault("stdout", subprocess.DEVNULL)
    popen_kwargs.setdefault("stderr", subprocess.DEVNULL)
    return subprocess.Popen(
        [*cmd, str(test_file)], cwd=tmp_path, env=env, **popen_kwargs
    )


def _run_and_signal(
    tmp_path: Path, signum: int, extra_args: list[str], hang_teardown: bool = False
) -> int:
    """Run a sleeping pytest session and signal it once its test has started.

    :param tmp_path: Directory for the session's files and markers.
    :param signum: The signal to send.
    :param extra_args: More pytest arguments (e.g. ``-n 1`` for xdist).
    :param hang_teardown: Make teardown hang and send ``signum`` again once it
        starts, to check that the second signal still ends the run.
    :returns: The pytest process's exit code.
    """
    proc = _start_pytest(
        tmp_path,
        _SIGNAL_CONFTEST,
        _SIGNAL_TEST,
        extra_args,
        extra_env={"HANG_TEARDOWN": "1"} if hang_teardown else None,
    )
    try:
        _wait_for(tmp_path / "started", proc)
        proc.send_signal(signum)
        if hang_teardown:
            _wait_for(tmp_path / "teardown", proc)
            proc.send_signal(signum)
        return proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()


def _wait_for(path: Path, proc: subprocess.Popen, timeout: float = 30) -> None:
    """Wait for a marker file to appear while ``proc`` is still running.

    :param path: The marker file.
    :param proc: The pytest process expected to write it.
    :param timeout: Seconds to wait.
    :raises AssertionError: If ``proc`` exits first or ``timeout`` passes.
    """
    end = time.monotonic() + timeout
    while not path.exists():
        assert proc.poll() is None, f"pytest exited before {path.name} appeared"
        assert time.monotonic() < end, f"timed out waiting for {path.name}"
        time.sleep(0.1)


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP])
def test_signal_runs_teardown(tmp_path, signum):
    rc = _run_and_signal(tmp_path, signum, [])
    assert (tmp_path / "teardown").exists()
    assert rc == pytest.ExitCode.INTERRUPTED


def test_signal_to_xdist_controller_runs_worker_teardown(tmp_path):
    pytest.importorskip("xdist")
    _run_and_signal(tmp_path, signal.SIGTERM, ["-n", "1"])
    assert (tmp_path / "teardown").exists()


def test_second_signal_ends_hung_teardown(tmp_path):
    # The second signal interrupts the hung teardown, so the run still exits.
    rc = _run_and_signal(tmp_path, signal.SIGTERM, [], hang_teardown=True)
    # An interrupt escaping pytest makes Python exit by re-raising SIGINT.
    assert rc in (pytest.ExitCode.INTERRUPTED, -signal.SIGINT)


def test_signal_tests_are_self_contained():
    # Guard: the subprocess tests above depend only on libgbtest.abort_cleanup
    # importing without the rest of the test tree (no gbserver imports).
    src = Path(abort_cleanup.__file__).read_text()
    assert "from gbserver" not in src and "import gbserver" not in src


_TIMEOUT_CONFTEST = """\
import pytest
from libgbtest import abort_cleanup

@pytest.hookimpl(tryfirst=True, optionalhook=True)
def pytest_timeout_set_timer(item, settings):
    abort_cleanup.record_timeout_deadline(item, settings)
"""

_TIMEOUT_TEST = """\
import os, time
from pathlib import Path
import pytest
from libgbtest.abort_cleanup import CleanupWatchdog, get_timeout_deadline

MARKER = Path(os.environ["MARKER_DIR"])

@pytest.fixture(autouse=True)
def watchdog(request):
    w = CleanupWatchdog(
        get_timeout_deadline(request.node),
        abort=lambda: None,  # this test cannot be aborted: it is hung
        cleanup=lambda: (MARKER / "cleaned").write_text("yes"),
        abort_lead_seconds=2.0,
        cleanup_lead_seconds=1.0,
    )
    w.start()
    yield
    w.cancel()

def test_hangs():
    time.sleep(60)
"""


def test_watchdog_cleans_up_before_pytest_timeout_kill(tmp_path):
    pytest.importorskip("pytest_timeout")
    proc = _start_pytest(
        tmp_path,
        _TIMEOUT_CONFTEST,
        _TIMEOUT_TEST,
        ["--timeout=3", "--timeout-method=thread"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        output, _ = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert b"Timeout" in output  # pytest-timeout did kill it
    assert (tmp_path / "cleaned").exists()


def test_nohup_run_survives_sighup(tmp_path):
    # Under nohup SIGHUP is ignored; the run must keep going rather than abort.
    proc = _start_pytest(
        tmp_path,
        _SIGNAL_CONFTEST,
        _SIGNAL_TEST,
        preexec_fn=lambda: signal.signal(signal.SIGHUP, signal.SIG_IGN),
    )
    try:
        _wait_for(tmp_path / "started", proc)
        proc.send_signal(signal.SIGHUP)
        time.sleep(1)
        assert proc.poll() is None, "a nohup'd run was aborted by SIGHUP"
        assert not (tmp_path / "teardown").exists()
    finally:
        proc.kill()
        proc.wait(timeout=30)
