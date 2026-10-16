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

"""Run test cleanup even when a test run is aborted abnormally.

A test's teardown is what deletes the remote resources it created (pushed hf://
datasets, buildrunner pods, ...). Two kinds of abort skip it:

* **Termination signals.** SIGTERM / SIGHUP (``kill``, a CI cancel, a closed
  terminal) terminate Python immediately by default. pytest only tears down on
  ``KeyboardInterrupt`` (SIGINT), so :func:`map_termination_signals_to_sigint`
  handles them the way SIGINT is handled.
* **pytest-timeout's thread method.** On timeout it dumps stacks and calls
  ``os._exit(1)``: no ``finally``, no teardown. conftest records each test's kill
  time (:func:`record_timeout_deadline`) so a test can end all its waits before
  it (:func:`wait_deadline`, :func:`bounded_timeout`), and arm a
  :class:`CleanupWatchdog` for a test hung where it cannot fail on its own (e.g.
  a call blocked with no read timeout). The watchdog first asks the test to
  abort, so it fails and tears down normally, and only cleans up itself if
  teardown has still not finished just before the kill.

SIGKILL cannot be handled in-process. This module must not depend on gbserver: the
signal tests run it in a bare pytest subprocess.
"""

import logging
import signal
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import pytest

logger = logging.getLogger(__name__)

TERMINATION_SIGNALS = (signal.SIGTERM, signal.SIGHUP)
"""Signals handled like SIGINT, so pytest runs teardown."""


def map_termination_signals_to_sigint() -> dict[int, Any]:
    """Handle SIGTERM / SIGHUP the way SIGINT is handled, so teardown runs.

    Maps them to Python's SIGINT handler (raise ``KeyboardInterrupt``), which
    pytest answers by tearing down and exiting interrupted. Call it from
    ``pytest_configure``, not a session fixture: under xdist the controller runs
    no tests, so a session fixture never reaches it, and a signalled controller
    would die at once and take its workers down without teardown.

    Only signals still at their default action are mapped: one already ignored
    (``nohup`` ignores SIGHUP) or handled by someone else is left as it is.
    Uses ``signal.default_int_handler`` itself, not SIGINT's current handler: a
    process started in the background by a non-interactive shell inherits SIGINT
    as ignored, and copying that would make SIGTERM ignored too. A no-op off the
    main thread, where Python cannot set handlers.

    :returns: The previous handlers of the mapped signals, for
        :func:`restore_signal_handlers`.
    """
    if threading.current_thread() is not threading.main_thread():
        return {}
    return {
        signum: signal.signal(signum, signal.default_int_handler)
        for signum in TERMINATION_SIGNALS
        if signal.getsignal(signum) is signal.SIG_DFL
    }


def restore_signal_handlers(previous: dict[int, Any]) -> None:
    """Reinstall handlers saved by :func:`map_termination_signals_to_sigint`.

    :param previous: Signal number to the handler to put back.
    """
    if threading.current_thread() is not threading.main_thread():
        return
    for signum, handler in previous.items():
        signal.signal(signum, handler)


@dataclass(frozen=True)
class TimeoutDeadline:
    """When pytest-timeout will act on the running test."""

    kill_at: float
    """``time.monotonic()`` value at which the timeout fires."""
    method: str
    """pytest-timeout method: ``thread`` (``os._exit``) or ``signal`` (raises)."""
    disable_debugger_detection: bool
    """If False, pytest-timeout does not fire while a debugger is attached."""

    def remaining(self) -> float:
        """Seconds until the timeout fires (negative once past)."""
        return self.kill_at - time.monotonic()


TIMEOUT_DEADLINE_KEY = pytest.StashKey[TimeoutDeadline]()
"""``item.stash`` key under which conftest records the test's deadline."""


def record_timeout_deadline(item: pytest.Item, settings: Any) -> None:
    """Record when pytest-timeout will fire for ``item``.

    Called from conftest's ``pytest_timeout_set_timer`` hook, which pytest-timeout
    calls as it starts the item's timer.

    :param item: The test item being timed.
    :param settings: pytest-timeout's settings for the item.
    """
    item.stash[TIMEOUT_DEADLINE_KEY] = TimeoutDeadline(
        kill_at=time.monotonic() + settings.timeout,
        method=settings.method,
        disable_debugger_detection=bool(settings.disable_debugger_detection),
    )


def get_timeout_deadline(item: pytest.Item) -> Optional[TimeoutDeadline]:
    """Look up the deadline :func:`record_timeout_deadline` stored for ``item``.

    :param item: The test item.
    :returns: Its deadline, or None when pytest-timeout set no timer for it
        (no timeout configured, or pytest-timeout not installed).
    """
    return item.stash.get(TIMEOUT_DEADLINE_KEY, None)


def wait_deadline(
    deadline: Optional[TimeoutDeadline], margin_seconds: float
) -> Optional[float]:
    """The absolute time by which all of a test's waits must end.

    A test whose waits outlast pytest-timeout never fails on its own: it is
    killed first, skipping its teardown. Ending every wait ``margin_seconds``
    before the kill lets it fail normally, however many waits it runs in a row.

    :param deadline: The test's pytest-timeout deadline, or None if untimed.
    :param margin_seconds: Time to leave for stopping the work and tearing down.
    :returns: A ``time.monotonic()`` value, or None when the test is untimed.
    """
    if deadline is None:
        return None
    return deadline.kill_at - margin_seconds


def bounded_timeout(timeout: float, deadline: Optional[float]) -> float:
    """Shorten ``timeout`` so a wait starting now ends by ``deadline``.

    :param timeout: The wait's own timeout, in seconds.
    :param deadline: A :func:`wait_deadline` value, or None for no bound.
    :returns: The timeout to use, never negative.
    """
    if deadline is None:
        return timeout
    return max(min(timeout, deadline - time.monotonic()), 0.0)


class SerializedCleanup:
    """Runs an idempotent cleanup from several threads, one run at a time.

    The test's teardown and the :class:`CleanupWatchdog` both run it. The lock is
    held for the whole run, so a teardown that arrives while the watchdog is part
    way through waits for it, then runs its own pass for whatever was created
    since. The cleanup function must therefore skip what it already deleted. The
    final run (teardown's) closes it, so the watchdog cannot touch resources the
    rest of teardown releases afterwards.
    """

    def __init__(self, fn: Callable[[], None]):
        """:param fn: The cleanup to run; must be safe to run more than once."""
        self._fn = fn
        self._lock = threading.Lock()
        self._closed = False

    def run(self, caller: str, final: bool = False) -> bool:
        """Run the cleanup, waiting for any run already in progress.

        :param caller: Who is running it, for the log.
        :param final: Close it after this run, so later runs are skipped.
        :returns: True if this call ran it, False if it was already closed.
        :raises Exception: Whatever the cleanup raises (a final run still closes).
        """
        with self._lock:
            if self._closed:
                logger.info("Cleanup already finished; skipping it for %s", caller)
                return False
            try:
                logger.info("Running test cleanup from %s", caller)
                self._fn()
            finally:
                self._closed = final
        return True


def _is_debugging() -> bool:
    """Check for an attached debugger the way pytest-timeout does.

    pytest-timeout does not fire while one is attached (unless debugger
    detection is disabled), so neither should the watchdog. Without
    pytest-timeout installed there is no timeout kill to mirror, so this
    reports False rather than failing inside the watchdog's timer thread.

    :returns: True if pytest-timeout is installed and sees a debugger (or other
        trace function) active.
    """
    try:
        import pytest_timeout
    except ImportError:
        return False
    return pytest_timeout.is_debugging()


class CleanupWatchdog:
    """Gets a hung test cleaned up before pytest-timeout's thread method kills it.

    The thread method ends the process with ``os._exit``, so the test's teardown
    never runs. ``abort_lead_seconds`` before the kill the watchdog calls
    ``abort``, which should make the test fail so that its own teardown runs.
    ``cleanup_lead_seconds`` before the kill it calls ``cleanup``, for a test that
    could not be aborted. Each stage has its own daemon timer, so an ``abort``
    that blocks cannot hold up the cleanup.
    """

    def __init__(
        self,
        deadline: Optional[TimeoutDeadline],
        abort: Callable[[], Any],
        cleanup: Callable[[], Any],
        abort_lead_seconds: float,
        cleanup_lead_seconds: float,
    ):
        """
        :param deadline: The test's pytest-timeout deadline, or None if untimed.
        :param abort: Makes the test stop waiting and fail.
        :param cleanup: The cleanup to run; a :class:`SerializedCleanup` shared
            with the test's teardown, so the two never run it at once.
        :param abort_lead_seconds: How long before the kill to call ``abort``.
        :param cleanup_lead_seconds: How long before the kill to call ``cleanup``.
        """
        self._deadline = deadline
        self._stages = [
            (abort_lead_seconds, abort, "aborting the test"),
            (cleanup_lead_seconds, cleanup, "cleaning up"),
        ]
        self._timers: list[threading.Timer] = []

    def start(self) -> bool:
        """Arm the timers, if pytest-timeout would kill the process.

        Not armed for an untimed test or the signal method, whose timeout raises
        in the test and so already reaches teardown.

        :returns: True if armed.
        """
        if self._deadline is None or self._deadline.method != "thread":
            return False
        for lead_seconds, fn, action in self._stages:
            delay = max(self._deadline.remaining() - lead_seconds, 0.0)
            timer = threading.Timer(delay, self._fire, args=(fn, action))
            timer.daemon = True
            timer.name = f"abort-cleanup-watchdog ({action})"
            timer.start()
            self._timers.append(timer)
        return True

    def cancel(self) -> None:
        """Disarm the timers (a no-op if never armed or already fired)."""
        for timer in self._timers:
            timer.cancel()

    def _fire(self, fn: Callable[[], Any], action: str) -> None:
        """Timer callback: run one stage, as pytest-timeout is about to kill us.

        :param fn: The stage's callable.
        :param action: What the stage does, for the log.
        """
        assert self._deadline is not None
        if not self._deadline.disable_debugger_detection and _is_debugging():
            return  # pytest-timeout spares a debugged test, so leave it alone.
        logger.warning(
            "pytest-timeout fires in %.0fs and would skip teardown; %s",
            self._deadline.remaining(),
            action,
        )
        try:
            fn()
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Watchdog failed %s", action)
