"""Node-wide named mutexes for resources the enroot sandbox can't isolate.

enroot gives a container no network namespace, so two sandboxes on one node
(in one granite-evals process or in two LSF jobs) that serve on the same fixed
port, or use the same X display, collide. A benchmark takes a
:func:`node_locks` over the keys of what a task needs (``port-6379``,
``x99``, ...) around it.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import logging
import os
import socket
import sys
import tempfile
import time
from pathlib import Path

log = logging.getLogger(__name__)


class NodeLock:
    """A mutex shared by every process on the node that sees the same network
    namespace, named by ``key``.

    On Linux it is an abstract unix socket bound to ``@granite-lock-<key>``: the
    name lives in the network namespace, which enroot shares with the host, so
    it reaches across LSF jobs whose containers each have a private /tmp, and
    the kernel frees it when its holder dies (no stale lock files). Elsewhere
    (macOS, where sandboxes have their own network) it is an flock on a file in
    ``GRANITE_EVALS_LOCK_DIR`` or the temp dir."""

    def __init__(self, key: str, *, poll_s: float = 2.0, abstract: bool | None = None):
        self.key, self.poll_s = key, poll_s
        self.abstract = sys.platform == "linux" if abstract is None else abstract
        self._held = None

    def _try(self):
        if self.abstract:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                s.bind(f"\0granite-lock-{self.key}")
            except OSError as e:
                s.close()
                if e.errno == errno.EADDRINUSE:
                    return None
                raise
            return s
        import fcntl

        path = Path(os.environ.get("GRANITE_EVALS_LOCK_DIR") or tempfile.gettempdir()) / f"granite-{self.key}.lock"
        f = open(path, "a")  # noqa: SIM115 - held until release
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            f.close()
            return None
        return f

    def acquire(self) -> float:
        """Blocks until held; returns the seconds waited."""
        start = last_log = time.monotonic()
        while (held := self._try()) is None:
            if time.monotonic() - last_log >= 60:
                log.info("waiting for node lock %s (%.0f s)", self.key, time.monotonic() - start)
                last_log = time.monotonic()
            time.sleep(self.poll_s)
        self._held = held
        return time.monotonic() - start

    async def acquire_async(self) -> float:
        """:meth:`acquire` for asyncio: polls without blocking the loop, and a
        cancelled waiter holds nothing."""
        start = last_log = time.monotonic()
        while (held := self._try()) is None:
            if time.monotonic() - last_log >= 60:
                log.info("waiting for node lock %s (%.0f s)", self.key, time.monotonic() - start)
                last_log = time.monotonic()
            await asyncio.sleep(self.poll_s)
        self._held = held
        return time.monotonic() - start

    def release(self) -> None:
        if self._held is not None:
            self._held.close()
            self._held = None


@contextlib.contextmanager
def node_locks(keys: list[str], *, what: str = "", poll_s: float = 2.0):
    """Holds a NodeLock per key (taken in sorted order, so two holders of
    overlapping sets cannot deadlock); yields the seconds waited."""
    locks = [NodeLock(k, poll_s=poll_s) for k in sorted(set(keys))]
    waited = 0.0
    try:
        for lock in locks:
            waited += lock.acquire()
        if locks:
            log.info("%s: holding node locks %s (waited %.0f s)", what, [l.key for l in locks], waited)  # noqa: E741
        yield waited
    finally:
        for lock in reversed(locks):
            lock.release()


@contextlib.asynccontextmanager
async def async_node_locks(keys: list[str], *, what: str = "", poll_s: float = 2.0):
    """:func:`node_locks` for asyncio."""
    locks = [NodeLock(k, poll_s=poll_s) for k in sorted(set(keys))]
    waited = 0.0
    try:
        for lock in locks:
            waited += await lock.acquire_async()
        if locks:
            log.info("%s: holding node locks %s (waited %.0f s)", what, [l.key for l in locks], waited)  # noqa: E741
        yield waited
    finally:
        for lock in reversed(locks):
            lock.release()
