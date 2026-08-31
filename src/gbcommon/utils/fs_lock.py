#!/usr/bin/env python3

# Copyright Granite.Build Authors
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

"""Cross-process lock via atomic directory creation on a shared filesystem.

Unlike BSD ``flock`` (which two-node probes found to be node-local on the Blue
Vela GPFS mount), ``os.mkdir`` is atomic *and* coherent across nodes on the
shared filesystems the cluster uses (verified on GPFS and the AFM/COS-backed
CSI PVC). ``SharedFileSystemLock`` therefore serializes processes across
containers/nodes that share the mount, using ``mkdir`` to acquire and ``rmdir``
to release, so each lock cleans itself up on release rather than accumulating.
(``acquire`` creates the lock's parent directory if needed but does not remove
it on release, since a peer may be creating a sibling lock there; that shared
container dir persists.)

``mkdir`` has no kernel auto-release on holder death. Two options bound that:
callers that must never block indefinitely can use a finite ``timeout`` and
treat a ``False`` return as "proceed anyway"; callers that want stale locks
reclaimed can set a ``ttl`` (seconds) after which a still-held lock is broken.
``ttl`` defaults to ``None`` (never break a held lock) -- a TTL shorter than the
protected work would break a live holder mid-operation, so leave it off unless
the work is bounded well below the TTL.
"""

import logging
import os
import socket
import time
from pathlib import Path
from typing import Optional, Self, Union

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL_S = 1.0
DEFAULT_TIMEOUT_S = 10.0


class SharedFileSystemLock:
    """A cross-node advisory lock backed by atomic directory creation.

    Acquire creates ``lock_path`` (a directory) with ``os.mkdir``; whoever
    creates it holds the lock. Release removes it. An identity file inside the
    directory records ``host|pid`` so release only removes a lock this instance
    still owns.
    """

    def __init__(
        self,
        lock_path: Union[str, Path],
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        poll_interval: float = DEFAULT_POLL_INTERVAL_S,
        ttl: Optional[float] = None,
    ) -> None:
        """
        Args:
            lock_path: Directory path used as the lock on the shared filesystem.
            timeout: Max seconds to wait to acquire before giving up.
            poll_interval: Seconds between acquisition attempts while waiting.
            ttl: If set, a held lock older than this many seconds is treated as
                stale and broken. ``None`` (default) never breaks a held lock.
        """
        self.lock_path = Path(lock_path)
        self.info_file = self.lock_path / "lock.info"
        self.timeout = float(timeout)
        self.poll_interval = float(poll_interval)
        self.ttl = ttl
        self.identity = f"host:{socket.gethostname()}|pid:{os.getpid()}"
        self._held = False

    @property
    def is_held(self) -> bool:
        """True if this instance currently holds the lock."""
        return self._held

    def acquire(self) -> bool:
        """Try to acquire the lock, waiting up to ``timeout`` seconds.

        Returns True if acquired. Returns False -- so best-effort callers can
        proceed without failing -- on timeout, if the lock directory cannot be
        created (e.g. a read-only or otherwise failing mount), or if the
        identity file cannot be written after creating it. The recorded identity
        is what release() uses to avoid removing a lock a peer now owns, so a
        lock we cannot attribute to ourselves is rolled back rather than held.
        """
        deadline = time.monotonic() + self.timeout
        # Create the container dir once up front rather than on every poll (a
        # contended wait can otherwise re-issue this mkdir hundreds of times).
        # The container is not removed on release, so nothing recreates it mid-
        # wait; only ``lock_path`` itself is contended below.
        try:
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(
                "SharedFileSystemLock: cannot create lock container %s (%s)",
                self.lock_path.parent,
                e,
            )
            return False
        while True:
            try:
                self.lock_path.mkdir()
            except FileExistsError:
                if self.ttl is not None and self._clear_if_stale():
                    continue
            except OSError as e:
                logger.warning(
                    "SharedFileSystemLock: cannot create lock %s (%s)",
                    self.lock_path,
                    e,
                )
                return False
            else:
                if not self._write_info():
                    # We created the dir but cannot record our identity in it.
                    # Without that, release() could not tell our own lock apart
                    # from one a stale-breaker later hands to a peer, so roll
                    # back and give up rather than hold an unattributable lock.
                    logger.warning(
                        "SharedFileSystemLock: created %s but could not write "
                        "identity; rolling back",
                        self.lock_path,
                    )
                    try:
                        self.lock_path.rmdir()
                    except OSError:
                        pass
                    return False
                self._held = True
                return True

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(self.poll_interval, remaining))

    def release(self) -> None:
        """Release the lock if this instance holds it (idempotent)."""
        if not self._held:
            return
        try:
            if self._owned_by_us():
                self.info_file.unlink(missing_ok=True)
                self.lock_path.rmdir()
        except OSError:
            # Another process may have force-cleared a stale lock; don't crash
            # cleanup over it.
            pass
        finally:
            self._held = False

    def _write_info(self) -> bool:
        # The recorded identity is load-bearing (release/ownership rely on it),
        # so acquire() treats a failed write as a failed acquire. Returns
        # whether the write succeeded.
        try:
            self.info_file.write_text(f"{self.identity}\n{time.time()}\n")
            return True
        except OSError:
            return False

    def _owned_by_us(self) -> bool:
        # acquire() only reports success after writing our identity, so a
        # missing/unreadable info file here means a peer broke our lock (a ttl
        # stale-break unlinks the info file before re-taking the dir). Never
        # remove a lock we cannot positively confirm is still ours.
        try:
            first_line = self.info_file.read_text().splitlines()[0].strip()
        except (OSError, IndexError):
            return False
        return first_line == self.identity

    def _clear_if_stale(self) -> bool:
        """Break the held lock if its recorded age exceeds ``ttl``."""
        try:
            created = float(self.info_file.read_text().splitlines()[1])
        except (OSError, IndexError, ValueError):
            return False
        if time.time() - created <= self.ttl:
            return False
        logger.warning(
            "SharedFileSystemLock: breaking stale lock %s (age exceeds ttl=%ss)",
            self.lock_path,
            self.ttl,
        )
        try:
            self.info_file.unlink(missing_ok=True)
            self.lock_path.rmdir()
        except OSError:
            return False
        return True

    def __enter__(self) -> Self:
        if not self.acquire():
            raise TimeoutError(
                f"Could not acquire lock on {self.lock_path} within " f"{self.timeout}s"
            )
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()
