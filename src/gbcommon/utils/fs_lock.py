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
to release, so the lock also cleans itself up rather than accumulating files.

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

        Returns True if acquired, False on timeout or if the lock directory
        cannot be created (e.g. a read-only or otherwise failing mount) -- so
        best-effort callers can proceed without failing.
        """
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                self.lock_path.parent.mkdir(parents=True, exist_ok=True)
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
                self._write_info()
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

    def _write_info(self) -> None:
        # Best-effort metadata; the directory's existence is the lock, not this.
        try:
            self.info_file.write_text(f"{self.identity}\n{time.time()}\n")
        except OSError:
            pass

    def _owned_by_us(self) -> bool:
        try:
            first_line = self.info_file.read_text().splitlines()[0].strip()
        except (OSError, IndexError):
            # No/unreadable info file: we set ``_held`` on our own mkdir, so
            # treat the lock as ours to remove.
            return True
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
