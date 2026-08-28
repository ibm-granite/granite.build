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

"""Unit tests for ``SharedFileSystemLock`` (mkdir-based cross-node lock)."""

import time
from unittest.mock import patch

import pytest

from gbcommon.utils.fs_lock import SharedFileSystemLock


def test_acquire_creates_dir_and_release_removes_it(tmp_path):
    lock = SharedFileSystemLock(tmp_path / "a.lock", timeout=1)
    assert lock.acquire() is True
    assert lock.is_held is True
    assert lock.lock_path.is_dir()
    assert lock.info_file.exists()

    lock.release()
    assert lock.is_held is False
    assert not lock.lock_path.exists()  # cleaned up, does not accumulate


def test_acquire_times_out_when_peer_holds(tmp_path):
    lock_path = tmp_path / "b.lock"
    lock_path.mkdir()  # a peer already holds it
    lock = SharedFileSystemLock(lock_path, timeout=0)

    assert lock.acquire() is False
    assert lock.is_held is False
    assert lock_path.exists()  # we must not remove a lock we don't hold


def test_release_is_noop_when_not_held(tmp_path):
    lock_path = tmp_path / "c.lock"
    lock_path.mkdir()  # held by a peer
    lock = SharedFileSystemLock(lock_path, timeout=0)
    assert lock.acquire() is False
    lock.release()  # must not touch the peer's lock
    assert lock_path.exists()


def test_context_manager_acquires_and_releases(tmp_path):
    lock_path = tmp_path / "d.lock"
    with SharedFileSystemLock(lock_path, timeout=1) as lock:
        assert lock.is_held is True
        assert lock_path.is_dir()
    assert not lock_path.exists()


def test_context_manager_raises_on_timeout(tmp_path):
    lock_path = tmp_path / "e.lock"
    lock_path.mkdir()  # held by a peer
    with pytest.raises(TimeoutError):
        with SharedFileSystemLock(lock_path, timeout=0):
            pass
    assert lock_path.exists()  # peer's lock untouched


def test_ttl_breaks_a_stale_lock(tmp_path):
    lock_path = tmp_path / "f.lock"
    lock_path.mkdir()
    stale = time.time() - 1000
    (lock_path / "lock.info").write_text(f"host:dead|pid:1\n{stale}\n")

    lock = SharedFileSystemLock(lock_path, timeout=2, ttl=1)
    assert lock.acquire() is True, "stale lock past ttl should be broken and re-taken"
    assert lock.is_held is True
    lock.release()


def test_ttl_does_not_break_a_fresh_lock(tmp_path):
    lock_path = tmp_path / "g.lock"
    lock_path.mkdir()
    (lock_path / "lock.info").write_text(f"host:alive|pid:1\n{time.time()}\n")

    lock = SharedFileSystemLock(lock_path, timeout=0, ttl=100)
    assert lock.acquire() is False, "a fresh lock within ttl must not be broken"
    assert lock_path.exists()


def test_default_ttl_never_breaks_a_lock(tmp_path):
    """Without a ttl, even an ancient lock is left alone (best-effort default)."""
    lock_path = tmp_path / "h.lock"
    lock_path.mkdir()
    (lock_path / "lock.info").write_text(f"host:dead|pid:1\n{time.time() - 99999}\n")

    lock = SharedFileSystemLock(lock_path, timeout=0)  # ttl defaults to None
    assert lock.acquire() is False
    assert lock_path.exists()


def test_release_only_removes_lock_it_owns(tmp_path):
    lock_path = tmp_path / "i.lock"
    lock = SharedFileSystemLock(lock_path, timeout=1)
    assert lock.acquire() is True
    # Simulate a stale-breaker having handed the lock to someone else.
    lock.info_file.write_text("host:other|pid:999\n123.0\n")

    lock.release()
    assert lock.is_held is False
    assert lock_path.exists(), "must not remove a lock now owned by another holder"


def test_acquire_returns_false_when_mkdir_fails(tmp_path):
    lock = SharedFileSystemLock(tmp_path / "j.lock", timeout=1)
    with patch(
        "pathlib.Path.mkdir", side_effect=OSError("[Errno 30] Read-only file system")
    ):
        assert lock.acquire() is False
    assert lock.is_held is False
