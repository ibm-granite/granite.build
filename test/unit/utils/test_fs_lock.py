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

import os
import time
from unittest.mock import patch

import pytest

from gbcommon.utils.fs_lock import SharedFileSystemLock, _has_recent_activity


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


def test_context_manager_raises_infra_error_distinctly_from_timeout(tmp_path):
    """An infra failure (read-only mount) must not masquerade as a timeout.

    ``__enter__`` raising ``TimeoutError`` for an ``EROFS``/``ENOSPC`` acquire
    failure would report a broken mount as mere contention, with no way to tell
    them apart. The infra error propagates as itself (``TimeoutError`` is an
    ``OSError`` subclass, so the test also asserts it is *not* a ``TimeoutError``).
    """
    lock = SharedFileSystemLock(tmp_path / "m.lock", timeout=1)
    with patch("pathlib.Path.mkdir", side_effect=OSError(30, "Read-only file system")):
        with pytest.raises(OSError) as excinfo:
            with lock:
                pass
    assert not isinstance(
        excinfo.value, TimeoutError
    ), "a read-only/broken mount must not be reported as a lock timeout"
    assert excinfo.value.errno == 30


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


def test_acquire_rolls_back_when_identity_unwritable(tmp_path):
    """If the identity file can't be written, acquire rolls back and fails.

    Without a recorded identity, release() could not tell our lock apart from a
    peer's, so a lock we cannot attribute to ourselves must not be held.
    """
    lock_path = tmp_path / "k.lock"
    lock = SharedFileSystemLock(lock_path, timeout=1)
    with patch("pathlib.Path.write_text", side_effect=OSError("[Errno 28] No space")):
        assert lock.acquire() is False
    assert lock.is_held is False
    assert not lock_path.exists(), "the unattributable lock dir must be rolled back"


def test_release_keeps_lock_when_info_missing(tmp_path):
    """A missing info file means a peer broke our lock; release must not remove it.

    Guards the ttl stale-break race: the breaker moves the whole dir aside
    before re-taking it, so a legitimate holder finding no info file can no
    longer prove ownership and must leave the (now someone else's) lock in place.
    """
    lock_path = tmp_path / "l.lock"
    lock = SharedFileSystemLock(lock_path, timeout=1)
    assert lock.acquire() is True
    lock.info_file.unlink()  # simulate a stale-breaker mid re-acquire

    lock.release()
    assert lock.is_held is False
    assert lock_path.exists(), "must not remove a lock we can no longer prove is ours"


# --- progress-aware (liveness) reclamation --------------------------------


def _stale_peer_lock(lock_path, *, age_s: float) -> None:
    """Create a peer-held lock dir whose recorded start time is *age_s* ago."""
    lock_path.mkdir(parents=True)
    (lock_path / "lock.info").write_text(f"host:peer|pid:1\n{time.time() - age_s}\n")


def test_progress_keeps_a_slow_but_live_holder(tmp_path):
    """Past ttl but writing under progress_path => alive, not reclaimed.

    This is the finding-1 guard: a legitimately long download must never be
    abandoned mid-write just because it outran the ttl. With recent activity
    under progress_path the lock is left to its live holder.
    """
    lock_path = tmp_path / "repo" / ".gb-hfpull-locks" / "rev.lock"
    dest = tmp_path / "repo" / "rev"
    dest.mkdir(parents=True)
    (dest / "model.safetensors.incomplete").write_text("streaming")  # fresh write
    _stale_peer_lock(lock_path, age_s=1000)  # well past ttl

    lock = SharedFileSystemLock(lock_path, timeout=0, ttl=1, progress_path=dest)
    assert lock.acquire() is False, "a live holder (recent writes) must not be broken"
    assert lock_path.exists()


def test_progress_reclaims_a_dead_holder(tmp_path):
    """Past ttl AND no recent writes under progress_path => dead, reclaimed.

    The finding-2 guard: a crashed holder's lock is reclaimed within ttl of its
    last write rather than stalling every future puller until an operator clears
    it.
    """
    lock_path = tmp_path / "repo" / ".gb-hfpull-locks" / "rev.lock"
    dest = tmp_path / "repo" / "rev"
    dest.mkdir(parents=True)
    old = time.time() - 1000
    os.utime(dest, (old, old))  # no recent activity under dest
    _stale_peer_lock(lock_path, age_s=1000)

    lock = SharedFileSystemLock(lock_path, timeout=2, ttl=1, progress_path=dest)
    assert lock.acquire() is True, "a dead holder (no recent writes) must be reclaimed"
    assert lock.is_held is True
    lock.release()


def test_stale_break_leaves_no_graveyard_debris(tmp_path):
    """Reclaiming a stale lock must not leave a moved-aside dir behind."""
    lock_path = tmp_path / "repo" / ".gb-hfpull-locks" / "rev.lock"
    _stale_peer_lock(lock_path, age_s=1000)

    lock = SharedFileSystemLock(lock_path, timeout=2, ttl=1)  # no progress_path
    assert lock.acquire() is True
    lock.release()
    # Only the (now-removed) lock dir should have lived here -- no .stale/.released
    # graveyard copies accumulating on the shared cache.
    siblings = list(lock_path.parent.iterdir())
    assert siblings == [], f"unexpected leftover lock debris: {siblings}"


def test_ttl_breaks_lock_with_unreadable_info_via_dir_mtime(tmp_path):
    """A lock dir with no usable info line falls back to its own mtime for age.

    A holder that died between ``mkdir`` and writing its info leaves a dir with
    no timestamp; it must still be reclaimable (via the dir mtime) once past ttl,
    or such a dir would wedge every future waiter forever under ``timeout=None``.
    """
    lock_path = tmp_path / "n.lock"
    lock_path.mkdir()  # no lock.info written at all
    old = time.time() - 1000
    os.utime(lock_path, (old, old))

    lock = SharedFileSystemLock(lock_path, timeout=2, ttl=1)
    assert lock.acquire() is True, "a timestamp-less, old lock dir must be reclaimable"
    lock.release()


def test_timeout_none_reclaims_dead_holder_without_hanging(tmp_path):
    """timeout=None waits indefinitely for a live holder but reclaims a dead one.

    With no wall-clock deadline the loop only terminates by acquiring; a stale
    (past-ttl, no-progress) holder must therefore be reclaimed so the caller is
    not hung. (A pytest timeout would fire if this looped forever.)
    """
    lock_path = tmp_path / "repo" / ".gb-hfpull-locks" / "rev.lock"
    dest = tmp_path / "repo" / "rev"
    dest.mkdir(parents=True)
    old = time.time() - 1000
    os.utime(dest, (old, old))
    _stale_peer_lock(lock_path, age_s=1000)

    lock = SharedFileSystemLock(
        lock_path, timeout=None, poll_interval=0.01, ttl=1, progress_path=dest
    )
    assert lock.acquire() is True
    lock.release()


def test_still_owned_reflects_on_disk_ownership(tmp_path):
    """still_owned() re-reads the FS, so it flips False once a peer reclaims us."""
    lock = SharedFileSystemLock(tmp_path / "s.lock", timeout=1)
    assert lock.acquire() is True
    assert lock.still_owned() is True  # we hold it and the info records us

    # A peer reclaims: its stale-break recorded a different owner in our place.
    lock.info_file.write_text("host:other|pid:999\n123.0\n")
    assert lock.still_owned() is False, "a reclaimed lock is no longer ours"
    # is_held is only the in-memory flag; it does not re-check the filesystem.
    assert lock.is_held is True

    lock.release()  # must not remove the peer's lock
    assert lock.lock_path.exists()


def test_still_owned_false_when_not_held(tmp_path):
    lock = SharedFileSystemLock(tmp_path / "s2.lock", timeout=1)
    assert lock.still_owned() is False  # never acquired


def test_write_info_timestamp_is_integer_for_shell_interop(tmp_path):
    """The recorded start time is whole seconds so the shell parse can read it.

    The shell hfpull staleness check only accepts ``^[0-9]+$`` for the lock's
    start time; a float would fail its test and silently fall back to the lock
    dir mtime, so the anchor the Python and shell paths share must be an int.
    """
    lock = SharedFileSystemLock(tmp_path / "i.lock", timeout=1)
    assert lock.acquire() is True
    line2 = lock.info_file.read_text().splitlines()[1]
    assert line2.isdigit(), f"timestamp must be integer seconds, got {line2!r}"
    assert int(line2) > 0
    lock.release()


def test_acquire_reaps_leftover_graveyards(tmp_path):
    """Orphaned moved-aside lock dirs are swept on acquire, not accumulated."""
    container = tmp_path / ".gb-hfpull-locks"
    container.mkdir()
    (container / "x.lock.stale.1.2").mkdir()  # an interrupted stale-break
    (container / "x.lock.released.3.4").mkdir()  # an interrupted release
    (container / "other.lock").mkdir()  # a live sibling lock -- must be left alone

    lock = SharedFileSystemLock(container / "x.lock", timeout=1)
    assert lock.acquire() is True

    leftovers = sorted(
        p.name
        for p in container.iterdir()
        if ".stale." in p.name or ".released." in p.name
    )
    assert leftovers == [], f"graveyards not reaped: {leftovers}"
    assert (container / "other.lock").is_dir(), "a live sibling lock must not be reaped"
    lock.release()


def test_stale_check_is_throttled_not_run_every_poll(tmp_path):
    """The (expensive) staleness check runs on a throttle, not every poll.

    The mkdir attempt still runs every poll (to grab a released lock promptly),
    but the progress walk under a large ttl must not run once a second per
    waiter. With a large ttl and a short wait, only the initial check fires.
    """
    lock_path = tmp_path / "t.lock"
    lock_path.mkdir()  # a peer holds it (fresh, so never reclaimed here)
    lock = SharedFileSystemLock(lock_path, timeout=0.05, poll_interval=0.01, ttl=100)
    calls = []
    original = lock._clear_if_stale

    def counting():
        calls.append(1)
        return original()

    lock._clear_if_stale = counting  # type: ignore[method-assign]
    assert lock.acquire() is False  # peer holds a fresh lock; we time out
    assert len(calls) == 1, f"stale check should be throttled to once, got {len(calls)}"


def test_has_recent_activity_helper(tmp_path):
    """`_has_recent_activity` sees a freshly written nested file, not an old tree."""
    root = tmp_path / "tree"
    (root / "sub").mkdir(parents=True)
    nested = root / "sub" / "file.bin"
    nested.write_text("x")  # fresh

    now = time.time()
    assert _has_recent_activity(root, now - 60) is True

    old = now - 1000
    for p in (nested, root / "sub", root):
        os.utime(p, (old, old))
    assert _has_recent_activity(root, now - 60) is False
    assert _has_recent_activity(tmp_path / "missing", now - 60) is False
