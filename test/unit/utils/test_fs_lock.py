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

import logging
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

import gbcommon.utils.fs_lock as fs_lock
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


# --- removal must survive a mount without atomic directory rename ----------
#
# The removal path (release and stale-break) renames the lock dir aside with
# ``os.replace`` before deleting it, as a TOCTOU guard. But ``os.replace`` of a
# directory is a rename, and the object-store/AFM-backed FUSE mounts these locks
# target (see the module docstring) do not support renaming a directory: it
# raises OSError there. ``mkdir`` (acquire) works on such a mount but
# ``os.replace`` (removal) does not -- an asymmetry that leaks every lock. These
# tests pin the observed production hang (build e87ceebb: a holder's release
# left the lock behind and waiters re-logged "breaking stale lock" every poll
# forever under ``timeout=None``) by simulating that rename failure.

# ENOTSUP is what a directory rename raises on an object-store-backed FUSE mount.
_DIR_RENAME_UNSUPPORTED = OSError(95, "Operation not supported")


def test_release_removes_owned_lock_when_dir_rename_unsupported(tmp_path):
    """Release must free an owned lock even where directory rename is unsupported.

    Currently ``release`` removes the lock only via ``os.replace``; when that
    raises it is swallowed and the lock dir is left behind (a leak that blocks
    every waiter for the full ttl). Removal must fall back to a direct delete so
    an owned lock is actually freed.
    """
    lock = SharedFileSystemLock(tmp_path / "rn.lock", timeout=1)
    assert lock.acquire() is True

    with patch("os.replace", side_effect=_DIR_RENAME_UNSUPPORTED):
        lock.release()

    assert lock.is_held is False
    assert (
        not lock.lock_path.exists()
    ), "an owned lock must be freed even without dir rename"


def test_stale_lock_reclaimed_when_dir_rename_unsupported(tmp_path):
    """A dead holder's lock must be reclaimable where directory rename is unsupported.

    Same mount limitation as release: if the stale-break's ``os.replace`` raises
    and is swallowed, ``_clear_if_stale`` never clears the lock, so a waiter
    (``timeout=None`` in hfpull) re-logs "breaking stale lock" every poll and
    hangs forever. The break must fall back to a direct delete.
    """
    lock_path = tmp_path / "repo" / ".gb-hfpull-locks" / "rev.lock"
    dest = tmp_path / "repo" / "rev"
    dest.mkdir(parents=True)
    old = time.time() - 1000
    os.utime(dest, (old, old))  # no recent progress => holder looks dead
    _stale_peer_lock(lock_path, age_s=1000)

    lock = SharedFileSystemLock(
        lock_path, timeout=1, poll_interval=0.02, ttl=1, progress_path=dest
    )
    with patch("os.replace", side_effect=_DIR_RENAME_UNSUPPORTED):
        acquired = lock.acquire()

    assert (
        acquired is True
    ), "a dead holder's lock must be reclaimed even without dir rename"
    lock.release()


def test_abandoned_break_claim_does_not_wedge_stale_reclaim(tmp_path):
    """A crashed breaker's leftover claim must not deadlock future stale-breaks.

    The dir-rename fallback picks a single break winner via an atomic ``mkdir``
    of a ``<lock>.breaking`` marker. A breaker holds that marker only across one
    ``rmtree``, so a marker older than the ttl is a crashed breaker's abandoned
    claim; it must be reaped or every future waiter would block on it forever --
    reintroducing the very hang #354 fixes.
    """
    lock_path = tmp_path / "repo" / ".gb-hfpull-locks" / "rev.lock"
    dest = tmp_path / "repo" / "rev"
    dest.mkdir(parents=True)
    old = time.time() - 1000
    os.utime(dest, (old, old))  # no recent progress => holder looks dead
    _stale_peer_lock(lock_path, age_s=1000)
    # A previous breaker crashed still holding the claim; make it clearly aged.
    claim = lock_path.with_name(lock_path.name + ".breaking")
    claim.mkdir()
    os.utime(claim, (old, old))

    lock = SharedFileSystemLock(
        lock_path, timeout=2, poll_interval=0.02, ttl=1, progress_path=dest
    )
    with patch("os.replace", side_effect=_DIR_RENAME_UNSUPPORTED):
        acquired = lock.acquire()

    assert (
        acquired is True
    ), "an abandoned break-claim must be reaped, not deadlock the break"
    lock.release()


def test_rename_unsupported_notice_logged_once_per_process(tmp_path, caplog):
    """The dir-rename fallback notice fires once per process, not per release.

    On a mount without directory rename the fallback is the *normal* path, so a
    per-release WARNING would flood logs and look like a recurring fault. The
    "cannot rename" notice is emitted once at INFO; later fallbacks stay quiet
    (DEBUG).
    """
    fs_lock._rename_fallback_logged = False  # reset the process-wide flag
    with patch("os.replace", side_effect=OSError(39, "Directory not empty")):
        with caplog.at_level(logging.INFO, logger="gbcommon.utils.fs_lock"):
            for name in ("one.lock", "two.lock", "three.lock"):
                lock = SharedFileSystemLock(tmp_path / name, timeout=1)
                assert lock.acquire() is True
                lock.release()
                assert not lock.lock_path.exists()  # fallback still frees each lock

    notices = [
        r
        for r in caplog.records
        if r.levelno == logging.INFO and "cannot rename a directory" in r.getMessage()
    ]
    assert len(notices) == 1, f"expected one INFO notice, got {len(notices)}"


def test_release_logs_released_only_when_it_actually_removed(tmp_path, caplog):
    """release() logs 'released' only on an actual removal, not a peer-dir restore.

    In the stale-break-then-recreate race, _move_aside_and_remove restores the
    peer's recreated dir and returns False; release must not then claim the lock
    was 'released' (misleading in the log). Gate the INFO on the return value.
    """
    lock = SharedFileSystemLock(tmp_path / "rel.lock", timeout=1)
    assert lock.acquire() is True
    with caplog.at_level(logging.INFO, logger="gbcommon.utils.fs_lock"):
        with patch.object(lock, "_move_aside_and_remove", return_value=False):
            lock.release()
    # Match the message template (not the formatted path, which can contain the
    # test name "...released...").
    released = [r for r in caplog.records if "released" in str(r.msg)]
    assert released == [], "must not log 'released' when the lock was not removed"


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


def test_release_capture_restores_a_peer_recreated_dir(tmp_path):
    """Release must not delete a dir a peer recreated in the check/act window.

    ``os.replace`` captures whatever occupies ``lock_path``; if a peer stale-broke
    and re-created it with its own identity after our ownership check, the capture
    grabs the peer's *live* dir. ``verify_ours`` must then restore it rather than
    rmtree a peer's lock.
    """
    lock = SharedFileSystemLock(tmp_path / "r.lock", timeout=1)
    assert lock.acquire() is True
    # Simulate the recreated-by-a-peer state at capture time.
    lock.info_file.write_text("host:peer|pid:999\n123\n")

    removed = lock._move_aside_and_remove("released", verify_ours=True)

    assert removed is False, "must not remove a dir that is no longer ours"
    assert lock.lock_path.is_dir(), "the peer's recreated lock must be restored"
    assert lock._identity_at(lock.info_file) == "host:peer|pid:999"
    # No graveyard debris left from the capture/restore.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["r.lock"]


def test_still_owned_false_when_not_held(tmp_path):
    lock = SharedFileSystemLock(tmp_path / "s2.lock", timeout=1)
    assert lock.still_owned() is False  # never acquired


def test_ownership_retries_a_transient_read_error_not_eviction(tmp_path, monkeypatch):
    """A transient info-read glitch on a lock we hold must not read as eviction.

    A stale NFS handle / GPFS attribute-cache miss can make ``lock.info`` briefly
    unreadable on a lock that still names us. Concluding "not ours" there would,
    in the self-heal fence, abandon a live download to a re-wait while the on-disk
    dir still names us -- re-acquire then keeps failing until ttl. So a transient
    read *error* is retried; the second read succeeds and we are still the owner.
    """
    monkeypatch.setattr(fs_lock, "OWNERSHIP_READ_RETRY_INTERVAL_S", 0)  # no real wait
    lock = SharedFileSystemLock(tmp_path / "u.lock", timeout=1)
    assert lock.acquire() is True

    real_read = Path.read_text
    reads = {"n": 0}

    def flaky_read(self, *args, **kwargs):
        if self == lock.info_file:
            reads["n"] += 1
            if reads["n"] == 1:
                raise OSError(116, "Stale file handle")  # transient, dir still there
        return real_read(self, *args, **kwargs)

    with patch.object(Path, "read_text", flaky_read):
        assert lock.still_owned() is True, "a transient read glitch is not eviction"
    assert reads["n"] >= 2, "the transient read error should have been retried"
    lock.release()


def test_ownership_does_not_retry_a_genuinely_absent_info(tmp_path):
    """A moved-aside (absent) info file is a real handoff: conclude at once.

    Only an *error* on a present lock is ambiguous; a genuinely gone file
    (``FileNotFoundError`` -- a peer stale-broke and moved the dir aside) is
    definitively not ours, so it must not burn the retry budget on every eviction.
    """
    lock = SharedFileSystemLock(tmp_path / "v.lock", timeout=1)
    assert lock.acquire() is True
    lock.info_file.unlink()  # the dir was moved aside under us

    with patch("gbcommon.utils.fs_lock.time.sleep") as mock_sleep:
        assert lock.still_owned() is False
    mock_sleep.assert_not_called()  # no retry/backoff for a genuine handoff
    lock.release()


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
