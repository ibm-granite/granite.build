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

``mkdir`` has no kernel auto-release on holder death, so a lock left by a
crashed holder does not clean itself up. Callers bound that with ``ttl`` plus a
``progress_path``: a held lock older than ``ttl`` is reclaimed *only if* nothing
under ``progress_path`` has been written within ``ttl`` -- i.e. the holder's own
I/O is the liveness signal, so a live-but-slow holder is never broken while a
dead one is reclaimed within ``ttl`` of its last write. This lets a waiter block
indefinitely (``timeout=None``) for a *live* holder without hanging on a *dead*
one; the ``ttl`` window then only needs to exceed the longest plausible no-write
gap for a live holder (network stalls plus the shared FS's attribute-cache
latency), not the whole protected operation. A ``ttl`` without a
``progress_path`` falls back to breaking on age alone, which can break a live
holder that legitimately outlasts the ttl -- so pair the two, or use a finite
``timeout`` and treat a ``False`` return as "proceed anyway". ``ttl`` defaults to
``None`` (never break a held lock).

Reclaiming a stale lock and releasing an owned one both capture the lock dir with
an atomic ``os.replace`` before removing it, so two waiters racing to break the
same lock cannot both succeed and re-take it. ``os.replace`` acts on whatever
occupies the path, though, so a release racing a peer's stale-break-then-recreate
could capture the peer's *fresh* dir; release therefore re-checks the captured
dir's identity and restores it instead of deleting when it is no longer ours (a
narrow third-party-``mkdir``-during-restore residual remains, which a POSIX
rename cannot close; the on-acquire reaper cleans the orphan). This whole race is
unreachable when a ``progress_path`` is set -- a releasing holder has just
written, so a peer never declares it stale to begin with -- and only a
``ttl``-without-``progress_path`` caller can approach it.

Some shared mounts (object-store/AFM-backed FUSE) support ``os.mkdir`` but *not*
renaming a directory, so ``os.replace`` cannot run there at all -- left
unhandled, that leaks every lock and hangs waiters for the full ``ttl`` (issue
#354). When ``os.replace`` fails, removal falls back to a direct delete that
keeps the same guarantees using ``os.mkdir``: release deletes only after
re-confirming ownership, and the stale-break selects a single winner via an
atomic ``mkdir`` of a ``<lock>.breaking`` claim (reaped if a crashed breaker
leaves it behind).
"""

import logging
import os
import shutil
import socket
import time
from pathlib import Path
from typing import Optional, Self, Union

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL_S = 1.0
DEFAULT_TIMEOUT_S = 10.0

# Re-reads of lock.info before concluding a lock we hold is no longer ours, when
# the read *errors transiently* (stale NFS handle, GPFS attribute-cache miss, EIO)
# on a lock dir that still exists. Such a glitch on a lock that physically still
# names us must not be misread as eviction: in the self-heal fence that would
# abandon a live download to a re-wait while the on-disk dir still names us, so
# re-acquire keeps failing and the holder's own writes read as "recent activity"
# -- self-stalling until ttl. A *genuinely* absent info file (``FileNotFoundError``
# -- a peer moved the whole dir aside in a stale-break) is a real handoff and is
# concluded at once; likewise a read that returns a *different* owner. Retrying is
# reserved for the ambiguous transient-error case.
OWNERSHIP_READ_RETRIES = 3
OWNERSHIP_READ_RETRY_INTERVAL_S = 0.2

# Suffix for the break-claim marker used by the dir-rename fallback (issue #354).
# On mounts that do not support renaming a directory (object-store/AFM-backed
# FUSE), ``os.replace`` cannot capture the lock dir, so the stale-break selects a
# single winner by atomically ``mkdir``-ing ``<lock>.breaking`` instead -- the one
# operation verified atomic there. The winner alone deletes the stale lock; the
# marker is held only across that delete, so one older than ``ttl`` is a crashed
# breaker's and is reaped rather than left to wedge future breaks.
BREAK_CLAIM_SUFFIX = ".breaking"

# On mounts that cannot rename a directory the removal fallback is the *normal*
# path, so a per-call WARNING would flood the log. Announce it once per process
# at INFO (so an operator sees the fallback is in use) and drop to DEBUG after.
_rename_fallback_logged = False


def _note_rename_unsupported(lock_path: Path, err: OSError) -> None:
    """Log the directory-rename removal fallback once per process, then DEBUG."""
    global _rename_fallback_logged
    if not _rename_fallback_logged:
        _rename_fallback_logged = True
        logger.info(
            "SharedFileSystemLock: this filesystem cannot rename a directory "
            "(%s); using the direct-delete fallback for lock removal. Expected on "
            "object-store/AFM-backed caches; not logged again this process.",
            err,
        )
    else:
        logger.debug(
            "SharedFileSystemLock: os.replace of %s failed (%s); using the "
            "no-rename removal fallback",
            lock_path,
            err,
        )


def _has_recent_activity(root: Path, min_mtime: float) -> bool:
    """True if *root* or any file under it was modified at/after *min_mtime*.

    Liveness signal for stale-lock reclamation: a process actively writing under
    *root* keeps some mtime advancing, so a waiter can tell a live-but-slow
    holder from a dead one without the holder heartbeating. A single large file
    streaming in bumps only its own mtime (not its parent dir's), so file mtimes
    -- not just directory mtimes -- are checked. Returns as soon as one fresh
    entry is found, so the common "holder is alive" case is cheap; only a fully
    idle tree is walked in full before returning False. A missing *root* counts
    as no activity.
    """
    try:
        if root.stat().st_mtime >= min_mtime:
            return True
    except OSError:
        return False
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            try:
                if os.stat(os.path.join(dirpath, name)).st_mtime >= min_mtime:
                    return True
            except OSError:
                continue
    return False


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
        timeout: Optional[float] = DEFAULT_TIMEOUT_S,
        poll_interval: float = DEFAULT_POLL_INTERVAL_S,
        ttl: Optional[float] = None,
        progress_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Args:
            lock_path: Directory path used as the lock on the shared filesystem.
            timeout: Max seconds to wait to acquire before giving up. ``None``
                waits indefinitely -- safe only when ``ttl`` is set, since a dead
                holder is then reclaimed within ``ttl`` rather than hung on.
            poll_interval: Seconds between acquisition attempts while waiting.
            ttl: If set, a held lock older than this many seconds is treated as
                stale and reclaimed -- subject to ``progress_path`` below.
                ``None`` (default) never breaks a held lock.
            progress_path: Path whose recent write activity marks the holder as
                alive. When set (with ``ttl``), a lock past ``ttl`` is reclaimed
                only if nothing under this path was written within ``ttl``, so a
                live-but-slow holder is never broken. When ``None``, ``ttl``
                reclaims on age alone.
        """
        self.lock_path = Path(lock_path)
        self.info_file = self.lock_path / "lock.info"
        self.timeout = None if timeout is None else float(timeout)
        self.poll_interval = float(poll_interval)
        self.ttl = ttl
        self.progress_path = Path(progress_path) if progress_path is not None else None
        self.identity = f"host:{socket.gethostname()}|pid:{os.getpid()}"
        self._held = False
        # Set by acquire() when a False return was caused by an infrastructure
        # failure (unusable mount, unwritable identity file) rather than a
        # timeout, so __enter__ can distinguish the two. None means "timed out".
        self._last_acquire_error: Optional[OSError] = None
        # How often the (potentially expensive) staleness check runs while
        # waiting. The mkdir attempt still runs every poll so a released lock is
        # grabbed promptly, but the progress walk under ``progress_path`` is
        # throttled to this cadence rather than every poll -- a few extra seconds
        # to reclaim a *dead* lock is negligible against a ttl of minutes, and it
        # avoids each waiter walking the destination tree once a second.
        self._stale_check_interval = (
            max(self.poll_interval, min(self.ttl / 4.0, 30.0))
            if self.ttl is not None
            else self.poll_interval
        )
        self._next_stale_check = 0.0

    @property
    def is_held(self) -> bool:
        """True if this instance currently holds the lock.

        This is an in-memory flag set at acquire; it does *not* re-check the
        shared filesystem. Use :meth:`still_owned` to detect a peer having
        reclaimed a stale lock out from under us.
        """
        return self._held

    def still_owned(self) -> bool:
        """True if we hold the lock *and* the on-disk lock still records us.

        Re-reads ``lock.info`` on the shared filesystem, so unlike
        :attr:`is_held` it returns False once a peer has reclaimed our stale
        lock (its ttl stale-break moved our dir aside and recorded a new owner).
        Lets a long-running holder fence its own work: if it has been evicted it
        can stop rather than keep acting as if it still holds the lock.
        """
        return self._held and self._owned_by_us()

    def acquire(self) -> bool:
        """Try to acquire the lock, waiting up to ``timeout`` seconds.

        With ``timeout=None`` waits indefinitely, reclaiming a stale holder via
        ``ttl`` (see :meth:`_clear_if_stale`) so a *live* holder is waited on
        while a *dead* one does not hang the caller.

        Returns True if acquired. Returns False -- so best-effort callers can
        proceed without failing -- on timeout, if the lock directory cannot be
        created (e.g. a read-only or otherwise failing mount), or if the
        identity file cannot be written after creating it. The recorded identity
        is what release() uses to avoid removing a lock a peer now owns, so a
        lock we cannot attribute to ourselves is rolled back rather than held.
        """
        deadline = None if self.timeout is None else time.monotonic() + self.timeout
        self._last_acquire_error = None
        self._next_stale_check = 0.0  # check on the first contended poll
        announced_wait = False  # so the "waiting" INFO is logged once, not per poll
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
            self._last_acquire_error = e
            return False
        self._reap_graveyards()
        while True:
            try:
                self.lock_path.mkdir()
            except FileExistsError:
                # Throttle the staleness check: the mkdir above already runs every
                # poll (so we grab a released lock at once), but the progress walk
                # is expensive on a shared FS, so only run it every
                # ``_stale_check_interval``.
                now_mono = time.monotonic()
                if self.ttl is not None and now_mono >= self._next_stale_check:
                    self._next_stale_check = now_mono + self._stale_check_interval
                    if self._clear_if_stale():
                        continue
            except OSError as e:
                logger.warning(
                    "SharedFileSystemLock: cannot create lock %s (%s)",
                    self.lock_path,
                    e,
                )
                self._last_acquire_error = e
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
                logger.info(
                    "SharedFileSystemLock: acquired %s (%s)",
                    self.lock_path,
                    self.identity,
                )
                return True

            # Reached only when contended and about to wait (a successful acquire
            # returns above; a successful stale-break `continue`s). Announce the
            # wait once so the log shows who is blocked on whom, not every poll.
            if not announced_wait:
                announced_wait = True
                logger.info(
                    "SharedFileSystemLock: %s is held; waiting (%s)",
                    self.lock_path,
                    self.identity,
                )
            if deadline is None:
                time.sleep(self.poll_interval)
                continue
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
                # Capture our dir aside atomically, then delete it only after
                # re-confirming it is still ours (verify_ours): a peer that
                # stale-broke and re-created lock_path in the check/act window
                # would otherwise have its live dir deleted -- os.replace acts on
                # whatever occupies the path, which may now be the peer's. If the
                # captured dir is not ours, it is restored instead of removed.
                if self._move_aside_and_remove("released", verify_ours=True):
                    # Log only on an actual removal -- not when the capture
                    # restored a peer's dir it had recreated in the check/act
                    # window (return False). That race is unreachable with a
                    # progress_path set, as hfpull always does.
                    logger.info(
                        "SharedFileSystemLock: released %s (%s)",
                        self.lock_path,
                        self.identity,
                    )
        except OSError:
            # Another process may have force-cleared a stale lock; don't crash
            # cleanup over it.
            pass
        finally:
            self._held = False

    def _move_aside_and_remove(self, reason: str, *, verify_ours: bool = False) -> bool:
        """Atomically ``os.replace`` ``lock_path`` to a unique name, then delete.

        ``os.replace`` of a directory is atomic, so exactly one racer captures a
        given lock-dir inode; a loser gets ``ENOENT`` (the dir moved) and returns
        False. Removing the *moved* dir cannot disturb whoever wins a subsequent
        ``mkdir`` of ``lock_path``. Returns whether this caller did the removal.

        ``os.replace`` acts on whatever occupies ``lock_path`` at that instant,
        which -- for a release racing a peer's stale-break-then-recreate -- may
        already be the peer's fresh dir. With ``verify_ours`` the captured dir's
        identity is re-checked after the capture and, if it is no longer ours, it
        is restored rather than deleted (so a release never destroys a live
        peer's lock). A narrow residual remains -- a third party ``mkdir``-ing
        ``lock_path`` between our capture and restore -- which a POSIX rename
        cannot close; the reaper cleans the resulting orphan. Stale-breaks pass
        ``verify_ours=False`` (they legitimately remove a *peer's* dead dir); that
        path's own freshly-acquired-lock race is bounded instead by the age +
        progress checks in :meth:`_clear_if_stale`.
        """
        graveyard = self.lock_path.with_name(
            f"{self.lock_path.name}.{reason}.{os.getpid()}.{time.monotonic_ns()}"
        )
        try:
            os.replace(self.lock_path, graveyard)
        except FileNotFoundError:
            # A racer already moved lock_path aside; we lost. (Distinct from the
            # OSError below so a genuine handoff is not mistaken for a mount that
            # cannot rename.)
            return False
        except OSError as e:
            # The mount cannot rename a directory (object-store/AFM-backed FUSE:
            # os.mkdir works but os.replace does not), so the capture cannot run.
            # Fall back to a direct removal that keeps the same guarantees via
            # os.mkdir -- the primitive verified atomic here (issue #354). Logged
            # (once per process, then DEBUG), not swallowed: silently dropping this
            # is what let a leaked lock hang every waiter for the full ttl.
            _note_rename_unsupported(self.lock_path, e)
            return self._remove_without_rename(verify_ours=verify_ours)
        if verify_ours and self._identity_at(graveyard / "lock.info") != self.identity:
            # Captured a dir a peer recreated in the check/act window -- restore
            # it rather than delete a live peer's lock.
            try:
                os.replace(graveyard, self.lock_path)
            except OSError:
                pass  # lock_path re-occupied; leave the orphan for the reaper
            return False
        shutil.rmtree(graveyard, ignore_errors=True)
        return True

    def _remove_without_rename(self, *, verify_ours: bool) -> bool:
        """Remove ``lock_path`` directly, for mounts without atomic dir rename.

        Used only when :meth:`_move_aside_and_remove`'s ``os.replace`` capture is
        unavailable (issue #354). Preserves the same guarantees using ``os.mkdir``
        -- the one operation verified atomic/coherent on these shared mounts.
        Returns whether ``lock_path`` is gone afterwards.

        * release (``verify_ours=True``): re-confirm the on-disk lock still records
          us immediately before deleting, so a peer that stale-broke and recreated
          it is not destroyed; if it is no longer ours, leave it. (For hfpull this
          check/act window is unreachable anyway -- a releasing holder has just
          written under its ``progress_path``, so no peer declares it stale.)
        * stale-break (``verify_ours=False``): atomically ``mkdir`` a
          ``<lock>.breaking`` claim so exactly one waiter deletes the stale dir;
          losers back off and just retry their ``mkdir`` of ``lock_path`` (itself
          single-winner). A claim older than ``ttl`` is a crashed breaker's and is
          reaped, so it cannot wedge the break.
        """
        if verify_ours:
            if not self._owned_by_us():
                return False
            shutil.rmtree(self.lock_path, ignore_errors=True)
            return not self.lock_path.exists()

        claim = self.lock_path.with_name(self.lock_path.name + BREAK_CLAIM_SUFFIX)
        try:
            os.mkdir(claim)
        except FileExistsError:
            # Another waiter holds the break-claim. If it is an abandoned (aged)
            # claim from a crashed breaker, reap it so a later poll can retry;
            # otherwise a live breaker has it and we simply back off.
            self._reap_if_abandoned(claim)
            return False
        except OSError:
            return False
        try:
            shutil.rmtree(self.lock_path, ignore_errors=True)
        finally:
            shutil.rmtree(claim, ignore_errors=True)
        return not self.lock_path.exists()

    def _reap_if_abandoned(self, claim: Path) -> None:
        """Reap a break-claim only if it is older than ``ttl`` (crashed breaker).

        A live breaker holds the claim across a single ``rmtree`` (far younger
        than any sensible ``ttl``), so an aged claim is abandoned. Without a
        ``ttl`` the stale-break never runs, so no claim is ever created.
        """
        if self.ttl is None:
            return
        try:
            if time.time() - claim.stat().st_mtime > self.ttl:
                shutil.rmtree(claim, ignore_errors=True)
        except OSError:
            pass

    def _reap_graveyards(self) -> None:
        """Sweep leftover moved-aside lock dirs from the container (best-effort).

        :meth:`_move_aside_and_remove` renames a lock dir to ``<name>.stale.*`` /
        ``<name>.released.*`` and then ``rmtree``s it; an interrupted or
        transiently-failing ``rmtree`` (or the shell mirror's ``rm -rf``) can
        leave the renamed dir behind, and nothing else reaps the persistent
        container. Remove those orphans on acquire so they cannot accumulate
        unbounded over many crash/evict cycles. Only the moved-aside patterns are
        matched -- never a live ``*.lock`` dir -- and errors are ignored (a peer
        may be removing the same orphan concurrently).
        """
        container = self.lock_path.parent
        try:
            orphans = list(container.glob("*.stale.*")) + list(
                container.glob("*.released.*")
            )
        except OSError:
            return
        for orphan in orphans:
            shutil.rmtree(orphan, ignore_errors=True)

    def _write_info(self) -> bool:
        # The recorded identity is load-bearing (release/ownership rely on it),
        # so acquire() treats a failed write as a failed acquire. Returns
        # whether the write succeeded. The timestamp is written as whole seconds
        # (``int``) so the shell hfpull staleness parse -- which only accepts
        # ``^[0-9]+$`` -- can read a Python-created lock.info on a cache shared by
        # k8s (Python) and LSF/skypilot (shell) pullers, and vice versa (the
        # shell writes ``date +%s``). Sub-second precision is irrelevant: the ttl
        # comparisons are in whole seconds.
        try:
            self.info_file.write_text(f"{self.identity}\n{int(time.time())}\n")
            return True
        except OSError as e:
            self._last_acquire_error = e
            return False

    @staticmethod
    def _identity_at(info_file: Path) -> Optional[str]:
        """First line (the recorded owner identity) of *info_file*, or None."""
        try:
            return info_file.read_text().splitlines()[0].strip()
        except (OSError, IndexError):
            return None

    def _owned_by_us(self) -> bool:
        # acquire() only reports success after writing our identity, so the info
        # file names us for as long as we hold the lock. Reading a *different*
        # owner, or finding the file *gone* (``FileNotFoundError`` -- a ttl
        # stale-break moved the whole dir aside), is a real handoff: conclude "not
        # ours" at once. But a *transient* read error (stale NFS handle, GPFS
        # attribute-cache miss, EIO) on a lock dir that still exists must not be
        # misread as eviction -- in the fence that would abandon a live download
        # and then self-stall until ttl (the dir still names us, so re-acquire
        # keeps failing). Retry only that ambiguous case. Never remove/abandon a
        # lock we cannot positively confirm is ours.
        for attempt in range(OWNERSHIP_READ_RETRIES + 1):
            try:
                recorded = self.info_file.read_text().splitlines()[0].strip()
            except FileNotFoundError:
                return False  # the dir was moved aside: a genuine handoff
            except (OSError, IndexError):
                # Transient error, or a breaker that mkdir'd but has not yet
                # written its identity; retry before concluding.
                if attempt < OWNERSHIP_READ_RETRIES:
                    time.sleep(OWNERSHIP_READ_RETRY_INTERVAL_S)
                continue
            return recorded == self.identity
        return False

    def _lock_age_anchor(self) -> Optional[float]:
        """Best available creation time for the staleness check.

        The recorded timestamp when readable, else the lock dir's own mtime (a
        holder that died between ``mkdir`` and writing its info leaves a dir with
        no usable info line). ``None`` when neither is readable.
        """
        try:
            return float(self.info_file.read_text().splitlines()[1])
        except (OSError, IndexError, ValueError):
            pass
        try:
            return self.lock_path.stat().st_mtime
        except OSError:
            return None

    def _clear_if_stale(self) -> bool:
        """Reclaim the held lock if it is past ``ttl`` and shows no progress.

        Age past ``ttl`` alone is not death: a live holder can legitimately hold
        the lock longer than ``ttl`` on a slow transfer. When a ``progress_path``
        is set, only reclaim if nothing under it was written within ``ttl`` (the
        holder's own I/O is the liveness signal), so a live-but-slow holder is
        never broken. Without a ``progress_path``, age alone decides.
        """
        created = self._lock_age_anchor()
        if created is None:
            return False
        if time.time() - created <= self.ttl:
            return False
        if self.progress_path is not None and _has_recent_activity(
            self.progress_path, time.time() - self.ttl
        ):
            return False
        logger.warning(
            "SharedFileSystemLock: breaking stale lock %s (age exceeds ttl=%ss, "
            "no recent progress under %s)",
            self.lock_path,
            self.ttl,
            self.progress_path,
        )
        # Single-winner break: move the stale dir aside atomically so two waiters
        # racing to break it can't both succeed and re-take it. A loser's replace
        # fails; it just retries mkdir.
        return self._move_aside_and_remove("stale")

    def __enter__(self) -> Self:
        if self.acquire():
            return self
        # Distinguish a real infra failure (unusable mount, unwritable identity)
        # from contention: re-raise the former as itself so a read-only/broken
        # mount is not silently reported as a lock timeout. Note TimeoutError is
        # an OSError subclass, so callers catching OSError still catch both.
        if self._last_acquire_error is not None:
            raise self._last_acquire_error
        raise TimeoutError(
            f"Could not acquire lock on {self.lock_path} within {self.timeout}s"
        )

    def __exit__(self, *_exc: object) -> None:
        self.release()
