#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""Local cache of unpacked db-space snapshots, safe across threads and processes.

Layout, under a private root (``GBSERVER_DB_SPACE_CACHE_DIR``, default a
per-user dir in the temp dir)::

    <root>/s-<space>/<slot>/
        .lock                      per-key lock: shared to look up, exclusive
                                   to publish and prune
        <digest>-<id>/             a published copy; never renamed or rewritten
            .gb-snapshot-complete  manifest (each file and its size), last
            .gb-in-use             shared-locked while any process uses it
        .tmp-<id>/, .tmp-<id>.lock an extraction, and its writer's lock

The path names the space and slot, so a cached path maps back to its db:// URI
with no shared state (the ``s-`` prefix keeps a name like ``..`` a plain dir).
A copy is served only if its manifest's files are all present and unchanged.
Users hold a copy through a Lease; the copy is pruned only when no process
holds one, and the OS drops a dead process's holds. Pruning runs after a new
copy is published. Locks are POSIX ``flock``: the root must be on a local
filesystem.
"""

import errno
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
import threading
import urllib.parse
import uuid
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from gbserver.types.constants import ENV_VAR_DB_SPACE_CACHE_DIR
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

_KEY_LOCK = ".lock"
_MANIFEST = ".gb-snapshot-complete"
_IN_USE = ".gb-in-use"
_TMP_PREFIX = ".tmp-"
_SPACE_PREFIX = "s-"

# This process's holds: copy -> [fd with a shared in-use lock, lease count].
_held: Dict[Path, List[int]] = {}
_held_lock = threading.Lock()


def get_cache_root() -> Path:
    """The cache root: GBSERVER_DB_SPACE_CACHE_DIR, else a per-user temp dir.

    Per user: a shared, predictable path would let another local user plant a
    tree for a digest that is computable from the (readable) archive.
    """
    configured = os.getenv(ENV_VAR_DB_SPACE_CACHE_DIR)
    if configured:
        return Path(configured).expanduser()
    return Path(tempfile.gettempdir()) / f"gb-db-spaces-{os.getuid()}"


def _private_root() -> Path:
    """The cache root, created 0700; refused if not ours or writable by others."""
    root = get_cache_root()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    st = root.stat()
    if st.st_uid != os.getuid() or st.st_mode & 0o022:
        raise ValueError(
            f"refusing snapshot cache {root}: not owned by this user or writable"
            " by others"
        )
    return root


def key_dir(space_name: str, slot: str) -> Path:
    """The directory holding the copies of one space's snapshot slot."""
    space_dir = _SPACE_PREFIX + urllib.parse.quote(space_name, safe="")
    path = _private_root() / space_dir / slot
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def parse_cached_path(path: Path) -> Optional[Tuple[str, str, str]]:
    """``(space, slot, subpath)`` for a path inside a cached copy, else None."""
    try:
        rel = path.resolve().relative_to(get_cache_root().resolve())
    except ValueError:
        return None
    parts = rel.parts
    if (
        len(parts) < 3
        or not parts[0].startswith(_SPACE_PREFIX)
        or parts[2].startswith(_TMP_PREFIX)
    ):
        return None
    space_name = urllib.parse.unquote(parts[0][len(_SPACE_PREFIX) :])
    return space_name, parts[1], "/".join(parts[3:])


@contextmanager
def _locked(path: Path, mode: int) -> Iterator[None]:
    """Hold ``flock(mode)`` on ``path`` (created if missing) for the block."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, mode)
        yield
    finally:
        os.close(fd)


def _try_exclusive(path: Path) -> Optional[int]:
    """An fd holding ``LOCK_EX`` on ``path`` without waiting; None if held/gone."""
    try:
        fd = os.open(path, os.O_RDWR)
    except FileNotFoundError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        os.close(fd)
        if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
            return None
        raise
    return fd


def _file_sizes(copy: Path) -> Dict[str, int]:
    """Relative path -> size of every file in a copy, bookkeeping excluded."""
    sizes = {}
    for dirpath, _, names in os.walk(copy):
        for name in names:
            path = Path(dirpath, name)
            rel = path.relative_to(copy).as_posix()
            if rel not in (_MANIFEST, _IN_USE):
                sizes[rel] = path.lstat().st_size
    return sizes


def _is_complete(copy: Path) -> bool:
    """Whether every file the manifest lists is present with its size.

    Files added later (e.g. ``__pycache__``) don't count against a copy.
    """
    try:
        listed = json.loads((copy / _MANIFEST).read_text())["files"]
        return all((copy / rel).lstat().st_size == size for rel, size in listed.items())
    except (OSError, ValueError, KeyError, AttributeError):
        return False


def _find(keydir: Path, digest: str) -> Optional[Path]:
    """A complete published copy of ``digest``, if any."""
    for child in sorted(keydir.glob(f"{digest}-*")):
        if child.is_dir() and _is_complete(child):
            return child
    return None


def _acquire(copy: Path) -> None:
    """Count a lease on ``copy``; the first takes the shared in-use lock."""
    with _held_lock:
        held = _held.get(copy)
        if held is None:
            fd = os.open(copy / _IN_USE, os.O_RDONLY | os.O_CREAT, 0o600)
            fcntl.flock(fd, fcntl.LOCK_SH)
            _held[copy] = [fd, 1]
        else:
            held[1] += 1


def _release(copy: Path) -> None:
    """Drop a lease on ``copy``; the last drops the in-use lock."""
    with _held_lock:
        held = _held.get(copy)
        if held is None:
            return
        held[1] -= 1
        if held[1] <= 0:
            os.close(held[0])
            del _held[copy]


class Lease:
    """A hold on a cached copy: it isn't pruned while the lease is alive.

    Released by ``release()`` or when the lease is garbage collected; the OS
    releases all of a process's holds when it exits.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        _acquire(path)
        self._finalizer = weakref.finalize(self, _release, path)

    def release(self) -> None:
        """Release the hold now (idempotent)."""
        self._finalizer()


def release_holds() -> None:
    """Drop all of this process's holds, letting others prune (tests)."""
    with _held_lock:
        for fd, _ in _held.values():
            os.close(fd)
        _held.clear()


def _remove_if_unused(path: Path, lock: Path) -> None:
    """Delete ``path`` (and ``lock``) unless some process holds ``lock``."""
    fd = _try_exclusive(lock)
    if fd is None and lock.exists():
        return  # in use, or still being written
    try:
        shutil.rmtree(path, ignore_errors=True)
        if lock.parent != path:
            lock.unlink(missing_ok=True)
    finally:
        if fd is not None:
            os.close(fd)


def _prune(keydir: Path, keep: Path) -> None:
    """Remove every copy but ``keep`` that no process holds, and dead extractions."""
    for child in keydir.iterdir():
        if child == keep:
            continue
        if child.name.startswith(_TMP_PREFIX):
            if child.is_dir():
                _remove_if_unused(child, keydir / f"{child.name}.lock")
            elif child.suffix == ".lock" and not child.with_suffix("").exists():
                _remove_if_unused(child.with_suffix(""), child)  # orphan lock
        elif child.is_dir():
            _remove_if_unused(child, child / _IN_USE)


def _extract(
    keydir: Path, extract: Callable[[Path], None], label: str
) -> Tuple[Path, int]:
    """Extract into a temp dir; return it (manifest written) and its writer lock.

    The writer lock is taken before the dir exists and kept until the caller
    publishes or drops it, so a concurrent prune never takes a live extraction
    for a dead one.
    """
    tmp = keydir / f"{_TMP_PREFIX}{uuid.uuid4().hex}"
    writing = os.open(f"{tmp}.lock", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(writing, fcntl.LOCK_EX)
    try:
        tmp.mkdir(mode=0o700)
        extract(tmp)
        files = _file_sizes(tmp)
        (tmp / _IN_USE).touch()
        (tmp / _MANIFEST).write_text(json.dumps({"files": files}))
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        Path(f"{tmp}.lock").unlink(missing_ok=True)
        os.close(writing)
        raise
    logger.info("%s: unpacked %d files", label, len(files))
    return tmp, writing


def materialize(
    space_name: str,
    slot: str,
    content: str,
    extract: Callable[[Path], None],
    label: str,
) -> Lease:
    """A lease on the local copy of a snapshot, unpacking it if needed.

    Args:
        space_name: The gb_spaces row the snapshot belongs to.
        slot: Its slot ("space" or "base_uri_<i>").
        content: The stored archive; its digest identifies the copy.
        extract: Unpacks the archive into a given empty directory.
        label: For messages.
    """
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]
    keydir = key_dir(space_name, slot)
    lock = keydir / _KEY_LOCK
    with _locked(lock, fcntl.LOCK_SH):  # pruning needs LOCK_EX: can't interleave
        copy = _find(keydir, digest)
        if copy is not None:
            return Lease(copy)
    tmp, writing = _extract(keydir, extract, label)
    try:
        with _locked(lock, fcntl.LOCK_EX):
            copy = _find(keydir, digest)  # a concurrent extraction may have won
            if copy is None:
                copy = keydir / f"{digest}-{uuid.uuid4().hex[:12]}"
                os.rename(tmp, copy)  # a fresh name: never replaces anything
            lease = Lease(copy)
            _prune(keydir, keep=copy)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # no-op once published
        Path(f"{tmp}.lock").unlink(missing_ok=True)
        os.close(writing)
    return lease
