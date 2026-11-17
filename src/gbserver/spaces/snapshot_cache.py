#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""Local cache of unpacked db-space snapshots, safe across threads and processes.

Layout, under a private root (``GBSERVER_DB_SPACE_CACHE_DIR``, default a
per-user dir in the temp dir)::

    <root>/<space>/<slot>/
        .lock                      per-key lock: lookup, publish and prune
        <digest>-<id>/             a published copy; never renamed or rewritten
            .gb-snapshot-complete  manifest (file count, bytes), written last
            .gb-in-use             shared-locked by each process using the copy
        .tmp-<digest>-<id>/        an extraction, locked by its writer

The path names the space and slot, so a cached path maps back to its db:// URI
with no shared state. Copies are found by digest and checked against their
manifest; a damaged one is never served. A copy is deleted only when no
process holds its in-use lock. Holds are taken on resolve and kept for the
life of the process; the OS drops them when it exits or crashes, so a dead
process never pins a copy. Extraction runs outside the key lock, so it never
blocks readers; two racing extractions just publish two copies, and the
spare is pruned. Locks are POSIX ``flock``: the root must be on a local
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
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Iterator, Optional, Tuple

from gbserver.types.constants import ENV_VAR_DB_SPACE_CACHE_DIR
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

_KEY_LOCK = ".lock"
_MANIFEST = ".gb-snapshot-complete"
_IN_USE = ".gb-in-use"
_WRITING = ".gb-writing"
_TMP_PREFIX = ".tmp-"
_BOOKKEEPING = frozenset({_MANIFEST, _IN_USE, _WRITING})

# Copies this process holds a shared in-use lock on: path -> open fd.
_held: Dict[Path, int] = {}
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
    path = _private_root() / urllib.parse.quote(space_name, safe="") / slot
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def parse_cached_path(path: Path) -> Optional[Tuple[str, str, str]]:
    """``(space, slot, subpath)`` for a path inside a cached copy, else None."""
    try:
        rel = path.resolve().relative_to(get_cache_root().resolve())
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 3 or parts[2].startswith(_TMP_PREFIX):
        return None
    return urllib.parse.unquote(parts[0]), parts[1], "/".join(parts[3:])


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
    """An fd holding ``LOCK_EX`` on ``path`` without waiting, or None if held."""
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


def _tree_stats(copy: Path) -> Tuple[int, int]:
    """(file count, total bytes) of a copy's content, bookkeeping excluded."""
    files = size = 0
    for dirpath, _, names in os.walk(copy):
        for name in names:
            if Path(dirpath) == copy and name in _BOOKKEEPING:
                continue
            files += 1
            size += os.lstat(os.path.join(dirpath, name)).st_size
    return files, size


def _is_complete(copy: Path) -> bool:
    """Whether a copy's content still matches the manifest written after it."""
    try:
        manifest = json.loads((copy / _MANIFEST).read_text())
    except (OSError, ValueError):
        return False
    return [manifest.get("files"), manifest.get("bytes")] == list(_tree_stats(copy))


def _find(keydir: Path, digest: str) -> Optional[Path]:
    """A complete published copy of ``digest``, if any."""
    for child in sorted(keydir.glob(f"{digest}-*")):
        if child.is_dir() and _is_complete(child):
            return child
    return None


def _hold(copy: Path) -> None:
    """Take (once per process) a shared in-use lock on a copy, kept until exit."""
    with _held_lock:
        if copy in _held:
            return
        fd = os.open(copy / _IN_USE, os.O_RDONLY)
        fcntl.flock(fd, fcntl.LOCK_SH)
        _held[copy] = fd


def release_holds() -> None:
    """Drop this process's in-use locks, letting other processes prune (tests)."""
    with _held_lock:
        for fd in _held.values():
            os.close(fd)
        _held.clear()


def _remove_if_unused(path: Path, lock_name: str) -> None:
    """Delete ``path`` if no process holds its lock file (or it has none)."""
    lock = path / lock_name
    fd = _try_exclusive(lock) if lock.exists() else None
    if lock.exists() and fd is None:
        return  # in use, or still being written
    try:
        shutil.rmtree(path, ignore_errors=True)
    finally:
        if fd is not None:
            os.close(fd)


def _prune(keydir: Path, keep: Path) -> None:
    """Remove every copy but ``keep`` that no process uses, and dead extractions."""
    for child in keydir.iterdir():
        if child == keep or not child.is_dir():
            continue
        if child.name.startswith(_TMP_PREFIX):
            _remove_if_unused(child, _WRITING)
        else:
            _remove_if_unused(child, _IN_USE)


def _extract(
    keydir: Path, digest: str, extract: Callable[[Path], None], label: str
) -> Tuple[Path, int]:
    """Extract into a temp dir; return it, manifest written, and its writer lock.

    The caller keeps the writer lock until it publishes or drops the dir, so a
    concurrent prune never takes a finished extraction for a dead one.
    """
    tmp = keydir / f"{_TMP_PREFIX}{digest}-{uuid.uuid4().hex}"
    tmp.mkdir(mode=0o700)
    writing = os.open(tmp / _WRITING, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(writing, fcntl.LOCK_EX)
    try:
        extract(tmp)
        (tmp / _IN_USE).touch()
        files, size = _tree_stats(tmp)
        (tmp / _MANIFEST).write_text(json.dumps({"files": files, "bytes": size}))
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        os.close(writing)
        raise
    logger.info("%s: unpacked %d files (%d bytes)", label, files, size)
    return tmp, writing


def materialize(
    space_name: str,
    slot: str,
    content: str,
    extract: Callable[[Path], None],
    label: str,
) -> Path:
    """The local copy of a snapshot, unpacking it with ``extract`` if needed.

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
    with _locked(lock, fcntl.LOCK_EX):
        copy = _find(keydir, digest)
        if copy is not None:
            _hold(copy)
            _prune(keydir, keep=copy)
            return copy
    tmp, writing = _extract(keydir, digest, extract, label)
    try:
        with _locked(lock, fcntl.LOCK_EX):
            copy = _find(keydir, digest)  # a concurrent extraction may have won
            if copy is None:
                copy = keydir / f"{digest}-{uuid.uuid4().hex[:12]}"
                os.rename(tmp, copy)  # a fresh name: never replaces anything
                (copy / _WRITING).unlink(missing_ok=True)
            else:
                shutil.rmtree(tmp, ignore_errors=True)
            _hold(copy)
            _prune(keydir, keep=copy)
    finally:
        os.close(writing)
    return copy
