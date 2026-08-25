#!/usr/bin/env python3

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

"""Bounded, per-thread on-disk cache of hash-keyed subdirectories.

Several long-lived call paths cache pulled/cloned/synced content under a
``tempfile.mkdtemp()`` root stored in a class ``threading.local()``, keyed by a
content hash, and never reclaim it — so on the rest-server (and any thread-mode
build-watcher) the root grows one subdir per distinct key for the life of the
thread, filling ephemeral storage. See ``GitURI.get_repo_from_cache``,
``Environment.load_environment_config`` and ``Step.__init__``.

``BoundedThreadLocalCache`` keeps the design that makes those caches correct —
**per-thread isolation, so no locking is needed** even across the rest-server's
several worker processes and their threadpool threads — while bounding each
thread's footprint to ``max_entries`` most-recently-used subdirs. Past the cap
the least-recently-used subdir is ``rmtree``'d.

The cache manages only the *set* of subdirs and their LRU order; the caller
populates each subdir (clone/sync/copy) and decides whether an existing subdir
is reusable. ``path_for(key)`` returns a stable directory path for ``key`` on the
current thread and marks it most-recently-used; a subsequent call for the same
key returns the same path (unless it was evicted in between). Eviction only ever
deletes directories this cache created, so a caller that finds ``path.is_dir()``
false simply repopulates it, exactly as it did when the dir had never existed.
"""

import shutil
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class BoundedThreadLocalCache:
    """A per-thread, LRU-bounded set of hash-keyed subdirectories.

    Each instance owns its own ``threading.local()``; construct one per distinct
    cache (e.g. one for git clones, one for environment assets). All state is
    thread-local, so instances are safe to share as module/class attributes and
    call from many threads without locking.
    """

    def __init__(self, name: str, max_entries: int):
        # name is only used to make temp-dir prefixes and logs legible.
        self._name = name
        self._max_entries = max(1, int(max_entries))
        self._tl = threading.local()

    def _entries(self) -> "OrderedDict[str, Path]":
        if not hasattr(self._tl, "entries"):
            self._tl.entries = OrderedDict()
        return self._tl.entries

    def _root(self) -> Path:
        if not hasattr(self._tl, "root"):
            self._tl.root = Path(tempfile.mkdtemp(prefix=f"gb-{self._name}-"))
        return self._tl.root

    def path_for(self, key: str) -> Path:
        """Return this thread's subdir path for ``key``, marking it MRU.

        Does not create or populate the directory — the caller clones/syncs into
        the returned path. Requesting a key beyond ``max_entries`` evicts (and
        ``rmtree``s) the least-recently-used subdir on this thread.
        """
        entries = self._entries()
        root = self._root()
        path = entries.get(key)
        if path is not None:
            # Seen before: refresh recency and hand back the same path.
            entries.move_to_end(key)
            return path

        path = root / key
        entries[key] = path
        entries.move_to_end(key)
        self._evict_over_cap()
        return path

    def _evict_over_cap(self) -> None:
        entries = self._entries()
        while len(entries) > self._max_entries:
            _evicted_key, evicted_path = entries.popitem(last=False)  # LRU
            logger.info(
                "%s cache over cap (%d); evicting %s",
                self._name,
                self._max_entries,
                evicted_path,
            )
            shutil.rmtree(evicted_path, ignore_errors=True)

    def discard(self, key: str) -> None:
        """Forget and ``rmtree`` a single key's subdir on this thread, if present.

        Idempotent. Used for targeted reclaim (e.g. a caller that knows a key is
        no longer needed); not required for normal LRU operation.
        """
        entries = self._entries()
        path = entries.pop(key, None)
        if path is not None:
            shutil.rmtree(path, ignore_errors=True)

    def clear(self) -> None:
        """Drop and ``rmtree`` everything this thread has cached, incl. the root.

        Removing the whole root (not just the subdirs) avoids leaking an empty
        mkdtemp root per thread per clear; the root is recreated lazily on the
        next ``path_for``.
        """
        root = getattr(self._tl, "root", None)
        if root is not None:
            shutil.rmtree(root, ignore_errors=True)
            del self._tl.root
        if hasattr(self._tl, "entries"):
            self._entries().clear()

    def current_size(self) -> int:
        """Number of tracked subdirs on the current thread (for tests/metrics)."""
        return len(self._entries()) if hasattr(self._tl, "entries") else 0

    def root_if_created(self) -> Optional[Path]:
        """The thread-local root if it has been created, else None (for tests)."""
        return getattr(self._tl, "root", None)
