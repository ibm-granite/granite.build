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

"""In-memory, TTL'd cache of :class:`~gbserver.build.space.Space` objects.

Motivation
----------
On the rest-server, ``POST /validate`` constructs a fresh ``Space`` per request
(``buildrunner/validation.py``). Each construction pulls the space repo, parses
``space.yaml``, resolves base_uris, and syncs secrets — the secret sync in
particular may hit a remote secrets manager (``Space._fetch_secrets`` ->
``get_secrets_with_groups``). Caching the built ``Space`` amortizes all of that,
and the TTL doubles as the staleness bound: on expiry the Space is rebuilt (a
rebuild reuses ``GitURI``'s clone cache rather than force-cloning; freshness comes
from the rebuild happening at all), so remote ``space.yaml`` / secret changes are
picked up within the TTL.

(The ``/tmp`` checkout leak that first motivated this work is fixed in
``Space.__init__`` itself, which deletes its throwaway checkout before returning.
Most Spaces therefore hold no on-disk resource; the exception is a Space that
retained its checkout because a bundled local asset/store resolved inside it —
that one is reclaimed via ``Space.reclaim`` when this cache evicts or replaces
it.)

Design
------
* **Per-process, per-worker.** The rest-server runs several uvicorn worker
  processes; each imports this module and has its own dict + lock. No
  cross-process sharing, so no filesystem/flock coordination is needed.
* **Key = (space_uri, username).** ``Space`` resolves per-user secrets, so a
  built Space is user-specific; keying by user prevents serving one user's
  resolved secrets to another. Repo-clone de-duplication happens a layer down in
  ``GitURI``'s clone cache, not here.
* **TTL'd, count-bounded (LRU).** Entries expire after ``GBSERVER_SPACE_CACHE_TTL``
  (default 15 min) and the total is capped at ``GBSERVER_SPACE_CACHE_MAX_ENTRIES``
  (LRU eviction) as an OOM backstop — since the checkout is no longer tied to the
  Space's lifetime, the TTL can be longer, so the cap is what bounds memory from
  many distinct (space, user) pairs.
* **Thread-local re-application on every hit (load-bearing).** ``Space``
  construction records resolution state in *thread-locals* —
  ``URI.set_space_config`` / ``SpaceURI.set_baseuris`` AND the loaded
  ``Assetstore`` objects (``Assetstore``'s own thread-local dict) — and the
  ``/validate`` route is a synchronous handler run in Starlette's threadpool, so
  the thread that serves a cached hit is generally NOT the thread that built the
  Space, and downstream build validation reads those thread-locals when resolving
  ``space://`` URIs / assets. Every hit therefore re-applies the stored config,
  base_uris, and assetstores on the current thread before returning; otherwise
  validation silently resolves against the default ``["file:"]`` base_uris or
  hits ``AttributeError`` on the missing assetstore dict.
"""

import threading
import time
from collections import OrderedDict
from typing import Optional, Tuple

from gbcommon.uri.space import SpaceURI
from gbcommon.uri.uri import URI
from gbserver.asset.assetstore import Assetstore
from gbserver.build.space import Space
from gbserver.types.constants import (
    GBSERVER_SPACE_CACHE_ENABLED,
    GBSERVER_SPACE_CACHE_MAX_ENTRIES,
    GBSERVER_SPACE_CACHE_TTL,
)
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

_CacheKey = Tuple[str, Optional[str]]


class _CacheEntry:  # pylint: disable=too-few-public-methods
    """A cached Space plus the metadata the factory needs to manage it."""

    __slots__ = ("space", "built_at", "building")

    def __init__(self, space: Optional[Space], built_at: float):
        self.space = space
        self.built_at = built_at
        # Set while a thread is (re)building this key; other threads for the same
        # key wait on it instead of building a duplicate (thundering-herd guard).
        self.building: Optional[threading.Event] = None


# OrderedDict for LRU: most-recently-used moved to the end, LRU evicted from the
# front when the entry count exceeds GBSERVER_SPACE_CACHE_MAX_ENTRIES.
_cache: "OrderedDict[_CacheKey, _CacheEntry]" = OrderedDict()
_lock = threading.Lock()


def _evict_over_cap() -> None:
    """Evict least-recently-used entries past the cap. Caller holds ``_lock``.

    Never evicts an entry that is mid-build (``building`` set) — dropping its
    placeholder would strand threads waiting on the event. Most Spaces hold no
    on-disk resource, but one that retained its checkout (a bundled local
    asset/store resolved inside it) is reclaimed via ``Space.reclaim`` on evict.
    """
    max_entries = GBSERVER_SPACE_CACHE_MAX_ENTRIES
    if max_entries <= 0:
        return
    # Iterate from LRU end; skip in-flight builds so we never strand waiters.
    for key in list(_cache.keys()):
        if len(_cache) <= max_entries:
            break
        entry = _cache[key]
        if entry.building is not None:
            continue
        del _cache[key]
        if entry.space is not None:
            entry.space.reclaim()
        logger.info("space cache over cap (%d); evicted %r", max_entries, key)


def _apply_thread_local_state(space: Space) -> None:
    """Re-apply the Space's resolution state to the CURRENT thread.

    Reproduces the thread-local side effects that ``Space.__init__`` had on its
    building thread, so a cache hit served on a different thread resolves assets
    correctly. Three pieces of state, all thread-local:
      - URI.set_space_config / SpaceURI.set_baseuris (space config + base_uris);
      - Assetstore._thread_local.assetstores — the loaded store objects, without
        which Asset.get_assetstore raises AttributeError on a fresh threadpool
        thread (or resolves against the wrong stores on a stale one).
    The assetstore dict is additive and per-thread, so we merge this space's
    captured stores in (seeding the dict if the thread has none yet) rather than
    replacing it.
    """
    if space.space_config is not None:
        URI.set_space_config(space.space_config)
    SpaceURI.set_baseuris(base_uris=space.base_uris, space_secrets=space.secrets)
    Assetstore.merge_thread_local_assetstores(space.assetstores)


def _build_space(uri: str, username: Optional[str]) -> Space:
    """Build a fresh Space.

    ``force_fetch=False``: reuse GitURI's clone cache rather than re-cloning on
    every build. Freshness is provided by the cache TTL (an expired entry is
    rebuilt), not by forcing a clone on each miss — matching the pre-cache
    ``/validate`` behavior, which also did not force-fetch the space pull.
    """
    return Space(uri, username=username, force_fetch=False)


def get_cached_space(
    uri: str,
    username: Optional[str],
    *,
    ttl: float = GBSERVER_SPACE_CACHE_TTL,
) -> Space:
    """Return a cached ``Space`` for ``(uri, username)``, building on miss/expiry.

    When the cache is disabled (``GBSERVER_SPACE_CACHE_ENABLED=false``), builds a
    fresh Space per call. Either way the returned Space has its resolution state
    applied to the calling thread.
    """
    if not GBSERVER_SPACE_CACHE_ENABLED:
        # __init__ already applied the thread-local state on this (the calling)
        # thread, so no re-application is needed here.
        return _build_space(uri, username)

    key: _CacheKey = (URI.get_uristr(uri), username)

    while True:
        now = time.monotonic()
        wait_event: Optional[threading.Event] = None
        new_entry: Optional[_CacheEntry] = None
        building: Optional[threading.Event] = None
        # An expired Space we are replacing; its checkout (if any) is reclaimed
        # after the rebuild succeeds.
        superseded: Optional[Space] = None
        with _lock:
            entry = _cache.get(key)

            if entry is not None and entry.building is not None:
                # Another thread is building this key; wait and retry.
                wait_event = entry.building
            elif (
                entry is not None
                and entry.space is not None
                and (now - entry.built_at) < ttl
            ):
                # Fresh hit: mark MRU, re-apply thread-local state, return.
                _cache.move_to_end(key)
                space = entry.space
                _apply_thread_local_state(space)
                return space
            else:
                # Miss or expired: claim the build slot for this key (MRU).
                superseded = entry.space if entry is not None else None
                building = threading.Event()
                new_entry = _CacheEntry(space=None, built_at=now)
                new_entry.building = building
                _cache[key] = new_entry
                _cache.move_to_end(key)

        if wait_event is not None:
            # Don't hold the lock while another thread builds; retry after.
            wait_event.wait(timeout=60)
            continue

        # We own the build slot. Build outside the lock (it does network I/O).
        try:
            space = _build_space(uri, username)
        except Exception:
            # Build failed: drop our placeholder so the next caller retries
            # rather than waiting on an Event that never fires.
            with _lock:
                cur = _cache.get(key)
                if cur is new_entry:
                    del _cache[key]
            building.set()
            raise

        with _lock:
            new_entry.space = space
            new_entry.built_at = time.monotonic()
            new_entry.building = None
            _cache.move_to_end(key)
            # Now that this entry holds a real Space (building is cleared), it's
            # eligible for eviction, so enforce the cap.
            _evict_over_cap()
        building.set()

        # Reclaim the replaced Space's checkout, if it retained one. Done after
        # the new build succeeds and outside the lock; the rebuilt Space has its
        # own (or no) checkout, so this frees only the superseded one.
        if superseded is not None:
            superseded.reclaim()

        _apply_thread_local_state(space)
        return space


def clear_space_cache() -> None:
    """Drop all cached Spaces (reclaiming any retained checkouts). Tests/shutdown."""
    with _lock:
        for entry in _cache.values():
            if entry.space is not None:
                entry.space.reclaim()
        _cache.clear()
