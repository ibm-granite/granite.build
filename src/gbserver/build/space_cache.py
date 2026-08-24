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
``space.yaml``, resolves base_uris, and syncs secrets — and leaks a ``/tmp``
checkout (see ``Space.__init__``). Over the pod's lifetime that fills the
container's ephemeral storage and the pod is evicted. Caching the built ``Space``
amortizes the construction cost AND bounds the checkout churn (a cache hit builds
nothing), while the TTL doubles as the staleness bound: on expiry the Space is
rebuilt with a fresh, forced pull so remote ``space.yaml`` changes are picked up.

Design
------
* **Per-process, per-worker.** The rest-server runs several uvicorn worker
  processes; each imports this module and has its own dict + lock. No
  cross-process sharing, so no filesystem/flock coordination is needed.
* **Key = (space_uri, username).** ``Space`` resolves per-user secrets, so a
  built Space is user-specific; keying by user prevents serving one user's
  resolved secrets to another. Repo-clone de-duplication happens a layer down in
  ``GitURI``'s clone cache, not here.
* **Thread-local re-application on every hit (load-bearing).** ``Space``
  construction records resolution state in *thread-locals*
  (``URI.set_space_config`` / ``SpaceURI.set_baseuris``), and the ``/validate``
  route is a synchronous handler run in Starlette's threadpool — so the thread
  that serves a cached hit is generally NOT the thread that built the Space.
  Every hit therefore re-applies the stored resolution state on the current
  thread before returning; otherwise validation silently resolves against the
  default ``["file:"]`` base_uris.
* **Reclaim-on-replace with a grace sweep.** The ``Space``'s backing checkout
  must stay alive while the Space is cached (base_uris/assetstores reference it).
  When an entry is replaced (miss after TTL expiry), the old Space's checkout is
  not deleted inline — an in-flight request may still hold it — but queued and
  swept after a grace period on later calls.
"""

import threading
import time
from typing import Dict, Optional, Tuple

from gbcommon.uri.space import SpaceURI
from gbcommon.uri.uri import URI
from gbserver.build.space import Space
from gbserver.types.constants import (
    GBSERVER_SPACE_CACHE_ENABLED,
    GBSERVER_SPACE_CACHE_TTL,
)
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

# Grace period before a replaced Space's checkout is actually reclaimed, giving
# any in-flight request that obtained the old Space time to finish using it.
_RECLAIM_GRACE_SECONDS = 120.0

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


_cache: Dict[_CacheKey, _CacheEntry] = {}
_lock = threading.Lock()
# (reclaim_at, space) pairs for Spaces whose checkout is pending deletion.
_reclaim_queue: list = []


def _apply_thread_local_state(space: Space) -> None:
    """Re-apply the Space's resolution state to the CURRENT thread.

    Reproduces the thread-local side effects that ``Space.__init__`` had on its
    building thread, so a cache hit served on a different thread resolves assets
    correctly. See the module docstring.
    """
    if space.space_config is not None:
        URI.set_space_config(space.space_config)
    SpaceURI.set_baseuris(base_uris=space.base_uris, space_secrets=space.secrets)


def _sweep_reclaim_queue(now: float) -> None:
    """Delete checkouts whose grace period has elapsed. Caller holds ``_lock``."""
    if not _reclaim_queue:
        return
    still_pending = []
    for reclaim_at, space in _reclaim_queue:
        if now >= reclaim_at:
            space.reclaim()
        else:
            still_pending.append((reclaim_at, space))
    _reclaim_queue[:] = still_pending


def _build_space(uri: str, username: Optional[str]) -> Space:
    """Build a fresh Space with a forced pull (so the definition is current).

    ``manage_tmpdir=False``: this cache owns the checkout and reclaims it on
    eviction, so it must not also be registered for atexit cleanup.
    """
    return Space(uri, username=username, force_fetch=True, manage_tmpdir=False)


def get_cached_space(
    uri: str,
    username: Optional[str],
    *,
    ttl: float = GBSERVER_SPACE_CACHE_TTL,
) -> Space:
    """Return a cached ``Space`` for ``(uri, username)``, building on miss/expiry.

    When the cache is disabled (``GBSERVER_SPACE_CACHE_ENABLED=false``), builds a
    fresh Space per call with ``manage_tmpdir=True`` so its checkout is still
    reclaimed at process exit — disabling the cache never reintroduces the leak.

    The returned Space has its resolution state applied to the calling thread.
    """
    if not GBSERVER_SPACE_CACHE_ENABLED:
        # manage_tmpdir=True -> atexit-reclaimed; __init__ already applied the
        # thread-local state on this (the calling) thread.
        return Space(uri, username=username, force_fetch=True, manage_tmpdir=True)

    key: _CacheKey = (URI.get_uristr(uri), username)

    while True:
        now = time.monotonic()
        wait_event: Optional[threading.Event] = None
        # Only meaningful on the build-slot branch below; initialized here so it
        # is always bound before the post-lock code that reads it.
        old_space: Optional[Space] = None
        new_entry: Optional[_CacheEntry] = None
        building: Optional[threading.Event] = None
        with _lock:
            _sweep_reclaim_queue(now)
            entry = _cache.get(key)

            if entry is not None and entry.building is not None:
                # Another thread is building this key; wait and retry.
                wait_event = entry.building
            elif (
                entry is not None
                and entry.space is not None
                and (now - entry.built_at) < ttl
            ):
                # Fresh hit: re-apply thread-local state on THIS thread, return.
                space = entry.space
                _apply_thread_local_state(space)
                return space
            else:
                # Miss or expired: claim the build slot for this key.
                building = threading.Event()
                new_entry = _CacheEntry(space=None, built_at=now)
                new_entry.building = building
                # Preserve the old space (if any) so we can reclaim it after swap.
                old_space = entry.space if entry is not None else None
                _cache[key] = new_entry

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
            if old_space is not None:
                _reclaim_queue.append(
                    (time.monotonic() + _RECLAIM_GRACE_SECONDS, old_space)
                )
        building.set()

        _apply_thread_local_state(space)
        return space


def clear_space_cache() -> None:
    """Drop all cached Spaces and reclaim their checkouts. For tests/shutdown."""
    with _lock:
        for entry in _cache.values():
            if entry.space is not None:
                entry.space.reclaim()
        _cache.clear()
        for _reclaim_at, space in _reclaim_queue:
            space.reclaim()
        _reclaim_queue.clear()
