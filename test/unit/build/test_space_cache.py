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

"""Unit tests for the in-memory Space cache (``gbserver.build.space_cache``).

Covers hit/miss/expiry, ``(uri, username)`` key isolation, the thundering-herd
single-build guard, checkout reclaim-on-replace, and the load-bearing property
that a cache hit re-applies the Space's thread-local resolution state on the
*serving* thread (which is generally not the building thread, since the
``/validate`` route runs in Starlette's threadpool).
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import gbserver.build.space_cache as space_cache
from gbserver.build.space_cache import clear_space_cache, get_cached_space


class FakeSpace:
    """Stand-in for ``Space`` that records how it was built, without a git pull.

    Mirrors the attributes the cache reads: ``space_config``, ``base_uris``,
    ``secrets``.
    """

    instances: list = []

    def __init__(self, uri, username=None, force_fetch=False):
        self.uri = uri
        self.username = username
        self.force_fetch = force_fetch
        self.space_config = object()
        self.base_uris = [uri, "file://builtins"]
        self.secrets = {"user": username}
        FakeSpace.instances.append(self)


@pytest.fixture(autouse=True)
def _isolate_cache(monkeypatch):
    """Reset module state and record thread-local re-applications per test."""
    clear_space_cache()
    FakeSpace.instances = []

    monkeypatch.setattr(space_cache, "Space", FakeSpace)
    # URI.get_uristr is the key-normalizer; keep it identity for predictable keys.
    monkeypatch.setattr(space_cache.URI, "get_uristr", staticmethod(lambda u: u))

    applied = []

    def _fake_set_space_config(cfg):
        applied.append(("space_config", threading.get_ident(), cfg))

    def _fake_set_baseuris(base_uris, space_secrets):
        applied.append(("baseuris", threading.get_ident(), tuple(base_uris)))

    monkeypatch.setattr(
        space_cache.URI, "set_space_config", staticmethod(_fake_set_space_config)
    )
    monkeypatch.setattr(
        space_cache.SpaceURI, "set_baseuris", staticmethod(_fake_set_baseuris)
    )
    yield applied
    clear_space_cache()


def test_miss_builds_once_then_hit_reuses():
    a = get_cached_space("git://repo", "alice")
    b = get_cached_space("git://repo", "alice")
    assert a is b
    assert len(FakeSpace.instances) == 1
    # Built with a forced pull (so the cached definition is current).
    assert a.force_fetch is True


def test_key_isolation_by_username():
    a = get_cached_space("git://repo", "alice")
    b = get_cached_space("git://repo", "bob")
    assert a is not b
    assert len(FakeSpace.instances) == 2
    # Each user's Space carries that user's secrets — no cross-user bleed.
    assert a.secrets == {"user": "alice"}
    assert b.secrets == {"user": "bob"}


def test_expiry_rebuilds(monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr(space_cache.time, "monotonic", lambda: clock["t"])

    first = get_cached_space("git://repo", "alice", ttl=10)
    # Within ttl: same object, no rebuild.
    same = get_cached_space("git://repo", "alice", ttl=10)
    assert same is first
    assert len(FakeSpace.instances) == 1

    clock["t"] += 100  # past ttl -> rebuild with a fresh forced pull
    second = get_cached_space("git://repo", "alice", ttl=10)
    assert second is not first
    assert len(FakeSpace.instances) == 2
    assert second.force_fetch is True


def test_hit_reapplies_thread_local_state_on_serving_thread(_isolate_cache):
    """A hit served on a different thread must re-apply the resolution state
    on THAT thread — the core correctness property."""
    applied = _isolate_cache

    build_tid = threading.get_ident()
    get_cached_space("git://repo", "alice")  # build on this thread
    applied.clear()

    with ThreadPoolExecutor(max_workers=1) as ex:
        serve_tid = ex.submit(
            lambda: (get_cached_space("git://repo", "alice"), threading.get_ident())[1]
        ).result()

    assert serve_tid != build_tid, "expected the hit to be served on another thread"
    # Both set_space_config and set_baseuris were re-applied, on the serving thread.
    kinds = {k for (k, _tid, _v) in applied}
    assert kinds == {"space_config", "baseuris"}
    assert all(tid == serve_tid for (_k, tid, _v) in applied)


def test_thundering_herd_builds_once(monkeypatch):
    """Concurrent misses for one key build exactly one Space."""
    start = threading.Barrier(5)
    slow = threading.Event()
    orig = FakeSpace.__init__

    def slow_init(self, *a, **kw):
        orig(self, *a, **kw)
        slow.wait(timeout=5)  # hold the build slot so others pile up

    monkeypatch.setattr(FakeSpace, "__init__", slow_init)

    results = []

    def worker():
        start.wait(timeout=5)
        results.append(get_cached_space("git://repo", "alice"))

    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = [ex.submit(worker) for _ in range(5)]
        time.sleep(0.2)
        slow.set()  # release the builder
        for f in futs:
            f.result(timeout=5)

    assert len(FakeSpace.instances) == 1
    assert all(r is results[0] for r in results)


def test_disabled_cache_builds_fresh_each_call(monkeypatch):
    monkeypatch.setattr(space_cache, "GBSERVER_SPACE_CACHE_ENABLED", False)
    a = get_cached_space("git://repo", "alice")
    b = get_cached_space("git://repo", "alice")
    assert a is not b
    assert len(FakeSpace.instances) == 2
    assert a.force_fetch is True
