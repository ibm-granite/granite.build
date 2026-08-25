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

"""Unit tests for BoundedThreadLocalCache (bounds Leaks B/C/D).

The cache replaces the previously-unbounded thread-local ``mkdtemp`` roots in
``GitURI``/``Environment``/``Step``. These tests pin the two properties that
matter: it stays bounded (LRU-evicts + ``rmtree``s past the cap) and it is
per-thread isolated (each thread gets its own root, so no locking is needed).
"""

import threading
from concurrent.futures import ThreadPoolExecutor

from gbserver.utils.bounded_cache import BoundedThreadLocalCache


def _populate(path):
    """Simulate a caller cloning/syncing into the returned slot."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "content.txt").write_text("x")


def test_same_key_returns_same_path_and_reuses():
    cache = BoundedThreadLocalCache("t", max_entries=4)
    p1 = cache.path_for("k1")
    p2 = cache.path_for("k1")
    assert p1 == p2
    assert cache.current_size() == 1


def test_evicts_lru_past_cap_and_rmtrees():
    cache = BoundedThreadLocalCache("t", max_entries=2)
    a = cache.path_for("a")
    _populate(a)
    b = cache.path_for("b")
    _populate(b)
    # Touch 'a' so 'b' becomes the LRU, then add 'c' to force one eviction.
    cache.path_for("a")
    c = cache.path_for("c")
    _populate(c)

    assert cache.current_size() == 2
    # 'b' was least-recently-used -> evicted and deleted from disk.
    assert not b.exists()
    # 'a' (touched) and 'c' (newest) survive.
    assert a.exists()
    assert c.exists()


def test_discard_removes_single_key():
    cache = BoundedThreadLocalCache("t", max_entries=4)
    a = cache.path_for("a")
    _populate(a)
    cache.discard("a")
    assert not a.exists()
    assert cache.current_size() == 0
    # Idempotent.
    cache.discard("a")


def test_clear_removes_all():
    cache = BoundedThreadLocalCache("t", max_entries=4)
    for k in ("a", "b", "c"):
        _populate(cache.path_for(k))
    cache.clear()
    assert cache.current_size() == 0


def test_clear_removes_root_dir():
    """clear() removes the mkdtemp root too, not just subdirs (no per-clear leak)."""
    cache = BoundedThreadLocalCache("t", max_entries=4)
    _populate(cache.path_for("a"))
    root = cache.root_if_created()
    assert root is not None and root.exists()
    cache.clear()
    assert not root.exists()
    # Root is recreated lazily on next use, under a new path.
    new_path = cache.path_for("b")
    assert new_path.parent.exists()


def test_per_thread_isolation():
    """Each thread must get its own root and its own entry set — the property
    that lets the real caches run lock-free across threadpool threads."""
    cache = BoundedThreadLocalCache("t", max_entries=4)

    main_root = cache.path_for("k").parent
    main_tid = threading.get_ident()

    def other_thread():
        p = cache.path_for("k")
        return p.parent, threading.get_ident(), cache.current_size()

    with ThreadPoolExecutor(max_workers=1) as ex:
        other_root, other_tid, other_size = ex.submit(other_thread).result()

    assert other_tid != main_tid
    # Different thread -> different root dir, independent size accounting.
    assert other_root != main_root
    assert other_size == 1


def test_max_entries_floor_of_one():
    """A nonsensical cap is clamped to at least 1 (never 0/negative)."""
    cache = BoundedThreadLocalCache("t", max_entries=0)
    a = cache.path_for("a")
    _populate(a)
    b = cache.path_for("b")
    _populate(b)
    assert cache.current_size() == 1
    assert not a.exists()
    assert b.exists()
