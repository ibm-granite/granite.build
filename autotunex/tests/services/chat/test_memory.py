from __future__ import annotations

from autotunex.services.chat.memory import ConversationMemory


def test_put_then_get_returns_the_stored_messages() -> None:
    mem = ConversationMemory(max_threads=10, ttl_seconds=100.0, clock=lambda: 0.0)

    mem.put("t1", [{"role": "user", "content": "hi"}])

    assert mem.get("t1") == [{"role": "user", "content": "hi"}]


def test_get_unknown_thread_is_empty() -> None:
    mem = ConversationMemory(max_threads=10, ttl_seconds=100.0, clock=lambda: 0.0)

    assert mem.get("missing") == []


def test_expired_thread_is_dropped() -> None:
    now = {"t": 0.0}
    mem = ConversationMemory(max_threads=10, ttl_seconds=10.0, clock=lambda: now["t"])
    mem.put("t1", [{"role": "user", "content": "hi"}])

    now["t"] = 20.0

    assert mem.get("t1") == []


def test_lru_evicts_the_oldest_when_over_capacity() -> None:
    mem = ConversationMemory(max_threads=2, ttl_seconds=1000.0, clock=lambda: 0.0)
    mem.put("a", [{"role": "user", "content": "a"}])
    mem.put("b", [{"role": "user", "content": "b"}])

    mem.put("c", [{"role": "user", "content": "c"}])

    assert mem.get("a") == []
    assert mem.get("b") != []
    assert mem.get("c") != []


def test_get_refreshes_recency_so_the_untouched_thread_is_evicted() -> None:
    mem = ConversationMemory(max_threads=2, ttl_seconds=1000.0, clock=lambda: 0.0)
    mem.put("a", [{"role": "user", "content": "a"}])
    mem.put("b", [{"role": "user", "content": "b"}])

    mem.get("a")
    mem.put("c", [{"role": "user", "content": "c"}])

    assert mem.get("b") == []
    assert mem.get("a") != []
    assert mem.get("c") != []
