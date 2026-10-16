# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""In-process, bounded per-thread conversation store for the chat agent.

Replaces langgraph's ``InMemorySaver``. State is lost on process restart and is
NOT shared across workers — for a multi-worker deployment, back this with a
shared store. Bounded by an LRU cap and an idle TTL so a long-running process
does not grow without limit.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any


class ConversationMemory:
    """Keyed by ``thread_id``; each value is the OpenAI-shaped message list."""

    def __init__(
        self,
        *,
        max_threads: int,
        ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max = max_threads
        self._ttl = ttl_seconds
        self._clock = clock
        self._store: OrderedDict[str, tuple[float, list[dict[str, Any]]]] = OrderedDict()

    def get(self, thread_id: str) -> list[dict[str, Any]]:
        """Return the stored messages, or ``[]`` if unknown or expired."""
        entry = self._store.get(thread_id)
        if entry is None:
            return []
        stamped_at, messages = entry
        if self._clock() - stamped_at > self._ttl:
            del self._store[thread_id]
            return []
        self._store.move_to_end(thread_id)
        return messages

    def put(self, thread_id: str, messages: list[dict[str, Any]]) -> None:
        """Store (replacing) the messages for ``thread_id`` and evict if over cap."""
        self._store[thread_id] = (self._clock(), messages)
        self._store.move_to_end(thread_id)
        while len(self._store) > self._max:
            self._store.popitem(last=False)
