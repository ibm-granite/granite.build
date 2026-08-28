# src/autotunex/services/local/cancellation.py
# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Process-wide cancellation registry for in-process ``local`` runs.

A ``local`` run executes on this process (Ray is a global singleton — the trainer
"assumes one local run at a time"), and the running task lives on the *submit-time*
:class:`~autotunex.services.local.runner.LocalJobRunner` instance. A cancel request,
however, is served by a *fresh* runner instance with no handle to that task. This
module bridges the two: the running job registers a cancel :class:`threading.Event`
here, and the cancel request looks it up by ``job_id`` and sets it.

The registry is intentionally per-process. That matches ``local`` being a
single-process affordance; a cancel that lands on a worker not running the job
finds nothing active (and the caller proceeds to write ``terminated`` / delete —
correct, and it also cleans up a run orphaned by an API restart).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from uuid import UUID


@dataclass
class _CancelToken:
    """One in-flight run's cancel signal."""

    event: threading.Event = field(default_factory=threading.Event)


_LOCK = threading.Lock()
_REGISTRY: dict[UUID, _CancelToken] = {}


def register(job_id: UUID) -> None:
    """Register a fresh cancel token for ``job_id`` (called at run start)."""
    with _LOCK:
        _REGISTRY[job_id] = _CancelToken()


def unregister(job_id: UUID) -> None:
    """Drop ``job_id``'s token (called in the run's ``finally``). Idempotent."""
    with _LOCK:
        _REGISTRY.pop(job_id, None)


def request_cancel(job_id: UUID) -> bool:
    """Signal cancellation for ``job_id``; return ``False`` if no run is registered."""
    with _LOCK:
        token = _REGISTRY.get(job_id)
    if token is None:
        return False
    token.event.set()
    return True


def is_cancelled(job_id: UUID) -> bool:
    """Return whether cancellation has been requested for a registered ``job_id``."""
    with _LOCK:
        token = _REGISTRY.get(job_id)
    return token is not None and token.event.is_set()


def is_active(job_id: UUID) -> bool:
    """Return whether a run is currently registered for ``job_id`` in this process."""
    with _LOCK:
        return job_id in _REGISTRY


def cancel_event(job_id: UUID) -> threading.Event | None:
    """Return ``job_id``'s cancel event (for the trainer's watcher), or ``None``."""
    with _LOCK:
        token = _REGISTRY.get(job_id)
    return token.event if token is not None else None
