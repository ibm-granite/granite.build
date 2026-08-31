# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""The BuildStatusReader seam and the value it returns.

A reader turns a build id into a normalized :class:`BuildState`, or raises one
of the errors below. It is a ``Protocol`` (structural typing), per the repo's
protocol-based dependency inversion: tests supply a hand-written fake with no
HTTP, and a future event-driven reader becomes a second provider rather than a
second code path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID


@dataclass(frozen=True)
class BuildState:
    """The fields we read from gbserver's ``StoredBuild``, normalized.

    ``created_at`` / ``updated_at`` are kept as the ISO 8601 strings gbserver
    sends, not parsed to ``datetime``: their destination columns
    (``gb_tasks.started_at`` / ``updated_at``) are ``String(255)``, so parsing
    only to re-serialize would add a failure mode for no gain. ``raw`` is the
    whole ``/status`` response body; on a terminal state the loop transforms it
    (with the build events) into the UX ``build_status`` shape persisted on the
    TUNING task — see :func:`autotunex.services.reconcile.build_detail`.
    """

    build_id: UUID
    status: str
    failure_reason: str | None
    created_at: str | None
    updated_at: str | None
    raw: dict[str, Any]


class BuildStatusError(Exception):
    """Base for a status read that produced no usable :class:`BuildState`."""


class BuildStatusUnavailableError(BuildStatusError):
    """Transient failure (timeout, connection error, 5xx). Retry next sweep."""


class BuildNotFoundError(BuildStatusError):
    """gbserver returned 404 for the build id."""


class BuildStatusAuthError(BuildStatusError):
    """gbserver returned 401/403 — almost always an expired GB token."""


class MalformedBuildStatusError(BuildStatusError):
    """A 2xx response whose body lacked ``status.build.status``."""


class BuildStatusReader(Protocol):
    """Reads one build's status from the cluster. Must not block long."""

    async def read(self, build_id: UUID) -> BuildState:
        """Return the build's normalized status, or raise a ``BuildStatusError``."""
        ...

    async def read_events(self, build_id: UUID) -> dict[str, Any]:
        """Return the build's raw event-log body, or raise a ``BuildStatusError``.

        Fetched only at the terminal transition to assemble ``build_history`` for
        the Status tab; see :func:`autotunex.services.reconcile.build_detail`.
        """
        ...
