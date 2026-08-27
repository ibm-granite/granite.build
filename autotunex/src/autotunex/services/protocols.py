# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Service-layer interfaces.

These Protocols mark the seams where real machine-learning execution will be
plugged in. Only :class:`JobRunner` has an implementation today (a no-op); the
rest are declared so that the intended boundaries are explicit and reviewable
before any backend is chosen.
"""

from __future__ import annotations

from typing import Any, Protocol
from uuid import UUID

from autotunex.models.search_space import SearchSpace


class JobRunner(Protocol):
    """Hands an accepted job off for execution.

    Implementations must return promptly — never block the request. The real
    implementation will enqueue onto a task queue; see the open decision in
    ``CLAUDE.md``.
    """

    async def submit(self, job_id: UUID) -> None:
        """Schedule ``job_id`` for execution."""
        ...

    async def cancel(self, job_id: UUID) -> None:
        """Stop any live backend work for ``job_id``.

        A no-op when there is nothing to stop (no build submitted, no in-process
        run). Out-of-process backends return promptly; the local runner may wait a
        bounded time for a cooperative stop and raise
        ``JobCancellationInProgressError`` if the run does not stop in time.
        """
        ...


class SearchEngine(Protocol):
    """Proposes the next hyperparameter assignment to evaluate.

    Not implemented. A concrete engine (grid, random, Bayesian) decides which
    point in the search space a trial should test.
    """

    def suggest(self, space: SearchSpace, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Return concrete parameter values drawn from ``space``."""
        ...


class TrainingBackend(Protocol):
    """Runs one fine-tuning trial and reports its metrics.

    Not implemented. The training stack has not been chosen yet.
    """

    async def train(self, trial_id: UUID, params: dict[str, Any]) -> dict[str, float]:
        """Train with ``params`` and return the resulting metrics."""
        ...
