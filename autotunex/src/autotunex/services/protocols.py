# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Service-layer interfaces.

:class:`JobRunner` is the seam where job execution is plugged in. It is
deliberately *job*-level, not per-trial: every backend AutoTuneX supports
(``local``, and ``llmb``'s remote, bash and LSF variants) hands off a whole job
and lets the tuning core decide how to search within it. Hyperparameter search
itself lives in that core (``src/fm-tune/autotune/``), behind the
:class:`~autotunex.services.autotune.AutotuneCore` seam, so no service-layer
search or per-trial-training Protocol exists here.

The launch-time counterpart, ``services/launch/protocols.py``, describes how a
job is turned into a build spec and submitted to a cluster.
"""

from __future__ import annotations

from typing import Protocol
from uuid import UUID


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
