# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Reads a job's trials, scoped to the calling principal.

Separate from :class:`~autotunex.services.jobs.JobService` on purpose. Trials are
a job's unbounded child collection, so reading them is the same shape of problem
as reading its logs or its per-step metrics — resolve the caller's scope, prove
the job is visible, then page the children — and this module mirrors
:class:`~autotunex.services.metrics.MetricsService` rather than growing
``services/jobs.py`` with a third responsibility.

They were previously nested in the job detail response, which made every read of
a job (and every poll tick while it ran) pay for the whole trial list plus its
one-to-one ``results`` rows. See
``docs/superpowers/specs/2026-09-01-job-trials-endpoint-split-design.md``.
"""

from __future__ import annotations

from uuid import UUID

from autotunex.core.exceptions import JobNotFoundError
from autotunex.db.repositories.protocols import JobRepository, TrialRepository
from autotunex.models.auth import Principal
from autotunex.models.common import DataScope, Page
from autotunex.models.trial import TrialRead
from autotunex.services.mappers import trial_to_read
from autotunex.services.scoping import resolve_owner_filter, sees_nothing


class TrialService:
    """Reads one page of a job's trials, scoped to the principal."""

    def __init__(
        self,
        trial_repository: TrialRepository,
        job_repository: JobRepository,
        principal: Principal,
    ) -> None:
        self._trials = trial_repository
        self._jobs = job_repository
        self._principal = principal

    async def list_trials(
        self,
        job_id: UUID,
        *,
        limit: int = 50,
        offset: int = 0,
        scope: DataScope = DataScope.OWN,
    ) -> Page[TrialRead]:
        """Return one page of the job's trials, oldest first.

        Each trial carries the ``metric``/``metrics`` its one-to-one ``results``
        row reported, so a caller needs no second request to score the search.

        A job that exists and is visible but has run no trials yet returns an
        empty page, **not** a 404 — the pre-rewrite implementation raised 404
        there, conflating "no trials" with "no job".

        Raises:
            ScopeNotPermittedError: a non-admin requested ``scope=all``.
            JobNotFoundError: no such job, or it is not visible under ``scope``.
        """
        await self._require_visible(job_id, scope)
        rows, total = await self._trials.page(job_id, limit=limit, offset=offset)
        return Page[TrialRead](
            items=[trial_to_read(row) for row in rows],
            total=total,
            limit=limit,
            offset=offset,
        )

    async def _require_visible(self, job_id: UUID, scope: DataScope) -> None:
        """Resolve scope and 404 unless the caller may see ``job_id``.

        Same order as ``LogService`` and ``MetricsService``: the non-admin ``all``
        403 fires first (``resolve_owner_filter``), then the ``own``-no-identity
        short-circuit (``sees_nothing``), then the existence probe. So an unknown
        job, another owner's job, and a job belonging to an unresolvable caller
        are one indistinguishable 404.
        """
        owner_id = resolve_owner_filter(self._principal, scope)
        if sees_nothing(self._principal, scope):
            raise JobNotFoundError(job_id)
        if not await self._jobs.is_visible(job_id, owner_id=owner_id):
            raise JobNotFoundError(job_id)
