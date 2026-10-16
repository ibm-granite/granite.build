# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Reads a job/trial's per-step training metrics, scoped to the principal."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from autotunex.core.exceptions import JobNotFoundError
from autotunex.db.repositories.protocols import JobRepository, TrainingMetricsRepository
from autotunex.db.tables import TrainingMetricTable
from autotunex.models.auth import Principal
from autotunex.models.common import DataScope
from autotunex.models.metric import MetricPage
from autotunex.services.mappers import metric_point_to_read
from autotunex.services.scoping import resolve_owner_filter, sees_nothing


class MetricsService:
    """Reads a job's per-step (DB) training metrics, scoped to the principal."""

    def __init__(
        self,
        metrics_repository: TrainingMetricsRepository,
        job_repository: JobRepository,
        principal: Principal,
    ) -> None:
        self._metrics = metrics_repository
        self._jobs = job_repository
        self._principal = principal

    async def get_job_metrics(
        self, job_id: UUID, *, after_id: int, limit: int, scope: DataScope = DataScope.OWN
    ) -> MetricPage:
        """Return one ascending keyset page of the job's per-step metrics (all trials).

        Raises:
            ScopeNotPermittedError: a non-admin requested ``scope=all``.
            JobNotFoundError: no such job, or it is not visible under ``scope``.
        """
        await self._require_visible(job_id, scope)
        rows, has_more = await self._metrics.metrics_page(
            job_id, trial_id=None, after_id=after_id, limit=limit
        )
        return self._to_page(rows, has_more)

    async def get_trial_metrics(
        self,
        job_id: UUID,
        trial_id: str,
        *,
        after_id: int,
        limit: int,
        scope: DataScope = DataScope.OWN,
    ) -> MetricPage:
        """Return one ascending keyset page of one trial's per-step metrics.

        Raises:
            ScopeNotPermittedError: a non-admin requested ``scope=all``.
            JobNotFoundError: no such job, or it is not visible under ``scope``.
        """
        await self._require_visible(job_id, scope)
        rows, has_more = await self._metrics.metrics_page(
            job_id, trial_id=trial_id, after_id=after_id, limit=limit
        )
        return self._to_page(rows, has_more)

    async def _require_visible(self, job_id: UUID, scope: DataScope) -> None:
        """Resolve scope and 404 unless the caller may see ``job_id``.

        Same order as ``LogService``: the non-admin ``all`` 403 fires first
        (``resolve_owner_filter``), then the ``own``-no-identity short-circuit
        (``sees_nothing``), then the existence probe.
        """
        owner_id = resolve_owner_filter(self._principal, scope)
        if sees_nothing(self._principal, scope):
            raise JobNotFoundError(job_id)
        if not await self._jobs.is_visible(job_id, owner_id=owner_id):
            raise JobNotFoundError(job_id)

    def _to_page(self, rows: Sequence[TrainingMetricTable], has_more: bool) -> MetricPage:
        points = [metric_point_to_read(row) for row in rows]
        next_after_id = points[-1].id if has_more and points else None
        return MetricPage(metrics=points, has_more=has_more, next_after_id=next_after_id)
