# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""On-demand, admin-triggered reconcile of a single job against granite.build.

Unlike the background :class:`ReconcileLoop` (which only writes build detail at a
legal terminal transition), this is an explicit, authoritative sync: it always
re-fetches status + events, rewrites ``build_status`` and the output artifacts,
and forces ``jobs.status`` to whatever gbserver reports — **bypassing**
``check_transition``. That bypass is a deliberate, admin-only exception to the
state-machine invariant (see CLAUDE.md): it is the only way to repair a job stuck
in a wrong terminal state. ``to_run_status`` never maps back to a pre-run state,
so a forced write only ever lands on running/completed/error/terminated.
"""

from __future__ import annotations

from uuid import UUID

from autotunex.core.exceptions import (
    BuildReconcileUpstreamError,
    JobNotFoundError,
    JobNotReconcilableError,
)
from autotunex.core.logging import get_logger
from autotunex.db.repositories.protocols import JobRepository
from autotunex.models.job import JobRead
from autotunex.models.status import GbTaskType
from autotunex.services.mappers import job_to_read
from autotunex.services.reconcile.build_detail import build_status_detail, output_artifact_ref
from autotunex.services.reconcile.mapping import to_run_status
from autotunex.services.reconcile.protocols import BuildStatusError, BuildStatusReader

logger = get_logger(__name__)


class OnDemandReconciler:
    """Force one job to re-sync with granite.build (admin, on demand)."""

    def __init__(self, *, repository: JobRepository, reader: BuildStatusReader) -> None:
        self._repo = repository
        self._reader = reader

    async def reconcile(self, job_id: UUID) -> JobRead:
        """Re-sync one job and return its refreshed detail representation.

        Raises:
            JobNotFoundError: no such job.
            JobNotReconcilableError: the job has no ``TUNING`` build id to poll.
            BuildReconcileUpstreamError: gbserver could not be read for status.
        """
        job = await self._repo.get(job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        task = await self._repo.get_task(job_id, GbTaskType.TUNING)
        if task is None or task.build_id is None:
            raise JobNotReconcilableError(job_id)

        try:
            state = await self._reader.read(task.build_id)
        except BuildStatusError as exc:
            raise BuildReconcileUpstreamError(str(exc)) from exc

        try:
            events = await self._reader.read_events(task.build_id)
        except BuildStatusError as exc:
            logger.warning(
                "Could not read build events for %s (job %s): %s; build_history empty.",
                task.build_id,
                job_id,
                exc,
            )
            events = {}

        detail = build_status_detail(state.raw, events)
        artifact_id: UUID | None = None
        artifact_uri: str | None = None
        ref = output_artifact_ref(detail)
        if ref is not None:
            try:
                artifact_id = UUID(ref["artifact_id"])
                artifact_uri = ref["uri"] or None
            except ValueError:
                logger.warning(
                    "Build %s output artifact id %r is not a UUID; leaving artifact unset.",
                    task.build_id,
                    ref["artifact_id"],
                )

        mapped = to_run_status(state.status)
        # Force: write the gbserver-reported status directly, bypassing
        # check_transition (admin action). `None` means gbserver is undecided —
        # leave status unchanged but still refresh the detail below.
        new_status = mapped if mapped is not None else task.status
        await self._repo.upsert_task(
            job_id,
            GbTaskType.TUNING,
            status=new_status,
            build_status=detail,
            artifact_id=artifact_id,
            artifact_uri=artifact_uri,
            started_at=state.created_at,
            updated_at=state.updated_at,
        )
        if mapped is not None:
            await self._repo.set_status(job_id, mapped)

        refreshed = await self._repo.get(job_id)
        assert refreshed is not None  # just written; the row cannot vanish mid-request
        return job_to_read(refreshed)
