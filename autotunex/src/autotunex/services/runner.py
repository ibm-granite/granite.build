# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Job execution.

Two :class:`~autotunex.services.protocols.JobRunner` implementations live here.
``InProcessJobRunner`` submits an accepted job to a cluster off the request
path (see ``services/launch/``) and leaves it ``pending`` until the cluster
confirms the run is actually running — that confirmation, and ingesting
trials/results, are separate follow-on specs, not implemented here.
``NoOpJobRunner`` remains the default when no backend is configured
(``job_backend="none"``): it logs a warning and leaves the job ``pending``
forever, with nothing to advance it.

A queue-backed runner (the task queue itself) is still the open decision: when
one is adopted, add a sibling implementation (for example ``ArqJobRunner``)
and wire it into ``autotunex.api.deps.get_job_runner`` alongside these two.
Callers depend on the Protocol, not on any concrete class, so nothing else
needs to change.

See "Open decisions" in ``CLAUDE.md``.
"""

from __future__ import annotations

import asyncio
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from autotunex.core.exceptions import BuildCancelUpstreamError
from autotunex.core.logging import get_logger
from autotunex.db.repositories.sqlalchemy import (
    SqlAlchemyDatasetRepository,
    SqlAlchemyJobRepository,
)
from autotunex.db.tables import DatasetTable, JobTable
from autotunex.models.status import GbTaskType, RunStatus
from autotunex.services.jobs import check_transition
from autotunex.services.launch.protocols import (
    BuildCanceller,
    LaunchContext,
    LaunchHandle,
    TuningLauncher,
)

logger = get_logger(__name__)


class NoOpJobRunner:
    """Accepts submissions and does nothing with them.

    Satisfies :class:`autotunex.services.protocols.JobRunner`.
    """

    async def submit(self, job_id: UUID) -> None:
        """Log that ``job_id`` was accepted but will not be executed."""
        logger.warning(
            "Job %s accepted but not executed: no JobRunner is implemented. "
            "It will remain in status 'pending'.",
            job_id,
        )

    async def cancel(self, job_id: UUID) -> None:
        """No live work exists for a no-op-submitted job; nothing to stop."""
        logger.info("Cancel requested for job %s but no runner is active.", job_id)


def _context_from(job: JobTable, dataset: DatasetTable | None) -> LaunchContext:
    """Build a LaunchContext from a job's persisted snapshot and its dataset."""
    snapshot = job.config_snapshot or {}
    rl_tuner_type = snapshot.get("rl_tuner_type")
    # The snapshot records the configuration's display name (services/jobs.py);
    # it names the in-container config file. Fall back to "config" — the same
    # default the 2025 runner used — if a directly-written job lacks it.
    name = snapshot.get("name")
    config_name = name if isinstance(name, str) and name.strip() else "config"
    return LaunchContext(
        job_id=job.id,
        model=job.model,
        model_source=job.model_source,
        experiment_name=job.experiment_name,
        tuning_type=job.tuning_type,
        rl_tuner_type=rl_tuner_type if isinstance(rl_tuner_type, str) else None,
        config_name=config_name,
        config_data=snapshot.get("config_data") or {},
        dataset_name=dataset.name if dataset is not None else "",
        dataset_uri=dataset.artifact_url if dataset is not None else None,
        data_format=dataset.data_format if dataset is not None else "jsonl",
        autotune=bool(job.autotune),
        seed=job.seed,
        reward_function_code=job.reward_function_code,
        reward_function_name=job.reward_function_name,
    )


class InProcessJobRunner:
    """Submits accepted jobs to a cluster off the request path.

    Satisfies :class:`autotunex.services.protocols.JobRunner`. Mirrors
    ``InProcessDatasetUploadRunner``: ``submit`` schedules ``process`` and
    returns; ``process`` opens its OWN session (the request session is long
    closed) and is directly awaitable so tests are deterministic.

    A successful submission leaves the job ``pending`` — "submitted" is not
    "running"; the reconcile follow-on advances it once the cluster confirms.
    Only failure writes status (``pending → error``).
    """

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        launcher: TuningLauncher,
        canceller: BuildCanceller,
    ) -> None:
        self._session_factory = session_factory
        self._launcher = launcher
        self._canceller = canceller
        self._tasks: set[asyncio.Task[None]] = set()

    async def submit(self, job_id: UUID) -> None:
        """Fire ``process`` as a background task and return immediately."""
        task = asyncio.create_task(self.process(job_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def process(self, job_id: UUID) -> None:
        """Assemble the launch, submit it, and record the handle or the failure."""
        async with self._session_factory() as session:
            jobs = SqlAlchemyJobRepository(session)
            datasets = SqlAlchemyDatasetRepository(session)
            job = await jobs.get(job_id)
            if job is None:
                logger.error("Job %s vanished before launch; nothing to submit.", job_id)
                return
            try:
                dataset = await datasets.get(job.dataset_id)
                handle: LaunchHandle = await self._launcher.launch(_context_from(job, dataset))
            # Broad except is intentional: any launch failure must land the job in
            # a terminal `error` state rather than leaving it silently pending.
            except Exception:
                logger.exception("Launch failed for job %s", job_id)
                # A freshly submitted job is `pending`; pending → error is always legal.
                check_transition(job.status, RunStatus.ERROR)
                await jobs.set_status(job_id, RunStatus.ERROR)
                await jobs.upsert_task(job_id, GbTaskType.TUNING, status=RunStatus.ERROR)
                return
            # Success: record the build handle; the job deliberately stays `pending`.
            await jobs.upsert_task(
                job_id,
                GbTaskType.TUNING,
                status=RunStatus.PENDING,
                build_id=handle.build_id,
                pr_url=handle.pr_url,
            )

    async def cancel(self, job_id: UUID) -> None:
        """Cancel the job's granite.build build, if one was submitted.

        Reads the ``TUNING`` build id from its own session (the request session is
        gone). No build id means nothing reached the cluster — a no-op. A CLI
        failure surfaces as ``BuildCancelUpstreamError`` (502) so the service
        leaves the job intact for a retry. The job's move to ``terminated`` is the
        service's job (and reconcile will confirm ``cancelled → terminated``).
        """
        async with self._session_factory() as session:
            task = await SqlAlchemyJobRepository(session).get_task(job_id, GbTaskType.TUNING)
        if task is None or task.build_id is None:
            logger.info("Job %s has no build to cancel; nothing to stop.", job_id)
            return
        try:
            await self._canceller.cancel(task.build_id)
        except Exception as exc:  # CLI/subprocess failure → upstream 502
            raise BuildCancelUpstreamError(str(exc)) from exc
