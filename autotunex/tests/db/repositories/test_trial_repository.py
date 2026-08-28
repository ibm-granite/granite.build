"""SqlAlchemyTrialRepository.terminate_running, against real SQLite."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from autotunex.db.repositories.sqlalchemy import SqlAlchemyTrialRepository
from autotunex.db.tables import JobTable, TrialTable
from autotunex.models.status import RunStatus


async def test_terminate_running_moves_only_running_trials_to_terminated(
    session: AsyncSession, job: JobTable
) -> None:
    trials = SqlAlchemyTrialRepository(session)
    await trials.upsert(job.id, "ray_0001", status=RunStatus.RUNNING, config=None)
    await trials.upsert(job.id, "ray_0002", status=RunStatus.COMPLETED, config=None)

    await trials.terminate_running(job.id)

    running = await session.get(TrialTable, "ray_0001")
    completed = await session.get(TrialTable, "ray_0002")
    assert running is not None
    assert running.status == RunStatus.TERMINATED
    assert completed is not None
    assert completed.status == RunStatus.COMPLETED
