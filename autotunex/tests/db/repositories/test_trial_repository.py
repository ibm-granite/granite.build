"""SqlAlchemyTrialRepository — terminate_running and the paged read, against real SQLite."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from autotunex.db.repositories.sqlalchemy import SqlAlchemyTrialRepository
from autotunex.db.tables import JobTable, ResultTable, TrialTable
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


async def test_page_returns_trials_oldest_first_with_the_unpaginated_total(
    session: AsyncSession, job: JobTable
) -> None:
    session.add_all(
        [
            TrialTable(
                id="later",
                job_id=job.id,
                status=RunStatus.COMPLETED,
                created_at=datetime(2026, 3, 2, tzinfo=UTC),
            ),
            TrialTable(
                id="earlier",
                job_id=job.id,
                status=RunStatus.RUNNING,
                created_at=datetime(2026, 3, 1, tzinfo=UTC),
            ),
        ]
    )
    await session.commit()

    trials, total = await SqlAlchemyTrialRepository(session).page(job.id, limit=1, offset=0)

    assert [trial.id for trial in trials] == ["earlier"]
    assert total == 2


async def test_page_loads_each_trial_s_one_to_one_result(
    session: AsyncSession, job: JobTable
) -> None:
    """Eager-loaded, so the mapper can read ``trial.result`` on a detached row."""
    session.add(TrialTable(id="t1", job_id=job.id, status=RunStatus.COMPLETED))
    await session.commit()
    session.add(
        ResultTable(
            id=uuid4(), job_id=job.id, trial_id="t1", metric="eval_loss", metrics={"eval_loss": 0.4}
        )
    )
    await session.commit()

    trials, _ = await SqlAlchemyTrialRepository(session).page(job.id, limit=50, offset=0)

    assert trials[0].result is not None
    assert trials[0].result.metric == "eval_loss"


async def test_page_of_a_job_with_no_trials_is_empty_rather_than_an_error(
    session: AsyncSession, job: JobTable
) -> None:
    trials, total = await SqlAlchemyTrialRepository(session).page(job.id, limit=50, offset=0)

    assert list(trials) == []
    assert total == 0


async def test_page_never_returns_another_job_s_trials(
    session: AsyncSession, job: JobTable
) -> None:
    """The ``job_id`` filter is the only scoping this method does."""
    other_job_id = uuid4()
    session.add(TrialTable(id="mine", job_id=job.id, status=RunStatus.COMPLETED))
    await session.commit()

    trials, total = await SqlAlchemyTrialRepository(session).page(other_job_id, limit=50, offset=0)

    assert list(trials) == []
    assert total == 0
