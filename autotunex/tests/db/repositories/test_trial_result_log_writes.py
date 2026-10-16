"""Trial/Result write repositories and job ``append_log``, against real SQLite."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from autotunex.db.repositories.sqlalchemy import (
    SqlAlchemyJobRepository,
    SqlAlchemyResultRepository,
    SqlAlchemyTrialRepository,
)
from autotunex.db.tables import JobTable, ResultTable, TrialTable
from autotunex.models.status import RunStatus


async def test_trial_upsert_inserts_a_new_trial(session: AsyncSession, job: JobTable) -> None:
    trials = SqlAlchemyTrialRepository(session)

    await trials.upsert(job.id, "ray_0001", status=RunStatus.RUNNING, config={"lr": 1})

    row = await session.get(TrialTable, "ray_0001")
    assert row is not None
    assert row.status == RunStatus.RUNNING
    assert row.config == {"lr": 1}


async def test_trial_upsert_updates_in_place(session: AsyncSession, job: JobTable) -> None:
    trials = SqlAlchemyTrialRepository(session)
    await trials.upsert(job.id, "ray_0001", status=RunStatus.RUNNING, config={"lr": 1})

    await trials.upsert(job.id, "ray_0001", status=RunStatus.COMPLETED, config={"lr": 2})

    count = await session.scalar(
        select(func.count()).select_from(TrialTable).where(TrialTable.id == "ray_0001")
    )
    row = await session.get(TrialTable, "ray_0001")
    assert count == 1  # updated, not duplicated
    assert row is not None
    assert row.status == RunStatus.COMPLETED
    assert row.config == {"lr": 2}


async def test_trial_set_status_transitions_to_completed(
    session: AsyncSession, job: JobTable
) -> None:
    trials = SqlAlchemyTrialRepository(session)
    await trials.upsert(job.id, "ray_0001", status=RunStatus.RUNNING, config=None)

    await trials.set_status("ray_0001", RunStatus.COMPLETED)

    row = await session.get(TrialTable, "ray_0001")
    assert row is not None
    assert row.status == RunStatus.COMPLETED


async def test_trial_set_status_is_a_no_op_when_the_trial_is_missing(
    session: AsyncSession, job: JobTable
) -> None:
    trials = SqlAlchemyTrialRepository(session)

    await trials.set_status("does_not_exist", RunStatus.ERROR)

    assert await session.get(TrialTable, "does_not_exist") is None


async def test_result_upsert_is_one_to_one(session: AsyncSession, job: JobTable) -> None:
    trials = SqlAlchemyTrialRepository(session)
    results = SqlAlchemyResultRepository(session)
    await trials.upsert(job.id, "ray_0001", status=RunStatus.RUNNING, config=None)

    await results.upsert(job.id, "ray_0001", metric="loss", metrics={"loss": 0.5})
    await results.upsert(job.id, "ray_0001", metric="loss", metrics={"loss": 0.4})

    count = await session.scalar(
        select(func.count()).select_from(ResultTable).where(ResultTable.trial_id == "ray_0001")
    )
    row = (
        await session.execute(select(ResultTable).where(ResultTable.trial_id == "ray_0001"))
    ).scalar_one()
    assert count == 1  # second upsert updated the row, did not duplicate it
    assert row.metrics == {"loss": 0.4}


async def test_append_log_writes_a_row_visible_via_logs_page(
    session: AsyncSession, job: JobTable
) -> None:
    jobs = SqlAlchemyJobRepository(session)

    await jobs.append_log(
        job.id,
        trial_id="t01",
        level="INFO",
        filename="x.py",
        message="hello",
        iteration=1,
        epoch=0.5,
    )

    rows, has_more = await jobs.logs_page(job.id, trial_id="t01", before_id=0, limit=10)
    assert has_more is False
    assert [(row.message, row.iteration, row.epoch) for row in rows] == [("hello", 1, 0.5)]
