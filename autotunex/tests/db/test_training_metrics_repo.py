"""Repository-level tests for ``TrainingMetricsRepository``."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from autotunex.db.repositories.sqlalchemy import SqlAlchemyTrainingMetricsRepository
from autotunex.db.tables import JobTable, UserTable


async def _insert_steps(
    repo: SqlAlchemyTrainingMetricsRepository, job: JobTable, steps: tuple[int, ...]
) -> None:
    """Append one row per step, all on trial ``t1``."""
    for step in steps:
        await repo.insert(
            job.id,
            trial_id="t1",
            global_step=step,
            epoch=0.1,
            loss=float(step),
            grad_norm=1.0,
            learning_rate=1e-6,
            split="train",
            extra=None,
        )


async def test_first_page_returns_the_oldest_rows_and_flags_more(
    session: AsyncSession, user: UserTable, job: JobTable
) -> None:
    repo = SqlAlchemyTrainingMetricsRepository(session)
    await _insert_steps(repo, job, (10, 20, 30))

    rows, has_more = await repo.metrics_page(job.id, trial_id="t1", after_id=0, limit=2)

    assert [r.global_step for r in rows] == [10, 20]
    assert has_more is True


async def test_the_next_page_resumes_after_the_cursor(
    session: AsyncSession, user: UserTable, job: JobTable
) -> None:
    repo = SqlAlchemyTrainingMetricsRepository(session)
    await _insert_steps(repo, job, (10, 20, 30))
    first, _ = await repo.metrics_page(job.id, trial_id="t1", after_id=0, limit=2)

    rows, has_more = await repo.metrics_page(job.id, trial_id="t1", after_id=first[-1].id, limit=2)

    assert [r.global_step for r in rows] == [30]
    assert has_more is False


async def test_page_filters_by_trial(session: AsyncSession, user: UserTable, job: JobTable) -> None:
    repo = SqlAlchemyTrainingMetricsRepository(session)
    await repo.insert(
        job.id,
        trial_id="t1",
        global_step=1,
        epoch=None,
        loss=1.0,
        grad_norm=None,
        learning_rate=None,
        split="train",
        extra=None,
    )
    await repo.insert(
        job.id,
        trial_id="t2",
        global_step=1,
        epoch=None,
        loss=2.0,
        grad_norm=None,
        learning_rate=None,
        split="train",
        extra=None,
    )

    rows, _ = await repo.metrics_page(job.id, trial_id="t2", after_id=0, limit=50)

    assert [r.loss for r in rows] == [2.0]


async def test_page_without_a_trial_filter_spans_every_trial(
    session: AsyncSession, user: UserTable, job: JobTable
) -> None:
    repo = SqlAlchemyTrainingMetricsRepository(session)
    for trial in ("t1", "t2"):
        await repo.insert(
            job.id,
            trial_id=trial,
            global_step=1,
            epoch=None,
            loss=1.0,
            grad_norm=None,
            learning_rate=None,
            split="train",
            extra=None,
        )

    rows, _ = await repo.metrics_page(job.id, trial_id=None, after_id=0, limit=50)

    assert sorted(r.trial_id or "" for r in rows) == ["t1", "t2"]
