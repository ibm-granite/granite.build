"""SqlAlchemyJobRepository status + gb_task write methods, against real SQLite."""

from __future__ import annotations

from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from autotunex.db.repositories.sqlalchemy import SqlAlchemyJobRepository
from autotunex.db.tables import JobTable
from autotunex.models.status import GbTaskType, RunStatus


async def test_set_status_updates_the_job(session: AsyncSession, job: JobTable) -> None:
    repository = SqlAlchemyJobRepository(session)

    await repository.set_status(job.id, RunStatus.ERROR)

    reloaded = await repository.get(job.id)
    assert reloaded is not None
    assert reloaded.status == RunStatus.ERROR


async def test_upsert_task_inserts_then_updates_one_row(
    session: AsyncSession, job: JobTable
) -> None:
    repository = SqlAlchemyJobRepository(session)
    build_id = job.id  # any UUID; reuse the job's for convenience

    created = await repository.upsert_task(
        job.id, GbTaskType.TUNING, status=RunStatus.PENDING, build_id=build_id, pr_url="u"
    )
    updated = await repository.upsert_task(job.id, GbTaskType.TUNING, status=RunStatus.ERROR)

    assert created.id == updated.id  # same row, not a duplicate
    fetched = await repository.get_task(job.id, GbTaskType.TUNING)
    assert fetched is not None
    assert fetched.status == RunStatus.ERROR
    assert fetched.build_id == build_id
    assert fetched.pr_url == "u"  # preserved: a None on update does not clobber


async def test_upsert_task_persists_build_status_and_timestamps(
    session: AsyncSession, job: JobTable
) -> None:
    repository = SqlAlchemyJobRepository(session)
    await repository.upsert_task(job.id, GbTaskType.TUNING, status=RunStatus.RUNNING)

    updated = await repository.upsert_task(
        job.id,
        GbTaskType.TUNING,
        status=RunStatus.COMPLETED,
        build_status={"status": {"build": {"status": "success"}}},
        started_at="2026-08-07T00:00:00Z",
        updated_at="2026-08-07T00:05:00Z",
    )

    assert updated.status == RunStatus.COMPLETED
    assert updated.build_status == {"status": {"build": {"status": "success"}}}
    assert updated.started_at == "2026-08-07T00:00:00Z"
    assert updated.updated_at == "2026-08-07T00:05:00Z"


async def test_upsert_task_persists_artifact_id_and_uri(
    session: AsyncSession, job: JobTable
) -> None:
    repository = SqlAlchemyJobRepository(session)
    artifact_id = uuid4()

    updated = await repository.upsert_task(
        job.id,
        GbTaskType.TUNING,
        status=RunStatus.COMPLETED,
        artifact_id=artifact_id,
        artifact_uri="hf://huggingface.co/models/ibm-research/autotunex_x",
    )

    assert updated.artifact_id == artifact_id
    assert updated.artifact_uri == "hf://huggingface.co/models/ibm-research/autotunex_x"


async def test_upsert_task_none_artifact_on_update_does_not_clobber(
    session: AsyncSession, job: JobTable
) -> None:
    repository = SqlAlchemyJobRepository(session)
    artifact_id = uuid4()
    await repository.upsert_task(
        job.id,
        GbTaskType.TUNING,
        status=RunStatus.RUNNING,
        artifact_id=artifact_id,
        artifact_uri="hf://models/x",
    )

    updated = await repository.upsert_task(job.id, GbTaskType.TUNING, status=RunStatus.COMPLETED)

    assert updated.artifact_id == artifact_id  # preserved: a None on update does not clobber
    assert updated.artifact_uri == "hf://models/x"


async def test_get_task_returns_none_when_absent(session: AsyncSession, job: JobTable) -> None:
    repository = SqlAlchemyJobRepository(session)

    assert await repository.get_task(job.id, GbTaskType.TUNING) is None


async def test_list_reconcilable_returns_nonterminal_jobs_with_a_tuning_build_id(
    session: AsyncSession, job: JobTable
) -> None:
    repository = SqlAlchemyJobRepository(session)
    build_id = job.id  # any UUID
    await repository.upsert_task(
        job.id, GbTaskType.TUNING, status=RunStatus.PENDING, build_id=build_id
    )

    reconcilable = await repository.list_reconcilable()

    assert len(reconcilable) == 1
    assert reconcilable[0].job_id == job.id
    assert reconcilable[0].status == RunStatus.PENDING
    assert reconcilable[0].build_id == build_id


async def test_list_reconcilable_excludes_terminal_jobs(
    session: AsyncSession, job: JobTable
) -> None:
    repository = SqlAlchemyJobRepository(session)
    await repository.upsert_task(
        job.id, GbTaskType.TUNING, status=RunStatus.RUNNING, build_id=job.id
    )
    await repository.set_status(job.id, RunStatus.ERROR)  # ERROR is terminal

    assert await repository.list_reconcilable() == []


async def test_list_reconcilable_excludes_jobs_without_a_build_id(
    session: AsyncSession, job: JobTable
) -> None:
    repository = SqlAlchemyJobRepository(session)
    await repository.upsert_task(job.id, GbTaskType.TUNING, status=RunStatus.PENDING)

    assert await repository.list_reconcilable() == []


async def test_list_reconcilable_excludes_non_tuning_tasks(
    session: AsyncSession, job: JobTable
) -> None:
    repository = SqlAlchemyJobRepository(session)
    await repository.upsert_task(job.id, GbTaskType.RITS, status=RunStatus.PENDING, build_id=job.id)

    assert await repository.list_reconcilable() == []
