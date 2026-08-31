"""Tests for SqlAlchemyJobRepository.create."""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from autotunex.core.exceptions import JobReferenceConflictError
from autotunex.db.repositories.sqlalchemy import SqlAlchemyJobRepository
from autotunex.db.tables import (
    ConfigurationTable,
    DatasetTable,
    GbTaskTable,
    JobTable,
    LogEntryTable,
    ResultTable,
    TrialTable,
    UserTable,
)
from autotunex.models.status import GbTaskType, RunStatus


async def test_create_persists_a_pending_job_with_reward_function(
    session: AsyncSession,
    user: UserTable,
    configuration: ConfigurationTable,
    ready_dataset: DatasetTable,
) -> None:
    repository = SqlAlchemyJobRepository(session)

    job = await repository.create(
        user_id=str(user.id),
        config_id=configuration.id,
        dataset_id=ready_dataset.id,
        model="ibm/granite",
        model_source="huggingface",
        experiment_name="exp",
        tuning_type="optuna",
        seed=7,
        autotune=True,
        config_snapshot={"name": configuration.name},
        reward_function_code="def compute_score():\n    return 1.0\n",
        reward_function_name="compute_score",
    )

    assert job.status == RunStatus.PENDING
    assert job.seed == 7
    assert job.reward_function_code == "def compute_score():\n    return 1.0\n"
    assert job.reward_function_name == "compute_score"


async def test_create_raises_on_unknown_config_reference(
    session: AsyncSession, user: UserTable, ready_dataset: DatasetTable
) -> None:
    repository = SqlAlchemyJobRepository(session)

    with pytest.raises(JobReferenceConflictError):
        await repository.create(
            user_id=str(user.id),
            config_id=uuid4(),
            dataset_id=ready_dataset.id,
            model="ibm/granite",
            model_source="huggingface",
            experiment_name="exp",
            tuning_type=None,
            seed=42,
            autotune=True,
            config_snapshot={},
            reward_function_code=None,
            reward_function_name=None,
        )


async def test_delete_removes_the_job_and_returns_true(
    session: AsyncSession,
    user: UserTable,
    configuration: ConfigurationTable,
    ready_dataset: DatasetTable,
) -> None:
    repository = SqlAlchemyJobRepository(session)
    job = await repository.create(
        user_id=str(user.id),
        config_id=configuration.id,
        dataset_id=ready_dataset.id,
        model="m",
        model_source="huggingface",
        experiment_name="exp",
        tuning_type=None,
        seed=42,
        autotune=True,
        config_snapshot={},
        reward_function_code=None,
        reward_function_name=None,
    )

    deleted = await repository.delete(job.id)

    assert deleted is True
    assert await repository.get(job.id) is None


async def test_delete_cascades_to_trials_results_and_tasks(
    session: AsyncSession,
    user: UserTable,
    configuration: ConfigurationTable,
    ready_dataset: DatasetTable,
) -> None:
    repository = SqlAlchemyJobRepository(session)
    job = await repository.create(
        user_id=str(user.id),
        config_id=configuration.id,
        dataset_id=ready_dataset.id,
        model="m",
        model_source="huggingface",
        experiment_name="exp",
        tuning_type=None,
        seed=42,
        autotune=True,
        config_snapshot={},
        reward_function_code=None,
        reward_function_name=None,
    )
    trial = TrialTable(id="t1", job_id=job.id, config={"lr": 0.1})
    session.add(trial)
    await session.commit()
    session.add(ResultTable(id=uuid4(), job_id=job.id, trial_id="t1", metric="loss", metrics={}))
    session.add(GbTaskTable(id=uuid4(), job_id=job.id, type=GbTaskType.TUNING))
    await session.commit()

    deleted = await repository.delete(job.id)

    assert deleted is True
    assert await session.scalar(select(func.count()).select_from(TrialTable)) == 0
    assert await session.scalar(select(func.count()).select_from(ResultTable)) == 0
    assert await session.scalar(select(func.count()).select_from(GbTaskTable)) == 0


async def test_delete_cascades_to_log_entries(
    session: AsyncSession,
    user: UserTable,
    configuration: ConfigurationTable,
    ready_dataset: DatasetTable,
) -> None:
    repository = SqlAlchemyJobRepository(session)
    job = await repository.create(
        user_id=str(user.id),
        config_id=configuration.id,
        dataset_id=ready_dataset.id,
        model="m",
        model_source="huggingface",
        experiment_name="exp",
        tuning_type=None,
        seed=42,
        autotune=True,
        config_snapshot={},
        reward_function_code=None,
        reward_function_name=None,
    )
    session.add(LogEntryTable(job_id=job.id, level="INFO", message="started"))
    await session.commit()

    deleted = await repository.delete(job.id)

    assert deleted is True
    assert await session.scalar(select(func.count()).select_from(LogEntryTable)) == 0


async def test_delete_succeeds_when_a_log_entry_has_a_non_uuid_trial_id(
    session: AsyncSession,
    user: UserTable,
    configuration: ConfigurationTable,
    ready_dataset: DatasetTable,
) -> None:
    """A log entry's ``trial_id`` is a short opaque trial code, not a UUID.

    ``trials.id`` is a short ``VARCHAR(16)`` code and the tuning pipeline writes those codes
    into ``log_entries.trial_id`` (schema-review A2). The delete path eager-loads
    log entries to cascade them, so reading a non-UUID ``trial_id`` back must not
    raise ``badly formed hexadecimal UUID string``.
    """
    repository = SqlAlchemyJobRepository(session)
    job = await repository.create(
        user_id=str(user.id),
        config_id=configuration.id,
        dataset_id=ready_dataset.id,
        model="m",
        model_source="huggingface",
        experiment_name="exp",
        tuning_type=None,
        seed=42,
        autotune=True,
        config_snapshot={},
        reward_function_code=None,
        reward_function_name=None,
    )
    session.add(LogEntryTable(job_id=job.id, trial_id="323b5b11", level="INFO", message="started"))
    await session.commit()

    deleted = await repository.delete(job.id)

    assert deleted is True
    assert await session.scalar(select(func.count()).select_from(LogEntryTable)) == 0


async def test_delete_returns_false_for_an_unknown_job(session: AsyncSession) -> None:
    repository = SqlAlchemyJobRepository(session)

    assert await repository.delete(uuid4()) is False


async def test_delete_does_not_load_child_rows_into_memory(
    engine: AsyncEngine,
    session: AsyncSession,
    user: UserTable,
    configuration: ConfigurationTable,
    ready_dataset: DatasetTable,
) -> None:
    """Delete must not ``SELECT`` the child collections before removing them.

    The old cascade ``selectinload``ed every trial/result/log entry/task and
    let the ORM delete them row by row inside one transaction. For a real job
    that is millions of ``log_entries`` hydrated into memory while the delete's
    transaction stays open — the driver of the minute-long deletes and the
    ``Lock wait timeout exceeded`` on ``gb_tasks`` seen in production. A
    set-based ``DELETE ... WHERE job_id = :id`` per child table reads no child
    rows at all, so its cost is independent of how many each table holds.

    Counting statements would not catch the regression: SQLite batches the
    old per-row deletes into one ``executemany`` per table, so only the
    pre-load ``SELECT``s distinguish the two paths on this dialect.
    """
    repository = SqlAlchemyJobRepository(session)
    job = await repository.create(
        user_id=str(user.id),
        config_id=configuration.id,
        dataset_id=ready_dataset.id,
        model="m",
        model_source="huggingface",
        experiment_name="exp",
        tuning_type=None,
        seed=42,
        autotune=True,
        config_snapshot={},
        reward_function_code=None,
        reward_function_name=None,
    )
    session.add_all([TrialTable(id=f"t{i}", job_id=job.id, config={}) for i in range(5)])
    await session.commit()
    session.add_all(
        [
            ResultTable(id=uuid4(), job_id=job.id, trial_id=f"t{i}", metric="loss", metrics={})
            for i in range(5)
        ]
    )
    session.add_all(
        [LogEntryTable(job_id=job.id, level="INFO", message=f"line {i}") for i in range(6)]
    )
    session.add_all(
        [
            GbTaskTable(id=uuid4(), job_id=job.id, type=GbTaskType.TUNING),
            GbTaskTable(id=uuid4(), job_id=job.id, type=GbTaskType.RITS),
        ]
    )
    await session.commit()

    statements: list[str] = []

    @event.listens_for(engine.sync_engine, "before_cursor_execute")
    def _record(
        _conn: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        statements.append(statement)

    deleted = await repository.delete(job.id)

    assert deleted is True
    child_selects = [
        s
        for s in statements
        if s.lstrip().upper().startswith("SELECT")
        and any(f"from {t}" in s.lower() for t in ("trials", "results", "log_entries", "gb_tasks"))
    ]
    assert child_selects == [], child_selects


async def _seed_logs(session: AsyncSession, job: JobTable) -> None:
    """Seed six job-level lines (ids 1-6) and two trial lines (ids 7-8)."""
    for i in range(1, 7):
        session.add(
            LogEntryTable(id=i, job_id=job.id, trial_id=None, level="INFO", message=f"job line {i}")
        )
    for i in (7, 8):
        session.add(
            LogEntryTable(
                id=i, job_id=job.id, trial_id="abc123", level="INFO", message=f"trial line {i}"
            )
        )
    await session.commit()


async def test_logs_page_returns_job_level_lines_newest_first(
    session: AsyncSession, job: JobTable
) -> None:
    await _seed_logs(session, job)
    repository = SqlAlchemyJobRepository(session)

    rows, has_more = await repository.logs_page(job.id, trial_id=None, before_id=0, limit=50)

    assert [r.id for r in rows] == [6, 5, 4, 3, 2, 1]
    assert has_more is False


async def test_logs_page_signals_has_more_and_trims_to_limit(
    session: AsyncSession, job: JobTable
) -> None:
    await _seed_logs(session, job)
    repository = SqlAlchemyJobRepository(session)

    rows, has_more = await repository.logs_page(job.id, trial_id=None, before_id=0, limit=2)

    assert [r.id for r in rows] == [6, 5]
    assert has_more is True


async def test_logs_page_honors_the_before_id_cursor(session: AsyncSession, job: JobTable) -> None:
    await _seed_logs(session, job)
    repository = SqlAlchemyJobRepository(session)

    rows, has_more = await repository.logs_page(job.id, trial_id=None, before_id=5, limit=2)

    assert [r.id for r in rows] == [4, 3]
    assert has_more is True


async def test_logs_page_selects_only_the_named_trials_lines(
    session: AsyncSession, job: JobTable
) -> None:
    await _seed_logs(session, job)
    repository = SqlAlchemyJobRepository(session)

    rows, _has_more = await repository.logs_page(job.id, trial_id="abc123", before_id=0, limit=50)

    assert [r.id for r in rows] == [8, 7]
    assert all(r.trial_id == "abc123" for r in rows)


async def test_logs_page_is_empty_for_a_trial_id_of_another_job(
    session: AsyncSession, job: JobTable
) -> None:
    await _seed_logs(session, job)
    repository = SqlAlchemyJobRepository(session)

    rows, has_more = await repository.logs_page(job.id, trial_id="not-mine", before_id=0, limit=50)

    assert rows == []
    assert has_more is False


async def test_is_visible_is_true_for_an_existing_job(session: AsyncSession, job: JobTable) -> None:
    repository = SqlAlchemyJobRepository(session)

    assert await repository.is_visible(job.id) is True


async def test_is_visible_is_false_for_an_unknown_job(session: AsyncSession) -> None:
    repository = SqlAlchemyJobRepository(session)

    assert await repository.is_visible(uuid4()) is False


async def test_list_filters_jobs_by_experiment_name_substring(
    session: AsyncSession,
    user: UserTable,
    configuration: ConfigurationTable,
    ready_dataset: DatasetTable,
) -> None:
    repository = SqlAlchemyJobRepository(session)
    for name in ("alpha-sweep", "beta-run"):
        await repository.create(
            user_id=str(user.id),
            config_id=configuration.id,
            dataset_id=ready_dataset.id,
            model="ibm/granite",
            model_source="huggingface",
            experiment_name=name,
            tuning_type=None,
            seed=42,
            autotune=True,
            config_snapshot={},
            reward_function_code=None,
            reward_function_name=None,
        )

    rows, total = await repository.list(limit=20, offset=0, q="ALPHA")
    jobs = [job for job, _ in rows]

    assert total == 1
    assert [j.experiment_name for j in jobs] == ["alpha-sweep"]


async def test_list_filters_jobs_by_model_substring(
    session: AsyncSession,
    user: UserTable,
    configuration: ConfigurationTable,
    ready_dataset: DatasetTable,
) -> None:
    repository = SqlAlchemyJobRepository(session)
    await repository.create(
        user_id=str(user.id),
        config_id=configuration.id,
        dataset_id=ready_dataset.id,
        model="meta-llama/Llama-3",
        model_source="huggingface",
        experiment_name="x",
        tuning_type=None,
        seed=42,
        autotune=True,
        config_snapshot={},
        reward_function_code=None,
        reward_function_name=None,
    )

    rows, total = await repository.list(limit=20, offset=0, q="llama")
    jobs = [job for job, _ in rows]

    assert total == 1
    assert jobs[0].model == "meta-llama/Llama-3"


async def test_list_filters_jobs_by_status_substring(
    session: AsyncSession,
    user: UserTable,
    configuration: ConfigurationTable,
    ready_dataset: DatasetTable,
) -> None:
    repository = SqlAlchemyJobRepository(session)
    job = await repository.create(
        user_id=str(user.id),
        config_id=configuration.id,
        dataset_id=ready_dataset.id,
        model="ibm/granite",
        model_source="huggingface",
        experiment_name="x",
        tuning_type=None,
        seed=42,
        autotune=True,
        config_snapshot={},
        reward_function_code=None,
        reward_function_name=None,
    )
    await repository.set_status(job.id, RunStatus.RUNNING)

    rows, total = await repository.list(limit=20, offset=0, q="run")
    jobs = [job for job, _ in rows]

    assert total == 1
    assert jobs[0].status is RunStatus.RUNNING


async def test_list_escapes_like_wildcards_in_q(
    session: AsyncSession,
    user: UserTable,
    configuration: ConfigurationTable,
    ready_dataset: DatasetTable,
) -> None:
    repository = SqlAlchemyJobRepository(session)
    await repository.create(
        user_id=str(user.id),
        config_id=configuration.id,
        dataset_id=ready_dataset.id,
        model="ibm/granite",
        model_source="huggingface",
        experiment_name="plain",
        tuning_type=None,
        seed=42,
        autotune=True,
        config_snapshot={},
        reward_function_code=None,
        reward_function_name=None,
    )

    rows, total = await repository.list(limit=20, offset=0, q="%")

    assert total == 0
    assert rows == []


async def test_list_reports_finished_at_as_the_latest_task_update(
    session: AsyncSession,
    user: UserTable,
    configuration: ConfigurationTable,
    ready_dataset: DatasetTable,
) -> None:
    """finished_at is the newest gb_tasks.updated_at across the job's tasks."""
    repository = SqlAlchemyJobRepository(session)
    job = await repository.create(
        user_id=str(user.id),
        config_id=configuration.id,
        dataset_id=ready_dataset.id,
        model="ibm/granite",
        model_source="huggingface",
        experiment_name="exp",
        tuning_type=None,
        seed=42,
        autotune=True,
        config_snapshot={},
        reward_function_code=None,
        reward_function_name=None,
    )
    session.add_all(
        [
            GbTaskTable(
                job_id=job.id,
                type=GbTaskType.RITS,
                status=RunStatus.COMPLETED,
                updated_at="2026-08-11T09:00:00Z",
            ),
            GbTaskTable(
                job_id=job.id,
                type=GbTaskType.TUNING,
                status=RunStatus.COMPLETED,
                updated_at="2026-08-11T11:05:26Z",
            ),
        ]
    )
    await session.commit()

    rows, total = await repository.list(limit=20, offset=0)

    assert total == 1
    returned_job, finished_at = rows[0]
    assert returned_job.id == job.id
    assert finished_at == "2026-08-11T11:05:26Z"


async def test_list_reports_finished_at_none_when_a_job_has_no_tasks(
    session: AsyncSession,
    user: UserTable,
    configuration: ConfigurationTable,
    ready_dataset: DatasetTable,
) -> None:
    """A job with no build tasks has no computable run end."""
    repository = SqlAlchemyJobRepository(session)
    await repository.create(
        user_id=str(user.id),
        config_id=configuration.id,
        dataset_id=ready_dataset.id,
        model="ibm/granite",
        model_source="huggingface",
        experiment_name="exp",
        tuning_type=None,
        seed=42,
        autotune=True,
        config_snapshot={},
        reward_function_code=None,
        reward_function_name=None,
    )

    rows, total = await repository.list(limit=20, offset=0)

    assert total == 1
    assert rows[0][1] is None


async def test_get_by_build_id_returns_the_job_for_a_matching_task(
    session: AsyncSession,
    user: UserTable,
    configuration: ConfigurationTable,
    ready_dataset: DatasetTable,
) -> None:
    repository = SqlAlchemyJobRepository(session)
    job = await repository.create(
        user_id=str(user.id),
        config_id=configuration.id,
        dataset_id=ready_dataset.id,
        model="m",
        model_source="huggingface",
        experiment_name="exp",
        tuning_type=None,
        seed=42,
        autotune=True,
        config_snapshot={},
        reward_function_code=None,
        reward_function_name=None,
    )
    build_id = uuid4()
    session.add(
        GbTaskTable(
            job_id=job.id, type=GbTaskType.TUNING, status=RunStatus.RUNNING, build_id=build_id
        )
    )
    await session.commit()

    found = await repository.get_by_build_id(build_id)

    assert found is not None
    assert found.id == job.id


async def test_get_by_build_id_returns_none_for_an_unknown_build(
    session: AsyncSession,
) -> None:
    repository = SqlAlchemyJobRepository(session)

    found = await repository.get_by_build_id(uuid4())

    assert found is None


async def test_get_by_build_id_honours_the_owner_filter(
    session: AsyncSession,
    user: UserTable,
    configuration: ConfigurationTable,
    ready_dataset: DatasetTable,
) -> None:
    repository = SqlAlchemyJobRepository(session)
    job = await repository.create(
        user_id=str(user.id),
        config_id=configuration.id,
        dataset_id=ready_dataset.id,
        model="m",
        model_source="huggingface",
        experiment_name="exp",
        tuning_type=None,
        seed=42,
        autotune=True,
        config_snapshot={},
        reward_function_code=None,
        reward_function_name=None,
    )
    build_id = uuid4()
    session.add(GbTaskTable(job_id=job.id, type=GbTaskType.TUNING, build_id=build_id))
    await session.commit()

    found = await repository.get_by_build_id(build_id, owner_id=uuid4())

    assert found is None
