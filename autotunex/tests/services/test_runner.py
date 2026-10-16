"""InProcessJobRunner.process terminal behavior, with a fake launcher.

The runner opens its OWN session, so these build a sessionmaker on the shared
test engine and assert by reading rows back through a fresh session.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from autotunex.core.exceptions import BuildCancelUpstreamError
from autotunex.db.repositories.sqlalchemy import SqlAlchemyJobRepository
from autotunex.db.tables import ConfigurationTable, DatasetTable, JobTable, UserTable
from autotunex.models.status import DatasetStatus, GbTaskType, RunStatus
from autotunex.services.launch.protocols import LaunchContext, LaunchHandle
from autotunex.services.runner import InProcessJobRunner, NoOpJobRunner

BUILD_ID = UUID("22222222-2222-2222-2222-222222222222")


class _OkLauncher:
    def __init__(self) -> None:
        self.seen: list[LaunchContext] = []

    async def launch(self, ctx: LaunchContext) -> LaunchHandle:
        self.seen.append(ctx)
        return LaunchHandle(build_id=BUILD_ID, pr_url="https://x/pull/1")


class _FailingLauncher:
    async def launch(self, ctx: LaunchContext) -> LaunchHandle:
        raise RuntimeError("cluster refused the build")


class _RecordingCanceller:
    """Fake :class:`BuildCanceller` that records the build ids it was asked to cancel."""

    def __init__(self) -> None:
        self.calls: list[UUID] = []

    async def cancel(self, build_id: UUID) -> None:
        self.calls.append(build_id)


class _BoomCanceller:
    """Fake :class:`BuildCanceller` whose ``cancel`` always fails."""

    async def cancel(self, build_id: UUID) -> None:
        raise RuntimeError("cluster refused")


async def _seed_job(
    factory: async_sessionmaker[AsyncSession], *, build_id: UUID | None = None
) -> UUID:
    async with factory() as session:
        user = UserTable(id=uuid4(), email=f"{uuid4()}@example.com", role="user")
        session.add(user)
        await session.commit()
        config = ConfigurationTable(
            id=uuid4(),
            user_id=str(user.id),
            name="c",
            tuner_type="lora",
            rl_tuner_type=None,
            config_data={"a": 1},
        )
        dataset = DatasetTable(
            id=uuid4(),
            user_id=str(user.id),
            name="alpaca",
            data_format="jsonl",
            status=DatasetStatus.READY,
            artifact_url="s3://data/alpaca",
        )
        session.add_all([config, dataset])
        await session.commit()
        job = JobTable(
            id=uuid4(),
            user_id=str(user.id),
            status=RunStatus.PENDING,
            config_id=config.id,
            dataset_id=dataset.id,
            model="ibm/granite",
            model_source="huggingface",
            experiment_name="exp",
            tuning_type="lora",
            config_snapshot={
                "name": "c",
                "tuner_type": "lora",
                "rl_tuner_type": None,
                "config_data": {"a": 1},
            },
        )
        session.add(job)
        await session.commit()
        if build_id is not None:
            await SqlAlchemyJobRepository(session).upsert_task(
                job.id, GbTaskType.TUNING, status=RunStatus.PENDING, build_id=build_id
            )
        return job.id


async def test_process_records_the_build_and_leaves_the_job_pending(
    engine: AsyncEngine,
) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed_job(factory)
    launcher = _OkLauncher()
    runner = InProcessJobRunner(
        session_factory=factory, launcher=launcher, canceller=_RecordingCanceller()
    )

    await runner.process(job_id)

    async with factory() as session:
        repository = SqlAlchemyJobRepository(session)
        job = await repository.get(job_id)
        task = await repository.get_task(job_id, GbTaskType.TUNING)
    assert job is not None and job.status == RunStatus.PENDING
    assert task is not None and task.build_id == BUILD_ID
    assert launcher.seen[0].dataset_uri == "s3://data/alpaca"
    # _context_from carries the snapshot's config name and the dataset's format.
    assert launcher.seen[0].config_name == "c"
    assert launcher.seen[0].data_format == "jsonl"


async def test_process_marks_error_when_the_launcher_fails(engine: AsyncEngine) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed_job(factory)
    runner = InProcessJobRunner(
        session_factory=factory, launcher=_FailingLauncher(), canceller=_RecordingCanceller()
    )

    await runner.process(job_id)

    async with factory() as session:
        repository = SqlAlchemyJobRepository(session)
        job = await repository.get(job_id)
        task = await repository.get_task(job_id, GbTaskType.TUNING)
    assert job is not None and job.status == RunStatus.ERROR
    assert task is not None and task.status == RunStatus.ERROR


async def test_noop_runner_cancel_is_a_noop() -> None:
    await NoOpJobRunner().cancel(uuid4())  # must not raise


async def test_inprocess_cancel_calls_canceller_with_the_build_id(engine: AsyncEngine) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    build_id = uuid4()
    job_id = await _seed_job(factory, build_id=build_id)
    canceller = _RecordingCanceller()
    runner = InProcessJobRunner(
        session_factory=factory, launcher=_OkLauncher(), canceller=canceller
    )

    await runner.cancel(job_id)

    assert canceller.calls == [build_id]


async def test_inprocess_cancel_is_noop_when_no_build_id(engine: AsyncEngine) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed_job(factory)  # no TUNING task at all
    canceller = _RecordingCanceller()
    runner = InProcessJobRunner(
        session_factory=factory, launcher=_OkLauncher(), canceller=canceller
    )

    await runner.cancel(job_id)

    assert canceller.calls == []


async def test_inprocess_cancel_wraps_canceller_failure(engine: AsyncEngine) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed_job(factory, build_id=uuid4())
    runner = InProcessJobRunner(
        session_factory=factory, launcher=_OkLauncher(), canceller=_BoomCanceller()
    )

    with pytest.raises(BuildCancelUpstreamError, match="cluster refused"):
        await runner.cancel(job_id)
