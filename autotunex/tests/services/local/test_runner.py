"""LocalJobRunner.process terminal behavior, driven by fake trainers.

The runner opens its OWN sessions, so these build a sessionmaker on the shared
test engine and assert by reading rows back through a fresh session. Both fakes
drive the ``DbTrialSink`` from inside ``asyncio.to_thread`` (the runner invokes
the trainer there), so the sink's worker-thread → loop bridge is genuinely
exercised, not stubbed.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from autotunex.core.exceptions import JobCancellationInProgressError
from autotunex.db.repositories.sqlalchemy import SqlAlchemyJobRepository
from autotunex.db.tables import ConfigurationTable, DatasetTable, JobTable, UserTable
from autotunex.db.tables.results import ResultTable
from autotunex.db.tables.trials import TrialTable
from autotunex.models.status import DatasetStatus, GbTaskType, RunStatus
from autotunex.services.local import cancellation
from autotunex.services.local.protocols import LocalRunContext, TrialSink
from autotunex.services.local.runner import LocalJobRunner


class FakeTrainer:
    """Emits one trial that starts, reports a result, and completes."""

    def run(self, ctx: LocalRunContext, sink: TrialSink) -> None:
        sink.trial_started("t01", {"lr": 1})
        sink.trial_result("t01", "loss", {"loss": 0.5})
        sink.trial_completed("t01")


class BoomTrainer:
    """Starts one trial, then raises before it is ever completed."""

    def run(self, ctx: LocalRunContext, sink: TrialSink) -> None:
        sink.trial_started("t01", None)
        raise RuntimeError("boom")


class CapturingTrainer:
    """Records the context it is handed and returns without touching the sink."""

    def __init__(self) -> None:
        self.contexts: list[LocalRunContext] = []

    def run(self, ctx: LocalRunContext, sink: TrialSink) -> None:
        self.contexts.append(ctx)


class CooperativeTrainer:
    """Polls the cancellation registry and returns as soon as it is cancelled."""

    def run(self, ctx: LocalRunContext, sink: TrialSink) -> None:
        while not cancellation.is_cancelled(ctx.job_id):
            time.sleep(0.01)


class StubbornTrainer:
    """Ignores cancellation for the duration of one sleep, then returns normally."""

    def run(self, ctx: LocalRunContext, sink: TrialSink) -> None:
        time.sleep(0.5)


async def _wait_until(condition: Callable[[], bool], *, timeout_seconds: float = 2.0) -> None:
    """Poll ``condition`` every 10ms until it is true, or raise after ``timeout_seconds``."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    while not condition():
        if loop.time() >= deadline:
            raise AssertionError("condition was not met before the timeout")
        await asyncio.sleep(0.01)


async def _seed_job(factory: async_sessionmaker[AsyncSession], *, dataset_root: Path) -> UUID:
    """Seed a pending, runnable job and write its training file to ``dataset_root``.

    Returns the job id. The dataset is ``ready`` with a HuggingFace model source,
    and its generated ``train_file`` is created on disk under
    ``dataset_root/<dataset_id>/`` so the runner's disk-existence check passes.
    """
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
        await session.refresh(dataset, ["train_file", "validation_file"])

        # The on-disk name carries the format extension (matches LocalStorageBackend
        # and the runner), unlike the extensionless ``train_file`` generated column.
        dataset_dir = dataset_root / str(dataset.id)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / f"{dataset.name}_train.{dataset.data_format}").write_text("{}\n")

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
        return job.id


async def test_process_completes_the_job_and_persists_the_trial(
    engine: AsyncEngine, tmp_path: Path
) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed_job(factory, dataset_root=tmp_path)
    runner = LocalJobRunner(
        session_factory=factory,
        trainer=FakeTrainer(),
        output_root=tmp_path / "out",
        dataset_root=tmp_path,
        cancel_timeout=5.0,
    )

    await runner.process(job_id)

    async with factory() as session:
        repository = SqlAlchemyJobRepository(session)
        job = await repository.get(job_id)
        task = await repository.get_task(job_id, GbTaskType.TUNING)
        trial = await session.get(TrialTable, "t01")
        result = (
            await session.execute(select(ResultTable).where(ResultTable.trial_id == "t01"))
        ).scalar_one_or_none()
    assert job is not None and job.status == RunStatus.COMPLETED
    assert task is not None and task.status == RunStatus.COMPLETED
    assert trial is not None and trial.status == RunStatus.COMPLETED
    assert result is not None and result.metrics == {"loss": 0.5}


async def test_process_errors_the_job_and_sweeps_running_trials_on_failure(
    engine: AsyncEngine, tmp_path: Path
) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed_job(factory, dataset_root=tmp_path)
    runner = LocalJobRunner(
        session_factory=factory,
        trainer=BoomTrainer(),
        output_root=tmp_path / "out",
        dataset_root=tmp_path,
        cancel_timeout=5.0,
    )

    await runner.process(job_id)

    async with factory() as session:
        repository = SqlAlchemyJobRepository(session)
        job = await repository.get(job_id)
        task = await repository.get_task(job_id, GbTaskType.TUNING)
        trial = await session.get(TrialTable, "t01")
    assert job is not None and job.status == RunStatus.ERROR
    assert task is not None and task.status == RunStatus.ERROR
    assert trial is not None and trial.status == RunStatus.ERROR


async def test_process_hands_the_trainer_an_absolute_output_dir_for_a_relative_root(
    engine: AsyncEngine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative ``output_root`` still yields an absolute ``ctx.output_dir``.

    Ray Tune hands ``storage_path`` (derived from ``output_dir``) to pyarrow's
    ``FileSystem.from_uri``, which rejects a relative path with "URI has empty
    scheme". The runner therefore resolves the per-job output dir to an absolute
    path before the trainer passes it to Ray. ``chdir`` into ``tmp_path`` so the
    relative resolution and the runner's ``mkdir`` stay inside the temp tree.
    """
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed_job(factory, dataset_root=tmp_path)
    monkeypatch.chdir(tmp_path)
    trainer = CapturingTrainer()
    runner = LocalJobRunner(
        session_factory=factory,
        trainer=trainer,
        output_root=Path("relative/out"),
        dataset_root=tmp_path,
        cancel_timeout=5.0,
    )

    await runner.process(job_id)

    assert trainer.contexts
    assert trainer.contexts[0].output_dir.is_absolute()


async def test_process_errors_the_job_when_the_model_source_is_unsupported(
    engine: AsyncEngine, tmp_path: Path
) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed_job(factory, dataset_root=tmp_path)
    async with factory() as session:
        job = await session.get(JobTable, job_id)
        assert job is not None
        job.model_source = "dmf"
        await session.commit()
    runner = LocalJobRunner(
        session_factory=factory,
        trainer=FakeTrainer(),
        output_root=tmp_path / "out",
        dataset_root=tmp_path,
        cancel_timeout=5.0,
    )

    await runner.process(job_id)

    async with factory() as session:
        job = await SqlAlchemyJobRepository(session).get(job_id)
    assert job is not None and job.status == RunStatus.ERROR


async def test_process_errors_the_job_when_the_training_file_is_missing(
    engine: AsyncEngine, tmp_path: Path
) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    # dataset_root points at an empty tree, so no training file exists on disk.
    job_id = await _seed_job(factory, dataset_root=tmp_path / "seeded")
    runner = LocalJobRunner(
        session_factory=factory,
        trainer=FakeTrainer(),
        output_root=tmp_path / "out",
        dataset_root=tmp_path / "empty",
        cancel_timeout=5.0,
    )

    await runner.process(job_id)

    async with factory() as session:
        job = await SqlAlchemyJobRepository(session).get(job_id)
    assert job is not None and job.status == RunStatus.ERROR


async def test_cancelled_local_run_ends_terminated_not_completed(
    engine: AsyncEngine, tmp_path: Path
) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed_job(factory, dataset_root=tmp_path)
    runner = LocalJobRunner(
        session_factory=factory,
        trainer=CooperativeTrainer(),
        output_root=tmp_path / "out",
        dataset_root=tmp_path,
        cancel_timeout=5.0,
    )
    task = asyncio.create_task(runner.process(job_id))
    await _wait_until(lambda: cancellation.is_active(job_id))

    await runner.cancel(job_id)  # signals + waits for the run to stop
    await task

    async with factory() as session:
        repository = SqlAlchemyJobRepository(session)
        job = await repository.get(job_id)
        gb_task = await repository.get_task(job_id, GbTaskType.TUNING)
    assert job is not None and job.status == RunStatus.TERMINATED
    assert gb_task is not None and gb_task.status == RunStatus.TERMINATED


async def test_cancel_times_out_when_run_will_not_stop(engine: AsyncEngine, tmp_path: Path) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed_job(factory, dataset_root=tmp_path)
    runner = LocalJobRunner(
        session_factory=factory,
        trainer=StubbornTrainer(),
        output_root=tmp_path / "out",
        dataset_root=tmp_path,
        cancel_timeout=0.2,
    )
    task = asyncio.create_task(runner.process(job_id))
    await _wait_until(lambda: cancellation.is_active(job_id))

    with pytest.raises(JobCancellationInProgressError):
        await runner.cancel(job_id)
    await task  # cleanup: the run finishes and unregisters shortly after


async def test_cancel_on_unregistered_job_is_a_noop(engine: AsyncEngine, tmp_path: Path) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    runner = LocalJobRunner(
        session_factory=factory,
        trainer=FakeTrainer(),
        output_root=tmp_path / "out",
        dataset_root=tmp_path,
        cancel_timeout=0.2,
    )

    await runner.cancel(uuid4())  # must not raise
