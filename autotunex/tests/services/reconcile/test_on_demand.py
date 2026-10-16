"""OnDemandReconciler against real in-memory SQLite with a fake reader.

Unlike the background loop's tests, these assert the force behaviour: detail is
always rewritten, and ``jobs.status`` is set to whatever gbserver reports even
across a terminal->terminal correction ``check_transition`` would forbid.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from autotunex.core.exceptions import (
    BuildReconcileUpstreamError,
    JobNotFoundError,
    JobNotReconcilableError,
)
from autotunex.db.repositories.sqlalchemy import SqlAlchemyJobRepository
from autotunex.db.tables import ConfigurationTable, DatasetTable, GbTaskTable, JobTable, UserTable
from autotunex.models.job import JobRead
from autotunex.models.status import DatasetStatus, GbTaskType, RunStatus
from autotunex.services.reconcile.on_demand import OnDemandReconciler
from autotunex.services.reconcile.protocols import (
    BuildState,
    BuildStatusReader,
    BuildStatusUnavailableError,
)

BUILD_ID = UUID("22222222-2222-2222-2222-222222222222")


class _FakeReader:
    """Hand-written BuildStatusReader: returns a canned status and events."""

    def __init__(
        self,
        state: BuildState,
        *,
        events: dict[str, Any] | None = None,
        events_error: Exception | None = None,
    ) -> None:
        self._state = state
        self._events: dict[str, Any] = events if events is not None else {"events": []}
        self._events_error = events_error

    async def read(self, build_id: UUID) -> BuildState:
        return self._state

    async def read_events(self, build_id: UUID) -> dict[str, Any]:
        if self._events_error is not None:
            raise self._events_error
        return self._events


class _RaisingReader:
    """A reader whose status read always raises."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    async def read(self, build_id: UUID) -> BuildState:
        raise self._error

    async def read_events(self, build_id: UUID) -> dict[str, Any]:
        return {"events": []}


def _state(status: str, raw: dict[str, Any] | None = None) -> BuildState:
    return BuildState(
        build_id=BUILD_ID,
        status=status,
        failure_reason=None,
        created_at="2026-08-07T00:00:00Z",
        updated_at="2026-08-07T00:05:00Z",
        raw=raw if raw is not None else {"status": {"build": {"status": status}}},
    )


_ARTIFACT_RAW: dict[str, Any] = {
    "status": {
        "build": {"status": "failed", "uuid": "dcd60f52"},
        "target_runs": [
            {
                "target": {"name": "custom", "uuid": "t-1", "status": "failed"},
                "input_artifacts": [],
                "output_artifacts": [
                    {
                        "uuid": "d4affa76-52a8-4f57-bd5b-db49470fed5f",
                        "uri": "hf://huggingface.co/models/ibm-research/autotunex_a69082b7",
                    }
                ],
                "steps": [],
            }
        ],
    }
}

_EVENTS: dict[str, Any] = {
    "events": [
        {"build_event": {"timestamp": "2026-08-07T00:04:00Z", "payload": {"msg": "done `ok`"}}}
    ]
}


async def _seed(
    factory: async_sessionmaker[AsyncSession],
    *,
    job_status: RunStatus = RunStatus.RUNNING,
    build_id: UUID | None = BUILD_ID,
    with_task: bool = True,
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
            config_data={},
        )
        dataset = DatasetTable(
            id=uuid4(),
            user_id=str(user.id),
            name="d",
            data_format="jsonl",
            status=DatasetStatus.READY,
        )
        session.add_all([config, dataset])
        await session.commit()
        job = JobTable(
            id=uuid4(),
            user_id=str(user.id),
            status=job_status,
            config_id=config.id,
            dataset_id=dataset.id,
            model="m",
            model_source="huggingface",
            experiment_name="e",
            tuning_type="lora",
        )
        session.add(job)
        await session.commit()
        if with_task:
            await SqlAlchemyJobRepository(session).upsert_task(
                job.id, GbTaskType.TUNING, status=RunStatus.PENDING, build_id=build_id
            )
        return job.id


async def _reconcile(
    factory: async_sessionmaker[AsyncSession], job_id: UUID, reader: BuildStatusReader
) -> JobRead:
    async with factory() as session:
        reconciler = OnDemandReconciler(repository=SqlAlchemyJobRepository(session), reader=reader)
        return await reconciler.reconcile(job_id)


async def _task_of(factory: async_sessionmaker[AsyncSession], job_id: UUID) -> GbTaskTable:
    async with factory() as session:
        task = await SqlAlchemyJobRepository(session).get_task(job_id, GbTaskType.TUNING)
    assert task is not None
    return task


async def test_reconcile_refreshes_transformed_build_status_and_history(
    engine: AsyncEngine,
) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed(factory)

    read = await _reconcile(factory, job_id, _FakeReader(_state("success"), events=_EVENTS))

    assert read.status == RunStatus.COMPLETED
    task = await _task_of(factory, job_id)
    assert task.build_status is not None
    assert "status" not in task.build_status
    assert task.build_status["details"]["status"] == "success"
    assert task.build_status["build_history"] == [
        {"time": "2026-08-07T00:04:00Z", "description": "done ok"}
    ]


async def test_reconcile_populates_artifact_from_output_artifacts(engine: AsyncEngine) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed(factory)

    read = await _reconcile(factory, job_id, _FakeReader(_state("failed", raw=_ARTIFACT_RAW)))

    assert read.status == RunStatus.ERROR
    task = await _task_of(factory, job_id)
    assert str(task.artifact_id) == "d4affa76-52a8-4f57-bd5b-db49470fed5f"
    assert task.artifact_uri == "hf://huggingface.co/models/ibm-research/autotunex_a69082b7"


async def test_reconcile_forces_status_bypassing_the_state_machine(engine: AsyncEngine) -> None:
    # A job wrongly parked in a terminal state (the split-brain case): a legal
    # transition would forbid completed -> error, but the force must apply it.
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed(factory, job_status=RunStatus.COMPLETED)

    read = await _reconcile(factory, job_id, _FakeReader(_state("failed")))

    assert read.status == RunStatus.ERROR


async def test_reconcile_leaves_status_when_gbserver_is_undecided(engine: AsyncEngine) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed(factory, job_status=RunStatus.RUNNING)

    read = await _reconcile(factory, job_id, _FakeReader(_state("submitted")))

    assert read.status == RunStatus.RUNNING
    task = await _task_of(factory, job_id)
    assert task.build_status is not None  # detail still refreshed
    assert task.build_status["details"]["status"] == "submitted"


async def test_reconcile_is_resilient_to_events_failure(engine: AsyncEngine) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed(factory)

    read = await _reconcile(
        factory,
        job_id,
        _FakeReader(_state("success"), events_error=BuildStatusUnavailableError()),
    )

    assert read.status == RunStatus.COMPLETED
    task = await _task_of(factory, job_id)
    assert task.build_status is not None
    assert task.build_status["build_history"] == []


async def test_reconcile_raises_when_job_missing(engine: AsyncEngine) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)

    with pytest.raises(JobNotFoundError):
        await _reconcile(factory, uuid4(), _FakeReader(_state("success")))


async def test_reconcile_raises_when_no_build_id(engine: AsyncEngine) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed(factory, build_id=None)

    with pytest.raises(JobNotReconcilableError):
        await _reconcile(factory, job_id, _FakeReader(_state("success")))


async def test_reconcile_maps_status_read_failure_to_upstream(engine: AsyncEngine) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed(factory)

    with pytest.raises(BuildReconcileUpstreamError):
        await _reconcile(factory, job_id, _RaisingReader(BuildStatusUnavailableError("down")))
