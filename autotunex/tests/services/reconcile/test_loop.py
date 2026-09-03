"""ReconcileLoop.sweep_once against real in-memory SQLite with a fake reader."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from autotunex.db.repositories.sqlalchemy import (
    SqlAlchemyJobRepository,
    SqlAlchemyTrialRepository,
)
from autotunex.db.tables import (
    ConfigurationTable,
    DatasetTable,
    JobTable,
    TrialTable,
    UserTable,
)
from autotunex.models.status import DatasetStatus, GbTaskType, RunStatus
from autotunex.services.reconcile.loop import ReconcileLoop
from autotunex.services.reconcile.protocols import (
    BuildNotFoundError,
    BuildState,
    BuildStatusAuthError,
    BuildStatusUnavailableError,
    MalformedBuildStatusError,
)

BUILD_ID = UUID("22222222-2222-2222-2222-222222222222")


class _FakeReader:
    """Hand-written BuildStatusReader: returns canned status/events or raises."""

    def __init__(
        self,
        state: BuildState | None = None,
        *,
        error: Exception | None = None,
        events: dict[str, Any] | None = None,
        events_error: Exception | None = None,
    ) -> None:
        self._state = state
        self._error = error
        self._events: dict[str, Any] = events if events is not None else {"events": []}
        self._events_error = events_error
        self.reads: list[UUID] = []
        self.event_reads: list[UUID] = []

    async def read(self, build_id: UUID) -> BuildState:
        self.reads.append(build_id)
        if self._error is not None:
            raise self._error
        assert self._state is not None
        return self._state

    async def read_events(self, build_id: UUID) -> dict[str, Any]:
        self.event_reads.append(build_id)
        if self._events_error is not None:
            raise self._events_error
        return self._events


# One event with a msg, so build_history is non-empty and backticks are stripped.
_EVENTS: dict[str, Any] = {
    "events": [
        {"build_event": {"timestamp": "2026-08-07T00:04:00Z", "payload": {"msg": "done `ok`"}}}
    ]
}


def _state(status: str) -> BuildState:
    return BuildState(
        build_id=BUILD_ID,
        status=status,
        failure_reason=None,
        created_at="2026-08-07T00:00:00Z",
        updated_at="2026-08-07T00:05:00Z",
        raw={"status": {"build": {"status": status}}},
    )


def _state_with_output_artifact() -> BuildState:
    """A failed build that still registered a model output artifact."""
    raw = {
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
    return BuildState(
        build_id=BUILD_ID,
        status="failed",
        failure_reason=None,
        created_at="2026-08-07T00:00:00Z",
        updated_at="2026-08-07T00:05:00Z",
        raw=raw,
    )


async def _seed(
    factory: async_sessionmaker[AsyncSession],
    *,
    job_status: RunStatus = RunStatus.PENDING,
    build_id: UUID | None = BUILD_ID,
    with_task: bool = True,
    with_trial: bool = False,
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
        if with_trial:
            session.add(TrialTable(id="t1", job_id=job.id, status=RunStatus.COMPLETED, config={}))
            await session.commit()
        return job.id


async def _status(factory: async_sessionmaker[AsyncSession], job_id: UUID) -> RunStatus:
    async with factory() as session:
        job = await SqlAlchemyJobRepository(session).get(job_id)
    assert job is not None
    return job.status


async def test_pending_job_advances_to_running(engine: AsyncEngine) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed(factory)
    loop = ReconcileLoop(session_factory=factory, reader=_FakeReader(_state("running")))

    await loop.sweep_once()

    assert await _status(factory, job_id) == RunStatus.RUNNING


@pytest.mark.parametrize(
    ("cluster", "expected"),
    [
        ("success", RunStatus.COMPLETED),
        ("failed", RunStatus.ERROR),
        ("invalid", RunStatus.ERROR),
        ("cancelled", RunStatus.TERMINATED),
    ],
)
async def test_running_job_advances_to_each_terminal_state(
    engine: AsyncEngine, cluster: str, expected: RunStatus
) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed(factory, job_status=RunStatus.RUNNING)
    loop = ReconcileLoop(session_factory=factory, reader=_FakeReader(_state(cluster)))

    await loop.sweep_once()

    assert await _status(factory, job_id) == expected


async def test_no_write_for_a_not_yet_decisive_status(engine: AsyncEngine) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed(factory)
    loop = ReconcileLoop(session_factory=factory, reader=_FakeReader(_state("submitted")))

    await loop.sweep_once()

    assert await _status(factory, job_id) == RunStatus.PENDING


async def test_terminal_jobs_are_never_polled(engine: AsyncEngine) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    await _seed(factory, job_status=RunStatus.COMPLETED)
    reader = _FakeReader(_state("running"))
    loop = ReconcileLoop(session_factory=factory, reader=reader)

    await loop.sweep_once()

    assert reader.reads == []  # excluded by the query, reader never consulted


async def test_jobs_without_a_build_id_are_never_polled(engine: AsyncEngine) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    await _seed(factory, build_id=None)
    reader = _FakeReader(_state("running"))
    loop = ReconcileLoop(session_factory=factory, reader=reader)

    await loop.sweep_once()

    assert reader.reads == []


async def test_racing_transition_is_swallowed(engine: AsyncEngine) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed(factory, job_status=RunStatus.RUNNING)
    # running -> running is not a legal transition; the loser is skipped, not raised.
    loop = ReconcileLoop(session_factory=factory, reader=_FakeReader(_state("running")))

    await loop.sweep_once()  # must not raise

    assert await _status(factory, job_id) == RunStatus.RUNNING


async def test_terminal_persists_transformed_build_status_with_history(
    engine: AsyncEngine,
) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed(factory, job_status=RunStatus.RUNNING)
    reader = _FakeReader(_state("success"), events=_EVENTS)
    loop = ReconcileLoop(session_factory=factory, reader=reader)

    await loop.sweep_once()

    async with factory() as session:
        task = await SqlAlchemyJobRepository(session).get_task(job_id, GbTaskType.TUNING)
    assert task is not None
    assert task.status == RunStatus.COMPLETED
    # The transformed {details, targets, build_history} shape, not the raw body.
    assert task.build_status is not None
    assert "status" not in task.build_status
    assert task.build_status["details"]["status"] == "success"
    assert task.build_status["build_history"] == [
        {"time": "2026-08-07T00:04:00Z", "description": "done ok"}
    ]
    assert task.started_at == "2026-08-07T00:00:00Z"
    assert task.updated_at == "2026-08-07T00:05:00Z"
    assert reader.event_reads == [BUILD_ID]  # events fetched once, only at terminal


async def test_events_not_fetched_for_a_non_terminal_transition(engine: AsyncEngine) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed(factory)  # pending -> running
    reader = _FakeReader(_state("running"), events=_EVENTS)
    loop = ReconcileLoop(session_factory=factory, reader=reader)

    await loop.sweep_once()

    assert await _status(factory, job_id) == RunStatus.RUNNING
    assert reader.event_reads == []  # running is not terminal; no events round trip


async def test_terminal_history_is_empty_when_events_are_unavailable(engine: AsyncEngine) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed(factory, job_status=RunStatus.RUNNING)
    reader = _FakeReader(_state("success"), events_error=BuildStatusUnavailableError())
    loop = ReconcileLoop(session_factory=factory, reader=reader)

    await loop.sweep_once()  # must not raise; the terminal write still happens

    async with factory() as session:
        task = await SqlAlchemyJobRepository(session).get_task(job_id, GbTaskType.TUNING)
    assert task is not None
    assert task.status == RunStatus.COMPLETED
    assert task.build_status is not None
    assert task.build_status["build_history"] == []
    assert task.build_status["details"]["status"] == "success"


async def test_terminal_populates_artifact_id_and_uri_from_output_artifacts(
    engine: AsyncEngine,
) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed(factory, job_status=RunStatus.RUNNING)
    loop = ReconcileLoop(session_factory=factory, reader=_FakeReader(_state_with_output_artifact()))

    await loop.sweep_once()

    async with factory() as session:
        task = await SqlAlchemyJobRepository(session).get_task(job_id, GbTaskType.TUNING)
    assert task is not None
    assert task.status == RunStatus.ERROR  # failed build, artifact still recorded
    assert str(task.artifact_id) == "d4affa76-52a8-4f57-bd5b-db49470fed5f"
    assert task.artifact_uri == "hf://huggingface.co/models/ibm-research/autotunex_a69082b7"


async def test_reconcile_never_modifies_trials(engine: AsyncEngine) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed(factory, job_status=RunStatus.RUNNING, with_trial=True)
    loop = ReconcileLoop(session_factory=factory, reader=_FakeReader(_state("failed")))

    await loop.sweep_once()

    async with factory() as session:
        job = await SqlAlchemyJobRepository(session).get(job_id)
        # Read through the trial repository: the job detail no longer eager-loads
        # trials, so this is now the only read path for them.
        trials, total = await SqlAlchemyTrialRepository(session).page(job_id, limit=50, offset=0)
    assert job is not None and job.status == RunStatus.ERROR
    # The pre-existing completed trial is untouched — the 2025 cascade bug.
    assert total == 1
    assert [trial.status for trial in trials] == [RunStatus.COMPLETED]


async def test_auth_error_does_not_raise_and_leaves_the_job(engine: AsyncEngine) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed(factory)
    loop = ReconcileLoop(session_factory=factory, reader=_FakeReader(error=BuildStatusAuthError()))

    await loop.sweep_once()  # must not raise

    assert await _status(factory, job_id) == RunStatus.PENDING


async def test_not_found_leaves_the_job_untouched(engine: AsyncEngine) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed(factory)
    loop = ReconcileLoop(session_factory=factory, reader=_FakeReader(error=BuildNotFoundError()))

    await loop.sweep_once()  # must not raise

    assert await _status(factory, job_id) == RunStatus.PENDING


async def test_unavailable_error_leaves_the_job_untouched(engine: AsyncEngine) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed(factory)
    loop = ReconcileLoop(
        session_factory=factory, reader=_FakeReader(error=BuildStatusUnavailableError())
    )

    await loop.sweep_once()  # must not raise

    assert await _status(factory, job_id) == RunStatus.PENDING


async def test_malformed_status_leaves_the_job_untouched(engine: AsyncEngine) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed(factory)
    loop = ReconcileLoop(
        session_factory=factory, reader=_FakeReader(error=MalformedBuildStatusError())
    )

    await loop.sweep_once()  # must not raise

    assert await _status(factory, job_id) == RunStatus.PENDING


async def test_pending_job_advances_to_completed_on_a_late_observed_success(
    engine: AsyncEngine,
) -> None:
    # The job never caught the running transit (e.g. a restart mid-build) but
    # gbserver already reports success; pending -> completed must be applied,
    # not skipped as an illegal transition.
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed(factory)
    loop = ReconcileLoop(session_factory=factory, reader=_FakeReader(_state("success")))

    await loop.sweep_once()

    assert await _status(factory, job_id) == RunStatus.COMPLETED
