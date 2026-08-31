"""Unit tests for JobService, isolated from the database.

The fake repository below is a plain class, not a mock: because the seam is a
Protocol, structural typing is enough, and mypy verifies conformance via the
annotated assignment in ``test_doubles_satisfy_their_protocols``.

``JobService.create`` (below) is exercised against the real
``SqlAlchemy*Repository`` implementations instead of the fake, since its
guards depend on genuine ownership-scoped configuration/dataset lookups that
the fake does not model.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine, Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from autotunex.core.exceptions import (
    BuildCancelUpstreamError,
    BuildNotFoundError,
    CallerNotProvisionedError,
    ConfigurationNotFoundError,
    DatasetNotReadyForJobError,
    InvalidStateTransitionError,
    JobNotCancellableError,
    JobNotFoundError,
    MissingRewardFunctionError,
    ScopeNotPermittedError,
)
from autotunex.db.repositories.protocols import JobRepository, ReconcilableJob
from autotunex.db.repositories.sqlalchemy import (
    SqlAlchemyConfigurationRepository,
    SqlAlchemyDatasetRepository,
    SqlAlchemyJobRepository,
)
from autotunex.db.tables import (
    ConfigurationTable,
    DatasetTable,
    GbTaskTable,
    JobTable,
    LogEntryTable,
    UserTable,
)
from autotunex.models.auth import Principal
from autotunex.models.common import DataScope
from autotunex.models.job import ALLOWED_JOB_TRANSITIONS, JobCreate
from autotunex.models.status import TERMINAL_RUN_STATUSES, GbTaskType, RunStatus
from autotunex.services.jobs import JobService, check_transition
from autotunex.services.runner import NoOpJobRunner

ADMIN = Principal(email="admin@example.com", provider="session", user_id=uuid4(), is_admin=True)
"""An unrestricted principal, standing in for both a real admin and standalone mode."""

DEFAULT_SEED_OWNER = Principal(
    email="seed-owner@example.com",
    provider="session",
    user_id=UUID("00000000-0000-0000-0000-000000000001"),
    is_admin=False,
)
"""Owns the rows ``FakeJobRepository.seed()`` creates by default, so the shared
``service`` fixture below exercises real (non-empty) results under the new
own-by-default scoping instead of an admin whose own-scope has nothing to see."""


class FakeJobRepository:
    """In-memory job store, seeded directly rather than through a write path."""

    def __init__(self) -> None:
        self.jobs: dict[UUID, JobTable] = {}
        self.builds: dict[UUID, UUID] = {}  # build_id -> job_id
        self.tasks: dict[tuple[UUID, GbTaskType], GbTaskTable] = {}

    def seed(
        self,
        *,
        status: RunStatus = RunStatus.PENDING,
        owner_id: str | None = None,
        build_id: UUID | None = None,
    ) -> JobTable:
        """Add a job straight into the store, as the tuning pipeline would.

        ``job_to_summary`` reads ``job.user``, ``job.configuration`` and
        ``job.dataset`` (see ``services/mappers.py``), so the fake attaches
        plain, unpersisted rows to those relationships rather than leaving them
        ``None`` — a real repository never returns a job with a missing parent,
        because the read path's inner joins make that impossible.
        """
        now = datetime.now(UTC)
        job = JobTable(
            id=uuid4(),
            status=status,
            user_id=owner_id or "00000000-0000-0000-0000-000000000001",
            config_id=uuid4(),
            dataset_id=uuid4(),
            model="m",
            model_source="huggingface",
            experiment_name="exp",
            created_at=now,
            updated_at=now,
        )
        job.user = UserTable(id=UUID(job.user_id), email="tester@example.com")
        job.configuration = ConfigurationTable(
            id=job.config_id, user_id=job.user_id, name="lora-sweep"
        )
        job.dataset = DatasetTable(
            id=job.dataset_id, user_id=job.user_id, name="alpaca", description="d"
        )
        # num_trials is a column_property backed by a correlated subquery, which
        # only resolves for a session-bound row. A detached fake stands in with a
        # plain attribute instead of running a query.
        job.num_trials = 0
        self.jobs[job.id] = job
        if build_id is not None:
            self.builds[build_id] = job.id
        return job

    async def get(self, job_id: UUID, *, owner_id: UUID | None = None) -> JobTable | None:
        job = self.jobs.get(job_id)
        if job is None:
            return None
        if owner_id is not None and job.user_id != str(owner_id):
            return None
        return job

    async def get_by_build_id(
        self, build_id: UUID, *, owner_id: UUID | None = None
    ) -> JobTable | None:
        """Resolve ``build_id`` to a seeded job, honouring the owner filter.

        Mirrors the real repository's resolve-then-get: look up the job id
        registered for ``build_id``, then reuse :meth:`get` so the owner filter
        is applied identically.
        """
        job_id = self.builds.get(build_id)
        if job_id is None:
            return None
        return await self.get(job_id, owner_id=owner_id)

    async def list(
        self, *, limit: int, offset: int, owner_id: UUID | None = None, q: str | None = None
    ) -> tuple[Sequence[tuple[JobTable, str | None]], int]:
        matching = [
            job for job in self.jobs.values() if owner_id is None or job.user_id == str(owner_id)
        ]
        if q:
            needle = q.lower()
            matching = [
                job
                for job in matching
                if needle in job.experiment_name.lower()
                or needle in job.model.lower()
                or needle in str(job.status.value).lower()
            ]
        ordered = sorted(matching, key=lambda job: job.created_at, reverse=True)
        # The real query supplies finished_at (latest gb_tasks.updated_at); these
        # seeded jobs carry no build tasks, so it is None — as it is for any job
        # without one. The scope/paging tests here do not depend on the value.
        page = ordered[offset : offset + limit]
        return [(job, None) for job in page], len(ordered)

    async def create(
        self,
        *,
        user_id: str,
        config_id: UUID,
        dataset_id: UUID,
        model: str,
        model_source: str,
        experiment_name: str,
        tuning_type: str | None,
        seed: int,
        autotune: bool,
        config_snapshot: dict[str, Any],
        reward_function_code: str | None,
        reward_function_name: str | None,
    ) -> JobTable:
        """Not exercised by any test here; present only to satisfy the Protocol.

        ``JobService.create`` is tested against the real
        ``SqlAlchemyJobRepository``, not this fake — see the ``create`` tests
        below.
        """
        job = JobTable(
            id=uuid4(),
            status=RunStatus.PENDING,
            user_id=user_id,
            config_id=config_id,
            dataset_id=dataset_id,
            model=model,
            model_source=model_source,
            experiment_name=experiment_name,
            tuning_type=tuning_type,
            seed=seed,
            autotune=autotune,
            config_snapshot=config_snapshot,
            reward_function_code=reward_function_code,
            reward_function_name=reward_function_name,
        )
        self.jobs[job.id] = job
        return job

    async def set_status(self, job_id: UUID, status: RunStatus) -> None:
        """Not exercised by any test here; present only to satisfy the Protocol."""
        job = self.jobs.get(job_id)
        if job is not None:
            job.status = status

    async def delete(self, job_id: UUID, *, owner_id: UUID | None = None) -> bool:
        """Remove a job from the in-memory store, honouring the owner filter."""
        job = self.jobs.get(job_id)
        if job is None:
            return False
        if owner_id is not None and job.user_id != str(owner_id):
            return False
        del self.jobs[job_id]
        return True

    async def get_task(self, job_id: UUID, task_type: GbTaskType) -> GbTaskTable | None:
        """Not exercised by any test here; present only to satisfy the Protocol."""
        return None

    async def upsert_task(
        self,
        job_id: UUID,
        task_type: GbTaskType,
        *,
        status: RunStatus,
        build_id: UUID | None = None,
        pr_url: str | None = None,
        build_status: dict[str, Any] | None = None,
        artifact_id: UUID | None = None,
        artifact_uri: str | None = None,
        started_at: str | None = None,
        updated_at: str | None = None,
    ) -> GbTaskTable:
        """Record the task's status in-memory, so ``JobService.cancel`` can call it."""
        task = GbTaskTable(job_id=job_id, type=task_type, status=status, build_id=build_id)
        self.tasks[(job_id, task_type)] = task
        return task

    async def list_reconcilable(self) -> Sequence[ReconcilableJob]:
        """Not exercised by any test here; present only to satisfy the Protocol."""
        return []

    async def is_visible(self, job_id: UUID, *, owner_id: UUID | None = None) -> bool:
        """Not exercised by any test here; present only to satisfy the Protocol."""
        job = self.jobs.get(job_id)
        if job is None:
            return False
        return owner_id is None or job.user_id == str(owner_id)

    async def logs_page(
        self, job_id: UUID, *, trial_id: str | None, before_id: int, limit: int
    ) -> tuple[Sequence[LogEntryTable], bool]:
        """Not exercised by any test here; present only to satisfy the Protocol."""
        return [], False

    async def append_log(
        self,
        job_id: UUID,
        *,
        trial_id: str | None,
        level: str | None,
        filename: str | None,
        message: str | None,
        iteration: int | None,
        epoch: float | None,
    ) -> None:
        """Not exercised by any test here; present only to satisfy the Protocol."""
        return None


def _read_service(
    repository: FakeJobRepository, principal: Principal, session: AsyncSession
) -> JobService:
    """Build a ``JobService`` for the read-path tests below.

    ``get``/``list`` never touch ``configuration_repository``, ``dataset_repository``
    or ``runner``, so real ``SqlAlchemy*Repository`` instances (backed by the
    test session) and the real, side-effect-free ``NoOpJobRunner`` stand in —
    there is nothing here worth faking.
    """
    return JobService(
        repository=repository,
        configuration_repository=SqlAlchemyConfigurationRepository(session),
        dataset_repository=SqlAlchemyDatasetRepository(session),
        principal=principal,
        runner=NoOpJobRunner(),
    )


@pytest.fixture
def repository() -> FakeJobRepository:
    return FakeJobRepository()


@pytest.fixture
def service(repository: FakeJobRepository, session: AsyncSession) -> JobService:
    return _read_service(repository, DEFAULT_SEED_OWNER, session)


@pytest.fixture
def fake_runner() -> _RecordingRunner:
    """A JobRunner fake for the cancel/delete tests below, recording ``cancel`` calls."""
    return _RecordingRunner()


@pytest.fixture
def job_service(
    repository: FakeJobRepository, fake_runner: _RecordingRunner, session: AsyncSession
) -> JobService:
    """A JobService over the fake repository, backed by ``fake_runner``.

    For the cancel/delete tests, which must observe whether the runner was
    asked to stop live work.
    """
    return JobService(
        repository=repository,
        configuration_repository=SqlAlchemyConfigurationRepository(session),
        dataset_repository=SqlAlchemyDatasetRepository(session),
        principal=DEFAULT_SEED_OWNER,
        runner=fake_runner,
    )


@pytest.fixture
def seed_job(repository: FakeJobRepository) -> Callable[..., Coroutine[Any, Any, JobTable]]:
    """Seed a job owned by ``DEFAULT_SEED_OWNER`` at a given status.

    Async to match the plan's ``await seed_job(status=...)`` call shape, even
    though ``FakeJobRepository.seed`` itself is synchronous.
    """

    async def _seed(*, status: RunStatus = RunStatus.PENDING) -> JobTable:
        return repository.seed(status=status)

    return _seed


def test_doubles_satisfy_their_protocols() -> None:
    """A type-level assertion: mypy fails here if the Protocol drifts."""
    repository: JobRepository = FakeJobRepository()

    assert repository is not None


async def test_get_returns_the_seeded_job(
    service: JobService, repository: FakeJobRepository
) -> None:
    seeded = repository.seed()

    job = await service.get(seeded.id)

    assert job.id == seeded.id


async def test_get_raises_for_an_unknown_job(service: JobService) -> None:
    with pytest.raises(JobNotFoundError):
        await service.get(uuid4())


async def test_list_reports_the_total_independently_of_the_page(
    service: JobService, repository: FakeJobRepository
) -> None:
    for _ in range(3):
        repository.seed()

    page = await service.list(limit=2, offset=0)

    assert page.total == 3
    assert len(page.items) == 2


# Scope: every caller — admin included — sees only their own rows by default;
# an admin widens to all rows with scope=all; a non-admin asking for all is 403.


async def test_a_provisioned_user_sees_only_their_own_jobs(
    repository: FakeJobRepository, session: AsyncSession
) -> None:
    owner_id = uuid4()
    mine = repository.seed(owner_id=str(owner_id))
    repository.seed()  # someone else's

    principal = Principal(
        email="u@example.com", provider="session", user_id=owner_id, is_admin=False
    )
    service = _read_service(repository, principal, session)

    page = await service.list(limit=20, offset=0)

    assert page.total == 1
    assert page.items[0].id == mine.id


async def test_a_provisioned_user_cannot_get_another_user_s_job(
    repository: FakeJobRepository, session: AsyncSession
) -> None:
    other = repository.seed()
    principal = Principal(
        email="u@example.com", provider="session", user_id=uuid4(), is_admin=False
    )
    service = _read_service(repository, principal, session)

    with pytest.raises(JobNotFoundError):
        await service.get(other.id)


async def test_an_admin_sees_only_their_own_jobs_by_default(
    repository: FakeJobRepository, session: AsyncSession
) -> None:
    repository.seed()  # owned by the default seed owner, not the admin
    repository.seed()

    page = await _read_service(repository, ADMIN, session).list(limit=20, offset=0)

    assert page.total == 0


async def test_an_admin_sees_every_job_with_scope_all(
    repository: FakeJobRepository, session: AsyncSession
) -> None:
    repository.seed()
    repository.seed()

    page = await _read_service(repository, ADMIN, session).list(
        limit=20, offset=0, scope=DataScope.ALL
    )

    assert page.total == 2


async def test_an_admin_can_get_another_user_s_job_with_scope_all(
    repository: FakeJobRepository, session: AsyncSession
) -> None:
    other = repository.seed()

    job = await _read_service(repository, ADMIN, session).get(other.id, scope=DataScope.ALL)

    assert job.id == other.id


async def test_an_admin_cannot_get_another_user_s_job_by_default(
    repository: FakeJobRepository, session: AsyncSession
) -> None:
    other = repository.seed()

    with pytest.raises(JobNotFoundError):
        await _read_service(repository, ADMIN, session).get(other.id)


async def test_a_non_admin_requesting_scope_all_is_forbidden_on_list(
    repository: FakeJobRepository, session: AsyncSession
) -> None:
    principal = Principal(
        email="u@example.com", provider="session", user_id=uuid4(), is_admin=False
    )
    service = _read_service(repository, principal, session)

    with pytest.raises(ScopeNotPermittedError):
        await service.list(limit=20, offset=0, scope=DataScope.ALL)


async def test_a_non_admin_requesting_scope_all_is_forbidden_on_get(
    repository: FakeJobRepository, session: AsyncSession
) -> None:
    existing = repository.seed()
    principal = Principal(
        email="u@example.com", provider="session", user_id=uuid4(), is_admin=False
    )
    service = _read_service(repository, principal, session)

    with pytest.raises(ScopeNotPermittedError):
        await service.get(existing.id, scope=DataScope.ALL)


async def test_get_by_build_id_returns_the_seeded_job(
    service: JobService, repository: FakeJobRepository
) -> None:
    build_id = uuid4()
    seeded = repository.seed(build_id=build_id)

    job = await service.get_by_build_id(build_id)

    assert job.id == seeded.id


async def test_get_by_build_id_raises_for_an_unknown_build(service: JobService) -> None:
    with pytest.raises(BuildNotFoundError):
        await service.get_by_build_id(uuid4())


async def test_get_by_build_id_of_another_owners_build_is_not_found(
    repository: FakeJobRepository, session: AsyncSession
) -> None:
    build_id = uuid4()
    repository.seed(owner_id=str(uuid4()), build_id=build_id)
    caller = Principal(email="me@example.com", provider="session", user_id=uuid4(), is_admin=False)
    service = _read_service(repository, caller, session)

    with pytest.raises(BuildNotFoundError):
        await service.get_by_build_id(build_id)


async def test_get_by_build_id_admin_scope_all_reaches_any_owner(
    repository: FakeJobRepository, session: AsyncSession
) -> None:
    build_id = uuid4()
    seeded = repository.seed(owner_id=str(uuid4()), build_id=build_id)

    job = await _read_service(repository, ADMIN, session).get_by_build_id(
        build_id, scope=DataScope.ALL
    )

    assert job.id == seeded.id


async def test_get_by_build_id_non_admin_scope_all_is_refused(
    service: JobService,
) -> None:
    with pytest.raises(ScopeNotPermittedError):
        await service.get_by_build_id(uuid4(), scope=DataScope.ALL)


async def test_an_authenticated_but_unprovisioned_caller_sees_an_empty_page(
    repository: FakeJobRepository, session: AsyncSession
) -> None:
    repository.seed()
    principal = Principal(
        email="ghost@example.com", provider="session", user_id=None, is_admin=False
    )
    service = _read_service(repository, principal, session)

    page = await service.list(limit=20, offset=0)

    assert page.total == 0
    assert page.items == []


async def test_an_authenticated_but_unprovisioned_caller_gets_404_on_any_job(
    repository: FakeJobRepository, session: AsyncSession
) -> None:
    existing = repository.seed()
    principal = Principal(
        email="ghost@example.com", provider="session", user_id=None, is_admin=False
    )
    service = _read_service(repository, principal, session)

    with pytest.raises(JobNotFoundError):
        await service.get(existing.id)


# The job state machine. Only these transitions are legal; terminal states accept
# none, which is what stops a completed job from being quietly restarted.


def test_pending_may_start_running() -> None:
    check_transition(RunStatus.PENDING, RunStatus.RUNNING)


def test_running_may_pause() -> None:
    check_transition(RunStatus.RUNNING, RunStatus.PAUSED)


def test_paused_may_resume() -> None:
    check_transition(RunStatus.PAUSED, RunStatus.RUNNING)


def test_running_may_complete() -> None:
    check_transition(RunStatus.RUNNING, RunStatus.COMPLETED)


def test_pending_may_complete_when_a_finished_build_is_observed_late() -> None:
    # The reconcile loop can observe a build that already reached "success"
    # while our job never caught the running transit (e.g. a restart mid-build);
    # that build genuinely ran, so pending -> completed must be legal, not raise.
    check_transition(RunStatus.PENDING, RunStatus.COMPLETED)


def test_pending_may_not_pause() -> None:
    with pytest.raises(InvalidStateTransitionError):
        check_transition(RunStatus.PENDING, RunStatus.PAUSED)


@pytest.mark.parametrize("terminal", sorted(TERMINAL_RUN_STATUSES))
def test_terminal_states_accept_no_transitions(terminal: RunStatus) -> None:
    assert ALLOWED_JOB_TRANSITIONS[terminal] == frozenset()


@pytest.mark.parametrize("terminal", sorted(TERMINAL_RUN_STATUSES))
def test_a_terminal_job_cannot_be_restarted(terminal: RunStatus) -> None:
    with pytest.raises(InvalidStateTransitionError):
        check_transition(terminal, RunStatus.RUNNING)


def test_every_status_has_an_entry_so_lookups_cannot_keyerror() -> None:
    assert set(ALLOWED_JOB_TRANSITIONS) == set(RunStatus)


def test_no_status_may_transition_to_itself() -> None:
    for status, allowed in ALLOWED_JOB_TRANSITIONS.items():
        assert status not in allowed


# JobService.create, against the real SqlAlchemy*Repository implementations —
# the guards below (ownership, dataset readiness, the online-RL reward
# requirement) depend on genuine scoped lookups that FakeJobRepository does not
# model.


class _RecordingRunner:
    """A JobRunner fake that records the ids it was asked to submit or cancel."""

    def __init__(self) -> None:
        self.submitted: list[object] = []
        self.cancelled: list[object] = []

    async def submit(self, job_id: object) -> None:
        self.submitted.append(job_id)

    async def cancel(self, job_id: object) -> None:
        self.cancelled.append(job_id)


def _service(session: AsyncSession, principal: Principal, runner: _RecordingRunner) -> JobService:
    return JobService(
        repository=SqlAlchemyJobRepository(session),
        configuration_repository=SqlAlchemyConfigurationRepository(session),
        dataset_repository=SqlAlchemyDatasetRepository(session),
        principal=principal,
        runner=runner,
    )


def _owner(user: UserTable) -> Principal:
    return Principal(email=user.email, provider="session", user_id=user.id, is_admin=False)


async def test_create_returns_a_pending_job_and_submits_it(
    session: AsyncSession,
    user: UserTable,
    configuration: ConfigurationTable,
    ready_dataset: DatasetTable,
) -> None:
    runner = _RecordingRunner()
    service = _service(session, _owner(user), runner)

    job = await service.create(
        JobCreate(
            config_id=configuration.id,
            dataset_id=ready_dataset.id,
            model="ibm/granite",
            experiment_name="exp one",
        )
    )

    assert job.status.value == "pending"
    assert job.tuning_type == configuration.tuner_type
    assert job.config_snapshot is not None
    assert job.config_snapshot["name"] == configuration.name
    assert runner.submitted == [job.id]


async def test_create_by_unprovisioned_caller_raises(
    session: AsyncSession, configuration: ConfigurationTable, ready_dataset: DatasetTable
) -> None:
    principal = Principal(
        email="ghost@example.com", provider="session", user_id=None, is_admin=False
    )
    service = _service(session, principal, _RecordingRunner())

    with pytest.raises(CallerNotProvisionedError):
        await service.create(
            JobCreate(
                config_id=configuration.id,
                dataset_id=ready_dataset.id,
                model="m",
                experiment_name="e",
            )
        )


async def test_create_with_another_users_config_raises_not_found(
    session: AsyncSession, user: UserTable, ready_dataset: DatasetTable
) -> None:
    service = _service(session, _owner(user), _RecordingRunner())

    with pytest.raises(ConfigurationNotFoundError):
        await service.create(
            JobCreate(
                config_id=uuid4(),
                dataset_id=ready_dataset.id,
                model="m",
                experiment_name="e",
            )
        )


async def test_create_with_unready_dataset_raises(
    session: AsyncSession,
    user: UserTable,
    configuration: ConfigurationTable,
    dataset: DatasetTable,
) -> None:
    service = _service(session, _owner(user), _RecordingRunner())

    with pytest.raises(DatasetNotReadyForJobError):
        await service.create(
            JobCreate(
                config_id=configuration.id,
                dataset_id=dataset.id,
                model="m",
                experiment_name="e",
            )
        )


async def test_create_online_rl_without_reward_raises(
    session: AsyncSession, user: UserTable, ready_dataset: DatasetTable
) -> None:
    rl_config = ConfigurationTable(
        id=uuid4(),
        user_id=str(user.id),
        name="ppo-sweep",
        tuner_type="optuna",
        rl_tuner_type="ppo",
        config_data={"x": 1},
    )
    session.add(rl_config)
    await session.commit()
    service = _service(session, _owner(user), _RecordingRunner())

    with pytest.raises(MissingRewardFunctionError):
        await service.create(
            JobCreate(
                config_id=rl_config.id,
                dataset_id=ready_dataset.id,
                model="m",
                experiment_name="e",
            )
        )


# JobService.cancel — the backend is told to stop, the job is forced to
# `terminated`, idempotent for an already-terminated job, and a 409 for one
# that has already finished with nothing left to stop.


async def test_cancel_running_job_calls_runner_and_drives_to_terminated(
    job_service: JobService,
    fake_runner: _RecordingRunner,
    seed_job: Callable[..., Coroutine[Any, Any, JobTable]],
) -> None:
    job = await seed_job(status=RunStatus.RUNNING)

    result = await job_service.cancel(job.id)

    assert fake_runner.cancelled == [job.id]
    assert result.status == RunStatus.TERMINATED


async def test_cancel_is_idempotent_for_already_terminated(
    job_service: JobService,
    fake_runner: _RecordingRunner,
    seed_job: Callable[..., Coroutine[Any, Any, JobTable]],
) -> None:
    job = await seed_job(status=RunStatus.TERMINATED)

    result = await job_service.cancel(job.id)

    assert result.status == RunStatus.TERMINATED
    assert fake_runner.cancelled == []  # no backend call, no re-write


@pytest.mark.parametrize("terminal", [RunStatus.COMPLETED, RunStatus.ERROR])
async def test_cancel_rejects_a_job_with_nothing_left_to_stop(
    job_service: JobService,
    seed_job: Callable[..., Coroutine[Any, Any, JobTable]],
    terminal: RunStatus,
) -> None:
    job = await seed_job(status=terminal)

    with pytest.raises(JobNotCancellableError):
        await job_service.cancel(job.id)


async def test_cancel_unknown_job_is_404(job_service: JobService) -> None:
    with pytest.raises(JobNotFoundError):
        await job_service.cancel(uuid4())


# JobService.delete — ownership scope. Delete now auto-cancels any live backend
# work for a non-terminal job before removing it, rather than refusing a running
# or paused job with a 409 (the behavior this replaces).


async def test_delete_removes_a_pending_job(
    service: JobService, repository: FakeJobRepository
) -> None:
    seeded = repository.seed(status=RunStatus.PENDING)

    await service.delete(seeded.id)

    assert seeded.id not in repository.jobs


@pytest.mark.parametrize("terminal", [RunStatus.COMPLETED, RunStatus.ERROR, RunStatus.TERMINATED])
async def test_delete_removes_a_terminal_job(
    service: JobService, repository: FakeJobRepository, terminal: RunStatus
) -> None:
    seeded = repository.seed(status=terminal)

    await service.delete(seeded.id)

    assert seeded.id not in repository.jobs


@pytest.mark.parametrize("blocked", [RunStatus.RUNNING, RunStatus.PAUSED])
async def test_delete_running_job_cancels_first_then_deletes(
    job_service: JobService,
    fake_runner: _RecordingRunner,
    seed_job: Callable[..., Coroutine[Any, Any, JobTable]],
    blocked: RunStatus,
) -> None:
    job = await seed_job(status=blocked)

    await job_service.delete(job.id)

    assert fake_runner.cancelled == [job.id]
    with pytest.raises(JobNotFoundError):
        await job_service.get(job.id)


@pytest.mark.parametrize("terminal", [RunStatus.COMPLETED, RunStatus.ERROR, RunStatus.TERMINATED])
async def test_delete_terminal_job_does_not_cancel(
    job_service: JobService,
    fake_runner: _RecordingRunner,
    seed_job: Callable[..., Coroutine[Any, Any, JobTable]],
    terminal: RunStatus,
) -> None:
    job = await seed_job(status=terminal)

    await job_service.delete(job.id)

    assert fake_runner.cancelled == []


async def test_delete_aborts_and_row_survives_when_cancel_fails(
    repository: FakeJobRepository,
    session: AsyncSession,
) -> None:
    job = repository.seed(status=RunStatus.RUNNING)

    class _FailingRunner:
        async def submit(self, job_id: object) -> None: ...

        async def cancel(self, job_id: object) -> None:
            raise BuildCancelUpstreamError("cluster refused")

    service = JobService(
        repository=repository,
        configuration_repository=SqlAlchemyConfigurationRepository(session),
        dataset_repository=SqlAlchemyDatasetRepository(session),
        principal=DEFAULT_SEED_OWNER,
        runner=_FailingRunner(),
    )

    with pytest.raises(BuildCancelUpstreamError):
        await service.delete(job.id)

    assert job.id in repository.jobs  # cancel raised before delete; the row is untouched


async def test_delete_of_an_unknown_job_raises_not_found(service: JobService) -> None:
    with pytest.raises(JobNotFoundError):
        await service.delete(uuid4())


async def test_delete_of_another_users_job_raises_not_found(
    repository: FakeJobRepository, session: AsyncSession
) -> None:
    other = repository.seed()  # default owner, not the caller
    principal = Principal(
        email="u@example.com", provider="session", user_id=uuid4(), is_admin=False
    )
    service = _read_service(repository, principal, session)

    with pytest.raises(JobNotFoundError):
        await service.delete(other.id)

    assert other.id in repository.jobs  # untouched


async def test_an_unprovisioned_caller_cannot_delete_any_job(
    repository: FakeJobRepository, session: AsyncSession
) -> None:
    existing = repository.seed()
    principal = Principal(
        email="ghost@example.com", provider="session", user_id=None, is_admin=False
    )
    service = _read_service(repository, principal, session)

    with pytest.raises(JobNotFoundError):
        await service.delete(existing.id)


async def test_a_provisioned_user_can_delete_their_own_job(
    repository: FakeJobRepository, session: AsyncSession
) -> None:
    owner_id = uuid4()
    mine = repository.seed(owner_id=str(owner_id))
    principal = Principal(
        email="u@example.com", provider="session", user_id=owner_id, is_admin=False
    )
    service = _read_service(repository, principal, session)

    await service.delete(mine.id)

    assert mine.id not in repository.jobs


async def test_delete_of_own_running_job_as_non_admin_cancels_then_deletes(
    repository: FakeJobRepository, session: AsyncSession
) -> None:
    owner_id = uuid4()
    mine = repository.seed(status=RunStatus.RUNNING, owner_id=str(owner_id))
    principal = Principal(
        email="u@example.com", provider="session", user_id=owner_id, is_admin=False
    )
    runner = _RecordingRunner()
    service = JobService(
        repository=repository,
        configuration_repository=SqlAlchemyConfigurationRepository(session),
        dataset_repository=SqlAlchemyDatasetRepository(session),
        principal=principal,
        runner=runner,
    )

    await service.delete(mine.id)

    assert runner.cancelled == [mine.id]
    assert mine.id not in repository.jobs


async def test_delete_of_another_users_running_job_raises_not_found_not_conflict(
    repository: FakeJobRepository, session: AsyncSession
) -> None:
    other = repository.seed(status=RunStatus.RUNNING)  # default owner, not the caller
    principal = Principal(
        email="u@example.com", provider="session", user_id=uuid4(), is_admin=False
    )
    service = _read_service(repository, principal, session)

    with pytest.raises(JobNotFoundError):
        await service.delete(other.id)

    assert other.id in repository.jobs


async def test_an_admin_can_delete_another_user_s_job_with_scope_all(
    repository: FakeJobRepository, session: AsyncSession
) -> None:
    seeded = repository.seed()  # default owner, not the admin
    service = _read_service(repository, ADMIN, session)

    await service.delete(seeded.id, scope=DataScope.ALL)

    with pytest.raises(JobNotFoundError):
        await service.get(seeded.id, scope=DataScope.ALL)


async def test_an_admin_cannot_delete_another_user_s_job_by_default(
    repository: FakeJobRepository, session: AsyncSession
) -> None:
    seeded = repository.seed()  # default owner, not the admin
    service = _read_service(repository, ADMIN, session)

    with pytest.raises(JobNotFoundError):
        await service.delete(seeded.id)

    assert seeded.id in repository.jobs


async def test_a_non_admin_requesting_scope_all_is_forbidden_on_delete(
    repository: FakeJobRepository, session: AsyncSession
) -> None:
    existing = repository.seed()
    principal = Principal(
        email="u@example.com", provider="session", user_id=uuid4(), is_admin=False
    )
    service = _read_service(repository, principal, session)

    with pytest.raises(ScopeNotPermittedError):
        await service.delete(existing.id, scope=DataScope.ALL)
