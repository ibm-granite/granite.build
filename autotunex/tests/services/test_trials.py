"""Unit tests for TrialService, isolated from the database.

Mirrors ``test_metrics.py``: hand-written fakes for the two repositories the
service touches, so these tests are about scope resolution and the visibility
gate rather than SQL. The paged read itself is covered against real SQLite in
``tests/db/repositories/test_trial_repository.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from autotunex.core.exceptions import JobNotFoundError, ScopeNotPermittedError
from autotunex.db.tables import TrialTable
from autotunex.models.auth import Principal
from autotunex.models.common import DataScope
from autotunex.models.status import RunStatus
from autotunex.services.trials import TrialService

OWNER = uuid4()
ADMIN = Principal(email="admin@example.com", provider="session", user_id=uuid4(), is_admin=True)
USER = Principal(email="u@example.com", provider="session", user_id=OWNER, is_admin=False)
UNPROVISIONED = Principal(email="x@example.com", provider="oidc", user_id=None, is_admin=False)


class FakeJobRepository:
    """In-memory stand-in exposing only the visibility probe TrialService touches."""

    def __init__(self) -> None:
        self.visible = True
        self.seen_args: dict[str, object] = {}
        self.probes = 0

    async def is_visible(self, job_id: UUID, *, owner_id: UUID | None = None) -> bool:
        self.probes += 1
        self.seen_args["owner_id"] = owner_id
        return self.visible


class FakeTrialRepository:
    """Records what the service asked for and returns a canned page."""

    def __init__(self, rows: list[TrialTable] | None = None, total: int | None = None) -> None:
        self.rows: list[TrialTable] = rows or []
        self.total = total if total is not None else len(self.rows)
        self.seen_args: dict[str, object] = {}

    async def page(
        self, job_id: UUID, *, limit: int, offset: int
    ) -> tuple[Sequence[TrialTable], int]:
        self.seen_args.update(job_id=job_id, limit=limit, offset=offset)
        return self.rows, self.total


def _trial(trial_id: str) -> TrialTable:
    return TrialTable(
        id=trial_id,
        job_id=uuid4(),
        status=RunStatus.COMPLETED,
        config={"learning_rate": 3e-5},
        created_at=datetime(2026, 8, 26, tzinfo=UTC),
        updated_at=datetime(2026, 8, 26, tzinfo=UTC),
    )


def _service(
    jobs: FakeJobRepository,
    trials: FakeTrialRepository,
    principal: Principal = USER,
) -> TrialService:
    # Both fakes implement only the methods TrialService calls, not the full
    # protocols (mirrors test_metrics.py); mypy checks the full interface here.
    return TrialService(
        trial_repository=trials,  # type: ignore[arg-type]
        job_repository=jobs,  # type: ignore[arg-type]
        principal=principal,
    )


async def test_it_returns_a_page_carrying_the_unpaginated_total() -> None:
    trials = FakeTrialRepository([_trial("t1")], total=137)

    page = await _service(FakeJobRepository(), trials).list_trials(uuid4(), limit=1, offset=0)

    assert [trial.id for trial in page.items] == ["t1"]
    assert page.total == 137
    assert (page.limit, page.offset) == (1, 0)


async def test_it_forwards_limit_and_offset_to_the_repository() -> None:
    trials = FakeTrialRepository()

    await _service(FakeJobRepository(), trials).list_trials(uuid4(), limit=25, offset=50)

    assert trials.seen_args["limit"] == 25
    assert trials.seen_args["offset"] == 50


async def test_a_visible_job_with_no_trials_is_an_empty_page_not_a_404() -> None:
    """A job with no trials yet is not a missing job; the pre-rewrite code conflated them."""
    page = await _service(FakeJobRepository(), FakeTrialRepository()).list_trials(uuid4())

    assert page.items == []
    assert page.total == 0


async def test_404_when_the_job_is_not_visible() -> None:
    jobs = FakeJobRepository()
    jobs.visible = False

    with pytest.raises(JobNotFoundError):
        await _service(jobs, FakeTrialRepository()).list_trials(uuid4())


async def test_404_for_an_own_scope_caller_with_no_identity() -> None:
    """An unresolvable caller under ``scope=own`` owns nothing, so it sees nothing."""
    jobs = FakeJobRepository()

    with pytest.raises(JobNotFoundError):
        await _service(jobs, FakeTrialRepository(), UNPROVISIONED).list_trials(uuid4())

    assert jobs.probes == 0  # short-circuited before touching the database


async def test_403_when_a_non_admin_requests_scope_all() -> None:
    with pytest.raises(ScopeNotPermittedError):
        await _service(FakeJobRepository(), FakeTrialRepository()).list_trials(
            uuid4(), scope=DataScope.ALL
        )


async def test_the_scope_403_precedes_the_existence_probe() -> None:
    """A non-admin must not learn whether a job exists by asking for ``scope=all``."""
    jobs = FakeJobRepository()
    jobs.visible = False

    with pytest.raises(ScopeNotPermittedError):
        await _service(jobs, FakeTrialRepository()).list_trials(uuid4(), scope=DataScope.ALL)

    assert jobs.probes == 0


async def test_admin_scope_all_passes_owner_none() -> None:
    jobs = FakeJobRepository()

    await _service(jobs, FakeTrialRepository(), ADMIN).list_trials(uuid4(), scope=DataScope.ALL)

    assert jobs.seen_args["owner_id"] is None


async def test_admin_default_own_scope_passes_the_admins_own_id() -> None:
    """Being an admin does not widen the read; the ``scope`` parameter does."""
    jobs = FakeJobRepository()

    await _service(jobs, FakeTrialRepository(), ADMIN).list_trials(uuid4())

    assert jobs.seen_args["owner_id"] == ADMIN.user_id
