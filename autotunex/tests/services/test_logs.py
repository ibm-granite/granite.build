"""Unit tests for LogService, isolated from the database and gb."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from autotunex.core.exceptions import (
    GbLogsUnavailableError,
    JobNotFoundError,
    ScopeNotPermittedError,
)
from autotunex.db.tables import GbTaskTable, LogEntryTable
from autotunex.models.auth import Principal
from autotunex.models.common import DataScope
from autotunex.models.status import GbTaskType, RunStatus
from autotunex.services.logs import LogService

OWNER = uuid4()
ADMIN = Principal(email="admin@example.com", provider="session", user_id=uuid4(), is_admin=True)
USER = Principal(email="u@example.com", provider="session", user_id=OWNER, is_admin=False)
UNPROVISIONED = Principal(email="x@example.com", provider="oidc", user_id=None, is_admin=False)


class FakeJobRepository:
    """In-memory stand-in exposing only what LogService touches."""

    def __init__(self) -> None:
        self.visible = True
        self.rows: list[LogEntryTable] = []
        self.task: GbTaskTable | None = None
        self.seen_args: dict[str, object] = {}

    async def is_visible(self, job_id: UUID, *, owner_id: UUID | None = None) -> bool:
        self.seen_args["owner_id"] = owner_id
        return self.visible

    async def logs_page(
        self, job_id: UUID, *, trial_id: str | None, before_id: int, limit: int
    ) -> tuple[Sequence[LogEntryTable], bool]:
        self.seen_args["trial_id"] = trial_id
        return self.rows[:limit], len(self.rows) > limit

    async def get_task(self, job_id: UUID, task_type: GbTaskType) -> GbTaskTable | None:
        return self.task

    # The remaining JobRepository methods are unused here; LogService never calls
    # them, and this fake is not asserted against the full Protocol.


class FakeGbLogReader:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.fetch_all: bool | None = None

    async def fetch(self, build_id: str, *, fetch_all: bool) -> list[str]:
        self.fetch_all = fetch_all
        return self.lines


def _line(i: int) -> LogEntryTable:
    return LogEntryTable(
        id=i,
        job_id=uuid4(),
        trial_id=None,
        level="INFO",
        message=f"line {i}",
        timestamp=datetime(2026, 8, 10, tzinfo=UTC),
    )


def _service(repo: FakeJobRepository, principal: Principal = USER) -> LogService:
    # FakeJobRepository intentionally implements only the methods LogService
    # calls, not the full JobRepository protocol (see its docstring) — mypy
    # checks the full interface at this call site regardless.
    return LogService(
        repository=repo,  # type: ignore[arg-type]
        principal=principal,
        gb_log_reader=FakeGbLogReader([]),
    )


async def test_get_job_logs_returns_a_page_with_cursor() -> None:
    repo = FakeJobRepository()
    repo.rows = [_line(3), _line(2)]

    page = await _service(repo).get_job_logs(uuid4(), before_id=0, limit=1)

    assert [e.id for e in page.logs] == [3]
    assert page.has_more is True
    assert page.next_before_id == 3


async def test_get_job_logs_404_when_job_not_visible() -> None:
    repo = FakeJobRepository()
    repo.visible = False

    with pytest.raises(JobNotFoundError):
        await _service(repo).get_job_logs(uuid4(), before_id=0, limit=50)


async def test_get_logs_404_for_own_scope_caller_with_no_identity() -> None:
    repo = FakeJobRepository()

    with pytest.raises(JobNotFoundError):
        await _service(repo, UNPROVISIONED).get_job_logs(uuid4(), before_id=0, limit=50)


async def test_get_trial_logs_forwards_the_trial_id() -> None:
    repo = FakeJobRepository()

    await _service(repo).get_trial_logs(uuid4(), "abc123", before_id=0, limit=50)

    assert repo.seen_args["trial_id"] == "abc123"


async def test_admin_scope_all_passes_owner_none() -> None:
    repo = FakeJobRepository()

    await _service(repo, ADMIN).get_job_logs(uuid4(), before_id=0, limit=50, scope=DataScope.ALL)

    assert repo.seen_args["owner_id"] is None


async def test_admin_default_own_scope_passes_admins_own_id() -> None:
    repo = FakeJobRepository()

    await _service(repo, ADMIN).get_job_logs(uuid4(), before_id=0, limit=50)

    assert repo.seen_args["owner_id"] == ADMIN.user_id


async def test_provisioned_user_scope_passes_their_id() -> None:
    repo = FakeJobRepository()

    await _service(repo, USER).get_job_logs(uuid4(), before_id=0, limit=50)

    assert repo.seen_args["owner_id"] == OWNER


async def test_non_admin_scope_all_is_forbidden_before_any_repo_call() -> None:
    repo = FakeJobRepository()

    with pytest.raises(ScopeNotPermittedError):
        await _service(repo, USER).get_job_logs(uuid4(), before_id=0, limit=50, scope=DataScope.ALL)

    assert "owner_id" not in repo.seen_args  # 403 fires before is_visible


async def test_get_gb_logs_delegates_to_the_reader() -> None:
    repo = FakeJobRepository()
    repo.task = GbTaskTable(
        job_id=uuid4(), type=GbTaskType.TUNING, status=RunStatus.RUNNING, build_id=uuid4()
    )
    reader = FakeGbLogReader(["a", "b"])
    service = LogService(repository=repo, principal=USER, gb_log_reader=reader)  # type: ignore[arg-type]

    lines = await service.get_gb_logs(uuid4(), fetch_all=True)

    assert lines == ["a", "b"]
    assert reader.fetch_all is True


async def test_get_gb_logs_503_when_no_build_handle() -> None:
    repo = FakeJobRepository()
    repo.task = None

    with pytest.raises(GbLogsUnavailableError):
        await _service(repo).get_gb_logs(uuid4(), fetch_all=False)
