"""Unit tests for MetricsService, isolated from the database."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from autotunex.core.exceptions import JobNotFoundError, ScopeNotPermittedError
from autotunex.db.tables import TrainingMetricTable
from autotunex.models.auth import Principal
from autotunex.models.common import DataScope
from autotunex.services.metrics import MetricsService

OWNER = uuid4()
ADMIN = Principal(email="admin@example.com", provider="session", user_id=uuid4(), is_admin=True)
USER = Principal(email="u@example.com", provider="session", user_id=OWNER, is_admin=False)
UNPROVISIONED = Principal(email="x@example.com", provider="oidc", user_id=None, is_admin=False)


class FakeJobRepository:
    """In-memory stand-in exposing only the visibility probe MetricsService touches."""

    def __init__(self) -> None:
        self.visible = True
        self.seen_args: dict[str, object] = {}

    async def is_visible(self, job_id: UUID, *, owner_id: UUID | None = None) -> bool:
        self.seen_args["owner_id"] = owner_id
        return self.visible


class FakeMetricsRepository:
    """Records what the service asked for and returns a canned page."""

    def __init__(self, rows: list[TrainingMetricTable] | None = None) -> None:
        self.rows: list[TrainingMetricTable] = rows or []
        self.has_more = False
        self.seen_args: dict[str, object] = {}

    async def metrics_page(
        self, job_id: UUID, *, trial_id: str | None, after_id: int, limit: int
    ) -> tuple[Sequence[TrainingMetricTable], bool]:
        self.seen_args.update(trial_id=trial_id, after_id=after_id, limit=limit)
        return self.rows, self.has_more


def _point(row_id: int, *, step: int) -> TrainingMetricTable:
    return TrainingMetricTable(
        id=row_id,
        job_id=uuid4(),
        trial_id="t1",
        global_step=step,
        epoch=0.5,
        loss=1.25,
        grad_norm=None,
        learning_rate=None,
        split="train",
        extra=None,
        created_at=datetime(2026, 8, 26, tzinfo=UTC),
    )


def _service(
    jobs: FakeJobRepository,
    metrics: FakeMetricsRepository,
    principal: Principal = USER,
) -> MetricsService:
    # Both fakes implement only the methods MetricsService calls, not the full
    # protocols (mirrors tests/services/test_logs.py); mypy checks the full
    # interface at this call site regardless.
    return MetricsService(
        metrics_repository=metrics,  # type: ignore[arg-type]
        job_repository=jobs,  # type: ignore[arg-type]
        principal=principal,
    )


async def test_get_job_metrics_returns_an_ascending_page_with_cursor() -> None:
    metrics = FakeMetricsRepository([_point(1, step=10), _point(2, step=20)])
    metrics.has_more = True

    page = await _service(FakeJobRepository(), metrics).get_job_metrics(
        uuid4(), after_id=0, limit=2
    )

    assert [p.global_step for p in page.metrics] == [10, 20]
    assert page.has_more is True
    assert page.next_after_id == 2


async def test_next_after_id_is_none_on_the_last_page() -> None:
    metrics = FakeMetricsRepository([_point(7, step=70)])

    page = await _service(FakeJobRepository(), metrics).get_job_metrics(
        uuid4(), after_id=0, limit=50
    )

    assert page.has_more is False
    assert page.next_after_id is None


async def test_get_job_metrics_404_when_the_job_is_not_visible() -> None:
    jobs = FakeJobRepository()
    jobs.visible = False

    with pytest.raises(JobNotFoundError):
        await _service(jobs, FakeMetricsRepository()).get_job_metrics(uuid4(), after_id=0, limit=50)


async def test_get_metrics_404_for_an_own_scope_caller_with_no_identity() -> None:
    jobs = FakeJobRepository()

    with pytest.raises(JobNotFoundError):
        await _service(jobs, FakeMetricsRepository(), UNPROVISIONED).get_job_metrics(
            uuid4(), after_id=0, limit=50
        )


async def test_get_metrics_403_when_a_non_admin_requests_scope_all() -> None:
    jobs = FakeJobRepository()

    with pytest.raises(ScopeNotPermittedError):
        await _service(jobs, FakeMetricsRepository()).get_job_metrics(
            uuid4(), after_id=0, limit=50, scope=DataScope.ALL
        )


async def test_the_scope_403_precedes_the_existence_probe() -> None:
    jobs = FakeJobRepository()
    jobs.visible = False

    with pytest.raises(ScopeNotPermittedError):
        await _service(jobs, FakeMetricsRepository()).get_trial_metrics(
            uuid4(), "t1", after_id=0, limit=50, scope=DataScope.ALL
        )

    assert "owner_id" not in jobs.seen_args


async def test_get_trial_metrics_forwards_the_trial_id() -> None:
    metrics = FakeMetricsRepository()

    await _service(FakeJobRepository(), metrics).get_trial_metrics(
        uuid4(), "abc123", after_id=5, limit=50
    )

    assert metrics.seen_args["trial_id"] == "abc123"
    assert metrics.seen_args["after_id"] == 5


async def test_admin_scope_all_passes_owner_none() -> None:
    jobs = FakeJobRepository()

    await _service(jobs, FakeMetricsRepository(), ADMIN).get_job_metrics(
        uuid4(), after_id=0, limit=50, scope=DataScope.ALL
    )

    assert jobs.seen_args["owner_id"] is None


async def test_admin_default_own_scope_passes_the_admins_own_id() -> None:
    jobs = FakeJobRepository()

    await _service(jobs, FakeMetricsRepository(), ADMIN).get_job_metrics(
        uuid4(), after_id=0, limit=50
    )

    assert jobs.seen_args["owner_id"] == ADMIN.user_id
