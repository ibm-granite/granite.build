"""POST /jobs/{id}/reconcile over HTTP.

Admin gating rides on the reader dependency (``get_reconcile_reader`` depends on
``require_admin``), so a non-admin is refused before the availability check runs.
The 200/404 cases override ``get_reconcile_reader`` with a fake reader; the 403
and 503 cases leave it real to exercise the gate and the llmb-availability check.
"""

from __future__ import annotations

from collections.abc import Callable
from http import HTTPStatus
from typing import Any
from uuid import UUID, uuid4

from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from autotunex.api.deps import get_reconcile_reader
from autotunex.db.repositories.sqlalchemy import SqlAlchemyJobRepository
from autotunex.db.tables import ConfigurationTable, DatasetTable, JobTable, UserTable
from autotunex.models.auth import Principal
from autotunex.models.status import GbTaskType, RunStatus
from autotunex.services.reconcile.protocols import BuildState
from tests.conftest import API

BUILD_ID = UUID("22222222-2222-2222-2222-222222222222")

_EVENTS: dict[str, Any] = {
    "events": [
        {"build_event": {"timestamp": "2026-08-07T00:04:00Z", "payload": {"msg": "done `ok`"}}}
    ]
}


class _FakeReader:
    def __init__(self, state: BuildState, *, events: dict[str, Any]) -> None:
        self._state = state
        self._events = events

    async def read(self, build_id: UUID) -> BuildState:
        return self._state

    async def read_events(self, build_id: UUID) -> dict[str, Any]:
        return self._events


def _state(status: str) -> BuildState:
    return BuildState(
        build_id=BUILD_ID,
        status=status,
        failure_reason=None,
        created_at="2026-08-07T00:00:00Z",
        updated_at="2026-08-07T00:05:00Z",
        raw={"status": {"build": {"status": status}}},
    )


def _admin() -> Principal:
    return Principal(email="admin@example.com", provider="session", user_id=uuid4(), is_admin=True)


def _regular() -> Principal:
    return Principal(email="reg@example.com", provider="session", user_id=uuid4(), is_admin=False)


async def _seed_job_with_build(session: AsyncSession, *, status: RunStatus) -> UUID:
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
    dataset = DatasetTable(id=uuid4(), user_id=str(user.id), name="d", data_format="jsonl")
    session.add_all([config, dataset])
    await session.commit()
    job = JobTable(
        id=uuid4(),
        user_id=str(user.id),
        status=status,
        config_id=config.id,
        dataset_id=dataset.id,
        model="m",
        model_source="huggingface",
        experiment_name="e",
        tuning_type="lora",
    )
    session.add(job)
    await session.commit()
    await SqlAlchemyJobRepository(session).upsert_task(
        job.id, GbTaskType.TUNING, status=RunStatus.PENDING, build_id=BUILD_ID
    )
    return job.id


async def test_admin_can_reconcile_returns_refreshed_job(
    client: AsyncClient,
    app: FastAPI,
    session: AsyncSession,
    as_principal: Callable[[Principal], None],
) -> None:
    job_id = await _seed_job_with_build(session, status=RunStatus.RUNNING)
    as_principal(_admin())
    app.dependency_overrides[get_reconcile_reader] = lambda: _FakeReader(
        _state("success"), events=_EVENTS
    )

    response = await client.post(f"{API}/jobs/{job_id}/reconcile")

    assert response.status_code == HTTPStatus.OK
    body = response.json()
    assert body["status"] == "completed"
    assert body["tasks"][0]["build_status"]["build_history"]


async def test_non_admin_gets_403(
    client: AsyncClient,
    session: AsyncSession,
    as_principal: Callable[[Principal], None],
) -> None:
    job_id = await _seed_job_with_build(session, status=RunStatus.RUNNING)
    as_principal(_regular())

    response = await client.post(f"{API}/jobs/{job_id}/reconcile")

    assert response.status_code == HTTPStatus.FORBIDDEN


async def test_unavailable_when_not_llmb_returns_503(
    client: AsyncClient,
    session: AsyncSession,
    as_principal: Callable[[Principal], None],
) -> None:
    job_id = await _seed_job_with_build(session, status=RunStatus.RUNNING)
    as_principal(_admin())

    response = await client.post(f"{API}/jobs/{job_id}/reconcile")

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE


async def test_missing_job_returns_404(
    client: AsyncClient,
    app: FastAPI,
    as_principal: Callable[[Principal], None],
) -> None:
    as_principal(_admin())
    app.dependency_overrides[get_reconcile_reader] = lambda: _FakeReader(
        _state("success"), events=_EVENTS
    )

    response = await client.post(f"{API}/jobs/{uuid4()}/reconcile")

    assert response.status_code == HTTPStatus.NOT_FOUND
