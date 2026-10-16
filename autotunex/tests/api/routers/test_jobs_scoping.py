"""End-to-end proof that a caller only ever sees their own jobs over HTTP.

The `job` fixture (``tests/conftest.py``) is owned by the ``user`` fixture's
email, ``tester@example.com``. These tests swap the resolved principal per test via
``as_principal`` and check what ``GET /jobs`` and ``GET /jobs/{id}`` return.
"""

from __future__ import annotations

from collections.abc import Callable
from http import HTTPStatus
from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from autotunex.db.tables import ConfigurationTable, DatasetTable, JobTable, UserTable
from autotunex.models.auth import Principal
from autotunex.models.status import RunStatus
from tests.conftest import API


@pytest.fixture
async def other_users_job(
    session: AsyncSession, configuration: ConfigurationTable, dataset: DatasetTable
) -> JobTable:
    """A second job, owned by somebody other than the ``user`` fixture.

    Without a row that must be excluded, a positive scoping assertion proves
    nothing: the ``job`` fixture creates one job in total, so
    "correctly filtered to mine" and "no filtering at all, and mine happens to be
    the only row" produce identical responses. Reuses ``configuration`` and
    ``dataset`` deliberately — the repository's inner joins only require them to
    exist, and it is ``jobs.user_id`` alone that decides ownership.
    """
    other = UserTable(id=uuid4(), email="not-the-owner@example.com", role="user")
    session.add(other)
    await session.commit()
    other_job = JobTable(
        id=uuid4(),
        user_id=str(other.id),
        status=RunStatus.PENDING,
        seed=7,
        config_id=configuration.id,
        dataset_id=dataset.id,
        model="ibm-granite/granite-3.0-8b-instruct",
        model_source="huggingface",
        experiment_name="not-yours",
        tuning_type="lora",
    )
    session.add(other_job)
    await session.commit()
    return other_job


async def test_the_owner_sees_their_own_job_in_the_list(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    job: JobTable,
    other_users_job: JobTable,
    user: UserTable,
) -> None:
    as_principal(Principal(email=user.email, provider="session", user_id=user.id, is_admin=False))

    response = await client.get(f"{API}/jobs")

    assert response.status_code == HTTPStatus.OK
    assert [item["id"] for item in response.json()["items"]] == [str(job.id)]
    assert response.json()["total"] == 1


async def test_the_owner_can_fetch_their_own_job_by_id(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    job: JobTable,
    other_users_job: JobTable,
    user: UserTable,
) -> None:
    as_principal(Principal(email=user.email, provider="session", user_id=user.id, is_admin=False))

    response = await client.get(f"{API}/jobs/{job.id}")

    assert response.status_code == HTTPStatus.OK
    assert response.json()["id"] == str(job.id)


async def test_the_owner_cannot_fetch_the_other_user_s_job_by_id(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    other_users_job: JobTable,
    user: UserTable,
) -> None:
    """The negative half of the pair above, on the same two-job database."""
    as_principal(Principal(email=user.email, provider="session", user_id=user.id, is_admin=False))

    response = await client.get(f"{API}/jobs/{other_users_job.id}")

    assert response.status_code == HTTPStatus.NOT_FOUND


async def test_a_different_user_sees_an_empty_list(
    client: AsyncClient, as_principal: Callable[[Principal], None], job: JobTable
) -> None:
    as_principal(
        Principal(
            email="someone-else@example.com", provider="session", user_id=uuid4(), is_admin=False
        )
    )

    response = await client.get(f"{API}/jobs")

    assert response.json() == {"items": [], "total": 0, "limit": 20, "offset": 0}


async def test_a_different_user_gets_a_404_for_someone_else_s_job(
    client: AsyncClient, as_principal: Callable[[Principal], None], job: JobTable
) -> None:
    as_principal(
        Principal(
            email="someone-else@example.com", provider="session", user_id=uuid4(), is_admin=False
        )
    )

    response = await client.get(f"{API}/jobs/{job.id}")

    assert response.status_code == HTTPStatus.NOT_FOUND


async def test_an_admin_sees_only_their_own_jobs_by_default(
    client: AsyncClient, as_principal: Callable[[Principal], None], job: JobTable
) -> None:
    as_principal(
        Principal(email="admin@example.com", provider="session", user_id=uuid4(), is_admin=True)
    )

    response = await client.get(f"{API}/jobs")

    assert response.json() == {"items": [], "total": 0, "limit": 20, "offset": 0}


async def test_an_admin_sees_every_job_with_scope_all(
    client: AsyncClient, as_principal: Callable[[Principal], None], job: JobTable
) -> None:
    as_principal(
        Principal(email="admin@example.com", provider="session", user_id=uuid4(), is_admin=True)
    )

    response = await client.get(f"{API}/jobs?scope=all")

    assert response.json()["total"] == 1


async def test_an_admin_can_fetch_another_user_s_job_with_scope_all(
    client: AsyncClient, as_principal: Callable[[Principal], None], job: JobTable
) -> None:
    as_principal(
        Principal(email="admin@example.com", provider="session", user_id=uuid4(), is_admin=True)
    )

    response = await client.get(f"{API}/jobs/{job.id}?scope=all")

    assert response.status_code == HTTPStatus.OK
    assert response.json()["id"] == str(job.id)


async def test_a_non_admin_requesting_scope_all_is_forbidden(
    client: AsyncClient, as_principal: Callable[[Principal], None], job: JobTable
) -> None:
    as_principal(
        Principal(email="u@example.com", provider="session", user_id=uuid4(), is_admin=False)
    )

    response = await client.get(f"{API}/jobs?scope=all")

    assert response.status_code == HTTPStatus.FORBIDDEN
    assert response.headers["content-type"] == "application/problem+json"


async def test_an_authenticated_but_unprovisioned_caller_sees_an_empty_list(
    client: AsyncClient, as_principal: Callable[[Principal], None], job: JobTable
) -> None:
    as_principal(
        Principal(email="ghost@example.com", provider="session", user_id=None, is_admin=False)
    )

    response = await client.get(f"{API}/jobs")

    assert response.json() == {"items": [], "total": 0, "limit": 20, "offset": 0}


async def test_standalone_default_scopes_to_its_own_owner(
    client: AsyncClient, job: JobTable
) -> None:
    """No override at all: the standalone admin now sees only its own rows.

    The ``job`` fixture is owned by ``tester@example.com``, a different owner than
    the standalone system account, so the default (own) scope excludes it. In a
    real standalone deployment every row is owned by that one account, so this
    exclusion never appears there — it is visible only because the fixture seeds
    a row under a foreign owner.
    """
    response = await client.get(f"{API}/jobs")

    assert response.json() == {"items": [], "total": 0, "limit": 20, "offset": 0}


async def test_standalone_admin_sees_every_job_with_scope_all(
    client: AsyncClient, job: JobTable
) -> None:
    response = await client.get(f"{API}/jobs?scope=all")

    assert response.json()["total"] == 1
