# tests/api/routers/test_users.py
"""User-management endpoints over HTTP.

Principal resolution is swapped per test with ``as_principal``; ``require_admin``
reads the resulting ``is_admin`` flag, so these tests are independent of which
credential kind is configured.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from http import HTTPStatus
from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from autotunex.core.auth.disabled import SYSTEM_STANDALONE_EMAIL
from autotunex.db.tables import ConfigurationTable, DatasetTable, JobTable, UserTable
from autotunex.models.auth import Principal
from tests.conftest import API


def _admin(user_id: object) -> Principal:
    return Principal(email="admin@example.com", provider="session", user_id=user_id, is_admin=True)


def _regular(user_id: object) -> Principal:
    return Principal(email="reg@example.com", provider="session", user_id=user_id, is_admin=False)


@pytest.fixture
async def admin_user(session: AsyncSession) -> UserTable:
    admin = UserTable(id=uuid4(), email="admin@example.com", role="admin")
    session.add(admin)
    await session.commit()
    return admin


async def test_admin_lists_users(
    client: AsyncClient, as_principal: Callable[[Principal], None], admin_user: UserTable
) -> None:
    as_principal(_admin(admin_user.id))

    response = await client.get(f"{API}/users")

    assert response.status_code == HTTPStatus.OK
    assert response.json()["total"] == 1
    assert response.json()["items"][0]["email"] == "admin@example.com"


async def test_a_non_admin_cannot_list_users(
    client: AsyncClient, as_principal: Callable[[Principal], None], admin_user: UserTable
) -> None:
    as_principal(_regular(admin_user.id))

    response = await client.get(f"{API}/users")

    assert response.status_code == HTTPStatus.FORBIDDEN


async def test_admin_gets_a_user_by_id(
    client: AsyncClient, as_principal: Callable[[Principal], None], admin_user: UserTable
) -> None:
    as_principal(_admin(admin_user.id))

    response = await client.get(f"{API}/users/{admin_user.id}")

    assert response.status_code == HTTPStatus.OK
    assert response.json()["id"] == str(admin_user.id)


async def test_getting_an_unknown_user_is_404(
    client: AsyncClient, as_principal: Callable[[Principal], None], admin_user: UserTable
) -> None:
    as_principal(_admin(admin_user.id))

    response = await client.get(f"{API}/users/{uuid4()}")

    assert response.status_code == HTTPStatus.NOT_FOUND


async def test_a_non_admin_cannot_get_a_user(
    client: AsyncClient, as_principal: Callable[[Principal], None], admin_user: UserTable
) -> None:
    as_principal(_regular(admin_user.id))

    response = await client.get(f"{API}/users/{admin_user.id}")

    assert response.status_code == HTTPStatus.FORBIDDEN


async def test_admin_promotes_a_user_to_admin(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    session: AsyncSession,
    admin_user: UserTable,
) -> None:
    target = UserTable(id=uuid4(), email="target@example.com", role="user")
    session.add(target)
    await session.commit()
    as_principal(_admin(admin_user.id))

    response = await client.patch(f"{API}/users/{target.id}", json={"role": "admin"})

    assert response.status_code == HTTPStatus.OK
    assert response.json()["role"] == "admin"


async def test_changing_your_own_role_is_a_conflict(
    client: AsyncClient, as_principal: Callable[[Principal], None], admin_user: UserTable
) -> None:
    as_principal(_admin(admin_user.id))

    response = await client.patch(f"{API}/users/{admin_user.id}", json={"role": "user"})

    assert response.status_code == HTTPStatus.CONFLICT


async def test_demoting_the_last_admin_is_a_conflict(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    session: AsyncSession,
) -> None:
    sole_admin = UserTable(id=uuid4(), email="only-admin@example.com", role="admin")
    session.add(sole_admin)
    await session.commit()
    # A standalone-style admin caller with no counted users row of its own, so
    # the own-role guard does not pre-empt the last-admin guard.
    as_principal(Principal(email=None, provider="disabled", user_id=None, is_admin=True))

    response = await client.patch(f"{API}/users/{sole_admin.id}", json={"role": "user"})

    assert response.status_code == HTTPStatus.CONFLICT


async def test_changing_an_unknown_users_role_is_404(
    client: AsyncClient, as_principal: Callable[[Principal], None], admin_user: UserTable
) -> None:
    as_principal(_admin(admin_user.id))

    response = await client.patch(f"{API}/users/{uuid4()}", json={"role": "admin"})

    assert response.status_code == HTTPStatus.NOT_FOUND


async def test_an_invalid_role_value_is_422(
    client: AsyncClient, as_principal: Callable[[Principal], None], admin_user: UserTable
) -> None:
    as_principal(_admin(admin_user.id))

    response = await client.patch(f"{API}/users/{admin_user.id}", json={"role": "root"})

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


async def test_a_non_admin_cannot_change_a_role(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    session: AsyncSession,
    admin_user: UserTable,
) -> None:
    target = UserTable(id=uuid4(), email="target@example.com", role="user")
    session.add(target)
    await session.commit()
    as_principal(_regular(admin_user.id))

    response = await client.patch(f"{API}/users/{target.id}", json={"role": "admin"})

    assert response.status_code == HTTPStatus.FORBIDDEN


async def test_me_metadata_returns_counts_for_the_caller(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    configuration: ConfigurationTable,
    dataset: DatasetTable,
    job: JobTable,
) -> None:
    # `user` owns exactly one configuration, dataset, and job via the fixtures.
    as_principal(_regular(user.id))

    response = await client.get(f"{API}/users/me/metadata")

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {
        "number_of_jobs": 1,
        "number_of_configurations": 1,
        "number_of_datasets": 1,
    }


async def test_me_metadata_returns_zeros_for_an_unresolvable_caller(
    client: AsyncClient, as_principal: Callable[[Principal], None]
) -> None:
    as_principal(Principal(email=None, provider="disabled", user_id=None, is_admin=True))

    response = await client.get(f"{API}/users/me/metadata")

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {
        "number_of_jobs": 0,
        "number_of_configurations": 0,
        "number_of_datasets": 0,
    }


async def test_the_user_list_reports_the_last_login_time(
    client: AsyncClient, as_principal: Callable[[Principal], None], session: AsyncSession
) -> None:
    """Backs the Users table's "Last login on" column."""
    logged_in_at = datetime(2026, 3, 4, 5, 6, 7, tzinfo=UTC)
    admin = UserTable(
        id=uuid4(), email="admin@example.com", role="admin", last_login_at=logged_in_at
    )
    session.add(admin)
    await session.commit()
    as_principal(_admin(admin.id))

    response = await client.get(f"{API}/users")

    assert response.json()["items"][0]["last_login_at"] == "2026-03-04T05:06:07Z"


async def test_a_user_who_has_never_logged_in_reports_a_null_last_login(
    client: AsyncClient, as_principal: Callable[[Principal], None], admin_user: UserTable
) -> None:
    """Honest ``null`` rather than a stand-in timestamp — see the column's docstring."""
    as_principal(_admin(admin_user.id))

    response = await client.get(f"{API}/users/{admin_user.id}")

    assert response.json()["last_login_at"] is None


async def test_using_the_app_makes_the_users_table_report_a_last_login(
    client: AsyncClient, session: AsyncSession
) -> None:
    """End-to-end guard for the reported bug, through the real principal chain.

    No ``as_principal`` override: the caller is resolved by the standalone provider
    exactly as a deployment would, so this exercises ``get_principal`` ->
    ``_record_activity`` -> ``touch_login`` and then reads the value back out of
    the endpoint the Users tab actually calls. Before ``last_login_at`` existed,
    that column was ``updated_at`` and never moved off the row's creation time.

    The row is created here, with ``last_login_at`` explicitly unset, precisely so
    provisioning does *not* run: ``provision`` stamps the column itself, so a test
    that let the caller be provisioned would still pass with the per-request
    refresh deleted — verified by removing it. Pre-existing row, no stamp, is the
    only arrangement that pins the activity path.
    """
    standalone = UserTable(
        id=uuid4(), email=SYSTEM_STANDALONE_EMAIL, role="admin", last_login_at=None
    )
    session.add(standalone)
    await session.commit()

    response = await client.get(f"{API}/users")

    assert response.status_code == HTTPStatus.OK
    assert response.json()["items"][0]["last_login_at"] is not None
