"""End-to-end proof that ownership scoping follows the impersonation overlay.

While an admin impersonates another user (a real admin principal via
``as_principal`` plus a valid ``autotunex_assume`` cookie), the effective owner
resolved by ``get_effective_principal`` is the *target*: owner-scoped reads
attribute to the target, while ``is_admin`` is preserved so the admin-only
``scope=all`` widening still works.
"""

from __future__ import annotations

from collections.abc import Callable
from http import HTTPStatus
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from autotunex.core.auth.impersonation import mint_assume_token
from autotunex.core.config import Settings
from autotunex.db.tables import ConfigurationTable, DatasetTable, JobTable, UserTable
from autotunex.models.auth import Principal
from autotunex.models.status import RunStatus
from tests.conftest import API, make_settings

_SECRET = "x" * 32


@pytest.fixture
def settings() -> Settings:
    """Test settings whose ``session_secret`` lets the overlay cookie be read.

    A ≥32-char ``session_secret`` is the only field ``get_effective_principal``
    needs to verify the ``autotunex_assume`` cookie. The default ``["disabled"]``
    providers are kept deliberately — enabling ``"session"`` would trip the
    ``Settings`` validator, which then demands the full OIDC/session endpoint set.
    """
    return make_settings(session_secret=_SECRET)


async def _seed_user(session: AsyncSession, *, email: str, role: str) -> UserTable:
    """Persist and return a fresh user with the given email and role."""
    user = UserTable(email=email, role=role)
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


async def test_listing_jobs_while_impersonating_returns_the_targets_jobs(
    app: FastAPI,
    session: AsyncSession,
    as_principal: Callable[[Principal], None],
    configuration: ConfigurationTable,
    dataset: DatasetTable,
    job: JobTable,
) -> None:
    admin = await _seed_user(session, email="admin@example.com", role="admin")
    target = await _seed_user(session, email="target@example.com", role="user")
    target_job = JobTable(
        id=uuid4(),
        user_id=str(target.id),
        status=RunStatus.PENDING,
        seed=7,
        config_id=configuration.id,
        dataset_id=dataset.id,
        model="ibm-granite/granite-3.0-8b-instruct",
        model_source="huggingface",
        experiment_name="targets-own",
        tuning_type="lora",
    )
    session.add(target_job)
    await session.commit()
    as_principal(Principal(email=admin.email, provider="session", user_id=admin.id, is_admin=True))
    cookie = mint_assume_token(target.id, secret=_SECRET, ttl_hours=8)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        cookies={"autotunex_assume": cookie},
    ) as client:
        response = await client.get(f"{API}/jobs")

    assert response.status_code == HTTPStatus.OK
    ids = {row["id"] for row in response.json()["items"]}
    assert ids == {str(target_job.id)}


async def test_scope_all_still_works_while_impersonating(
    app: FastAPI,
    session: AsyncSession,
    as_principal: Callable[[Principal], None],
) -> None:
    admin = await _seed_user(session, email="admin@example.com", role="admin")
    target = await _seed_user(session, email="target@example.com", role="user")
    as_principal(Principal(email=admin.email, provider="session", user_id=admin.id, is_admin=True))
    cookie = mint_assume_token(target.id, secret=_SECRET, ttl_hours=8)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        cookies={"autotunex_assume": cookie},
    ) as client:
        response = await client.get(f"{API}/jobs?scope=all")

    assert response.status_code == HTTPStatus.OK
