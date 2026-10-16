# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""The admin impersonation overlay, exercised end-to-end through ``GET /auth/me``.

``get_principal`` is overridden by ``as_principal`` (the real caller), so the
authenticator is never consulted and no auth provider needs enabling — only
``session_secret`` must be set for ``get_effective_principal`` to read the
overlay cookie. Every guard in that dependency falls back to the real principal,
so these tests pin the one path that applies the overlay and the two that must not.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from autotunex.api.deps import get_session
from autotunex.core.auth.impersonation import mint_assume_token
from autotunex.core.config import Settings, get_settings
from autotunex.db.tables import UserTable
from autotunex.main import create_app
from autotunex.models.auth import Principal
from tests.conftest import make_settings

_SECRET = "x" * 32


@pytest.fixture
def settings() -> Settings:
    """Override the conftest default so the overlay cookie is readable.

    ``session_secret`` is the only field the overlay path needs; the providers
    stay at the ``["disabled"]`` default because ``as_principal`` overrides
    ``get_principal`` and no authenticator is consulted.
    """
    return make_settings(session_secret=_SECRET)


async def _seed_user(session: AsyncSession, *, email: str, role: str) -> UserTable:
    user = UserTable(email=email, role=role)
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


@pytest.fixture
def app_no_secret(session: AsyncSession) -> Iterator[FastAPI]:
    """An app whose settings carry no ``session_secret``.

    Mirrors the conftest ``app`` fixture body but pins ``session_secret=None`` so
    ``POST /auth/assume`` reaches the ``ImpersonationUnavailableError`` (503)
    guard, which fires before the token is ever minted.
    """
    no_secret_settings = make_settings(session_secret=None)
    app = create_app(no_secret_settings)
    app.dependency_overrides[get_settings] = lambda: no_secret_settings
    app.dependency_overrides[get_session] = lambda: session

    yield app

    app.dependency_overrides.clear()


async def test_me_reflects_impersonation_for_an_admin_with_a_valid_overlay(
    app: FastAPI, session: AsyncSession, as_principal: Callable[[Principal], None]
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
        response = await client.get("/auth/me")

    body = response.json()
    assert body["email"] == "target@example.com"
    assert body["user_id"] == str(target.id)
    assert body["is_admin"] is True  # preserved
    assert body["impersonator"] == "admin@example.com"


async def test_me_ignores_the_overlay_for_a_non_admin(
    app: FastAPI, session: AsyncSession, as_principal: Callable[[Principal], None]
) -> None:
    caller = await _seed_user(session, email="user@example.com", role="user")
    target = await _seed_user(session, email="target@example.com", role="user")
    as_principal(
        Principal(email=caller.email, provider="session", user_id=caller.id, is_admin=False)
    )
    cookie = mint_assume_token(target.id, secret=_SECRET, ttl_hours=8)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        cookies={"autotunex_assume": cookie},
    ) as client:
        response = await client.get("/auth/me")

    body = response.json()
    assert body["email"] == "user@example.com"
    assert body["impersonator"] is None


async def test_me_ignores_an_overlay_pointing_at_a_missing_user(
    app: FastAPI, session: AsyncSession, as_principal: Callable[[Principal], None]
) -> None:
    admin = await _seed_user(session, email="admin@example.com", role="admin")
    as_principal(Principal(email=admin.email, provider="session", user_id=admin.id, is_admin=True))
    cookie = mint_assume_token(uuid4(), secret=_SECRET, ttl_hours=8)  # no such user

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        cookies={"autotunex_assume": cookie},
    ) as client:
        response = await client.get("/auth/me")

    body = response.json()
    assert body["email"] == "admin@example.com"
    assert body["impersonator"] is None


async def test_assume_sets_the_overlay_cookie_for_an_admin(
    app: FastAPI, session: AsyncSession, as_principal: Callable[[Principal], None]
) -> None:
    admin = await _seed_user(session, email="admin@example.com", role="admin")
    target = await _seed_user(session, email="target@example.com", role="user")
    as_principal(Principal(email=admin.email, provider="session", user_id=admin.id, is_admin=True))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(f"/auth/assume/{target.id}")

    assert response.status_code == 200
    assert response.json()["assumed_email"] == "target@example.com"
    assert "autotunex_assume" in response.cookies


async def test_assume_is_forbidden_for_a_non_admin(
    app: FastAPI, session: AsyncSession, as_principal: Callable[[Principal], None]
) -> None:
    caller = await _seed_user(session, email="user@example.com", role="user")
    target = await _seed_user(session, email="target@example.com", role="user")
    as_principal(
        Principal(email=caller.email, provider="session", user_id=caller.id, is_admin=False)
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(f"/auth/assume/{target.id}")

    assert response.status_code == 403


async def test_assume_rejects_assuming_yourself(
    app: FastAPI, session: AsyncSession, as_principal: Callable[[Principal], None]
) -> None:
    admin = await _seed_user(session, email="admin@example.com", role="admin")
    as_principal(Principal(email=admin.email, provider="session", user_id=admin.id, is_admin=True))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(f"/auth/assume/{admin.id}")

    assert response.status_code == 400


async def test_assume_returns_404_for_an_unknown_target(
    app: FastAPI, session: AsyncSession, as_principal: Callable[[Principal], None]
) -> None:
    admin = await _seed_user(session, email="admin@example.com", role="admin")
    as_principal(Principal(email=admin.email, provider="session", user_id=admin.id, is_admin=True))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(f"/auth/assume/{uuid4()}")

    assert response.status_code == 404


async def test_assume_is_unavailable_without_a_session_secret(
    app_no_secret: FastAPI, session: AsyncSession, as_principal: Callable[[Principal], None]
) -> None:
    admin = await _seed_user(session, email="admin@example.com", role="admin")
    target = await _seed_user(session, email="target@example.com", role="user")
    as_principal(Principal(email=admin.email, provider="session", user_id=admin.id, is_admin=True))

    async with AsyncClient(
        transport=ASGITransport(app=app_no_secret), base_url="http://test"
    ) as client:
        response = await client.post(f"/auth/assume/{target.id}")

    assert response.status_code == 503


async def test_unassume_clears_the_overlay_cookie(
    app: FastAPI, session: AsyncSession, as_principal: Callable[[Principal], None]
) -> None:
    admin = await _seed_user(session, email="admin@example.com", role="admin")
    as_principal(Principal(email=admin.email, provider="session", user_id=admin.id, is_admin=True))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/auth/unassume")

    assert response.status_code == 200
    # delete_cookie sets an expired Set-Cookie header for the name.
    assert "autotunex_assume" in response.headers.get("set-cookie", "")
