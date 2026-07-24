# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

import models as api
import pytest
from fastapi import HTTPException


class _Req:
    """Minimal Request stand-in exposing .cookies (what get_current_user reads)."""

    def __init__(self, cookies=None):
        self.cookies = cookies or {}


class _FakeUserDB:
    """In-memory stand-in for db_service.Database's user methods.

    DevAuthProvider provisions the dev user (like the OIDC /callback does) via
    user_service.User, which only calls get_user/insert_user on its db. This
    fake records inserts so tests can assert the row is created without a real
    MySQL pool.
    """

    def __init__(self, existing=None):
        # email -> row dict
        self.rows = dict(existing or {})
        self.inserts = []

    async def get_user(self, email):
        return self.rows.get(email)

    async def insert_user(self, email):
        self.inserts.append(email)
        self.rows[email] = {"id": f"id-{len(self.rows)}", "email": email}
        return self.rows[email]["id"]


def _patch_dev_db(monkeypatch, fake):
    """Point dependencies.get_database() at a fake so the provider's
    provisioning step never touches a real DB pool."""
    import dependencies

    monkeypatch.setattr(dependencies, "get_database", lambda: fake)


async def test_dev_provider_returns_dev_user(monkeypatch):
    from services.auth_providers.dev_provider import DevAuthProvider

    monkeypatch.delenv("DEV_USER_EMAIL", raising=False)
    monkeypatch.delenv("DEV_USER_ROLE", raising=False)
    _patch_dev_db(monkeypatch, _FakeUserDB())
    user = await DevAuthProvider().get_current_user(_Req())
    assert user.email == "dev@example.com"
    assert user.role == api.Roles("admin")


async def test_dev_provider_honors_env(monkeypatch):
    from services.auth_providers.dev_provider import DevAuthProvider

    monkeypatch.setenv("DEV_USER_EMAIL", "alice@x.com")
    monkeypatch.setenv("DEV_USER_ROLE", "user")
    _patch_dev_db(monkeypatch, _FakeUserDB())
    user = await DevAuthProvider().get_current_user(_Req())
    assert user.email == "alice@x.com"
    assert user.role == api.Roles("user")


async def test_dev_provider_provisions_missing_user(monkeypatch):
    """Regression: dev mode has no OIDC /callback to create the user row, so a
    fresh DB has no dev@example.com row and downstream get_user(email)["id"]
    lookups (e.g. GET /jobs) 500 with 'NoneType is not subscriptable'. The
    provider must create the row if absent."""
    from services.auth_providers.dev_provider import DevAuthProvider

    monkeypatch.setenv("DEV_USER_EMAIL", "dev@example.com")
    fake = _FakeUserDB()  # empty DB — no dev user yet
    _patch_dev_db(monkeypatch, fake)

    await DevAuthProvider().get_current_user(_Req())

    assert fake.inserts == ["dev@example.com"]  # row was created
    assert await fake.get_user("dev@example.com") is not None
    # downstream code does get_user(email)["id"] — this must now succeed
    assert (await fake.get_user("dev@example.com"))["id"]


async def test_dev_provider_does_not_duplicate_existing_user(monkeypatch):
    """If the dev user already exists, the provider must not insert it again."""
    from services.auth_providers.dev_provider import DevAuthProvider

    monkeypatch.setenv("DEV_USER_EMAIL", "dev@example.com")
    fake = _FakeUserDB(
        existing={"dev@example.com": {"id": "preexisting", "email": "dev@example.com"}}
    )
    _patch_dev_db(monkeypatch, fake)

    await DevAuthProvider().get_current_user(_Req())

    assert fake.inserts == []  # no duplicate insert


async def test_w3id_provider_valid_cookie(monkeypatch):
    import auth
    from services.auth_providers.w3id_provider import W3idAuthProvider

    token = auth.create_session_token(email="bob@x.com", role="admin")
    req = _Req(cookies={auth.SESSION_COOKIE: token})
    user = await W3idAuthProvider().get_current_user(req)
    assert user.email == "bob@x.com"
    assert user.role == api.Roles("admin")
    assert user.impersonating is None


async def test_w3id_provider_impersonation_precedence(monkeypatch):
    import auth
    from services.auth_providers.w3id_provider import W3idAuthProvider

    token = auth.create_session_token(
        email="admin@x.com",
        role="admin",
        impersonating="target@x.com",
        impersonator="admin@x.com",
    )
    req = _Req(cookies={auth.SESSION_COOKIE: token})
    user = await W3idAuthProvider().get_current_user(req)
    assert user.email == "target@x.com"  # impersonating wins
    assert user.role == api.Roles("admin")  # admin role retained
    assert user.impersonating == "target@x.com"


async def test_w3id_provider_missing_cookie_401():
    from services.auth_providers.w3id_provider import W3idAuthProvider

    with pytest.raises(HTTPException) as ei:
        await W3idAuthProvider().get_current_user(_Req())
    assert ei.value.status_code == 401


async def test_w3id_provider_invalid_token_401():
    import auth
    from services.auth_providers.w3id_provider import W3idAuthProvider

    req = _Req(cookies={auth.SESSION_COOKIE: "not-a-jwt"})
    with pytest.raises(HTTPException) as ei:
        await W3idAuthProvider().get_current_user(req)
    assert ei.value.status_code == 401


def test_login_routes_defaults_to_none():
    from services.auth_providers.dev_provider import DevAuthProvider

    assert DevAuthProvider().login_routes() is None
