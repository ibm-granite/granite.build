"""Fixtures for the chat-service tests.

Provides a session factory bound to the shared in-memory test engine (so a
``ToolContext`` opens sessions against the same database the fixtures seed) and a
provisioned, non-admin principal that owns a real ``users`` row — the identity
own-scope tool tests run as.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from autotunex.db.tables import UserTable
from autotunex.models.auth import Principal


@pytest.fixture
def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """An async sessionmaker bound to the shared in-memory test engine."""
    return async_sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)


@pytest.fixture
async def provisioned_principal(
    session_factory: async_sessionmaker[AsyncSession],
) -> Principal:
    """A non-admin principal owning a persisted ``users`` row (own-scope tests)."""
    user = UserTable(id=uuid4(), email="chat-tester@autotunex.local", role="user")
    async with session_factory() as db:
        db.add(user)
        await db.commit()
    return Principal(email=user.email, provider="disabled", user_id=user.id, is_admin=False)
