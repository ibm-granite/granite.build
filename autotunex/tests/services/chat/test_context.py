"""Tests for :mod:`autotunex.services.chat.context`."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from autotunex.core.config import Settings
from autotunex.models.auth import Principal
from autotunex.services.chat.context import ToolContext


async def test_services_are_scoped_to_the_principal(
    session_factory: async_sessionmaker[AsyncSession], provisioned_principal: Principal
) -> None:
    """A fresh, provisioned principal sees an empty page of its own jobs."""
    settings = Settings(job_backend="none")
    ctx = ToolContext(
        principal=provisioned_principal, settings=settings, session_factory=session_factory
    )

    async with ctx.services() as svc:
        page = await svc.job.list(limit=5, offset=0)

    assert page.total == 0
    assert svc.principal.user_id == provisioned_principal.user_id
