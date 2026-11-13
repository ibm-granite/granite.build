# Copyright Granite.Build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
