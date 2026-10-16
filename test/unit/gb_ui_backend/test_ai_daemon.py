#!/usr/bin/env python3

# Copyright LLM.build Authors
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

"""Regression tests for the AI daemon's SQLite (default analytics DB) compatibility.

Covers two review findings:
  - run_custom_categorization crashed with NameError (func never imported).
  - _upsert_gbserver_build used a Postgres-only on_conflict_do_update, which
    raises against the SQLite DB gbserver auto-configures by default.
"""

import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from gb_ui_backend.services.ai_daemon import AIDaemon, run_custom_categorization
from gb_ui_backend.services.ai_data_collectors import CompositeDataCollector
from gb_ui_backend.services.db_schema import Base, GbdBuild


@pytest_asyncio.fixture
async def sqlite_session_factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


@pytest.fixture
def daemon(sqlite_session_factory):
    return AIDaemon(
        session_factory=sqlite_session_factory,
        data_collector=CompositeDataCollector(),
        llm_base_url="",
        llm_api_key="",
        llm_models=[],
    )


class TestUpsertGbserverBuild:
    @pytest.mark.asyncio
    async def test_insert_new_build_on_sqlite(self, daemon, sqlite_session_factory):
        """A build not yet in gbd_builds should be inserted without raising."""
        build_id = str(uuid.uuid4())
        raw = {
            "uuid": build_id,
            "name": "my-build",
            "space_name": "space1",
            "username": "alice",
            "status": "failed",
            "created_time": datetime.now(timezone.utc),
            "updated_time": datetime.now(timezone.utc),
        }
        async with sqlite_session_factory() as session:
            result = await daemon._upsert_gbserver_build(session, raw)
            await session.commit()

        assert str(result.id) == build_id
        assert result.status == "failed"

    @pytest.mark.asyncio
    async def test_update_existing_build_on_sqlite(
        self, daemon, sqlite_session_factory
    ):
        """A build already in gbd_builds should be refreshed in place, not
        raise a Postgres-dialect CompileError on SQLite."""
        build_id = str(uuid.uuid4())
        raw = {
            "uuid": build_id,
            "name": "my-build",
            "space_name": "space1",
            "username": "alice",
            "status": "running",
            "created_time": datetime.now(timezone.utc),
            "updated_time": datetime.now(timezone.utc),
        }
        async with sqlite_session_factory() as session:
            await daemon._upsert_gbserver_build(session, raw)
            await session.commit()

        raw["status"] = "failed"
        async with sqlite_session_factory() as session:
            result = await daemon._upsert_gbserver_build(session, raw)
            await session.commit()

        assert result.status == "failed"
        async with sqlite_session_factory() as session:
            rows = (await session.execute(GbdBuild.__table__.select())).fetchall()
        assert len(rows) == 1


class TestRunCustomCategorization:
    @pytest.mark.asyncio
    async def test_runs_without_nameerror_on_empty_db(self, sqlite_session_factory):
        """Regression test: `func` was never imported, so this raised
        NameError immediately on every call. An empty DB should just return 0."""
        count = await run_custom_categorization(
            session_factory=sqlite_session_factory,
            llm_base_url="http://example.invalid",
            llm_api_key="key",
            llm_models=["some-model"],
            categories=["infra", "user-error"],
        )
        assert count == 0
