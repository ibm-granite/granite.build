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

"""Regression coverage for the build status-chart endpoint (Path 1: the
analytics-service database).

Guards the per-date / per-status / is-test aggregation and, specifically, the
reading of the aggregated build-count column. That column is labelled and read
back off the SQLAlchemy ``Row`` by attribute; a label that collides with a
``Row``/``Sequence`` method name (``count``, ``index``) resolves to the value
at runtime but trips static type checking, so the label must stay collision
free. This test fails loudly if the count stops being read correctly.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from gb_ui_backend.api.analytics import router
from gb_ui_backend.config import Config, get_config
from gb_ui_backend.services.db_schema import Base, GbdBuild, get_optional_db


def _make_build(username: str, status: str, when: datetime) -> GbdBuild:
    """Build a minimally-populated GbdBuild row for seeding.

    Args:
        username: owner; a ``test``-prefixed name marks the build as a test run.
        status: build status string (e.g. ``success``, ``failed``).
        when: ``created_at`` timestamp used for date grouping and the
            days-back window filter.

    Returns:
        An unsaved GbdBuild instance ready to add to a session.
    """
    return GbdBuild(
        id=uuid.uuid4(),
        name=f"build-{uuid.uuid4().hex[:8]}",
        space_name="space",
        username=username,
        status=status,
        created_at=when,
        updated_at=when,
    )


@pytest_asyncio.fixture
async def app_and_client():
    """FastAPI app + TestClient backed by an in-memory DB seeded with builds.

    Seeds (all on the same recent date):
      - two ``success`` builds for a normal user (not test runs)
      - one ``failed`` build for a ``test``-prefixed user (a test run)

    Yields:
        (app, client) with ``get_optional_db``/``get_config`` overridden.
    """
    # StaticPool keeps a single underlying connection for the whole engine, so
    # schema creation, seeding, and every per-request session share one in-memory
    # database. Without it each :memory: connection is a distinct DB and the test
    # only passes by the accident of the default pool reusing the same connection.
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", poolclass=StaticPool)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    when = datetime.now(timezone.utc) - timedelta(days=1)
    async with factory() as session:
        session.add_all(
            [
                _make_build("alice", "success", when),
                _make_build("alice", "success", when),
                _make_build("test_bob", "failed", when),
            ]
        )
        await session.commit()

    async def override_get_optional_db():
        async with factory() as session:
            yield session

    app = FastAPI()
    app.include_router(router, prefix="/api/analytics")
    app.dependency_overrides[get_optional_db] = override_get_optional_db
    app.dependency_overrides[get_config] = lambda: Config(
        database_url="sqlite+aiosqlite:///:memory:"
    )

    yield app, TestClient(app)
    await engine.dispose()


class TestBuildStatusChart:
    def test_counts_group_by_status_and_is_test(self, app_and_client):
        """The aggregated count must land in the right status bucket, split by
        whether the build is a test run."""
        _, client = app_and_client
        resp = client.get("/api/analytics/builds/status-chart")
        assert resp.status_code == 200
        points = resp.json()
        assert len(points) == 1  # all three builds share one date
        point = points[0]
        # Two non-test successes; one failed build belongs to a test user.
        assert point["success"] == 2
        assert point["failed_test"] == 1
        # The failed build was a test run, so the non-test failed bucket is empty.
        assert point["failed"] == 0
        assert point["success_test"] == 0

    def test_exclude_tests_zeroes_test_columns(self, app_and_client):
        """exclude_tests must zero the ``*_test`` columns while leaving the
        normal buckets intact."""
        _, client = app_and_client
        resp = client.get(
            "/api/analytics/builds/status-chart", params={"exclude_tests": "true"}
        )
        assert resp.status_code == 200
        point = resp.json()[0]
        assert point["success"] == 2  # unaffected
        assert point["failed_test"] == 0  # suppressed
