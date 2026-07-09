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

"""Saved-trend-analysis ownership must be derived from the server-trusted
X-User-Email identity header, not a client-supplied `author` field/query
param — otherwise any caller can impersonate another user's identity to
read/edit/delete their saved analyses.
"""

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from gb_ui_backend.api.analytics import router
from gb_ui_backend.config import Config, get_config
from gb_ui_backend.services.db_schema import Base, get_db


@pytest_asyncio.fixture
async def app_and_client():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def override_get_db():
        async with factory() as session:
            yield session

    app = FastAPI()
    app.include_router(router, prefix="/api/analytics")
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_config] = lambda: Config(
        database_url="sqlite+aiosqlite:///:memory:"
    )

    yield app, TestClient(app)
    await engine.dispose()


def _save_trend(client, headers=None):
    return client.post(
        "/api/analytics/builds/failure-trends/save",
        json={"data": {"labels": [], "categories": [], "total_analyzed": 0}},
        headers=headers or {},
    )


class TestSavedTrendOwnership:
    def test_delete_ignores_client_supplied_author(self, app_and_client):
        """Regression: delete used to accept an `author` query param and only
        checked ownership `if author and ...`, so omitting it bypassed the
        check entirely. The endpoint must derive identity from the trusted
        header instead."""
        _, client = app_and_client
        save_resp = _save_trend(client, headers={"X-User-Email": "alice@example.com"})
        update_id = save_resp.json()["update_id"]

        # No author query param, no identity header — must NOT be deletable.
        delete_resp = client.delete(f"/api/analytics/builds/failure-trends/{update_id}")
        assert delete_resp.status_code == 403

        # Spoofing the old client-supplied author param must not work either.
        delete_resp = client.delete(
            f"/api/analytics/builds/failure-trends/{update_id}",
            params={"author": "alice@example.com"},
        )
        assert delete_resp.status_code == 403

    def test_owner_can_delete_via_identity_header(self, app_and_client):
        _, client = app_and_client
        save_resp = _save_trend(client, headers={"X-User-Email": "alice@example.com"})
        update_id = save_resp.json()["update_id"]

        delete_resp = client.delete(
            f"/api/analytics/builds/failure-trends/{update_id}",
            headers={"X-User-Email": "alice@example.com"},
        )
        assert delete_resp.status_code == 200

    def test_mine_tab_scoped_to_trusted_identity(self, app_and_client):
        """The "mine" filter must use the trusted identity, not a
        client-supplied author string that could impersonate another user."""
        _, client = app_and_client
        _save_trend(client, headers={"X-User-Email": "alice@example.com"})
        _save_trend(client, headers={"X-User-Email": "bob@example.com"})

        resp = client.get(
            "/api/analytics/builds/failure-trends/history",
            params={"tab": "mine", "author": "bob@example.com"},
            headers={"X-User-Email": "alice@example.com"},
        )
        # Even though the query param claims to be bob, the trusted header
        # says alice — "mine" must return alice's own saved trend, not bob's
        # (which it would if the spoofable query param were trusted instead).
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_count"] == 1
        assert body["items"][0]["author"] == "alice@example.com"
