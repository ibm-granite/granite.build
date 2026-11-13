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

"""Tests for the health endpoints."""

from __future__ import annotations

from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy.exc import OperationalError

from autotunex.api.deps import get_session


async def test_health_reports_ok(client: AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_health_live_reports_liveness(client: AsyncClient) -> None:
    response = await client.get("/health/live")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_health_ready_reports_ready_when_database_reachable(client: AsyncClient) -> None:
    response = await client.get("/health/ready")

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "ready"
    assert body["database"] == "ok"


async def test_health_ready_returns_503_when_database_unreachable(
    app: FastAPI, client: AsyncClient
) -> None:
    class _UnreachableSession:
        """A stand-in session whose first query fails as a dead connection would."""

        async def execute(self, *_args: object, **_kwargs: object) -> object:
            raise OperationalError("SELECT 1", None, Exception("connection refused"))

    app.dependency_overrides[get_session] = lambda: _UnreachableSession()

    response = await client.get("/health/ready")

    body = response.json()
    assert response.status_code == 503
    assert body["status"] == 503
    assert body["title"] == "Service Unavailable"


async def test_health_reports_service_name_and_version(client: AsyncClient) -> None:
    response = await client.get("/health")

    body = response.json()
    assert body["service"] == "AutoTuneX API"
    assert body["version"]


async def test_openapi_schema_is_served(client: AsyncClient) -> None:
    response = await client.get("/openapi.json")

    assert response.status_code == 200
    assert "/api/v1/jobs" in response.json()["paths"]
