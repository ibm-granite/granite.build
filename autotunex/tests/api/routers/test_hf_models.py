# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace model search/card endpoints over HTTP; the Hub is a MockTransport."""

from __future__ import annotations

from collections.abc import Callable
from http import HTTPStatus

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from autotunex.api.deps import get_hf_http_client, get_session
from autotunex.core.config import get_settings
from autotunex.main import create_app
from tests.conftest import API, make_settings

PROBLEM_JSON = "application/problem+json"

Handler = Callable[[httpx.Request], httpx.Response]


def _wire(app: FastAPI, handler: Handler, *, lsf_cluster: str | None = None) -> list[httpx.Request]:
    """Allowlist example-org and route Hub traffic to ``handler``; return the requests seen."""
    seen: list[httpx.Request] = []

    def recording(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    settings = make_settings(lsf_cluster=lsf_cluster).model_copy(
        update={"hf_import_namespaces": ["example-org"]}
    )
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_hf_http_client] = lambda: httpx.AsyncClient(
        transport=httpx.MockTransport(recording)
    )
    return seen


def _search_hub(request: httpx.Request) -> httpx.Response:
    if request.url.params.get("author") == "example-org":
        return httpx.Response(200, json=[{"id": "example-org/autotunex_x", "private": True}])
    return httpx.Response(200, json=[{"id": "ibm-granite/granite-4.0-h-micro"}])


async def test_search_returns_private_then_public_ids(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "tok")
    _wire(app, _search_hub)

    response = await client.get(f"{API}/hf/models/search", params={"query": "x"})

    assert response.status_code == HTTPStatus.OK
    assert response.json() == ["example-org/autotunex_x", "ibm-granite/granite-4.0-h-micro"]


async def test_search_is_not_gated_on_dataset_import(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An LSF deployment closes the dataset-import gate; model search must still answer."""
    monkeypatch.setenv("HF_TOKEN", "tok")
    _wire(app, _search_hub, lsf_cluster="a-cluster")

    response = await client.get(f"{API}/hf/models/search", params={"query": "x"})

    assert response.status_code == HTTPStatus.OK


async def test_search_requires_a_query(app: FastAPI, client: AsyncClient) -> None:
    seen = _wire(app, _search_hub)

    response = await client.get(f"{API}/hf/models/search", params={"query": ""})

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert seen == []


async def test_search_outage_is_503(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    _wire(app, lambda request: httpx.Response(HTTPStatus.BAD_GATEWAY, text="bad gateway"))

    response = await client.get(f"{API}/hf/models/search", params={"query": "x"})

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


async def test_card_returns_plain_text(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "tok")
    seen = _wire(app, lambda request: httpx.Response(200, text="# Private card"))

    response = await client.get(
        f"{API}/hf/models/card", params={"repo_id": "example-org/autotunex_x"}
    )

    assert response.status_code == HTTPStatus.OK
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == "# Private card"
    assert seen[0].url.path == "/example-org/autotunex_x/raw/main/README.md"


async def test_card_for_an_unreadable_repo_is_404(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "tok")
    _wire(app, lambda request: httpx.Response(401, text="no"))

    response = await client.get(f"{API}/hf/models/card", params={"repo_id": "elsewhere/m"})

    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


async def test_card_rejects_a_repo_id_that_escapes_the_hub_path(
    app: FastAPI, client: AsyncClient
) -> None:
    seen = _wire(app, lambda request: httpx.Response(200, text="should not be reached"))

    response = await client.get(f"{API}/hf/models/card", params={"repo_id": "../../etc/passwd"})

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert seen == []


async def test_search_is_401_when_unauthenticated(session: AsyncSession) -> None:
    settings = make_settings(auth_providers=["api_key"], api_keys={"a" * 64: "someone@example.com"})
    unauthenticated_app = create_app(settings)
    unauthenticated_app.dependency_overrides[get_settings] = lambda: settings
    unauthenticated_app.dependency_overrides[get_session] = lambda: session

    async with AsyncClient(
        transport=ASGITransport(app=unauthenticated_app), base_url="http://testserver"
    ) as unauthenticated_client:
        response = await unauthenticated_client.get(
            f"{API}/hf/models/search", params={"query": "x"}
        )

    assert response.status_code == HTTPStatus.UNAUTHORIZED
