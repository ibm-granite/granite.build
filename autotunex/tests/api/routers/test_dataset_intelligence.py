"""Dataset-intelligence endpoints, end to end over HTTP."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI
from httpx import AsyncClient

from autotunex.api.deps import get_dataset_intelligence_service
from autotunex.core.exceptions import AutotuneCoreUnavailableError
from autotunex.services.dataset_intelligence import DatasetIntelligenceService
from autotunex.services.llm.base import ChatDelta
from tests.conftest import API, make_settings

PROBLEM_JSON = "application/problem+json"

DATASET_TYPES: dict[str, Any] = {
    "sft": {"columns": {"prompt": {"type": "str"}, "completion": {"type": "str"}}},
    "dpo": {"columns": {"prompt": {"type": "str"}}},
}


class FakeLlmClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    async def complete(
        self, *, system: str, user: str, response_schema: dict[str, Any] | None = None
    ) -> str:
        return self._responses.pop(0)

    def stream_chat(
        self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[ChatDelta]:
        # Dataset intelligence never streams — this satisfies the LlmClient
        # Protocol's `stream_chat` member without needing a real implementation.
        raise NotImplementedError


class FakeAutotuneCore:
    def __init__(
        self, *, dataset_types: dict[str, Any] | None = None, raises: Exception | None = None
    ) -> None:
        self._dataset_types = dataset_types if dataset_types is not None else DATASET_TYPES
        self._raises = raises

    async def get_config_template(self) -> dict[str, Any]:
        if self._raises is not None:
            raise self._raises
        return {}

    async def get_dataset_types(self) -> dict[str, Any]:
        if self._raises is not None:
            raise self._raises
        return self._dataset_types


def _use_service(
    app: FastAPI, responses: list[str], *, dataset_types: dict[str, Any] | None = None
) -> None:
    service = DatasetIntelligenceService(
        llm=FakeLlmClient(responses),
        settings=make_settings(),
        autotune=FakeAutotuneCore(dataset_types=dataset_types),
    )
    app.dependency_overrides[get_dataset_intelligence_service] = lambda: service


async def test_parse_strategy_returns_200(app: FastAPI, client: AsyncClient) -> None:
    _use_service(
        app,
        [json.dumps({"type": "direct_mapping", "input_field": "q", "output_field": "a"})],
    )

    response = await client.post(
        f"{API}/datasets/intelligence/parse-strategy",
        json={"sample": [{"q": "hi", "a": "yo"}], "data_format": "jsonl"},
    )

    assert response.status_code == HTTPStatus.OK
    assert response.json()["type"] == "direct_mapping"


async def test_suggest_mapping_returns_a_flat_mapping(app: FastAPI, client: AsyncClient) -> None:
    _use_service(
        app,
        [
            json.dumps(
                {
                    "dataset_format": "sft",
                    "tuning_type": "sft",
                    "column_mapping": {"prompt": "question", "completion": "answer"},
                }
            )
        ],
    )

    response = await client.post(
        f"{API}/datasets/intelligence/suggest-mapping",
        json={
            "column_names": ["question", "answer"],
            "column_samples": {"question": ["hi"], "answer": ["yo"]},
            "sample_data": [{"question": "hi", "answer": "yo"}],
        },
    )

    assert response.status_code == HTTPStatus.OK
    assert response.json()["column_mapping"] == {"prompt": "question", "completion": "answer"}


async def test_validate_strategy_needs_no_llm(app: FastAPI, client: AsyncClient) -> None:
    response = await client.post(
        f"{API}/datasets/intelligence/validate-strategy",
        json={
            "strategy": {"type": "direct_mapping", "input_field": "q", "output_field": "a"},
            "sample": [{"q": "hi", "a": "yo"}],
        },
    )

    assert response.status_code == HTTPStatus.OK
    body = response.json()
    assert body["success"] is True
    assert body["parsed_count"] == 1


async def test_parse_strategy_is_503_when_unconfigured(client: AsyncClient) -> None:
    # No override: default settings leave the LLM unconfigured.
    response = await client.post(
        f"{API}/datasets/intelligence/parse-strategy",
        json={"sample": [{"q": "hi"}], "data_format": "jsonl"},
    )

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


async def test_malformed_body_is_422(client: AsyncClient) -> None:
    response = await client.post(
        f"{API}/datasets/intelligence/parse-strategy",
        json={"sample": [{"q": "hi"}], "surprise": "x"},
    )

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


async def test_get_formats_returns_the_autotune_catalog(app: FastAPI, client: AsyncClient) -> None:
    _use_service(app, [])

    response = await client.get(f"{API}/datasets/intelligence/formats")

    assert response.status_code == HTTPStatus.OK
    body = response.json()
    assert set(body) == {"sft", "dpo"}
    assert "columns" in body["sft"]


async def test_get_formats_is_503_when_autotune_is_absent(
    app: FastAPI, client: AsyncClient
) -> None:
    # Inject a service whose AutotuneCore raises, so the unavailable path is exercised
    # whether or not ``autotune`` happens to be installed in this environment.
    service = DatasetIntelligenceService(
        llm=None,
        settings=make_settings(),
        autotune=FakeAutotuneCore(raises=AutotuneCoreUnavailableError()),
    )
    app.dependency_overrides[get_dataset_intelligence_service] = lambda: service

    response = await client.get(f"{API}/datasets/intelligence/formats")

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert response.headers["content-type"].startswith(PROBLEM_JSON)
