"""The OpenAI-compatible adapter against a mocked transport."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from autotunex.core.exceptions import LlmUnavailableError
from autotunex.services.llm.base import LlmClient
from autotunex.services.llm.openai_compatible import OpenAiCompatibleLlmClient


def _client(
    handler: object, *, structured_output: str = "json_object"
) -> OpenAiCompatibleLlmClient:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    http_client = httpx.AsyncClient(transport=transport)
    return OpenAiCompatibleLlmClient(
        http_client=http_client,
        base_url="https://gw.example/v1",
        api_key=SecretStr("sk-secret"),
        model="test-model",
        timeout_seconds=5.0,
        structured_output=structured_output,
    )


def test_the_adapter_satisfies_the_protocol() -> None:
    client: LlmClient = _client(lambda request: httpx.Response(200))

    assert client is not None


async def test_it_posts_the_expected_shape_and_returns_content() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = _json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}]})

    result = await _client(handler).complete(system="sys", user="usr")

    assert seen["url"] == "https://gw.example/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-secret"
    assert seen["body"]["model"] == "test-model"
    assert seen["body"]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]
    assert result == "hello"


def _capturing_handler(captured: dict[str, Any]) -> Any:  # noqa: ANN401
    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        captured["body"] = _json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    return handler


async def test_default_json_object_mode_sends_schemaless_json_mode() -> None:
    captured: dict[str, Any] = {}

    await _client(_capturing_handler(captured)).complete(
        system="s", user="u", response_schema={"type": "object"}
    )

    # Default is json_object: JSON mode, no schema embedded (Bedrock-safe).
    assert captured["body"]["response_format"] == {"type": "json_object"}


async def test_json_schema_mode_sends_the_full_schema() -> None:
    captured: dict[str, Any] = {}

    await _client(_capturing_handler(captured), structured_output="json_schema").complete(
        system="s", user="u", response_schema={"type": "object"}
    )

    assert captured["body"]["response_format"]["type"] == "json_schema"
    assert captured["body"]["response_format"]["json_schema"]["schema"] == {"type": "object"}


async def test_none_mode_omits_response_format() -> None:
    captured: dict[str, Any] = {}

    await _client(_capturing_handler(captured), structured_output="none").complete(
        system="s", user="u", response_schema={"type": "object"}
    )

    assert "response_format" not in captured["body"]


async def test_a_non_2xx_becomes_a_safe_llm_unavailable_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal upstream stacktrace SECRET")

    with pytest.raises(LlmUnavailableError) as exc:
        await _client(handler).complete(system="s", user="u")

    assert "SECRET" not in exc.value.detail


async def test_a_4xx_logs_the_upstream_body_but_never_returns_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text='{"error":"response_format not supported"}')

    with caplog.at_level("WARNING"), pytest.raises(LlmUnavailableError) as exc:
        await _client(handler).complete(system="s", user="u", response_schema={"type": "object"})

    # The upstream reason is logged server-side for diagnosis ...
    assert "response_format not supported" in caplog.text
    # ... but never leaks into the client-facing error detail.
    assert "response_format not supported" not in exc.value.detail


async def test_missing_content_becomes_llm_unavailable_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    with pytest.raises(LlmUnavailableError):
        await _client(handler).complete(system="s", user="u")
