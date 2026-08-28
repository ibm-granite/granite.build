"""The OpenAI-compatible streaming tool-calling adapter against a mocked transport."""

from __future__ import annotations

import httpx
import pytest
from pydantic import SecretStr

from autotunex.services.llm.openai_compatible import OpenAiCompatibleLlmClient


def _sse(*lines: str) -> bytes:
    return ("".join(f"data: {line}\n\n" for line in lines)).encode()


async def test_stream_chat_yields_text_then_finish() -> None:
    body = _sse(
        '{"choices":[{"delta":{"content":"Hel"},"finish_reason":null}]}',
        '{"choices":[{"delta":{"content":"lo"},"finish_reason":null}]}',
        '{"choices":[{"delta":{},"finish_reason":"stop"}]}',
        "[DONE]",
    )
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=body))

    async with httpx.AsyncClient(transport=transport) as http:
        client = OpenAiCompatibleLlmClient(
            http_client=http,
            base_url="https://gw/v1",
            api_key=SecretStr("k"),
            model="m",
            timeout_seconds=5.0,
        )
        deltas = [
            d
            async for d in client.stream_chat(
                messages=[{"role": "user", "content": "hi"}], tools=[]
            )
        ]

    text = "".join(d.content or "" for d in deltas)
    assert text == "Hello"
    assert deltas[-1].finish_reason == "stop"


async def test_stream_chat_accumulates_tool_call_fragments() -> None:
    body = _sse(
        '{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"list_jobs","arguments":""}}]},"finish_reason":null}]}',
        '{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]},"finish_reason":null}]}',
        '{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
        "[DONE]",
    )
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=body))

    async with httpx.AsyncClient(transport=transport) as http:
        client = OpenAiCompatibleLlmClient(
            http_client=http,
            base_url="https://gw/v1",
            api_key=SecretStr("k"),
            model="m",
            timeout_seconds=5.0,
        )
        fragments = [
            tc
            for d in [x async for x in client.stream_chat(messages=[], tools=[])]
            if d.tool_calls
            for tc in d.tool_calls
        ]

    assert fragments[0].name == "list_jobs"
    assert "".join(f.arguments or "" for f in fragments) == "{}"


async def test_stream_chat_maps_http_error_to_unavailable() -> None:
    from autotunex.core.exceptions import LlmUnavailableError

    transport = httpx.MockTransport(lambda request: httpx.Response(500, text="boom"))
    async with httpx.AsyncClient(transport=transport) as http:
        client = OpenAiCompatibleLlmClient(
            http_client=http,
            base_url="https://gw/v1",
            api_key=SecretStr("k"),
            model="m",
            timeout_seconds=5.0,
        )
        with pytest.raises(LlmUnavailableError):
            _ = [d async for d in client.stream_chat(messages=[], tools=[])]
