"""Chat endpoints, end to end over HTTP.

``POST /chat`` and ``POST /chat/stream`` both delegate to ``ChatService`` (see
``services/chat/service.py``); these tests fake that service entirely via
``app.dependency_overrides[get_chat_service]`` so no real LLM or outbound HTTP
client is needed. The 503 case is the exception: it deliberately does NOT
override the service, so ``get_chat_service`` itself raises
``LlmNotConfiguredError`` from unconfigured default test settings, proving the
router never needs ``app.state.http_client`` (never populated under
``ASGITransport``) to reach that response.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from autotunex.api.deps import get_chat_service, get_session
from autotunex.core.config import get_settings
from autotunex.main import create_app
from autotunex.models.auth import Principal
from autotunex.models.chat import ChatMessage, ChatResponse
from tests.conftest import API, make_settings

PROBLEM_JSON = "application/problem+json"


class FakeChatService:
    """A scripted double standing in for ``ChatService`` in these tests."""

    def __init__(self, *, output: str, stream_events: list[dict[str, Any]]) -> None:
        self._output = output
        self._stream_events = stream_events

    async def chat(
        self, *, messages: list[ChatMessage], principal: Principal, thread_id: str | None
    ) -> ChatResponse:
        return ChatResponse(output=self._output, context={})

    async def chat_stream(
        self, *, messages: list[ChatMessage], principal: Principal, thread_id: str | None
    ) -> AsyncIterator[dict[str, Any]]:
        for event in self._stream_events:
            yield event


def _use_service(app: FastAPI, service: FakeChatService) -> None:
    app.dependency_overrides[get_chat_service] = lambda: service


async def test_chat_endpoint_returns_the_assistant_response(
    app: FastAPI, client: AsyncClient
) -> None:
    _use_service(app, FakeChatService(output="hello there", stream_events=[]))

    response = await client.post(
        f"{API}/chat", json={"messages": [{"role": "user", "content": "hi"}]}
    )

    assert response.status_code == HTTPStatus.OK
    body = response.json()
    assert body["output"] == "hello there"
    assert body["context"] == {}


async def test_chat_stream_endpoint_returns_sse_events(app: FastAPI, client: AsyncClient) -> None:
    _use_service(
        app,
        FakeChatService(
            output="unused",
            stream_events=[
                {"type": "token", "text": "hel"},
                {"type": "token", "text": "lo"},
                {"type": "done"},
            ],
        ),
    )

    async with client.stream(
        "POST",
        f"{API}/chat/stream",
        json={"messages": [{"role": "user", "content": "hi"}], "context": {"a": 1}},
    ) as response:
        assert response.status_code == HTTPStatus.OK
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join([chunk async for chunk in response.aiter_text()])

    assert '"type": "token"' in body
    assert '"type": "done"' in body
    # The trailing context frame carries the caller's own context back, plus
    # the last user message's text.
    assert '"type": "context"' in body
    assert '"last_input": "hi"' in body
    assert '"a": 1' in body


async def test_chat_is_401_when_unauthenticated(session: AsyncSession) -> None:
    # A real credential-checking provider. The request below presents no
    # credential at all, so which key is registered is irrelevant — nothing
    # is offered for the Authenticator to verify.
    settings = make_settings(auth_providers=["api_key"], api_keys={"a" * 64: "someone@example.com"})
    unauthenticated_app = create_app(settings)
    unauthenticated_app.dependency_overrides[get_settings] = lambda: settings
    unauthenticated_app.dependency_overrides[get_session] = lambda: session

    async with AsyncClient(
        transport=ASGITransport(app=unauthenticated_app), base_url="http://testserver"
    ) as unauthenticated_client:
        response = await unauthenticated_client.post(
            f"{API}/chat", json={"messages": [{"role": "user", "content": "hi"}]}
        )

    assert response.status_code == HTTPStatus.UNAUTHORIZED
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


async def test_chat_is_503_when_the_llm_is_unconfigured(client: AsyncClient) -> None:
    # No override of get_chat_service: default test settings leave the LLM
    # unconfigured, so get_chat_service itself raises before ever touching
    # app.state.http_client (unset under ASGITransport — lifespan never runs).
    response = await client.post(
        f"{API}/chat", json={"messages": [{"role": "user", "content": "hi"}]}
    )

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


async def test_chat_stream_is_503_when_the_llm_is_unconfigured(client: AsyncClient) -> None:
    response = await client.post(
        f"{API}/chat/stream", json={"messages": [{"role": "user", "content": "hi"}]}
    )

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert response.headers["content-type"].startswith(PROBLEM_JSON)
