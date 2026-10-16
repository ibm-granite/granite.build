"""Schema tests for the chat API contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from autotunex.models.chat import ChatMessage, ChatRequest, ChatResponse


def test_chat_request_parses_messages_and_defaults() -> None:
    req = ChatRequest.model_validate({"messages": [{"role": "user", "content": "hi"}]})

    assert req.messages[0].role == "user"
    assert req.context == {}
    assert req.thread_id is None


def test_chat_request_requires_messages() -> None:
    with pytest.raises(ValidationError):
        ChatRequest.model_validate({})


def test_chat_request_accepts_context_and_thread_id() -> None:
    req = ChatRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "context": {"job_id": "abc123"},
            "thread_id": "thread-1",
        }
    )

    assert req.context == {"job_id": "abc123"}
    assert req.thread_id == "thread-1"


def test_chat_message_content_defaults_to_none() -> None:
    message = ChatMessage.model_validate({"role": "assistant"})

    assert message.content is None


def test_chat_response_defaults_context_to_empty_dict() -> None:
    response = ChatResponse.model_validate({"output": "done"})

    assert response.output == "done"
    assert response.context == {}
