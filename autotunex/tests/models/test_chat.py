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
