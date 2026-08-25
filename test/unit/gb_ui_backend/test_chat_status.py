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

"""Regression test for a code-review finding: chat_status() used to report
enabled=True even when the backend couldn't actually be constructed (e.g. an
API key env var is set but the matching package extra isn't installed) —
letting the frontend render the chat widget and then have every real
/chat/stream call 500, instead of the status check simply reporting the
real, already-checked-here outcome.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gb_ui_backend.api import chat as chat_module


@pytest.mark.asyncio
class TestChatStatus:
    async def test_disabled_when_config_says_not_configured(self, monkeypatch):
        monkeypatch.setattr(
            chat_module, "get_config", lambda: SimpleNamespace(chat_enabled=False)
        )

        result = await chat_module.chat_status()

        assert result.enabled is False
        assert result.backend is None

    async def test_enabled_when_backend_constructs_successfully(self, monkeypatch):
        monkeypatch.setattr(
            chat_module, "get_config", lambda: SimpleNamespace(chat_enabled=True)
        )
        backend = SimpleNamespace(
            describe=lambda: {
                "backend": "tool_loop",
                "provider": "anthropic",
                "model": "claude-sonnet-5",
            }
        )
        monkeypatch.setattr(chat_module, "get_backend", lambda: backend)

        result = await chat_module.chat_status()

        assert result.enabled is True
        assert result.backend == "tool_loop"
        assert result.model == "claude-sonnet-5"

    async def test_disabled_when_configured_but_backend_construction_fails(
        self, monkeypatch
    ):
        """Config says chat is configured, but constructing the actual
        backend raises (e.g. ANTHROPIC_API_KEY is set but `anthropic` isn't
        installed). Must report enabled=False, not True — reporting True
        here used to let the frontend render the widget and then have every
        real /chat/stream call 500 instead of the widget simply not
        appearing."""
        monkeypatch.setattr(
            chat_module, "get_config", lambda: SimpleNamespace(chat_enabled=True)
        )

        def _raise_backend():
            raise RuntimeError("anthropic is not installed")

        monkeypatch.setattr(chat_module, "get_backend", _raise_backend)

        result = await chat_module.chat_status()

        assert result.enabled is False
        assert result.backend is None
