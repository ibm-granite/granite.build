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

"""Regression test for a code-review finding: Config.chat_enabled
reimplemented, inline, the exact same "is Anthropic or OpenAI-compatible
configured" check that tool_loop_backend.py separately defined as
_has_anthropic_config()/_has_openai_compat_config() — two independently
maintained copies that could drift (e.g. /chat/status reporting enabled=true
while _build_provider() actually fails to construct). Both now call the
same has_anthropic_chat_config()/has_openai_compat_chat_config() functions
in config.py.
"""

from __future__ import annotations

from gb_ui_backend.config import (
    Config,
    has_anthropic_chat_config,
    has_openai_compat_chat_config,
)
from gb_ui_backend.services.chat_agents import tool_loop_backend


class TestChatConfigDedup:
    def test_tool_loop_backend_imports_the_same_functions_config_uses(self):
        """Guards against the two call sites drifting back into separate
        copies — both must be the literal same function object."""
        assert tool_loop_backend._has_anthropic_config is has_anthropic_chat_config
        assert (
            tool_loop_backend._has_openai_compat_config is has_openai_compat_chat_config
        )

    def test_chat_enabled_true_when_anthropic_configured(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        config = Config(_env_file=None)

        assert config.chat_enabled is True

    def test_chat_enabled_true_when_openai_compat_configured(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        config = Config(
            _env_file=None,
            chat_llm_base_url="http://fake",
            chat_llm_api_key="key",
        )

        assert config.chat_enabled is True

    def test_chat_enabled_false_when_neither_configured(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        # This dev environment's own .env ships a local Ollama default for
        # llm_base_url/llm_api_key (resolved_chat_llm_* falls back to them),
        # so those must be explicitly cleared too to genuinely exercise
        # "neither configured".
        config = Config(
            _env_file=None,
            llm_base_url="",
            llm_api_key="",
            chat_llm_base_url="",
            chat_llm_api_key="",
        )

        assert config.chat_enabled is False
