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

"""Regression tests for tool_loop_backend._build_provider's precedence.

Covers a code-review finding: provider selection used to implicitly favor
Anthropic whenever ANTHROPIC_API_KEY/ANTHROPIC_AUTH_TOKEN happened to be set,
even if an OpenAI-compatible endpoint was the one actually configured for
chat. GB_UI_CHAT_PROVIDER now makes the choice explicit and, left unset,
auto-detection favors the OpenAI-compatible config (the self-hosted-first
default) rather than Anthropic.

_build_anthropic_provider/_build_openai_compat_provider are monkeypatched to
sentinel-returning stubs so these tests don't require the optional
`chat-anthropic` extra to be installed.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gb_ui_backend import config as config_module
from gb_ui_backend.services.chat_agents import tool_loop_backend
from gb_ui_backend.services.chat_agents.tool_loop_backend import _build_provider

_ANTHROPIC_SENTINEL = object()
_OPENAI_SENTINEL = object()


@pytest.fixture(autouse=True)
def _stub_provider_constructors(monkeypatch):
    monkeypatch.setattr(
        tool_loop_backend,
        "_build_anthropic_provider",
        lambda config, prompt: _ANTHROPIC_SENTINEL,
    )
    monkeypatch.setattr(
        tool_loop_backend,
        "_build_openai_compat_provider",
        lambda config, prompt: _OPENAI_SENTINEL,
    )


@pytest.fixture(autouse=True)
def _clear_anthropic_env(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)


def _config(**kwargs) -> config_module.Config:
    # llm_base_url/llm_api_key default to "" here (overriding whatever the
    # developer's real .env/shell exports for AI analysis) so
    # resolved_chat_llm_base_url/_api_key reflect only what each test itself
    # configures — chat_llm_base_url/_api_key fall back to these when unset.
    kwargs.setdefault("llm_base_url", "")
    kwargs.setdefault("llm_api_key", "")
    return config_module.Config(_env_file=None, **kwargs)


class TestAutoDetection:
    def test_openai_compat_wins_when_only_it_is_configured(self):
        config = _config(chat_llm_base_url="http://fake", chat_llm_api_key="x")
        assert _build_provider(config, "sys") is _OPENAI_SENTINEL

    def test_anthropic_wins_when_only_it_is_configured(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
        config = _config()
        assert _build_provider(config, "sys") is _ANTHROPIC_SENTINEL

    def test_openai_compat_wins_the_tie_when_both_are_configured(self, monkeypatch):
        """The self-hosted-first default: an operator with ANTHROPIC_API_KEY
        exported for an unrelated reason shouldn't be silently routed to
        Claude over a local endpoint they configured on purpose."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
        config = _config(chat_llm_base_url="http://fake", chat_llm_api_key="x")
        assert _build_provider(config, "sys") is _OPENAI_SENTINEL

    def test_raises_when_neither_is_configured(self):
        config = _config()
        with pytest.raises(RuntimeError, match="not configured"):
            _build_provider(config, "sys")


class TestExplicitSelection:
    def test_explicit_anthropic_wins_even_when_openai_compat_is_also_configured(
        self, monkeypatch
    ):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
        config = _config(
            chat_provider="anthropic",
            chat_llm_base_url="http://fake",
            chat_llm_api_key="x",
        )
        assert _build_provider(config, "sys") is _ANTHROPIC_SENTINEL

    def test_explicit_openai_compat_wins_even_when_anthropic_is_also_configured(
        self, monkeypatch
    ):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
        config = _config(
            chat_provider="openai_compatible",
            chat_llm_base_url="http://fake",
            chat_llm_api_key="x",
        )
        assert _build_provider(config, "sys") is _OPENAI_SENTINEL

    def test_explicit_anthropic_without_credentials_raises(self):
        config = _config(chat_provider="anthropic")
        with pytest.raises(RuntimeError, match="GB_UI_CHAT_PROVIDER=anthropic"):
            _build_provider(config, "sys")

    def test_explicit_openai_compat_without_credentials_raises(self):
        config = _config(chat_provider="openai_compatible")
        with pytest.raises(RuntimeError, match="GB_UI_CHAT_PROVIDER=openai_compatible"):
            _build_provider(config, "sys")

    def test_invalid_chat_provider_value_rejected_at_config_construction(self):
        with pytest.raises(ValidationError, match="GB_UI_CHAT_PROVIDER"):
            _config(chat_provider="bogus")

    def test_blank_chat_provider_falls_back_to_auto_detection(self):
        config = _config(
            chat_provider="", chat_llm_base_url="http://fake", chat_llm_api_key="x"
        )
        assert config.chat_provider is None
        assert _build_provider(config, "sys") is _OPENAI_SENTINEL
