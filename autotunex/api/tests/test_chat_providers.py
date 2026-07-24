# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

import sys
import types

import pytest


def _install_fake_langchain_openai(monkeypatch):
    """Register a minimal fake langchain_openai so tests need no torch/transformers chain."""
    lo = types.ModuleType("langchain_openai")

    class ChatOpenAI:
        def __init__(
            self,
            *,
            model=None,
            api_key=None,
            base_url=None,
            max_tokens=None,
            temperature=None,
            **kwargs,
        ):
            self.model_name = model
            self.openai_api_base = base_url

    lo.ChatOpenAI = ChatOpenAI
    monkeypatch.setitem(sys.modules, "langchain_openai", lo)


def test_litellm_provider_builds_chatopenai(monkeypatch):
    _install_fake_langchain_openai(monkeypatch)
    from services.chat_providers.litellm_provider import LiteLLMChatProvider

    monkeypatch.setenv("LITELLM_API_KEY", "k")
    monkeypatch.setenv("LITELLM_URL", "https://litellm.example")
    monkeypatch.setenv("LITELLM_MODEL", "aws/claude-sonnet-4-6")
    llm = LiteLLMChatProvider().build_llm()
    assert llm.model_name == "aws/claude-sonnet-4-6"
    assert str(llm.openai_api_base) == "https://litellm.example/v1"


def test_litellm_provider_guard_raises_without_env(monkeypatch):
    # The guard fires before the `from langchain_openai import` line,
    # so the fake is not strictly required here, but install it anyway
    # for consistency and to ensure the import doesn't blow up if reached.
    _install_fake_langchain_openai(monkeypatch)
    from services.chat_providers.litellm_provider import LiteLLMChatProvider

    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    monkeypatch.delenv("LITELLM_URL", raising=False)
    with pytest.raises(ValueError) as ei:
        LiteLLMChatProvider().build_llm()
    assert "LITELLM_API_KEY and LITELLM_URL" in str(ei.value)


def test_openai_compat_provider_builds_chatopenai(monkeypatch):
    _install_fake_langchain_openai(monkeypatch)
    from services.chat_providers.openai_compat_provider import OpenAICompatChatProvider

    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("OPENAI_MODEL", "llama3.1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    llm = OpenAICompatChatProvider().build_llm()
    assert llm.model_name == "llama3.1"
    assert str(llm.openai_api_base) == "http://localhost:11434/v1"


def test_openai_compat_provider_default_model(monkeypatch):
    _install_fake_langchain_openai(monkeypatch)
    from services.chat_providers.openai_compat_provider import OpenAICompatChatProvider

    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    llm = OpenAICompatChatProvider().build_llm()
    assert llm.model_name == "gpt-4o-mini"
