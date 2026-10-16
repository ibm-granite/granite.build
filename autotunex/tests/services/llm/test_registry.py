"""The LLM client registry: configured -> adapter, unset -> 503."""

from __future__ import annotations

import httpx
import pytest

from autotunex.core.exceptions import LlmNotConfiguredError
from autotunex.services.llm.openai_compatible import OpenAiCompatibleLlmClient
from autotunex.services.llm.registry import get_llm_client
from tests.conftest import make_settings


def test_returns_an_adapter_when_configured() -> None:
    settings = make_settings(llm_base_url="https://gw.example/v1", llm_api_key="sk", llm_model="m")

    client = get_llm_client(settings, httpx.AsyncClient())

    assert isinstance(client, OpenAiCompatibleLlmClient)


def test_raises_503_when_unconfigured() -> None:
    with pytest.raises(LlmNotConfiguredError):
        get_llm_client(make_settings(), httpx.AsyncClient())
