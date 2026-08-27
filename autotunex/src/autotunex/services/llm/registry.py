# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""LLM client selection from settings."""

from __future__ import annotations

import httpx

from autotunex.core.config import Settings
from autotunex.core.exceptions import LlmNotConfiguredError
from autotunex.services.llm.base import LlmClient
from autotunex.services.llm.openai_compatible import OpenAiCompatibleLlmClient


def get_llm_client(settings: Settings, http_client: httpx.AsyncClient) -> LlmClient:
    """Return a configured adapter, or raise if the LLM feature is unset.

    The lazy raise keeps the intelligence endpoints mounted always, returning a
    clean 503 when no provider is configured rather than 404-ing or failing at
    import.

    Raises:
        LlmNotConfiguredError: the ``llm_*`` settings are unset.
    """
    if not settings.llm_configured:
        raise LlmNotConfiguredError()
    base_url, api_key, model = settings.llm_base_url, settings.llm_api_key, settings.llm_model
    # llm_configured guarantees these are set; re-checked to narrow for mypy.
    if base_url is None or api_key is None or model is None:  # pragma: no cover - defensive
        raise LlmNotConfiguredError()
    return OpenAiCompatibleLlmClient(
        http_client=http_client,
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_seconds=settings.llm_timeout_seconds,
        structured_output=settings.llm_structured_output,
    )
