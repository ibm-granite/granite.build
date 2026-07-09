"""
OpenAI-compatible LLM client with model fallback chain.

Works with any base_url:
  - RITS-style endpoints (base_url containing "rits.fmaas"):
      Auth:  RITS_API_KEY header
      URL:   <base_url>/<model-slug>/v1/chat/completions
      Model spec: "slug:full/model-name"  (e.g. "granite-4-h-small:ibm-granite/granite-4.0-h-small")
  - Ollama: http://localhost:11434/v1
  - OpenAI: https://api.openai.com/v1
  - Any other OpenAI-compatible endpoint
      Auth:  Authorization: Bearer <api_key>
      URL:   <base_url>/v1/chat/completions
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Optional

import httpx

logger = logging.getLogger(__name__)


def _parse_model_spec(spec: str) -> tuple[str, str]:
    """Return (url_slug, model_name) from a model spec string.

    RITS format:  "granite-4-h-small:ibm-granite/granite-4.0-h-small"
    Plain format: "granite-3.3-8b-instruct"  -> slug derived automatically
    """
    if ":" in spec:
        slug, name = spec.split(":", 1)
        return slug.strip(), name.strip()
    name = spec.strip()
    slug = name.split("/")[-1].replace(".", "-")
    return slug, name


class LLMClient:
    """Async OpenAI-compatible chat completions client with model fallback."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        models: list[str],
        timeout: int = 60,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.models = models
        self.timeout = timeout
        self._is_rits = "rits.fmaas" in self.base_url

    def _headers(self) -> dict[str, str]:
        if self._is_rits:
            return {"RITS_API_KEY": self.api_key, "Content-Type": "application/json"}
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _url(self, model_spec: str) -> tuple[str, str]:
        """Return (endpoint_url, model_name) for the given model spec."""
        slug, name = _parse_model_spec(model_spec)
        if self._is_rits:
            return f"{self.base_url}/{slug}/v1/chat/completions", name
        return f"{self.base_url}/v1/chat/completions", name

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        tools: Optional[list[dict]] = None,
    ) -> dict[str, Any]:
        """Try each model in order, returning the first successful response."""
        models_to_try = [model] if model else self.models
        last_exc: Exception = RuntimeError("No models configured")

        for m in models_to_try:
            try:
                return await self._call(m, messages, temperature, max_tokens, tools)
            except Exception as e:
                logger.warning("LLM: model %s failed: %s", m, e)
                last_exc = e

        raise last_exc

    async def _call(
        self,
        model_spec: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        tools: Optional[list[dict]],
    ) -> dict[str, Any]:
        url, model_name = self._url(model_spec)
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, headers=self._headers(), json=payload)
            resp.raise_for_status()
            return resp.json()

    async def chat_completion_stream(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """Yield text chunks from a streaming chat completion."""
        m = model or (self.models[0] if self.models else "default")
        url, model_name = self._url(m)
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                url,
                headers=self._headers(),
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except Exception:
                        continue
