# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""An OpenAI-compatible ``/chat/completions`` adapter over shared httpx.

Deliberately dumb: builds the request and returns the model's text. All JSON
extraction, Pydantic validation, and retries live in
:class:`~autotunex.services.dataset_intelligence.DatasetIntelligenceService`, so
exactly one place understands the JSON contract.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
from pydantic import SecretStr

from autotunex.core.exceptions import LlmUnavailableError
from autotunex.core.logging import get_logger
from autotunex.services.llm.base import ChatDelta, ToolCallDelta

logger = get_logger(__name__)


class OpenAiCompatibleLlmClient:
    """Speaks the OpenAI ``/chat/completions`` contract to any compatible gateway."""

    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient,
        base_url: str,
        api_key: SecretStr,
        model: str,
        timeout_seconds: float,
        structured_output: str = "json_object",
    ) -> None:
        self._http = http_client
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout_seconds
        self._structured_output = structured_output

    async def complete(
        self, *, system: str, user: str, response_schema: dict[str, Any] | None = None
    ) -> str:
        """POST one completion; return ``choices[0].message.content``.

        Any failure — timeout, connection error, non-2xx, unparseable body, or
        missing content — is mapped to :class:`LlmUnavailableError` with a safe
        message. The raw upstream detail is logged at ``warning``, never
        returned. The API key is unwrapped only here, for the header.

        Raises:
            LlmUnavailableError: the call failed or the response was unusable.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if response_schema is not None and self._structured_output != "none":
            # Gateways diverge on structured output (see Settings.llm_structured_output):
            # Bedrock's Claude rejects a full json_schema (unsupported keywords -> 400),
            # so json_object (schema-less JSON mode) is the portable default. The service
            # validates the parsed JSON regardless, so this is only a hint.
            if self._structured_output == "json_schema":
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "response", "schema": response_schema},
                }
            else:
                payload["response_format"] = {"type": "json_object"}
        try:
            response = await self._http.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key.get_secret_value()}"},
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()
            return str(data["choices"][0]["message"]["content"])
        except httpx.HTTPStatusError as exc:
            # A non-2xx carries the real reason in its body (e.g. litellm rejecting
            # an unsupported ``response_format``, or a context-length error). Log it
            # at ``warning`` — server-side only, never returned — bounded in length.
            logger.warning(
                "LLM request failed: %s | upstream response: %s",
                exc,
                exc.response.text[:2000],
            )
            raise LlmUnavailableError() from exc
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            logger.warning("LLM request failed: %s", exc)
            raise LlmUnavailableError() from exc

    async def stream_chat(
        self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[ChatDelta]:
        """Stream one assistant turn over SSE, yielding text and tool-call fragments.

        ``tools`` is only attached to the request (with ``tool_choice: "auto"``)
        when non-empty, so a tool-less caller does not pay for gateways that
        reject an empty ``tools`` array. Malformed SSE lines are skipped rather
        than raised, matching the leniency an upstream gateway's own quirks
        require; a failed connection or non-2xx status is mapped to
        :class:`LlmUnavailableError`, as :meth:`complete` does.

        Raises:
            LlmUnavailableError: the call failed or the connection was unusable.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        try:
            async with self._http.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key.get_secret_value()}"},
                timeout=self._timeout,
            ) as response:
                if response.is_error:
                    # The body is only readable once the stream context has read
                    # it; without this, raise_for_status()'s message (and our log
                    # line below) would report an empty body for a non-2xx reply.
                    await response.aread()
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if not data or data == "[DONE]":
                        continue
                    delta = self._parse_chunk(data)
                    if delta is not None:
                        yield delta
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "LLM stream failed: %s | upstream response: %s",
                exc,
                exc.response.text[:2000],
            )
            raise LlmUnavailableError() from exc
        except httpx.HTTPError as exc:
            logger.warning("LLM stream failed: %s", exc)
            raise LlmUnavailableError() from exc

    @staticmethod
    def _parse_chunk(data: str) -> ChatDelta | None:
        """Parse one SSE ``data:`` JSON chunk into a :class:`ChatDelta`.

        Returns ``None`` for a chunk that is not valid JSON or does not carry a
        recognizable ``choices[0]`` — skipped rather than raised, since one
        stray line should not abort an otherwise-good stream.
        """
        try:
            obj = json.loads(data)
            choice = obj["choices"][0]
        except (json.JSONDecodeError, KeyError, IndexError):
            return None
        delta = choice.get("delta") or {}
        tool_calls: list[ToolCallDelta] | None = None
        raw_calls = delta.get("tool_calls")
        if raw_calls:
            tool_calls = [
                ToolCallDelta(
                    index=tc.get("index", 0),
                    id=tc.get("id"),
                    name=(tc.get("function") or {}).get("name"),
                    arguments=(tc.get("function") or {}).get("arguments"),
                )
                for tc in raw_calls
            ]
        return ChatDelta(
            content=delta.get("content"),
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason"),
        )
