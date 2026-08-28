# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Reward-step LLM tools: generate sample solutions for reward test cases.

Sends each dataset prompt to the LLM seam to produce a plausible model answer,
so the user can preview how their reward function scores realistic outputs. A
prompt whose completion fails degrades to an empty string (index-aligned), and
the whole feature returns 503 when no LLM is configured.
"""

from __future__ import annotations

import asyncio

from autotunex.core.exceptions import LlmNotConfiguredError
from autotunex.models.reward import (
    ChatMessage,
    GenerateTestSolutionsRequest,
    GenerateTestSolutionsResponse,
)
from autotunex.services.llm.base import LlmClient


class RewardToolsService:
    """Generate sample reward-test solutions via the LLM seam."""

    def __init__(self, *, llm: LlmClient | None, max_concurrency: int = 5) -> None:
        self._llm = llm
        self._sem = asyncio.Semaphore(max_concurrency)

    def _client(self) -> LlmClient:
        if self._llm is None:
            raise LlmNotConfiguredError()
        return self._llm

    async def generate_test_solutions(
        self, request: GenerateTestSolutionsRequest
    ) -> GenerateTestSolutionsResponse:
        """Return one solution string per prompt (``""`` for a failed prompt)."""
        client = self._client()
        solutions = await asyncio.gather(*(self._one(client, prompt) for prompt in request.prompts))
        return GenerateTestSolutionsResponse(solutions=list(solutions))

    async def _one(self, client: LlmClient, prompt: list[ChatMessage]) -> str:
        system = "\n".join(m.get("content", "") for m in prompt if m.get("role") == "system")
        user = "\n".join(m.get("content", "") for m in prompt if m.get("role") != "system")
        async with self._sem:
            try:
                return await client.complete(system=system, user=user)
            except Exception:  # a failed prompt is data, not an error
                return ""
