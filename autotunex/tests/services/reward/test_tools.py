# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Tests for the generate-test-solutions service (fake LLM)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from autotunex.core.exceptions import LlmNotConfiguredError
from autotunex.models.reward import GenerateTestSolutionsRequest
from autotunex.services.llm.base import ChatDelta
from autotunex.services.reward.tools import RewardToolsService


class _FakeLlm:
    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail

    async def complete(
        self, *, system: str, user: str, response_schema: dict[str, Any] | None = None
    ) -> str:
        if self._fail:
            raise RuntimeError("boom")
        return f"answer to: {user}"

    def stream_chat(
        self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[ChatDelta]:  # pragma: no cover
        raise NotImplementedError


async def test_generates_one_solution_per_prompt() -> None:
    service = RewardToolsService(llm=_FakeLlm())

    response = await service.generate_test_solutions(
        GenerateTestSolutionsRequest(
            prompts=[
                [{"role": "user", "content": "q1"}],
                [{"role": "user", "content": "q2"}],
            ]
        )
    )

    assert len(response.solutions) == 2
    assert all(s for s in response.solutions)


async def test_failed_prompt_yields_empty_string() -> None:
    service = RewardToolsService(llm=_FakeLlm(fail=True))

    response = await service.generate_test_solutions(
        GenerateTestSolutionsRequest(prompts=[[{"role": "user", "content": "q1"}]])
    )

    assert response.solutions == [""]


async def test_raises_503_when_llm_unconfigured() -> None:
    service = RewardToolsService(llm=None)

    with pytest.raises(LlmNotConfiguredError):
        await service.generate_test_solutions(
            GenerateTestSolutionsRequest(prompts=[[{"role": "user", "content": "q"}]])
        )
