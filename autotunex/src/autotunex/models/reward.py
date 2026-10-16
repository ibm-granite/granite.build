# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Schemas for the online-RL reward step.

Two endpoints share this module because they are two halves of one wizard step:
``POST /reward-functions/validate`` (static-check + optionally sandbox-run a
user reward function) and ``POST /jobs/generate-test-solutions`` (LLM-generate
sample answers that seed the reward test cases). Field names match exactly what
``StepRewardFunction.svelte`` reads.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

ChatMessage = dict[str, str]
"""One OpenAI-style chat message: ``{"role": ..., "content": ...}``."""


class RewardTestCase(BaseModel):
    """One test case a reward function is scored against.

    ``extra='allow'`` because the frontend attaches arbitrary keys (beyond the
    canonical four) that are forwarded as kwargs to the reward function.
    """

    model_config = ConfigDict(extra="allow")

    data_source: str | None = None
    solution_str: str | None = None
    ground_truth: Any | None = None
    extra_info: dict[str, Any] | None = None


class RewardValidationRequest(BaseModel):
    """Request body for ``POST /reward-functions/validate``."""

    model_config = ConfigDict(extra="forbid")

    code: str
    function_name: str = "compute_score"
    test_execution: bool = False
    test_inputs: RewardTestCase | list[RewardTestCase] | None = None


class RewardValidationChecks(BaseModel):
    """The four static-analysis booleans the UI renders as status pills."""

    syntax_valid: bool
    security_valid: bool
    function_found: bool
    function_signature_valid: bool


class RewardCaseResult(BaseModel):
    """The outcome of running the reward function against one test case."""

    case: int
    inputs: dict[str, Any]
    return_value: Any | None = None
    return_type: str | None = None
    error: str | None = None


class RewardTestResult(BaseModel):
    """The execution phase's result; ``None`` on the response when not run."""

    executed: bool
    results: list[RewardCaseResult] = Field(default_factory=list)
    stdout: str = ""
    error: str | None = None
    execution_time_ms: float | None = None


class RewardValidationResponse(BaseModel):
    """Response body for ``POST /reward-functions/validate`` (always HTTP 200)."""

    success: bool
    validation: RewardValidationChecks
    security_issues: list[str] = Field(default_factory=list)
    syntax_errors: list[str] = Field(default_factory=list)
    test_result: RewardTestResult | None = None


class GenerateTestSolutionsRequest(BaseModel):
    """Request body for ``POST /jobs/generate-test-solutions``."""

    prompts: list[list[ChatMessage]] = Field(
        description="Each prompt is a chat-message array (a VERL prompt)."
    )


class GenerateTestSolutionsResponse(BaseModel):
    """Response: one solution string per input prompt, index-aligned."""

    solutions: list[str]
