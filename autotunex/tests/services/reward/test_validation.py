# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Tests for the reward validation service (fake executor, no real subprocess)."""

from __future__ import annotations

from typing import Any

from autotunex.models.reward import (
    RewardCaseResult,
    RewardTestResult,
    RewardValidationRequest,
)
from autotunex.services.reward.validation import RewardValidationService

_GOOD = "def compute_score(data_source, solution_str, **kw):\n    return 1.0\n"


class _FakeExecutor:
    def __init__(self, result: RewardTestResult) -> None:
        self._result = result

    async def execute(
        self, *, code: str, function_name: str, test_cases: list[dict[str, Any]]
    ) -> RewardTestResult:
        return self._result


def _service(result: RewardTestResult | None = None) -> RewardValidationService:
    return _service_with(result or RewardTestResult(executed=True))


def _service_with(result: RewardTestResult) -> RewardValidationService:
    return RewardValidationService(executor=_FakeExecutor(result))


async def test_valid_code_without_execution_succeeds() -> None:
    response = await _service().validate(RewardValidationRequest(code=_GOOD, test_execution=False))

    assert response.success is True
    assert response.validation.syntax_valid is True
    assert response.test_result is None


async def test_empty_code_fails() -> None:
    response = await _service().validate(RewardValidationRequest(code="   "))

    assert response.success is False
    assert response.syntax_errors == ["Code cannot be empty"]


async def test_blocked_import_fails_security() -> None:
    response = await _service().validate(
        RewardValidationRequest(code="import os\ndef compute_score(a, b):\n    return 1.0\n")
    )

    assert response.success is False
    assert response.validation.security_valid is False
    assert response.security_issues


async def test_execution_success_flows_case_results_through() -> None:
    result = RewardTestResult(
        executed=True,
        results=[RewardCaseResult(case=1, inputs={"solution_str": "a"}, return_value=1.0)],
    )
    response = await _service_with(result).validate(
        RewardValidationRequest(code=_GOOD, test_execution=True, test_inputs={"solution_str": "a"})
    )

    assert response.success is True
    assert response.test_result is not None
    assert response.test_result.results[0].return_value == 1.0


async def test_per_case_error_flips_success() -> None:
    result = RewardTestResult(
        executed=True,
        results=[RewardCaseResult(case=1, inputs={}, error="ZeroDivisionError: division by zero")],
    )
    response = await _service_with(result).validate(
        RewardValidationRequest(
            code=_GOOD, test_execution=True, test_inputs=[{"solution_str": "a"}]
        )
    )

    assert response.success is False


async def test_execution_skipped_when_static_checks_fail() -> None:
    response = await _service().validate(
        RewardValidationRequest(code="import os\n", test_execution=True)
    )

    assert response.test_result is not None
    assert response.test_result.executed is False
    assert response.test_result.error == "Cannot execute: validation failed"
