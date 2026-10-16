# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""The reward-function validation service.

Runs three in-process static phases (syntax, security, signature) and, only
when requested and all static phases pass, delegates execution to the
:class:`RewardExecutor` seam. A failing check is a normal ``success=false``
result (HTTP 200), never an exception.
"""

from __future__ import annotations

from typing import Any

from autotunex.models.reward import (
    RewardTestResult,
    RewardValidationChecks,
    RewardValidationRequest,
    RewardValidationResponse,
)
from autotunex.services.reward.constants import MAX_CODE_BYTES
from autotunex.services.reward.protocols import RewardExecutor
from autotunex.services.reward.static_analysis import (
    analyze_security,
    check_syntax,
    validate_function,
)

_DEFAULT_TEST_CASE = {
    "data_source": "Solve: What is the capital of France?",
    "solution_str": "The capital of France is Paris.",
}


class RewardValidationService:
    """Validate (and optionally sandbox-run) a user reward function."""

    def __init__(self, *, executor: RewardExecutor) -> None:
        self._executor = executor

    async def validate(self, request: RewardValidationRequest) -> RewardValidationResponse:
        """Return the full validation result; always a 200-shaped body."""
        code = request.code or ""
        if not code.strip():
            return self._failure(["Code cannot be empty"])
        if len(code.encode()) > MAX_CODE_BYTES:
            return self._failure(["Code exceeds maximum allowed size (50KB)"])

        syntax_ok, syntax_errors, tree = check_syntax(code)
        if not syntax_ok or tree is None:
            return self._failure(syntax_errors)

        security_issues = analyze_security(tree)
        found, sig_ok, fn_errors = validate_function(tree, request.function_name)
        checks = RewardValidationChecks(
            syntax_valid=True,
            security_valid=not security_issues,
            function_found=found,
            function_signature_valid=sig_ok,
        )
        static_ok = not security_issues and found and sig_ok

        test_result: RewardTestResult | None = None
        if request.test_execution:
            if not static_ok:
                test_result = RewardTestResult(
                    executed=False, error="Cannot execute: validation failed"
                )
            else:
                test_result = await self._executor.execute(
                    code=code,
                    function_name=request.function_name,
                    test_cases=self._test_cases(request),
                )

        success = static_ok and (
            test_result is None
            or (test_result.error is None and all(r.error is None for r in test_result.results))
        )
        return RewardValidationResponse(
            success=success,
            validation=checks,
            security_issues=security_issues,
            syntax_errors=fn_errors,
            test_result=test_result,
        )

    @staticmethod
    def _test_cases(request: RewardValidationRequest) -> list[dict[str, Any]]:
        raw = request.test_inputs
        if raw is None:
            return [dict(_DEFAULT_TEST_CASE)]
        cases = raw if isinstance(raw, list) else [raw]
        return [c.model_dump(exclude_none=False) for c in cases]

    @staticmethod
    def _failure(errors: list[str]) -> RewardValidationResponse:
        return RewardValidationResponse(
            success=False,
            validation=RewardValidationChecks(
                syntax_valid=False,
                security_valid=False,
                function_found=False,
                function_signature_valid=False,
            ),
            syntax_errors=errors,
        )
