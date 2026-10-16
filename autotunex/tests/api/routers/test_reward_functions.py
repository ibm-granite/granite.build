# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""``POST /reward-functions/validate``, end to end over HTTP.

A fake :class:`RewardExecutor` stands in for the real subprocess sandbox so
these tests are fast and deterministic; the sandbox itself is covered by
``tests/services/reward/test_subprocess_executor.py``.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from httpx import AsyncClient

from autotunex.api.deps import get_reward_executor
from autotunex.models.reward import RewardCaseResult, RewardTestResult
from tests.conftest import API

PROBLEM_JSON = "application/problem+json"

_GOOD_CODE = "def compute_score(data_source, solution_str, **kw):\n    return 1.0\n"


class _FakeRewardExecutor:
    """A canned :class:`RewardTestResult`, regardless of what is executed."""

    def __init__(self, result: RewardTestResult) -> None:
        self._result = result

    async def execute(
        self, *, code: str, function_name: str, test_cases: list[dict[str, Any]]
    ) -> RewardTestResult:
        return self._result


def _use_executor(app: FastAPI, result: RewardTestResult) -> None:
    app.dependency_overrides[get_reward_executor] = lambda: _FakeRewardExecutor(result)


async def test_valid_code_without_execution_succeeds(client: AsyncClient) -> None:
    response = await client.post(
        f"{API}/reward-functions/validate",
        json={"code": _GOOD_CODE, "test_execution": False},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["validation"]["syntax_valid"] is True
    assert body["test_result"] is None


async def test_blocked_import_fails_security(client: AsyncClient) -> None:
    response = await client.post(
        f"{API}/reward-functions/validate",
        json={"code": "import os\ndef compute_score(a, b):\n    return 1.0\n"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert body["validation"]["security_valid"] is False
    assert body["security_issues"]


async def test_empty_code_fails(client: AsyncClient) -> None:
    response = await client.post(f"{API}/reward-functions/validate", json={"code": "   "})

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert body["syntax_errors"] == ["Code cannot be empty"]


async def test_execution_returns_the_fake_executors_case_result(
    app: FastAPI, client: AsyncClient
) -> None:
    _use_executor(
        app,
        RewardTestResult(
            executed=True,
            results=[RewardCaseResult(case=1, inputs={"solution_str": "a"}, return_value=1.0)],
        ),
    )

    response = await client.post(
        f"{API}/reward-functions/validate",
        json={
            "code": _GOOD_CODE,
            "test_execution": True,
            "test_inputs": {"solution_str": "a"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["test_result"]["executed"] is True
    assert body["test_result"]["results"][0]["return_value"] == 1.0


async def test_a_per_case_error_flips_success_to_false(app: FastAPI, client: AsyncClient) -> None:
    _use_executor(
        app,
        RewardTestResult(
            executed=True,
            results=[
                RewardCaseResult(case=1, inputs={}, error="ZeroDivisionError: division by zero")
            ],
        ),
    )

    response = await client.post(
        f"{API}/reward-functions/validate",
        json={"code": _GOOD_CODE, "test_execution": True, "test_inputs": [{"solution_str": "a"}]},
    )

    assert response.status_code == 200
    assert response.json()["success"] is False


async def test_extra_field_is_rejected_as_bad_request(client: AsyncClient) -> None:
    response = await client.post(
        f"{API}/reward-functions/validate",
        json={"code": _GOOD_CODE, "unexpected_field": "x"},
    )

    assert response.status_code == 422
