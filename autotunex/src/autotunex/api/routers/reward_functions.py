# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Reward-function validation endpoint (online-RL step).

Static-checks and optionally sandbox-runs a user-supplied Python reward
function. A failing check is a normal ``success=false`` result (HTTP 200); the
declared 503 covers a sandbox or (future) upstream failure.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Any

from fastapi import APIRouter

from autotunex.api.deps import RewardValidationServiceDep
from autotunex.models.common import ProblemDetail
from autotunex.models.reward import RewardValidationRequest, RewardValidationResponse

router = APIRouter(prefix="/reward-functions", tags=["reward-functions"])

_PROBLEM_RESPONSE = {"model": ProblemDetail, "content": {"application/problem+json": {}}}
_RESPONSES: dict[int | str, dict[str, Any]] = {
    HTTPStatus.UNAUTHORIZED: _PROBLEM_RESPONSE,
    HTTPStatus.BAD_REQUEST: _PROBLEM_RESPONSE,
    HTTPStatus.SERVICE_UNAVAILABLE: _PROBLEM_RESPONSE,
}


@router.post("/validate", summary="Validate a reward function", responses=_RESPONSES)
async def validate_reward_function(
    body: RewardValidationRequest, service: RewardValidationServiceDep
) -> RewardValidationResponse:
    """Static-check and optionally sandbox-execute a reward function."""
    return await service.validate(body)
