# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""LLM-backed dataset-intelligence endpoints.

Stateless helpers that suggest a parsing strategy or column mapping for a
client-supplied sample, and validate a strategy with no LLM call. Every body is
one or two lines: parse, delegate, serialize. Mounted at
``/datasets/intelligence``; the router-level ``get_principal`` dependency (see
``main.py``) 401s unauthenticated callers before the service runs.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Any

from fastapi import APIRouter

from autotunex.api.deps import DatasetIntelligenceServiceDep
from autotunex.models.common import ProblemDetail
from autotunex.models.dataset_intelligence import (
    ColumnMappingSuggestion,
    ParseStrategyRequest,
    ParsingStrategy,
    SuggestMappingRequest,
    ValidateStrategyRequest,
    ValidationResult,
)

router = APIRouter(prefix="/datasets/intelligence", tags=["datasets"])

_PROBLEM_RESPONSE = {"model": ProblemDetail, "content": {"application/problem+json": {}}}
_AUTH_RESPONSES: dict[int | str, dict[str, Any]] = {
    HTTPStatus.UNAUTHORIZED: _PROBLEM_RESPONSE,
    HTTPStatus.BAD_REQUEST: _PROBLEM_RESPONSE,
}
_LLM_RESPONSES: dict[int | str, dict[str, Any]] = {
    HTTPStatus.UNPROCESSABLE_ENTITY: _PROBLEM_RESPONSE,
    HTTPStatus.BAD_GATEWAY: _PROBLEM_RESPONSE,
    HTTPStatus.SERVICE_UNAVAILABLE: _PROBLEM_RESPONSE,
    **_AUTH_RESPONSES,
}


@router.post("/parse-strategy", summary="Suggest a parsing strategy", responses=_LLM_RESPONSES)
async def parse_strategy(
    body: ParseStrategyRequest, service: DatasetIntelligenceServiceDep
) -> ParsingStrategy:
    """Suggest how to turn a raw sample into ``{input, output}`` training pairs."""
    return await service.generate_parsing_strategy(
        sample=body.sample, data_format=body.data_format, custom_prompt=body.custom_prompt
    )


@router.post("/suggest-mapping", summary="Suggest a column mapping", responses=_LLM_RESPONSES)
async def suggest_mapping(
    body: SuggestMappingRequest, service: DatasetIntelligenceServiceDep
) -> ColumnMappingSuggestion:
    """Suggest a flat ``{target: source}`` mapping onto a training format."""
    return await service.suggest_column_mapping(
        column_names=body.column_names,
        column_samples=body.column_samples,
        sample_data=body.sample_data,
        target_format=body.target_format,
    )


@router.post(
    "/validate-strategy",
    summary="Validate a parsing strategy (no LLM)",
    responses={HTTPStatus.UNPROCESSABLE_ENTITY: _PROBLEM_RESPONSE, **_AUTH_RESPONSES},
)
async def validate_strategy(
    body: ValidateStrategyRequest, service: DatasetIntelligenceServiceDep
) -> ValidationResult:
    """Dry-run a parsing strategy against a sample with no LLM call."""
    return service.validate_strategy(body.strategy, body.sample)


@router.get(
    "/formats",
    summary="List dataset formats",
    responses={HTTPStatus.SERVICE_UNAVAILABLE: _PROBLEM_RESPONSE, **_AUTH_RESPONSES},
)
async def list_formats(service: DatasetIntelligenceServiceDep) -> dict[str, Any]:
    """Return autotune's dataset-type catalog, keyed by type name."""
    return await service.list_formats()
