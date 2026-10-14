# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace model search and model cards, read server-side.

Not gated on ``hf_import_available``: that gate is about importing datasets into
this deployment's storage, while these routes only read the Hub.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Annotated, Any

from fastapi import APIRouter, Query
from fastapi.responses import PlainTextResponse

from autotunex.api.deps import HfModelServiceDep
from autotunex.models.common import ProblemDetail
from autotunex.models.hf_import import HfRepoId

router = APIRouter(prefix="/hf/models", tags=["huggingface"])

_PROBLEM_RESPONSE = {"model": ProblemDetail, "content": {"application/problem+json": {}}}
_AUTH_RESPONSES: dict[int | str, dict[str, Any]] = {
    HTTPStatus.UNAUTHORIZED: _PROBLEM_RESPONSE,
    HTTPStatus.BAD_REQUEST: _PROBLEM_RESPONSE,
}


@router.get(
    "/search",
    summary="Search HuggingFace for a model",
    responses={HTTPStatus.SERVICE_UNAVAILABLE: _PROBLEM_RESPONSE, **_AUTH_RESPONSES},
)
async def search_hf_models(
    service: HfModelServiceDep,
    query: str = Query(min_length=1),
    limit: int = Query(default=20, ge=1, le=100),
) -> list[str]:
    """Model repo ids matching ``query``; private ones only from allowlisted namespaces."""
    return await service.search(query=query, limit=limit)


@router.get(
    "/card",
    summary="Read a HuggingFace model's card",
    response_class=PlainTextResponse,
    responses={
        HTTPStatus.NOT_FOUND: _PROBLEM_RESPONSE,
        HTTPStatus.SERVICE_UNAVAILABLE: _PROBLEM_RESPONSE,
        **_AUTH_RESPONSES,
    },
)
async def get_hf_model_card(
    service: HfModelServiceDep, repo_id: Annotated[HfRepoId, Query()]
) -> PlainTextResponse:
    """``repo_id``'s README.md from ``main``, as plain text."""
    return PlainTextResponse(await service.card(repo_id=repo_id))
