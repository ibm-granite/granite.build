# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Configuration CRUD endpoints.

Unlike jobs, configurations are created, updated and deleted through this API —
they are the one resource whose full CRUD is wanted (see ``CLAUDE.md`` open
decision 6). Updates are ``PUT``-only full replacements; there is no ``PATCH``.
Every body here is one or two lines: parse, delegate, serialize.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Query, Response

from autotunex.api.deps import ConfigurationServiceDep
from autotunex.models.common import DataScope, Page, ProblemDetail
from autotunex.models.configuration import ConfigurationCreate, ConfigurationRead

router = APIRouter(prefix="/configurations", tags=["configurations"])

_PROBLEM_RESPONSE = {
    "model": ProblemDetail,
    "content": {"application/problem+json": {}},
}
_AUTH_RESPONSES: dict[int | str, dict[str, Any]] = {
    HTTPStatus.UNAUTHORIZED: _PROBLEM_RESPONSE,
    HTTPStatus.BAD_REQUEST: _PROBLEM_RESPONSE,
}


@router.post(
    "",
    summary="Create a configuration",
    status_code=HTTPStatus.CREATED,
    responses={
        HTTPStatus.FORBIDDEN: _PROBLEM_RESPONSE,
        HTTPStatus.CONFLICT: _PROBLEM_RESPONSE,
        HTTPStatus.UNPROCESSABLE_ENTITY: _PROBLEM_RESPONSE,
        **_AUTH_RESPONSES,
    },
)
async def create_configuration(
    body: ConfigurationCreate, service: ConfigurationServiceDep
) -> ConfigurationRead:
    """Create a configuration owned by the calling principal."""
    return await service.create(body)


@router.get(
    "",
    summary="List configurations",
    responses={HTTPStatus.FORBIDDEN: _PROBLEM_RESPONSE, **_AUTH_RESPONSES},
)
async def list_configurations(
    service: ConfigurationServiceDep,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    scope: DataScope = Query(default=DataScope.OWN),
    q: str | None = Query(default=None, description="Case-insensitive substring filter"),
) -> Page[ConfigurationRead]:
    """Return one page of the caller's configurations, newest first."""
    return await service.list(limit=limit, offset=offset, scope=scope, q=q)


@router.get(
    "/template",
    summary="Get the configuration starter template",
    responses={HTTPStatus.SERVICE_UNAVAILABLE: _PROBLEM_RESPONSE, **_AUTH_RESPONSES},
)
async def get_configuration_template(service: ConfigurationServiceDep) -> dict[str, Any]:
    """Return the autotune-provided starter template for a new configuration.

    Declared before ``/{configuration_id}`` so the literal path ``template`` is
    matched here rather than parsed as a configuration id.
    """
    return await service.get_template()


@router.get(
    "/{configuration_id}",
    summary="Get a configuration",
    responses={
        HTTPStatus.NOT_FOUND: _PROBLEM_RESPONSE,
        HTTPStatus.FORBIDDEN: _PROBLEM_RESPONSE,
        **_AUTH_RESPONSES,
    },
)
async def get_configuration(
    configuration_id: UUID,
    service: ConfigurationServiceDep,
    scope: DataScope = Query(default=DataScope.OWN),
) -> ConfigurationRead:
    """Return a single configuration."""
    return await service.get(configuration_id, scope=scope)


@router.put(
    "/{configuration_id}",
    summary="Replace a configuration",
    responses={
        HTTPStatus.NOT_FOUND: _PROBLEM_RESPONSE,
        HTTPStatus.CONFLICT: _PROBLEM_RESPONSE,
        HTTPStatus.UNPROCESSABLE_ENTITY: _PROBLEM_RESPONSE,
        HTTPStatus.FORBIDDEN: _PROBLEM_RESPONSE,
        **_AUTH_RESPONSES,
    },
)
async def replace_configuration(
    configuration_id: UUID,
    body: ConfigurationCreate,
    service: ConfigurationServiceDep,
    scope: DataScope = Query(default=DataScope.OWN),
) -> ConfigurationRead:
    """Fully replace a configuration's mutable fields."""
    return await service.update(configuration_id, body, scope=scope)


@router.delete(
    "/{configuration_id}",
    summary="Delete a configuration",
    status_code=HTTPStatus.NO_CONTENT,
    responses={
        HTTPStatus.NOT_FOUND: _PROBLEM_RESPONSE,
        HTTPStatus.CONFLICT: _PROBLEM_RESPONSE,
        HTTPStatus.FORBIDDEN: _PROBLEM_RESPONSE,
        **_AUTH_RESPONSES,
    },
)
async def delete_configuration(
    configuration_id: UUID,
    service: ConfigurationServiceDep,
    scope: DataScope = Query(default=DataScope.OWN),
) -> Response:
    """Delete a configuration, returning an empty ``204``."""
    await service.delete(configuration_id, scope=scope)
    return Response(status_code=HTTPStatus.NO_CONTENT)
