# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""User-management endpoints.

Three admin-only routes (list, get-by-id, change-role) and one self-service
route (``/me/metadata``). A user is an identity, not an owned resource, so the
admin routes are gated unconditionally by the ``require_admin`` dependency
rather than the ``?scope=all`` model the owned resources use — there is no "own
user" view to widen. Every body is one or two lines: parse, delegate, serialize.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query

from autotunex.api.deps import UserServiceDep, require_admin
from autotunex.models.common import Page, ProblemDetail
from autotunex.models.user import UserMetadata, UserRead, UserRoleUpdate

router = APIRouter(prefix="/users", tags=["users"])

_PROBLEM_RESPONSE = {
    "model": ProblemDetail,
    "content": {"application/problem+json": {}},
}
_AUTH_RESPONSES: dict[int | str, dict[str, Any]] = {
    HTTPStatus.UNAUTHORIZED: _PROBLEM_RESPONSE,
    HTTPStatus.BAD_REQUEST: _PROBLEM_RESPONSE,
}


@router.get(
    "",
    summary="List users (admin only)",
    dependencies=[Depends(require_admin)],
    responses={HTTPStatus.FORBIDDEN: _PROBLEM_RESPONSE, **_AUTH_RESPONSES},
)
async def list_users(
    service: UserServiceDep,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> Page[UserRead]:
    """Return one page of all users, newest first."""
    return await service.list(limit=limit, offset=offset)


@router.get(
    "/me/metadata",
    summary="Get the calling user's usage metadata",
    responses=_AUTH_RESPONSES,
)
async def get_my_metadata(service: UserServiceDep) -> UserMetadata:
    """Return the caller's own job/configuration/dataset counts.

    Declared before ``/{user_id}`` so the literal path is matched here. Open to
    any authenticated caller — not admin-gated.
    """
    return await service.my_metadata()


@router.get(
    "/{user_id}",
    summary="Get a user (admin only)",
    dependencies=[Depends(require_admin)],
    responses={
        HTTPStatus.NOT_FOUND: _PROBLEM_RESPONSE,
        HTTPStatus.FORBIDDEN: _PROBLEM_RESPONSE,
        **_AUTH_RESPONSES,
    },
)
async def get_user(user_id: UUID, service: UserServiceDep) -> UserRead:
    """Return a single user."""
    return await service.get(user_id)


@router.patch(
    "/{user_id}",
    summary="Change a user's role (admin only)",
    dependencies=[Depends(require_admin)],
    responses={
        HTTPStatus.NOT_FOUND: _PROBLEM_RESPONSE,
        HTTPStatus.CONFLICT: _PROBLEM_RESPONSE,
        HTTPStatus.UNPROCESSABLE_ENTITY: _PROBLEM_RESPONSE,
        HTTPStatus.FORBIDDEN: _PROBLEM_RESPONSE,
        **_AUTH_RESPONSES,
    },
)
async def change_user_role(
    user_id: UUID, body: UserRoleUpdate, service: UserServiceDep
) -> UserRead:
    """Change a user's role."""
    return await service.set_role(user_id, body.role)
