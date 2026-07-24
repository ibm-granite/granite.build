# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

import logging

import dependencies
import models as api
from auth import get_current_user
from fastapi import APIRouter, Depends, HTTPException
from services import (  # for the type annotations user_service.User / dmf_service.Dmf
    dmf_service,
    user_service,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/api/users",
    tags=["User"],
    summary="List all users (Admin only)",
    response_description="Array of all registered users",
)
async def get_users(
    auth_user: api.AuthUser = Depends(get_current_user),
    user: user_service.User = Depends(dependencies.get_user_service),
) -> list[api.User]:
    """
    Retrieve a list of all registered users in the system.

    **Authorization:** Admin only

    Returns user information including:
    - User IDs and email addresses
    - Roles (admin/user)
    - Registration timestamps
    - Last login information

    **Returns:**
    - Array of User objects

    **Errors:**
    - 403: Forbidden if user is not an admin

    **Use Cases:**
    - User management
    - Role administration
    - Usage monitoring
    """
    return await user.get_users()


@router.get(
    "/api/user/metadata",
    tags=["User"],
    summary="Get current user metadata",
    response_description="User metadata including statistics and preferences",
)
async def get_user_metadata(
    auth_user: api.AuthUser = Depends(get_current_user),
    user: user_service.User = Depends(dependencies.get_user_service),
) -> api.UserMetadata:
    """
    Retrieve metadata and statistics for the authenticated user.

    Returns comprehensive user information including:
    - Total jobs created
    - Total datasets uploaded
    - Storage usage statistics
    - Recent activity
    - User preferences
    - Account settings

    **Returns:**
    - User metadata object with statistics and settings

    **Use Cases:**
    - Display user dashboard
    - Show usage statistics
    - Profile management
    """
    user_id = (await user.get_user(auth_user.email))["id"]
    logger.info(f"Get user metadata for user {user_id}")
    return await user.get_user_metadata(user_id)


@router.get("/api/user/{id}", tags=["User"])
async def get_user_data(
    id: str,
    auth_user: api.AuthUser = Depends(get_current_user),
    user: user_service.User = Depends(dependencies.get_user_service),
) -> api.User:
    if auth_user.role != api.Roles.ADMIN:
        raise HTTPException(
            status_code=401, detail={"status": 401, "message": "Unauthorized"}
        )

    return await user.get_user_detail(id)


@router.get("/api/user/{id}/dmf/models", tags=["DMF"])
async def get_published_models_by_user_id(
    id: str,
    auth_user: api.AuthUser = Depends(get_current_user),
    dmf: dmf_service.Dmf = Depends(dependencies.get_dmf_service),
) -> list[api.ModelInfo]:
    """
    Get Published models from Data Model Factory
    """
    if auth_user.role != api.Roles.ADMIN:
        raise HTTPException(
            status_code=401, detail={"status": 401, "message": "Unauthorized"}
        )
    results = await dmf.get_models(user_id=id)
    return results
