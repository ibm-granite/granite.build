#!/usr/bin/env python3

# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Literal, cast

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel

from gbserver.api.utils import get_lh_token_if_needed, is_space_admin, is_super_admin
from gbserver.spaces.user_spaces_list import user_spaces_list
from gbserver.storage.singleton_storage import get_admin_storage
from gbserver.storage.stored_space import StoredSpace
from gbserver.storage.stored_space_user import StoredSpaceUser
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

spaces_api = FastAPI()


class ListSpacesResponse(BaseModel):
    spaces: list[StoredSpace]


class AddMemberRequest(BaseModel):
    username: str
    role: Literal["admin", "member"]


class UpdateMemberRequest(BaseModel):
    role: Literal["admin", "member"]


class AddMemberResponse(BaseModel):
    member: StoredSpaceUser


class ListMembersResponse(BaseModel):
    members: list[StoredSpaceUser]


def _require_member_management_access(request: Request, space_name: str) -> None:
    """Verify that the caller can manage members of the given space.

    Raises HTTPException with:
    - 501 if lakehouse_space_membership feature flag is True
    - 404 if the named space does not exist
    - 401 if the requesting user is neither admin of the space nor super-admin
    """
    from gbserver.types.constants import GB_ENVIRONMENT_CONFIG

    if GB_ENVIRONMENT_CONFIG.feature_flags.get("lakehouse_space_membership", True):
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Space member management is not available in lakehouse mode",
        )
    storage = get_admin_storage()
    space = storage.space_storage.get_by_name(space_name)
    if space is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Space '{space_name}' not found",
        )
    if not (is_space_admin(request, space_name) or is_super_admin(request)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin access required to manage space members",
        )


@spaces_api.get("/")
def list_spaces(
    name: str = "",
) -> ListSpacesResponse:
    storage = get_admin_storage()
    if name is not None and len(name) > 0:
        row_filter = {"name": name}
    else:
        row_filter = None
    items = cast(List[StoredSpace], storage.space_storage.get_by_where(row_filter))
    resp = ListSpacesResponse(spaces=items)
    return resp


@spaces_api.get("/spaces_for_user")
def spaces_for_user(request: Request):
    """Get a users spaces with admin details depending on LH access"""
    try:
        username = request.state.data["user"].email
        lh_token = get_lh_token_if_needed(request)

        list = user_spaces_list(username, lh_token=lh_token)

        return {"spaces": list}

    except Exception as e:
        logger.error("Failed to get a users spaces list error: %s", e)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="secret not found!")


@spaces_api.get("/{space_name}/members")
def list_members(request: Request, space_name: str) -> ListMembersResponse:
    """List all members of a space. Requires space admin or super-admin."""
    _require_member_management_access(request, space_name)
    members = get_admin_storage().space_user_storage.get_by_space(space_name)
    return ListMembersResponse(members=members)


@spaces_api.post("/{space_name}/members", status_code=status.HTTP_201_CREATED)
def add_member(request: Request, space_name: str, body: AddMemberRequest) -> AddMemberResponse:
    """Add a member to a space. Requires space admin or super-admin."""
    _require_member_management_access(request, space_name)
    storage = get_admin_storage()
    existing = storage.space_user_storage.get_by_space_and_username(space_name, body.username)
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"User '{body.username}' is already a member of space '{space_name}'",
        )
    new_member = StoredSpaceUser(space_name=space_name, username=body.username, role=body.role)
    storage.space_user_storage.add(new_member)
    return AddMemberResponse(member=new_member)


@spaces_api.patch("/{space_name}/members/{username}")
def update_member(
    request: Request, space_name: str, username: str, body: UpdateMemberRequest
) -> AddMemberResponse:
    """Update a space member's role. Requires space admin or super-admin."""
    _require_member_management_access(request, space_name)
    storage = get_admin_storage()
    member = storage.space_user_storage.get_by_space_and_username(space_name, username)
    if member is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{username}' is not a member of space '{space_name}'",
        )
    member.role = body.role
    storage.space_user_storage.update(member)
    return AddMemberResponse(member=member)


@spaces_api.delete("/{space_name}/members/{username}")
def delete_member(request: Request, space_name: str, username: str):
    """Remove a member from a space. Requires space admin or super-admin."""
    _require_member_management_access(request, space_name)
    storage = get_admin_storage()
    member = storage.space_user_storage.get_by_space_and_username(space_name, username)
    if member is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{username}' is not a member of space '{space_name}'",
        )
    storage.space_user_storage.delete(member.uuid)
    return {"result": "success"}
