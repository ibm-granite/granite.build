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

from gbserver.api.utils import (
    NO_ACCESSIBLE_SPACE,
    get_row_filter,
    is_space_admin,
    is_super_admin,
    scope_space_name_filter,
)
from gbserver.spaces.user_spaces_list import user_spaces_list
from gbserver.storage.singleton_storage import get_admin_storage
from gbserver.storage.stored_space import StoredSpace
from gbserver.storage.stored_space_user import StoredSpaceUser

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
    - 404 if the named space does not exist
    - 401 if the requesting user is neither admin of the space nor super-admin
    """
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
    request: Request,
    name: str = "",
) -> ListSpacesResponse:
    scoped_name = scope_space_name_filter(request, name)
    if scoped_name is NO_ACCESSIBLE_SPACE:
        return ListSpacesResponse(spaces=[])
    storage = get_admin_storage()
    row_filter = get_row_filter(name=scoped_name)
    items = cast(List[StoredSpace], storage.space_storage.get_by_where(row_filter))
    resp = ListSpacesResponse(spaces=items)
    return resp


@spaces_api.get("/spaces_for_user")
def spaces_for_user(request: Request):
    """Get a user's spaces with admin details.

    No try/except here on purpose: there's no legitimate "not found" case for
    this route (a user with zero spaces just gets an empty list), and a real
    storage error must surface as a 5xx rather than being caught and turned
    into a wrong status code -- the previous broad except did exactly that,
    converting a transient DB error into a misleading 404.
    """
    username = request.state.data["user"].email
    return {"spaces": user_spaces_list(username)}


@spaces_api.get("/{space_name}/members")
def list_members(request: Request, space_name: str) -> ListMembersResponse:
    """List all members of a space. Requires space admin or super-admin."""
    _require_member_management_access(request, space_name)
    members = get_admin_storage().space_user_storage.get_by_space(space_name)
    return ListMembersResponse(members=members)


@spaces_api.post("/{space_name}/members", status_code=status.HTTP_201_CREATED)
def add_member(
    request: Request, space_name: str, body: AddMemberRequest
) -> AddMemberResponse:
    """Add a member to a space. Requires space admin or super-admin."""
    _require_member_management_access(request, space_name)
    storage = get_admin_storage()
    existing = storage.space_user_storage.get_by_space_and_username(
        space_name, body.username
    )
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"User '{body.username}' is already a member of space '{space_name}'",
        )
    new_member = StoredSpaceUser(
        space_name=space_name, username=body.username, role=body.role
    )
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
