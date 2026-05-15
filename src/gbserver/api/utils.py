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


from typing import Any, Optional

from fastapi import HTTPException, Request, status
from pydantic import BaseModel

from gbserver.spaces.user_spaces_list import space_admin_check
from gbserver.storage.storage import Pagination, QueryControl, SortOrder, TaggedItem
from gbserver.types.constants import PUBLIC_SPACE_NAME, SYSTEM_TAG_PREFIX


def get_row_filter(**kwargs):
    filter = {}
    for key, value in kwargs.items():
        if value is not None:
            if isinstance(value, list) or isinstance(value, str):
                keep_value = len(value) > 0
            else:
                keep_value = True
            if keep_value:
                filter[key] = value
    return filter


def is_space_admin(request: Request, space_name: str) -> bool:
    """Determine if the user making the given request is an admin in the given named space."""
    username = request.state.data["user"].email
    return space_admin_check(username=username, space_name=space_name)


def is_super_admin(request: Request) -> bool:
    """Determine if the user making the given request is an admin of the public space"""
    return is_space_admin(request, PUBLIC_SPACE_NAME)


def has_space_write_access(
    request: Request, username_on_target: str, space_name
) -> tuple[bool, str]:
    """See if the requesting user has write access to an asset owned/created by the given username in the given space.
    Throws an HTTPException if the requesting user is not found in the request

    Args:
        request (Request): _description_
        username_on_target (str): _description_
        space_name (_type_): _description_

    Raises:
        HTTPException: if user id is not found in the request.

    Returns:
        tuple[bool,str]: first element indicates if the requester has write access and the 2nd is the user_id found in the request.
    """
    user_id = request.state.data["user"].login
    if user_id is None:
        # if the above dereference fails somewhere it will trigger an exception anyway
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Can not determine user id!"
        )
    is_owner = username_on_target == user_id
    if is_owner:
        return True, user_id
    has_access = is_super_admin(request) or is_space_admin(request, space_name)
    return has_access, user_id


def confirm_space_write_access(
    request: Request, username_on_target: str, space_name: str
) -> None:
    """See if the requesting user has write access to an asset owned/created by the given username in the given space and raise
    and HTTP exception if not.

    Args:
        request (Request): _description_
        username_on_target (str): _description_
        space_name (_type_): _description_

    Raises:
        HTTPException: if user id is not found in the request.
        HTTPException: if user name on the target does not have write access to the target.
    """
    has_access, user_id = has_space_write_access(
        request, username_on_target=username_on_target, space_name=space_name
    )
    if not has_access:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"User {user_id} does not have write access to item in space {space_name}",
        )


def split_tags(tags: Optional[list[str]]) -> tuple[list[str], list[str]]:
    """Split the list of tags into lists of system and and non-system tags"""
    sys_tags = []
    nonsys_tags = []
    if tags is not None:
        for tag in tags:
            assert isinstance(tag, str)
            if tag.startswith(SYSTEM_TAG_PREFIX):
                sys_tags.append(tag)
            else:
                nonsys_tags.append(tag)
    return sys_tags, nonsys_tags


def get_tags_to_set(
    is_super: bool, tagged_item: TaggedItem, tags: Optional[list[str]], appending: bool
) -> list[str]:
    """Used by tag-setting APIs to determine what tags should replace the existing tags with protection and handling of system tags

    Args:
        is_super (bool): if true, the the returned tags can modify/add system tags.
        tagged_item (Any): the items with a 'tags' attribute that we will set with the returned value
        tags (Optional[list[str]]): new tag values to either set or append.
        appending (bool): true if the tags will be appended, otherwise they are to be set and overwrite tag values.

    Raises:
        HTTPException: if setting or appending system tags as a non-admin
        HTTPException: if removing system tags as a non-admin via a non-append call.

    Returns:
        list[str]: list of tags to be applied to the tagged item to effect the requested change.
    """
    existing_sys_tags, existing_user_tags = split_tags(tagged_item.tags)
    new_sys_tags, new_user_tags = split_tags(tags)
    if len(new_sys_tags) != 0 and not is_super:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Can not set or append system tags as a non-admin",
        )

    new_tags = []
    if appending:
        # When appending, keep everything and add more
        new_tags.extend(existing_sys_tags)
        new_tags.extend(new_sys_tags)
        new_tags.extend(existing_user_tags)
        new_tags.extend(new_user_tags)
    else:
        # When setting: preserve existing system tags (unless super user), set user tags to new values
        if not is_super:
            # Always preserve system tags for non-super users
            new_tags.extend(existing_sys_tags)
        else:
            # Add any new system tags (only works for super users)
            new_tags.extend(new_sys_tags)
        new_tags.extend(new_user_tags)  # Set user tags to the new values

    return new_tags


class ListAppendOrSet(BaseModel):
    """Holds a request to either set or append tags to a given TaggedItem."""

    append: Optional[list[str]] = None
    set: Optional[list[str]] = None


def apply_tag_update(
    tagged_item: TaggedItem, tag_update: ListAppendOrSet, is_super_user: bool
):
    """Apply tags set or append request to the given tagged item, with protections for system tags.

    Args:
        tagged_item (TaggedItem): item holding tags to modify
        tag_update (ListAppendOrSet): specifies set or appending of tags.
        is_super_user (bool): true if modification of system tags should be allowed.

    Raises:
        HTTPException: When trying to both append and set tags.
        HTTPException: When trying to modify system tags as a non-super user.
    """
    if tag_update.append and tag_update.set:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Can not both append to and set the tags!",
        )
    if tag_update.append:
        tagged_item.tags = get_tags_to_set(
            is_super_user, tagged_item, tag_update.append, True
        )
    elif tag_update.set or tag_update.set == []:
        tagged_item.tags = get_tags_to_set(
            is_super_user, tagged_item, tag_update.set, False
        )


def get_query_control(
    sort_order_specs: Optional[list[str]], page_index: int, page_size: int
) -> Optional[QueryControl]:
    """Parse the given params to create the QueryControl option, or None if no relevant parameters.
    Pagination is generated only when page_index >=0 and page_size > 0.
    Ordering is generated only when the list of sort specs is non-zero length.
    If both fail the above conditions None is returned.

    Args:
        sort_order_specs (list[str]): List of strings in the form <column>[:(asc|desc)]
        page_index (int): specifies the 0-based page index.
        page_size (int): specifies the page size.

    Raises:
        HTTPException: If any of the sorted_by_specs are malformed.

    Returns:
        Optional[QueryControl]: None if no ordering or pagination is specified, otherwise a QueryControl with 1 or both controls contained.
    """
    if sort_order_specs and len(sort_order_specs) > 0:
        sort_orders = []
        for spec in sort_order_specs:
            try:
                so = SortOrder.parse(spec)
                sort_orders.append(so)
            except Exception as e:
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST, f"Can't parse sort order spec: {e}"
                )
    else:
        sort_orders = None

    pagination = (
        None
        if page_index < 0 or page_size <= 0
        else Pagination(index=page_index, size=page_size)
    )

    query_control = (
        None
        if pagination is None and sort_orders is None
        else QueryControl(sort_orders=sort_orders, pagination=pagination)
    )

    return query_control
