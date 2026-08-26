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

"""
Storage-based implementation of space access control.

This module provides a concrete implementation of ISpaceAccessManager that
uses the gb_space_users table as the source of truth for space membership
and role (admin/member).
"""

from typing import Union

from fastapi import status
from fastapi.responses import JSONResponse

from gbserver.spaces.space_access_manager import ISpaceAccessManager, SpaceAccessInfo
from gbserver.storage.singleton_storage import get_admin_storage
from gbserver.types.constants import PUBLIC_SPACE_NAME
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class StorageSpaceAccessManager(ISpaceAccessManager):
    """Storage-based implementation of space access control.

    Uses the gb_space_users table to determine which spaces a user can
    access and whether they hold the admin role. The caller is responsible
    for resolving the user identity (email) before calling these methods.
    """

    def get_user_spaces_with_access(self, username: str) -> list[SpaceAccessInfo]:
        """Get list of spaces the user has access to via gb_space_users.

        Does not catch storage errors -- a real DB failure here must surface
        as a 5xx to the caller, not resolve to an empty list. This is used by
        scope_space_name_filter() (api/utils.py) to scope every list
        endpoint's authorization; silently returning [] on error would make a
        transient outage indistinguishable from "no accessible space",
        masking the outage as an empty/zero-count result instead of failing.

        Args:
            username: User email address.

        Returns:
            List of SpaceAccessInfo for each space the user is a member of.
            Empty list if the user has no memberships (and no public space).
        """
        storage = get_admin_storage()
        memberships = storage.space_user_storage.get_by_username(username)

        # One batched lookup for all membership rows' spaces instead of one
        # get_by_name() round trip per membership. Skipped entirely (rather
        # than passed an empty name list) when there are no memberships --
        # this is now on the hot path for every list/count/tags request, and
        # an empty list still means a real column.in_([]) query and round
        # trip to the DB for a result we already know is empty.
        spaces_by_name = (
            {
                space.name: space
                for space in storage.space_storage.get_by_where(
                    where={"name": [m.space_name for m in memberships]}
                )
            }
            if memberships
            else {}
        )

        result = []
        for membership in memberships:
            space = spaces_by_name.get(membership.space_name)
            if space is None:
                logger.warning(
                    "StorageSpaceAccessManager: space %r referenced in gb_space_users "
                    "does not exist in gb_spaces; skipping",
                    membership.space_name,
                )
                continue
            result.append(
                SpaceAccessInfo(
                    space=space,
                    is_admin=(membership.role == "admin"),
                )
            )

        has_public = any(s.space.name == PUBLIC_SPACE_NAME for s in result)
        if not has_public:
            public_space = storage.space_storage.get_by_name(PUBLIC_SPACE_NAME)
            if public_space is not None:
                result.append(SpaceAccessInfo(space=public_space, is_admin=False))

        return result

    def is_space_admin(self, username: str, space_name: str) -> bool:
        """Check if the user is an admin of the specified space.

        Args:
            username: User email address.
            space_name: Name of the space to check.

        Returns:
            True if the user has the admin role in the space, False otherwise.
        """
        try:
            storage = get_admin_storage()
            membership = storage.space_user_storage.get_by_space_and_username(
                space_name, username
            )
            return membership is not None and membership.role == "admin"
        except Exception as e:
            logger.error("StorageSpaceAccessManager: error in is_space_admin: %s", e)
            return False

    def has_space_access(self, username: str, space_name: str) -> bool:
        """Check if the user has access (any role) to the specified space.

        All authenticated users have implicit access to the public space,
        even if no StoredSpace row for it exists yet -- unlike
        get_user_spaces_with_access() below, which only lists the public
        space when a real row exists. That's intentional (this is a
        single-object convenience check so e.g. builds submitted under a
        not-yet-created public space still resolve), not a bug: a
        previous attempt to unify them by synthesizing a placeholder row in
        get_user_spaces_with_access() broke an established contract in
        test_user_spaces_list and was reverted. The asymmetry is fail-closed
        (public content is never over-exposed, only under-listed), so it's
        left as-is rather than re-attempted.

        In practice the "no row yet" state is a pre-setup condition, not a
        steady-state early-deployment risk: registering the public space via
        `gbserver create-spaces` (docs/spaces/README.md) -- or automatically,
        for a standalone server, via `gbserver standalone --space-dir` -- is a
        documented one-time deployment step. A deployment that has completed
        that step never hits this divergence; one that hasn't will see list
        endpoints under-report until it does, which is the accepted trade-off.

        Args:
            username: User email address.
            space_name: Name of the space to check.

        Returns:
            True if the user has any membership in the space, False otherwise.
        """
        if space_name == PUBLIC_SPACE_NAME:
            return True
        try:
            storage = get_admin_storage()
            membership = storage.space_user_storage.get_by_space_and_username(
                space_name, username
            )
            return membership is not None
        except Exception as e:
            logger.error("StorageSpaceAccessManager: error in has_space_access: %s", e)
            return False

    def has_build_access(
        self, username: str, build_id: str
    ) -> Union[bool, JSONResponse]:
        """Check if the user has access to the specified build via its space.

        Args:
            username: User email address.
            build_id: UUID of the build to check.

        Returns:
            True if user has access, False if no access,
            or JSONResponse with 404 if the build is not found.
        """
        try:
            storage = get_admin_storage()
            build = storage.build_storage.get_by_uuid(build_id)
            if build is None:
                return JSONResponse(
                    status_code=status.HTTP_404_NOT_FOUND,
                    content={"detail": "Build not found!"},
                )
            return self.has_space_access(username, build.space_name)  # type: ignore[union-attr]
        except Exception as e:
            logger.error("StorageSpaceAccessManager: error in has_build_access: %s", e)
            return False
