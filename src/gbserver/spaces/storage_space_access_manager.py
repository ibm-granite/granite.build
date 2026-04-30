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

        Args:
            username: User email address.

        Returns:
            List of SpaceAccessInfo for each space the user is a member of.
            Returns empty list on error or if the user has no memberships.
        """
        try:
            storage = get_admin_storage()
            memberships = storage.space_user_storage.get_by_username(username)

            result = []
            for membership in memberships:
                space = storage.space_storage.get_by_name(membership.space_name)
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
        except Exception as e:
            logger.error("StorageSpaceAccessManager: error in get_user_spaces_with_access: %s", e)
            return []

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
            membership = storage.space_user_storage.get_by_space_and_username(space_name, username)
            return membership is not None and membership.role == "admin"
        except Exception as e:
            logger.error("StorageSpaceAccessManager: error in is_space_admin: %s", e)
            return False

    def has_space_access(self, username: str, space_name: str) -> bool:
        """Check if the user has access (any role) to the specified space.

        All authenticated users have implicit access to the public space.

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
            membership = storage.space_user_storage.get_by_space_and_username(space_name, username)
            return membership is not None
        except Exception as e:
            logger.error("StorageSpaceAccessManager: error in has_space_access: %s", e)
            return False

    def has_build_access(self, username: str, build_id: str) -> Union[bool, JSONResponse]:
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
            return self.has_space_access(username, build.space_name)
        except Exception as e:
            logger.error("StorageSpaceAccessManager: error in has_build_access: %s", e)
            return False
