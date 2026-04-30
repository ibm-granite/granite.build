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
Lakehouse-based implementation of space access control.

This module provides the concrete implementation of ISpaceAccessManager
that uses Lakehouse namespace permissions as the source of truth for
space access control.
"""

from typing import List, Union, cast

from fastapi import status
from fastapi.responses import JSONResponse

from gbserver.spaces.lakehouse_list_user_namespaces import (
    has_access_to_lakehouse_namespace,
    lakehouse_user_namespaces_admin_details,
)
from gbserver.spaces.space_access_manager import ISpaceAccessManager, SpaceAccessInfo
from gbserver.storage.singleton_storage import SingletonAdminStorage, get_admin_storage
from gbserver.storage.stored_space import StoredSpace
from gbserver.types.constants import PUBLIC_SPACE_NAME
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class LakehouseSpaceAccessManager(ISpaceAccessManager):
    """Lakehouse-based implementation of space access control.

    This implementation uses Lakehouse namespace permissions to determine
    user access to spaces. It queries the Lakehouse API to check if users
    have write access and admin privileges on namespaces corresponding to spaces.

    The Lakehouse token must be provided at construction time. A new instance
    should be created per request so that each request uses its own token.
    """

    def __init__(self, lh_token: str) -> None:
        self._lh_token = lh_token

    def get_user_spaces_with_access(self, username: str) -> list[SpaceAccessInfo]:
        """Get list of spaces that the user has access to via Lakehouse.

        This method:
        1. Retrieves all spaces from storage
        2. Extracts their Lakehouse namespaces
        3. Queries Lakehouse for user's access to those namespaces
        4. Returns spaces where user has write access, with admin flag

        Args:
            username: User email address

        Returns:
            List of SpaceAccessInfo objects for accessible spaces.
            Returns empty list on error.
        """
        try:
            space_list = []
            storage = get_admin_storage()
            storage_items = cast(List[StoredSpace], storage.space_storage.get_by_where(None))
            storage_namespaces = [item.lakehouse_namespace for item in storage_items]

            if storage_namespaces and self._lh_token:
                storage_namespaces = list(set(storage_namespaces))  # Remove duplicates
                namespaces = lakehouse_user_namespaces_admin_details(
                    self._lh_token,
                    storage_namespaces,
                )

                if namespaces:
                    for item in storage_items:
                        for namespace_details in namespaces:
                            if item.lakehouse_namespace == namespace_details["namespace"]:
                                space_list.append(
                                    SpaceAccessInfo(
                                        space=item,
                                        is_admin=namespace_details["is_admin"],
                                    )
                                )
            return space_list
        except Exception as e:
            logger.error(f"Error get user spaces list: {e}")
            return []

    def is_space_admin(self, username: str, space_name: str) -> bool:
        """Check if user is an admin of the specified space via Lakehouse.

        Args:
            username: User email address
            space_name: Name of the space to check

        Returns:
            True if user is admin of the space, False otherwise
        """
        try:
            spaces_list = self.get_user_spaces_with_access(username)
            space = next((s for s in spaces_list if s.space.name == space_name), None)

            return space is not None and space.is_admin
        except Exception as e:
            logger.error(f"Error get space admin check: {e}")
            return False

    def has_space_access(self, username: str, space_name: str) -> bool:
        """Check if user has write access to the specified space via Lakehouse.

        Args:
            username: User email address
            space_name: Name of the space to check

        Returns:
            True if user has write access, False otherwise
        """
        try:
            storage = get_admin_storage()
            items = cast(List[StoredSpace], storage.space_storage.get_by_where(None))
            space = next((s for s in items if s.name == space_name), None)
            hasAccess = False

            if space:
                hasAccess = has_access_to_lakehouse_namespace(
                    self._lh_token, space.lakehouse_namespace
                )

            return hasAccess
        except Exception as e:
            logger.error(f"Error get space access check: {e}")
            return False

    def has_build_access(self, username: str, build_id: str) -> Union[bool, JSONResponse]:
        """Check if user has access to the specified build via its space.

        Args:
            username: User email address
            build_id: UUID of the build to check

        Returns:
            True if user has access, False if no access,
            or JSONResponse with 404 if build not found
        """
        try:
            storage: SingletonAdminStorage = get_admin_storage()
            build = storage.build_storage.get_by_uuid(build_id)
            if build is None:
                return JSONResponse(
                    status_code=status.HTTP_404_NOT_FOUND,
                    content={
                        "detail": "Build not found!",
                    },
                )

            return self.has_space_access(username, build.space_name)  # type: ignore[union-attr]
        except Exception as e:
            logger.error(f"Error get build space access check: {e}")
            return False
