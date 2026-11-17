#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""
Standalone implementation of space access control.

In standalone mode there is no Lakehouse or external authorization backend.
All stored spaces are accessible, and the user is treated as admin.
"""

from typing import List, Union, cast

from fastapi.responses import JSONResponse

from gbserver.spaces.space_access_manager import ISpaceAccessManager, SpaceAccessInfo
from gbserver.storage.singleton_storage import SingletonAdminStorage, get_admin_storage
from gbserver.storage.stored_space import StoredSpace
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class StandaloneSpaceAccessManager(ISpaceAccessManager):
    """Standalone implementation — all stored spaces are accessible."""

    def get_user_spaces_with_access(self, username: str) -> list[SpaceAccessInfo]:
        storage = get_admin_storage()
        items = cast(List[StoredSpace], storage.space_storage.get_by_where(None))
        return [SpaceAccessInfo(space=item, is_admin=True) for item in items]

    def is_space_admin(self, username: str, space_name: str) -> bool:
        return True

    def has_space_access(self, username: str, space_name: str) -> bool:
        return True

    def has_build_access(
        self, username: str, build_id: str
    ) -> Union[bool, JSONResponse]:
        storage: SingletonAdminStorage = get_admin_storage()
        build = storage.build_storage.get_by_uuid(build_id)
        if build is None:
            return JSONResponse(
                status_code=404,
                content={"detail": "Build not found!"},
            )
        return True
