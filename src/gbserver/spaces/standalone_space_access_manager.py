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
