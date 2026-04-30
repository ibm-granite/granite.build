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
Base storage interface and implementation for space user memberships.
"""

from typing import List, Optional

from gbserver.storage.storage import (
    CREATED_TIME_FIELD_NAME,
    UPDATED_TIME_FIELD_NAME,
    BaseItemStorage,
    IItemStorage,
)
from gbserver.storage.stored_space_user import StoredSpaceUser
from gbserver.types.constants import GB_SPACE_USERS_TABLE_NAME


class ISpaceUserStorage(IItemStorage[StoredSpaceUser]):
    """Interface for space user membership storage implementations."""

    def get_by_space(self, space_name: str) -> List[StoredSpaceUser]:
        """Get all user memberships for a given space.

        Args:
            space_name: Name of the space.

        Returns:
            List of StoredSpaceUser records for that space.
        """
        raise NotImplementedError(f"{self.__class__.__name__} did not implement this method")

    def get_by_username(self, username: str) -> List[StoredSpaceUser]:
        """Get all space memberships for a given user.

        Args:
            username: Username to look up.

        Returns:
            List of StoredSpaceUser records for that user.
        """
        raise NotImplementedError(f"{self.__class__.__name__} did not implement this method")

    def get_by_space_and_username(
        self, space_name: str, username: str
    ) -> Optional[StoredSpaceUser]:
        """Look up the unique membership for a user in a specific space.

        Args:
            space_name: Name of the space.
            username: Username of the member.

        Raises:
            ValueError: if more than one record is found.

        Returns:
            StoredSpaceUser if found, None otherwise.
        """
        raise NotImplementedError(f"{self.__class__.__name__} did not implement this method")


class BaseSpaceUserStorage(BaseItemStorage[StoredSpaceUser], ISpaceUserStorage):
    """Base Space User Storage implementation."""

    def __init__(self, **kwargs) -> None:
        kwargs["item_class"] = StoredSpaceUser
        if kwargs.get("table_name") is None:
            kwargs["table_name"] = GB_SPACE_USERS_TABLE_NAME
        super().__init__(**kwargs)

    def _get_column_values(self, item: StoredSpaceUser) -> dict:
        fields_to_include = {"space_name", "username", "role"}
        json = item.model_dump(include=fields_to_include)
        json[CREATED_TIME_FIELD_NAME] = item.created_time
        json[UPDATED_TIME_FIELD_NAME] = item.updated_time
        return json

    @classmethod
    def _get_sample_item(cls) -> StoredSpaceUser:
        return StoredSpaceUser(
            space_name="sample_space",
            username="sample_user",
            role="member",
        )

    def get_by_space(self, space_name: str) -> List[StoredSpaceUser]:
        return self.get_by_where({"space_name": space_name})

    def get_by_username(self, username: str) -> List[StoredSpaceUser]:
        return self.get_by_where({"username": username})

    def get_by_space_and_username(
        self, space_name: str, username: str
    ) -> Optional[StoredSpaceUser]:
        results = self.get_by_where({"space_name": space_name, "username": username})
        if not results:
            return None
        if len(results) > 1:
            raise ValueError(
                f"Found {len(results)} records for space={space_name!r}, "
                f"username={username!r}; expected at most 1"
            )
        return results[0]
