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
Lakehouse storage implementation for space user memberships.
"""

from typing import List, Optional

from gbserver.storage.lh.lh_storage import BaseLakehouseItemStorage
from gbserver.storage.space_user_storage import ISpaceUserStorage
from gbserver.storage.stored_space_user import StoredSpaceUser
from gbserver.types.constants import GB_SPACE_USERS_TABLE_NAME


class LhSpaceUserStorage(BaseLakehouseItemStorage, ISpaceUserStorage):
    # TODO: this class should also inherit from BaseSpaceUserStorage to pick up
    # shared logic, similar to the recognized tech debt in LhSpaceStorage.

    def __init__(self, **kwargs) -> None:
        kwargs["item_class"] = StoredSpaceUser
        if kwargs.get("table_name") is None:
            kwargs["table_name"] = GB_SPACE_USERS_TABLE_NAME
        kwargs["unique_fields"] = ["uuid", "space_name", "username"]
        super().__init__(**kwargs)

    def _get_column_values(self, item: StoredSpaceUser) -> dict:
        fields_to_include = {"space_name", "username", "role"}
        json = item.model_dump(include=fields_to_include)
        return json

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
