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
SQL storage implementation for space user memberships.
"""

from typing import List, Optional

from sqlalchemy import func

from gbserver.storage.space_user_storage import BaseSpaceUserStorage, ISpaceUserStorage
from gbserver.storage.sql.sql_storage import BaseSQLItemStorage
from gbserver.storage.stored_space_user import StoredSpaceUser


class SQLSpaceUserStorage(
    BaseSQLItemStorage[StoredSpaceUser],
    BaseSpaceUserStorage,
    ISpaceUserStorage,
):
    """SQL-based storage for space user memberships."""

    def __init__(self, **kwargs) -> None:
        # Enforce uniqueness: a user can only have one role per space
        kwargs["unique_columns"] = {("space_name", "username"): None}
        # Index both columns used in the domain query methods
        kwargs["indexed_columns"] = ["space_name", "username"]
        super().__init__(**kwargs)

    def _ensure_initialized(self) -> None:
        """Ensure the SQLAlchemy model and table are initialized.

        BaseItemStorage.__initialize_storage() is double-underscore private so
        subclasses cannot call it without fragile name-mangling. We replicate
        the same effect here using the protected API.
        """
        if self._sql_alchemy_model is None:
            sample = self._get_sample_item()
            item_dict = self._convert_item_to_row_dict(sample)
            self._create_or_adjust_schema_item_dict(item_dict)

    def _query_by_username_ci(
        self, username: str, space_name: Optional[str] = None
    ) -> List[StoredSpaceUser]:
        """Case-insensitive username query using LOWER() on both the column and the input value."""
        self._ensure_initialized()

        session = self._get_session_without_retry()
        try:
            model = self._sql_alchemy_model
            query = session.query(model).filter(
                func.lower(model.username) == username.lower()
            )
            if space_name is not None:
                query = query.filter(model.space_name == space_name)
            rows = query.all()
            results = []
            assert self._column_types is not None
            for row in rows:
                row_dict = {k: getattr(row, k) for k in self._column_types}
                results.append(self._convert_row_dict_to_item(row_dict))
            return results
        finally:
            session.close()

    def get_by_username(self, username: str) -> List[StoredSpaceUser]:
        return self._query_by_username_ci(username)

    def get_by_space_and_username(
        self, space_name: str, username: str
    ) -> Optional[StoredSpaceUser]:
        results = self._query_by_username_ci(username, space_name=space_name)
        if not results:
            return None
        if len(results) > 1:
            raise ValueError(
                f"Found {len(results)} records for space={space_name!r}, "
                f"username={username!r}; expected at most 1"
            )
        return results[0]
