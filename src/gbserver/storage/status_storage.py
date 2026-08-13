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
Base storage interface and implementation for the generic gb_status key-value
store.
"""

from typing import Any, Dict, Optional

from gbserver.storage.storage import (
    CREATED_TIME_FIELD_NAME,
    BaseItemStorage,
    IItemStorage,
)
from gbserver.storage.stored_status import StoredStatus
from gbserver.types.constants import GB_STATUS_TABLE_NAME


class IStatusStorage(IItemStorage[StoredStatus]):
    """Interface for the generic key/JSON-value status storage."""

    def get_value(self, key: str) -> Optional[Dict[str, Any]]:
        """Get the JSON value stored under ``key``, or None if not set."""
        raise NotImplementedError

    def set_value(self, key: str, value: Dict[str, Any]) -> None:
        """Set (upsert) the JSON value stored under ``key``."""
        raise NotImplementedError


class BaseStatusStorage(BaseItemStorage[StoredStatus], IStatusStorage):
    """Base storage implementation for the generic gb_status key-value store."""

    def __init__(self, **kwargs) -> None:
        kwargs["item_class"] = StoredStatus
        if kwargs.get("table_name") is None:
            kwargs["table_name"] = GB_STATUS_TABLE_NAME
        super().__init__(**kwargs)

    def _get_column_values(self, item: StoredStatus) -> dict:
        json = {CREATED_TIME_FIELD_NAME: item.created_time}
        return json

    @classmethod
    def _get_sample_item(cls) -> StoredStatus:
        """Return a sample item for use by BaseItemStorage to initialize schema."""
        return StoredStatus(uuid="sample-status-key", value={"sample": "value"})

    def get_value(self, key: str) -> Optional[Dict[str, Any]]:
        item = self.get_by_uuid(key)
        return item.value if item is not None else None

    def set_value(self, key: str, value: Dict[str, Any]) -> None:
        item = StoredStatus(uuid=key, value=value)
        self.update(item, create_if_not_exist=True)
