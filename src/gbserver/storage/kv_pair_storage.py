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
Base storage interface and implementation for the generic gb_kv_pairs
key-value store.
"""

from typing import Any, Dict, Optional

from gbserver.storage.storage import (
    CREATED_TIME_FIELD_NAME,
    BaseItemStorage,
    IItemStorage,
)
from gbserver.storage.stored_kv_pair import StoredKeyValuePair
from gbserver.types.constants import GB_KV_PAIRS_TABLE_NAME


class IKeyValuePairStorage(IItemStorage[StoredKeyValuePair]):
    """Interface for the generic key/JSON-value storage."""

    def get_by_key(self, key: str) -> Optional[StoredKeyValuePair]:
        """Look up the unique row stored under ``key``, or None if not set.

        Raises:
            ValueError: if more than one row is found for ``key``.
        """
        raise NotImplementedError

    def get_value(self, key: str) -> Optional[Dict[str, Any]]:
        """Get the JSON value stored under ``key``, or None if not set."""
        raise NotImplementedError

    def set_value(self, key: str, value: Dict[str, Any]) -> None:
        """Set (upsert) the JSON value stored under ``key``."""
        raise NotImplementedError


class BaseKeyValuePairStorage(
    BaseItemStorage[StoredKeyValuePair], IKeyValuePairStorage
):
    """Base storage implementation for the generic gb_kv_pairs key-value store."""

    def __init__(self, **kwargs) -> None:
        kwargs["item_class"] = StoredKeyValuePair
        if kwargs.get("table_name") is None:
            kwargs["table_name"] = GB_KV_PAIRS_TABLE_NAME
        super().__init__(**kwargs)

    def _get_column_values(self, item: StoredKeyValuePair) -> dict:
        # `key` must be a real column: it is the lookup identity (see
        # get_by_key) and what SQLKeyValuePairStorage's index and unique
        # constraint attach to.
        json = {"key": item.key, CREATED_TIME_FIELD_NAME: item.created_time}
        return json

    @classmethod
    def _get_sample_item(cls) -> StoredKeyValuePair:
        """Return a sample item for use by BaseItemStorage to initialize schema."""
        return StoredKeyValuePair(key="sample-status-key", value={"sample": "value"})

    def get_by_key(self, key: str) -> Optional[StoredKeyValuePair]:
        """Look up the unique row stored under ``key``, or None if not set."""
        return self._get_by_single_field(  # type: ignore[return-value]
            column_name="key", column_value=key, allow_multiple=False
        )

    def get_value(self, key: str) -> Optional[Dict[str, Any]]:
        item = self.get_by_key(key)
        if item is None:
            return None
        return item.value

    def set_value(self, key: str, value: Dict[str, Any]) -> None:
        # The base class's update()/get_by_uuid() are all uuid-keyed, so the
        # upsert has to resolve `key` to a row first. Reusing the existing row's
        # uuid is what makes this an update rather than a second row under the
        # same key.
        existing = self.get_by_key(key)
        if existing is None:
            self.add(StoredKeyValuePair(key=key, value=value))
            return
        # created_time means "when the key was first set", so carry the stored
        # value forward explicitly. update() would also preserve it by popping
        # the field, but relying on that made the invariant a side effect that
        # any future add()-based upsert would silently reset.
        item = StoredKeyValuePair(
            uuid=existing.uuid,
            key=key,
            value=value,
            created_time=existing.created_time,
        )
        self.update(item, create_if_not_exist=True)
