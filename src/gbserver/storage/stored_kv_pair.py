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
Generic key/JSON-value application runtime storage.

Provides a small key-value table (``gb_kv_pairs``) for app-level state that
doesn't warrant its own typed table, e.g. watcher checkpoints. This is a
general key value store for the app runtime, and is not a "KV cache" in the
AI models sense. Lookups are by the ``key`` column; ``uuid`` is a generated
row identifier as in every other stored item.
"""

import datetime
from typing import Any, Dict

from pydantic import Field

from gbserver.storage.storage import BaseStoredItem
from gbserver.utils.utils import get_utc_time


class StoredKeyValuePair(BaseStoredItem):
    """
    Persistent key/JSON-value pair.

    Attributes:
        key: Caller-supplied lookup key. Unique across the table — this, not
            ``uuid``, is the identity callers address rows by.
        value: Arbitrary JSON-serializable payload associated with ``key``.
        created_time: When the key was first set. Preserved across updates by
            ``BaseKeyValuePairStorage.set_value``.
    """

    key: str
    value: Dict[str, Any] = Field(default_factory=dict)
    # The name of this field must match that defined in storage.CREATED_TIME_FIELD_NAME
    created_time: datetime.datetime = Field(default_factory=get_utc_time)
