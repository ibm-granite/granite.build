#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

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
