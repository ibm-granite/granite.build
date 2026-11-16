#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

from typing import Any, Self

from gbserver.storage.storage import BaseStoredItem
from gbserver.types.buildevent import BuildEvent


class StoredEvent(BaseStoredItem):
    # Required initializations
    build_event: Any

    def __init__(self: Self, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(self.build_event, BuildEvent)
