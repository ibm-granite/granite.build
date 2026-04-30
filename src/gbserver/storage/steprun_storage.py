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

"""Steprun storage module."""

from datetime import datetime
from typing import Self

from gbserver.storage.storage import BaseItemStorage, IItemStorage
from gbserver.storage.stored_step_run import StoredStepRun
from gbserver.types.constants import GB_STEP_RUNS_TABLE_NAME


class IStoredStepRunStorage(IItemStorage[StoredStepRun]):
    """I Stored Step Run Storage implementation."""

    pass


class BaseStoredStepRunStorage(BaseItemStorage[StoredStepRun], IStoredStepRunStorage):
    """Base Stored Step Run Storage implementation."""

    def __init__(self: Self, **kwargs):
        kwargs["item_class"] = StoredStepRun
        if kwargs.get("table_name") is None:  # Allow for testing using alternate table names.
            kwargs["table_name"] = GB_STEP_RUNS_TABLE_NAME
        super().__init__(**kwargs)

    def _get_column_values(self: Self, item: StoredStepRun) -> dict:
        fields_to_include = {"name", "build_id", "target_id", "status"}
        json = item.model_dump(include=fields_to_include)
        json["status"] = item.status.name
        return json

    @classmethod
    def _get_sample_item(cls) -> StoredStepRun:
        """Get a sample item that is sufficient for initial creation of the table and its schema.
        The item will never be inserted to the storage instance.
        """
        item = StoredStepRun(
            build_id="build_id",
            target_id="target_id",
            definition_uri="https://some.uri",
            started_at=datetime.now(),
            finished_at=datetime.now(),
        )
        return item
