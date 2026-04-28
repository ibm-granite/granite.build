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

from datetime import datetime
from typing import Self

from gbserver.storage.storage import BaseItemStorage, IItemStorage
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.constants import GB_TARGET_RUNS_TABLE_NAME


class IStoredTargetRunStorage(IItemStorage[StoredTargetRun]):
    pass


class BaseStoredTargetRunStorage(
    BaseItemStorage[StoredTargetRun], IStoredTargetRunStorage
):

    def __init__(self: Self, **kwargs):
        kwargs["item_class"] = StoredTargetRun
        if (
            kwargs.get("table_name") is None
        ):  # Allow for testing using alternate table names.
            kwargs["table_name"] = GB_TARGET_RUNS_TABLE_NAME
        super().__init__(**kwargs)

    def _get_column_values(self: Self, item: StoredTargetRun) -> dict:
        fields_to_include = {
            "name",
            "build_id",
            "status",
            "target_hash",
            "skipped_for_prerun_target_id",
        }
        json = item.model_dump(include=fields_to_include)
        json["status"] = item.status.name
        return json

    @classmethod
    def _get_sample_item(cls) -> StoredTargetRun:
        """Implemented per superclass requirements to return an item for use by BaseItemStorage"""
        item = StoredTargetRun(
            build_id="build_id",
            environment_uri="space://some-env",
            started_at=datetime.now(),
            finished_at=datetime.now(),
        )
        return item
