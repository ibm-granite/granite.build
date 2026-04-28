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

from typing import Self

from gbserver.storage.lh.lh_storage import BaseLakehouseItemStorage
from gbserver.storage.steprun_storage import IStoredStepRunStorage
from gbserver.storage.stored_step_run import StoredStepRun
from gbserver.types.constants import GB_STEP_RUNS_TABLE_NAME


class LhStepRunStorage(BaseLakehouseItemStorage, IStoredStepRunStorage):
    # TODO: this class should also inherit from BaseStoredStepRunStorage to pickup a lot of the function below.

    def __init__(self: Self, **kwargs):
        kwargs["item_class"] = StoredStepRun
        if (
            kwargs.get("table_name") is None
        ):  # Allow for testing using alternate table names.
            kwargs["table_name"] = GB_STEP_RUNS_TABLE_NAME
        kwargs["unique_fields"] = ["uuid"]
        super().__init__(**kwargs)

    def _get_column_values(self: Self, item: StoredStepRun) -> dict:
        fields_to_include = ["name", "build_id", "target_id", "status"]
        json = item.model_dump(include=fields_to_include)
        json["status"] = item.status.name
        return json


if __name__ == "__main__":
    obj = StoredStepRun(
        build_id="some_build_uuid",
        target_id="sometargetid",
        definition_uri="http://definition",
        environment_uri="https://env",
    )
    print(f"Step: {obj}")
