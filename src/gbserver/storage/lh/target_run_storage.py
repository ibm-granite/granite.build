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

import json
from typing import Self, Union

from gbserver.storage.lh.lh_storage import BaseLakehouseItemStorage
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.storage.target_run_storage import IStoredTargetRunStorage
from gbserver.types.constants import GB_TARGET_RUNS_TABLE_NAME


class LhTargetRunStorage(BaseLakehouseItemStorage, IStoredTargetRunStorage):
    # TODO: this class should also inherit from BaseStoredTargetRunStorage to pickup a lot of the function below.

    def __init__(self: Self, **kwargs):
        kwargs["item_class"] = StoredTargetRun
        if kwargs.get("table_name") is None:  # Allow for testing using alternate table names.
            kwargs["table_name"] = GB_TARGET_RUNS_TABLE_NAME
        kwargs["unique_fields"] = ["uuid"]
        super().__init__(**kwargs)

    def _get_column_values(self: Self, item: StoredTargetRun) -> dict:
        fields_to_include = ["name", "build_id", "status"]
        json = item.model_dump(include=fields_to_include)  # type: ignore[arg-type]
        json["status"] = item.status.name
        return json

    def __make_artifacts(
        self: Self, uuid_list: list[str], as_input: bool
    ) -> dict[str, Union[str, list[str]]]:
        name = "input" if as_input else "output"
        artifacts = {}
        for i in range(0, len(uuid_list)):
            key = f"{name}[{i}]"
            value = uuid_list[i] if as_input else [uuid_list[i]]
            artifacts[key] = value
        return artifacts

    def _prep_json_before_deserialization(self, json_item: str) -> str:
        """Before deserialization and for backwards compatibility, convert the old in/output_artifact_ids uuid lists to in/output_artifacts fields.
        Change of fields made on/or about Marcy 28, 2025.
        """
        if "put_artifact_ids" in json_item:
            item_dict = json.loads(json_item)
            input_ids = item_dict.get("input_artifact_ids", None)
            if input_ids is not None:
                item_dict["input_artifacts"] = self.__make_artifacts(input_ids, as_input=True)
                item_dict.pop("input_artifact_ids")
            output_ids = item_dict.get("output_artifact_ids", None)
            if output_ids is not None:
                item_dict["output_artifacts"] = self.__make_artifacts(output_ids, as_input=False)
                item_dict.pop("output_artifact_ids")
            json_item = json.dumps(item_dict)
        return json_item


if __name__ == "__main__":
    obj = StoredTargetRun(
        name="mystep",
        build_id="some_build_uuid",
        definition_uri="http://definition",
        environment_uri="https://env",
    )
    print(f"Step: {obj}")
