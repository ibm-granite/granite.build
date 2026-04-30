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

from gbserver.storage.build_storage import BaseStoredBuildStorage, IStoredBuildStorage
from gbserver.storage.sql.sql_storage import BaseSQLItemStorage
from gbserver.storage.storage import UPDATED_TIME_FIELD_NAME
from gbserver.storage.stored_build import StoredBuild
from gbserver.types.constants import GB_BUILDS_TABLE_NAME


class SQLBuildStorage(BaseSQLItemStorage[StoredBuild], BaseStoredBuildStorage, IStoredBuildStorage):

    def __init__(self, **kwargs) -> None:
        kwargs["default_pagination_sort_by_column"] = UPDATED_TIME_FIELD_NAME
        kwargs["exact_liked_list_columns"] = {"tags": "tags"}
        super().__init__(**kwargs)
