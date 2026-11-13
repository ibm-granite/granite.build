# Copyright Granite.Build Authors
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

from typing import Optional

from gbserver.storage.sqlite.sqlite_storage import (
    SqliteArtifactRegistry,
    SqliteBuildStorage,
    SqliteEventStorage,
    SqliteKeyValuePairStorage,
    SqliteLineageJobStorage,
    SqliteLineageRowStorage,
    SqliteNodeFailureStorage,
    SqliteSpaceStorage,
    SqliteSpaceUserStorage,
    SqliteStepRunStorage,
    SqliteTargetRunStorage,
)
from gbserver.storage.storage_factory import StorageFactory


class SqliteStorageFactory(StorageFactory):
    def create_build_storage(self, table_name: Optional[str] = None):
        return SqliteBuildStorage(table_name=table_name)

    def create_target_storage(self, table_name: Optional[str] = None):
        return SqliteTargetRunStorage(table_name=table_name)

    def create_step_storage(self, table_name: Optional[str] = None):
        return SqliteStepRunStorage(table_name=table_name)

    def create_space_storage(self, table_name: Optional[str] = None):
        return SqliteSpaceStorage(table_name=table_name)

    def create_artifact_registry(self, table_name: Optional[str] = None):
        return SqliteArtifactRegistry(table_name=table_name)

    def create_event_storage(self, table_name: Optional[str] = None):
        return SqliteEventStorage(table_name=table_name)

    def create_node_failure_storage(self, table_name: Optional[str] = None):
        return SqliteNodeFailureStorage(table_name=table_name)

    def create_lineage_row_storage(self, table_name: Optional[str] = None):
        return SqliteLineageRowStorage(table_name=table_name)

    def create_lineage_job_storage(self, table_name: Optional[str] = None):
        return SqliteLineageJobStorage(table_name=table_name)

    def create_space_user_storage(self, table_name: Optional[str] = None):
        return SqliteSpaceUserStorage(table_name=table_name)

    def create_kv_pair_storage(self, table_name: Optional[str] = None):
        return SqliteKeyValuePairStorage(table_name=table_name)
