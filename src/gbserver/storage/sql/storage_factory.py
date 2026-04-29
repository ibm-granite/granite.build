from typing import Optional

from gbserver.storage.sql.artifact_registry import SQLArtifactRegistry
from gbserver.storage.sql.build_storage import SQLBuildStorage
from gbserver.storage.sql.event_storage import SQLEventStorage
from gbserver.storage.sql.node_failure_storage import SQLNodeFailureStorage
from gbserver.storage.sql.space_storage import SQLSpaceStorage
from gbserver.storage.sql.space_user_storage import SQLSpaceUserStorage
from gbserver.storage.sql.steprun_storage import SQLStepRunStorage
from gbserver.storage.sql.target_run_storage import SQLTargetRunStorage
from gbserver.storage.storage_factory import StorageFactory


class SQLStorageFactory(StorageFactory):
    def create_build_storage(self, table_name: Optional[str] = None):
        return SQLBuildStorage(table_name=table_name)

    def create_target_storage(self, table_name: Optional[str] = None):
        return SQLTargetRunStorage(table_name=table_name)

    def create_step_storage(self, table_name: Optional[str] = None):
        return SQLStepRunStorage(table_name=table_name)

    def create_space_storage(self, table_name: Optional[str] = None):
        return SQLSpaceStorage(table_name=table_name)

    def create_artifact_registry(self, table_name: Optional[str] = None):
        return SQLArtifactRegistry(table_name=table_name)

    def create_event_storage(self, table_name: Optional[str] = None):
        return SQLEventStorage(table_name=table_name)

    def create_node_failure_storage(self, table_name: Optional[str] = None):
        return SQLNodeFailureStorage(table_name=table_name)

    def create_space_user_storage(self, table_name: Optional[str] = None):
        return SQLSpaceUserStorage(table_name=table_name)
