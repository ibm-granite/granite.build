from typing import Optional

from gbserver.storage.shadowed.artifact_registry import (
    LhSQLArtifactRegistry,
    SQLLhArtifactRegistry,
)
from gbserver.storage.shadowed.build_storage import LhSQLBuildStorage, SQLLhBuildStorage
from gbserver.storage.shadowed.space_storage import LhSQLSpaceStorage, SQLLhSpaceStorage
from gbserver.storage.shadowed.space_user_storage import (
    LhSQLSpaceUserStorage,
    SQLLhSpaceUserStorage,
)
from gbserver.storage.shadowed.steprun_storage import (
    LhSQLStepRunStorage,
    SQLLhStepRunStorage,
)
from gbserver.storage.shadowed.target_run_storage import (
    LhSQLTargetRunStorage,
    SQLLhTargetRunStorage,
)
from gbserver.storage.storage_factory import StorageFactory


class SQLLhStorageFactory(StorageFactory):
    def create_build_storage(self, table_name: Optional[str] = None):
        return SQLLhBuildStorage(table_name=table_name)

    def create_target_storage(self, table_name: Optional[str] = None):
        return SQLLhTargetRunStorage(table_name=table_name)

    def create_step_storage(self, table_name: Optional[str] = None):
        return SQLLhStepRunStorage(table_name=table_name)

    def create_space_storage(self, table_name: Optional[str] = None):
        return SQLLhSpaceStorage(table_name=table_name)

    def create_artifact_registry(self, table_name: Optional[str] = None):
        return SQLLhArtifactRegistry(table_name=table_name)

    def create_event_storage(self, table_name: Optional[str] = None):
        assert False, "Message storage not yet implemented"

    def create_node_failure_storage(self, table_name: Optional[str] = None):
        assert False, "Node failure storage not yet implemented for SQLLh shadowed storage"

    def create_space_user_storage(self, table_name: Optional[str] = None):
        return SQLLhSpaceUserStorage(table_name=table_name)


class LhSQLStorageFactory(StorageFactory):
    def create_build_storage(self, table_name: str):
        return LhSQLBuildStorage(table_name=table_name)

    def create_target_storage(self, table_name: str):
        return LhSQLTargetRunStorage(table_name=table_name)

    def create_step_storage(self, table_name: str):
        return LhSQLStepRunStorage(table_name=table_name)

    def create_space_storage(self, table_name: str):
        return LhSQLSpaceStorage(table_name=table_name)

    def create_artifact_registry(self, table_name: str):
        return LhSQLArtifactRegistry(table_name=table_name)

    def create_event_storage(self, table_name: Optional[str] = None):
        assert False, "Message storage not yet implemented"

    def create_node_failure_storage(self, table_name: Optional[str] = None):
        assert False, "Node failure storage not yet implemented for LhSQL shadowed storage"

    def create_space_user_storage(self, table_name: Optional[str] = None):
        return LhSQLSpaceUserStorage(table_name=table_name)
