from typing import Optional
from gbserver.storage.sqlite.sqlite_storage import SqliteArtifactRegistry, SqliteBuildStorage, SqliteEventStorage, SqliteNodeFailureStorage, SqliteSpaceStorage, SqliteSpaceUserStorage, SqliteStepRunStorage, SqliteTargetRunStorage
from gbserver.storage.storage_factory import StorageFactory


class SqliteStorageFactory(StorageFactory):
    def create_build_storage(self, table_name:Optional[str]=None):
        return SqliteBuildStorage(table_name=table_name)

    def create_target_storage(self, table_name:Optional[str]=None):
        return SqliteTargetRunStorage(table_name=table_name)

    def create_step_storage(self, table_name:Optional[str]=None):
        return SqliteStepRunStorage(table_name=table_name)

    def create_space_storage(self, table_name:Optional[str]=None):
        return SqliteSpaceStorage(table_name=table_name)

    def create_artifact_registry(self, table_name:Optional[str]=None):
        return SqliteArtifactRegistry(table_name=table_name)
    
    def create_event_storage(self, table_name:Optional[str]=None):
        return SqliteEventStorage(table_name=table_name)

    def create_node_failure_storage(self, table_name:Optional[str]=None):
        return SqliteNodeFailureStorage(table_name=table_name)

    def create_space_user_storage(self, table_name: Optional[str] = None):
        return SqliteSpaceUserStorage(table_name=table_name)