from typing import Optional
from gbserver.storage.lh.artifact_registry import LhArtifactRegistry
from gbserver.storage.lh.build_storage import LhBuildStorage
from gbserver.storage.lh.space_storage import LhSpaceStorage
from gbserver.storage.lh.space_user_storage import LhSpaceUserStorage
from gbserver.storage.lh.steprun_storage import LhStepRunStorage
from gbserver.storage.lh.target_run_storage import LhTargetRunStorage
from gbserver.storage.storage_factory import StorageFactory


class LhStorageFactory(StorageFactory):
    def create_build_storage(self, table_name:Optional[str]=None):
        return LhBuildStorage(table_name=table_name)

    def create_target_storage(self, table_name:Optional[str]=None):
        return LhTargetRunStorage(table_name=table_name)

    def create_step_storage(self, table_name:Optional[str]=None):
        return LhStepRunStorage(table_name=table_name)

    def create_space_storage(self, table_name:Optional[str]=None):
        return LhSpaceStorage(table_name=table_name)

    def create_artifact_registry(self, table_name:Optional[str]=None):
        return LhArtifactRegistry(table_name=table_name)

    def create_event_storage(self, table_name:Optional[str]=None):
        assert False, "Message storage not yet implemented"

    def create_node_failure_storage(self, table_name:Optional[str]=None):
        assert False, "Node failure storage not yet implemented for Lakehouse"

    def create_space_user_storage(self, table_name: Optional[str] = None):
        return LhSpaceUserStorage(table_name=table_name)