import abc
from abc import abstractmethod

from gbserver.storage.artifact_registry import IArtifactRegistry
from gbserver.storage.build_storage import IStoredBuildStorage
from gbserver.storage.event_storage import IStoredEventStorage
from gbserver.storage.space_storage import IStoredSpaceStorage
from gbserver.storage.target_run_storage import IStoredTargetRunStorage
from gbserver.storage.steprun_storage import IStoredStepRunStorage
from gbserver.storage.node_failure_storage import INodeFailureStorage
from gbserver.storage.space_user_storage import ISpaceUserStorage


class StorageFactory(abc.ABC):

    @abstractmethod
    def create_build_storage(self, table_name:str) -> IStoredBuildStorage:
        raise ValueError("Sub-class must implement")
    @abstractmethod
    def create_target_storage(self, table_name:str) -> IStoredTargetRunStorage:
        raise ValueError("Sub-class must implement")
    @abstractmethod
    def create_step_storage(self, table_name:str) -> IStoredStepRunStorage:
        raise ValueError("Sub-class must implement")
    @abstractmethod
    def create_space_storage(self, table_name:str) -> IStoredSpaceStorage:
        raise ValueError("Sub-class must implement")
    @abstractmethod
    def create_artifact_registry(self, table_name:str) -> IArtifactRegistry:
        raise ValueError("Sub-class must implement")
    @abstractmethod
    def create_event_storage(self, table_name:str) -> IStoredEventStorage:
        raise ValueError("Sub-class must implement")
    @abstractmethod
    def create_node_failure_storage(self, table_name:str) -> INodeFailureStorage:
        raise ValueError("Sub-class must implement")

    @abstractmethod
    def create_space_user_storage(self, table_name: str) -> ISpaceUserStorage:
        raise ValueError("Sub-class must implement")