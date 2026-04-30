"""Storage factory module."""

import abc
from abc import abstractmethod

from gbserver.storage.artifact_registry import IArtifactRegistry
from gbserver.storage.build_storage import IStoredBuildStorage
from gbserver.storage.event_storage import IStoredEventStorage
from gbserver.storage.node_failure_storage import INodeFailureStorage
from gbserver.storage.space_storage import IStoredSpaceStorage
from gbserver.storage.space_user_storage import ISpaceUserStorage
from gbserver.storage.steprun_storage import IStoredStepRunStorage
from gbserver.storage.target_run_storage import IStoredTargetRunStorage


class StorageFactory(abc.ABC):
    """Abstract factory for creating storage backend instances."""

    @abstractmethod
    def create_build_storage(self, table_name: str) -> IStoredBuildStorage:
        """Create build storage."""
        raise ValueError("Sub-class must implement")

    @abstractmethod
    def create_target_storage(self, table_name: str) -> IStoredTargetRunStorage:
        """Create target storage."""
        raise ValueError("Sub-class must implement")

    @abstractmethod
    def create_step_storage(self, table_name: str) -> IStoredStepRunStorage:
        """Create step storage."""
        raise ValueError("Sub-class must implement")

    @abstractmethod
    def create_space_storage(self, table_name: str) -> IStoredSpaceStorage:
        """Create space storage."""
        raise ValueError("Sub-class must implement")

    @abstractmethod
    def create_artifact_registry(self, table_name: str) -> IArtifactRegistry:
        """Create artifact registry."""
        raise ValueError("Sub-class must implement")

    @abstractmethod
    def create_event_storage(self, table_name: str) -> IStoredEventStorage:
        """Create event storage."""
        raise ValueError("Sub-class must implement")

    @abstractmethod
    def create_node_failure_storage(self, table_name: str) -> INodeFailureStorage:
        """Create node failure storage."""
        raise ValueError("Sub-class must implement")

    @abstractmethod
    def create_space_user_storage(self, table_name: str) -> ISpaceUserStorage:
        """Create space user storage."""
        raise ValueError("Sub-class must implement")
