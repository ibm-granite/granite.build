from typing import List, Optional, Union

from gbserver.storage.artifact_registration import ArtifactRegistration
from gbserver.storage.artifact_registry import IArtifactRegistry
from gbserver.storage.build_storage import IStoredBuildStorage
from gbserver.storage.event_storage import IStoredEventStorage
from gbserver.storage.stored_event import StoredEvent
from gbserver.storage.node_failure_storage import INodeFailureStorage
from gbserver.storage.shadowed.storage import BaseDualItemStorage
from gbserver.storage.space_storage import IStoredSpaceStorage
from gbserver.storage.space_user_storage import ISpaceUserStorage
from gbserver.storage.stored_space import StoredSpace
from gbserver.storage.stored_space_user import StoredSpaceUser
from gbserver.storage.sql.artifact_registry import SQLArtifactRegistry
from gbserver.storage.sql.build_storage import SQLBuildStorage
from gbserver.storage.sql.event_storage import SQLEventStorage
from gbserver.storage.sql.node_failure_storage import SQLNodeFailureStorage
from gbserver.storage.sql.space_storage import SQLSpaceStorage
from gbserver.storage.sql.space_user_storage import SQLSpaceUserStorage
from gbserver.storage.sql.steprun_storage import SQLStepRunStorage
from gbserver.storage.sql.target_run_storage import SQLTargetRunStorage
from gbserver.storage.sqlite.sqlite_storage import (
    SqliteArtifactRegistry,
    SqliteBuildStorage,
    SqliteEventStorage,
    SqliteNodeFailureStorage,
    SqliteSpaceStorage,
    SqliteSpaceUserStorage,
    SqliteStepRunStorage,
    SqliteTargetRunStorage,
)
from gbserver.storage.steprun_storage import IStoredStepRunStorage
from gbserver.storage.storage_factory import StorageFactory
from gbserver.storage.target_run_storage import IStoredTargetRunStorage


class DualBuildStorage(BaseDualItemStorage, IStoredBuildStorage):
    pass


class DualTargetRunStorage(BaseDualItemStorage, IStoredTargetRunStorage):
    pass


class DualStepRunStorage(BaseDualItemStorage, IStoredStepRunStorage):
    pass


class DualSpaceStorage(BaseDualItemStorage, IStoredSpaceStorage):

    def get_by_name(self, name: str) -> Optional[StoredSpace]:
        return self.primary.get_by_name(name)  # type: ignore[union-attr]


class DualArtifactRegistry(BaseDualItemStorage, IArtifactRegistry):

    def get_by_uri(
        self, uri: str, space_name: str = ""
    ) -> Union[List[ArtifactRegistration], Optional[ArtifactRegistration]]:
        return self.primary.get_by_uri(uri, space_name)  # type: ignore[union-attr]


class DualEventStorage(BaseDualItemStorage, IStoredEventStorage):

    def get_sorted_build_events(
        self, build_id: str, where: Optional[dict] = None
    ) -> list[StoredEvent]:
        return self.primary.get_sorted_build_events(build_id, where)  # type: ignore[union-attr]


class DualNodeFailureStorage(BaseDualItemStorage, INodeFailureStorage):
    pass


class DualSpaceUserStorage(BaseDualItemStorage, ISpaceUserStorage):

    def get_by_space(self, space_name: str) -> List[StoredSpaceUser]:
        return self.primary.get_by_space(space_name)  # type: ignore[union-attr]

    def get_by_username(self, username: str) -> List[StoredSpaceUser]:
        return self.primary.get_by_username(username)  # type: ignore[union-attr]

    def get_by_space_and_username(
        self, space_name: str, username: str
    ) -> Optional[StoredSpaceUser]:
        return self.primary.get_by_space_and_username(space_name, username)  # type: ignore[union-attr]


class DualSQLSqliteStorageFactory(StorageFactory):
    """SQL primary, SQLite secondary dual-write storage factory."""

    def create_build_storage(self, table_name: Optional[str] = None):
        return DualBuildStorage(
            table_name=table_name,
            primary_class=SQLBuildStorage,
            secondary_class=SqliteBuildStorage,
        )

    def create_target_storage(self, table_name: Optional[str] = None):
        return DualTargetRunStorage(
            table_name=table_name,
            primary_class=SQLTargetRunStorage,
            secondary_class=SqliteTargetRunStorage,
        )

    def create_step_storage(self, table_name: Optional[str] = None):
        return DualStepRunStorage(
            table_name=table_name,
            primary_class=SQLStepRunStorage,
            secondary_class=SqliteStepRunStorage,
        )

    def create_space_storage(self, table_name: Optional[str] = None):
        return DualSpaceStorage(
            table_name=table_name,
            primary_class=SQLSpaceStorage,
            secondary_class=SqliteSpaceStorage,
        )

    def create_artifact_registry(self, table_name: Optional[str] = None):
        return DualArtifactRegistry(
            table_name=table_name,
            primary_class=SQLArtifactRegistry,
            secondary_class=SqliteArtifactRegistry,
        )

    def create_event_storage(self, table_name: Optional[str] = None):
        return DualEventStorage(
            table_name=table_name,
            primary_class=SQLEventStorage,
            secondary_class=SqliteEventStorage,
        )

    def create_node_failure_storage(self, table_name: Optional[str] = None):
        return DualNodeFailureStorage(
            table_name=table_name,
            primary_class=SQLNodeFailureStorage,
            secondary_class=SqliteNodeFailureStorage,
        )

    def create_space_user_storage(self, table_name: Optional[str] = None):
        return DualSpaceUserStorage(
            table_name=table_name,
            primary_class=SQLSpaceUserStorage,
            secondary_class=SqliteSpaceUserStorage,
        )
