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

"""
Provides interfaces for interacting with persistent storage.
"""

import os
from typing import Optional

from pydantic import BaseModel

from gbserver.storage.artifact_registry import IArtifactRegistry
from gbserver.storage.build_storage import IStoredBuildStorage
from gbserver.storage.event_storage import IStoredEventStorage
from gbserver.storage.node_failure_storage import INodeFailureStorage
from gbserver.storage.space_storage import IStoredSpaceStorage
from gbserver.storage.space_user_storage import ISpaceUserStorage
from gbserver.storage.sql.storage_factory import SQLStorageFactory
from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory
from gbserver.storage.steprun_storage import IStoredStepRunStorage
from gbserver.storage.storage_factory import StorageFactory
from gbserver.storage.target_run_storage import IStoredTargetRunStorage
from gbserver.types.constants import (
    ENV_VAR_GBSERVER_ADMIN_TABLE_PREFIX,
    GB_ARTIFACT_REGISTRY_TABLE_NAME,
    GB_BUILDS_TABLE_NAME,
    GB_EVENTS_TABLE_NAME,
    GB_METADATA_STORAGE,
    GB_NODE_FAILURES_TABLE_NAME,
    GB_SPACE_USERS_TABLE_NAME,
    GB_SPACES_TABLE_NAME,
    GB_STEP_RUNS_TABLE_NAME,
    GB_TARGET_RUNS_TABLE_NAME,
)
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class SingletonAdminStorage(BaseModel):
    """The singleton class used to interact with persistent storage."""

    space_storage: IStoredSpaceStorage
    build_storage: IStoredBuildStorage
    target_storage: IStoredTargetRunStorage
    step_storage: IStoredStepRunStorage
    artifact_registry: IArtifactRegistry
    event_storage: IStoredEventStorage
    node_failure_storage: INodeFailureStorage
    space_user_storage: ISpaceUserStorage
    table_name_prefix: str


__STORAGE: Optional[SingletonAdminStorage] = None


__STORAGE_FACTORY: Optional[StorageFactory] = None


def set_storage_factory(factory: StorageFactory):
    """Set the storage factory to use.
    Primarily used by testing to test different storage implementations.
    """
    global __STORAGE_FACTORY
    __STORAGE_FACTORY = factory


def get_storage_factory() -> StorageFactory:
    """Get the configured or set StorageFactory.

    Raises:
        ValueError: If an unknown GB_METADATA_STORAGE value is set.

    Returns:
        StorageFactory: _description_
    """
    global __STORAGE_FACTORY
    if __STORAGE_FACTORY is None:
        if GB_METADATA_STORAGE == "sql":
            __STORAGE_FACTORY = SQLStorageFactory()
        elif GB_METADATA_STORAGE == "sqlite":
            __STORAGE_FACTORY = SqliteStorageFactory()
        else:
            raise ValueError(
                f"Unrecognized storage factory config {GB_METADATA_STORAGE}"
            )
        logger.info(
            "Using storage factory %s based on %s setting.",
            type(__STORAGE_FACTORY).__name__,
            GB_METADATA_STORAGE,
        )

    return __STORAGE_FACTORY


def get_admin_storage() -> SingletonAdminStorage:
    """Get the tuple of storage instances that the REST apis should be using.

    Returns:
        SingletonStorage
    """
    global __STORAGE
    if __STORAGE is None:
        gb_admin_table_prefix = os.getenv(ENV_VAR_GBSERVER_ADMIN_TABLE_PREFIX, "")
        if gb_admin_table_prefix != "":
            logger.warning(
                "Global admin table name prefix set to %s!!!",
                gb_admin_table_prefix,
            )
        __STORAGE = set_storage_prefix(gb_admin_table_prefix)
    return __STORAGE


def set_storage_prefix(table_prefix: Optional[str] = None) -> SingletonAdminStorage:
    """
    Creates tables with the given prefix in their name.
    If no prefix is given, the standard table names are used.
    Especially useful for running tests without affecting production tables.
    """
    if table_prefix is None:
        table_prefix = ""
    factory = get_storage_factory()
    build_storage = factory.create_build_storage(
        table_name=table_prefix + GB_BUILDS_TABLE_NAME
    )
    target_storage = factory.create_target_storage(
        table_name=table_prefix + GB_TARGET_RUNS_TABLE_NAME
    )
    step_storage = factory.create_step_storage(
        table_name=table_prefix + GB_STEP_RUNS_TABLE_NAME
    )
    space_storage = factory.create_space_storage(
        table_name=table_prefix + GB_SPACES_TABLE_NAME
    )
    artifact_storage = factory.create_artifact_registry(
        table_name=table_prefix + GB_ARTIFACT_REGISTRY_TABLE_NAME
    )
    event_storage = factory.create_event_storage(
        table_name=table_prefix + GB_EVENTS_TABLE_NAME
    )
    node_failure_storage = factory.create_node_failure_storage(
        table_name=table_prefix + GB_NODE_FAILURES_TABLE_NAME
    )
    space_user_storage = factory.create_space_user_storage(
        table_name=table_prefix + GB_SPACE_USERS_TABLE_NAME
    )

    # # Force the table creation as early as possible.
    # # This is primarily for tests which have multiple runnerjobs running simultaneiously in separate processes/jobs.
    # # But this is good practice in general so that the primary processes (build watcher and the rest server create these and not some other process)
    # build_storage.get_by_where({"uuid": ""})
    # target_storage.get_by_where({"uuid": ""})
    # step_storage.get_by_where({"uuid": ""})
    # space_storage.get_by_where({"uuid": ""})
    # artifact_storage.get_by_where({"uuid": ""})
    # event_storage.get_by_where({"uuid": ""})

    global __STORAGE
    __STORAGE = SingletonAdminStorage(
        build_storage=build_storage,
        target_storage=target_storage,
        step_storage=step_storage,
        artifact_registry=artifact_storage,
        space_storage=space_storage,
        event_storage=event_storage,
        node_failure_storage=node_failure_storage,
        space_user_storage=space_user_storage,
        table_name_prefix=table_prefix,
    )
    return __STORAGE
