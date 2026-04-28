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


import multiprocessing
import os
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar, Union

from filelock import FileLock
from pydantic import BaseModel
from sqlalchemy import Integer

from gbserver.storage.sql.artifact_registry import SQLArtifactRegistry
from gbserver.storage.sql.build_storage import SQLBuildStorage
from gbserver.storage.sql.event_storage import SQLEventStorage
from gbserver.storage.sql.node_failure_storage import SQLNodeFailureStorage
from gbserver.storage.sql.space_storage import SQLSpaceStorage
from gbserver.storage.sql.space_user_storage import SQLSpaceUserStorage
from gbserver.storage.sql.steprun_storage import SQLStepRunStorage
from gbserver.storage.sql.target_run_storage import SQLTargetRunStorage
from gbserver.storage.storage import BASE_ITEM_TYPE, IItemStorage, QueryControl

USER_HOME_LLMB_DIR_NAME = ".llmb"
SQLITE_DB_FILE_NAME = "llmb-server.db"


class SqliteStorageOverrides(BaseModel, Generic[BASE_ITEM_TYPE]):

    def __init__(self, **kwargs) -> None:
        # These initializatinos can't be done here because when the super().__init__() calls _get_connection_specs() below,
        # self is the super and not this class, so these fields aren't present (I think that is the problem anyway)
        # self._db_path = self._get_db_file_path()
        # lock_file = f"{self._db_path.absolute()}.lck"
        # self._db_file_lock = FileLock(lock_file)
        super().__init__(**kwargs)

        # Removing the explicit call to _get_connection_specs() here
        # It will be called automatically by the parent class
        # self._get_connection_specs()

    def _get_connection_specs(
        self,
    ) -> tuple[Optional[str], str, str, Optional[dict[str, str]]]:
        """Determines and returns the db connection specifications for this sub-class implementation.

        Returns:
            tuple[str, str,str,dict[str,str]]: A set of connection information as follows:
                database schema to use - None if not used
                database connection URL - including password if needed.
                obfuscated database connection URL - db url w/o password
                connection args - a dictionary of arguments used when creating the db engine.
        """
        db_schema = None
        # self.db_url = 'sqlite:////tmp/sql.db'   # Works
        # self.db_url = 'sqlite://'               # Hangs
        # self.db_url = 'duckdb:////tmp/database.duckdb'  # Works
        # self.db_url = 'duckdb:///:memory:'      # Hangs
        if not hasattr(
            self, "_db_path"
        ):  # Extra safety in case this method is called more than once.
            self._db_path = self._get_db_file_path()
            lock_file = f"{self._db_path.absolute()}.lck"
            self._db_file_lock = FileLock(lock_file)
        db_url = f"sqlite:///{str(self._db_path.absolute())}"
        db_obfuscated_url = db_url
        connect_args = {}
        return db_schema, db_url, db_obfuscated_url, connect_args

    def _get_db_file_path(self) -> Path:
        home_dir = os.getenv("HOME", None)
        if not home_dir:
            raise ValueError("Could not get home directory from HOME env var")

        llmb_dir = f"{home_dir}/{USER_HOME_LLMB_DIR_NAME}"
        if not os.path.exists(llmb_dir):
            try:
                os.makedirs(llmb_dir)
            except FileExistsError as e:
                pass  # Something else got in ahead of us
        db_file = f"{llmb_dir}/{SQLITE_DB_FILE_NAME}"
        return Path(db_file)

    def _get_autoincr_column_type(self) -> Any:
        """By default, the auto increment column type is BigInteger (originally for postgres), but sub-classes
        can override to define another type, as is required for sqlite (Integer).
        The returned value will be used as the 'type' parameter to the Column initializer.
        """
        return Integer

    def add(
        self, item: Union[BASE_ITEM_TYPE, list[BASE_ITEM_TYPE]]
    ) -> Union[str, list[str]]:
        with self._db_file_lock:
            return super().add(item)

    def get_by_uuid(
        self, uuids: Optional[Union[str, list[str]]]
    ) -> Union[BASE_ITEM_TYPE, list[BASE_ITEM_TYPE]]:
        with self._db_file_lock:
            return super().get_by_uuid(uuids)

    def get_by_where(
        self,
        where: Optional[Union[str, dict]] = None,
        query_control: Optional[QueryControl] = None,
    ) -> list[BASE_ITEM_TYPE]:
        with self._db_file_lock:
            return super().get_by_where(where, query_control)

    def delete_table(self):
        with self._db_file_lock:
            super().delete_table()

    def delete(self, uuids: Union[str, list[str]]) -> None:
        with self._db_file_lock:
            super().delete(uuids)

    def update(
        self,
        item: BASE_ITEM_TYPE,
        update_updated_time: bool = True,
        create_if_not_exist: bool = True,
    ) -> None:
        with self._db_file_lock:
            super().update(item, update_updated_time, create_if_not_exist)


class SqliteArtifactRegistry(SqliteStorageOverrides, SQLArtifactRegistry):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


class SqliteBuildStorage(SqliteStorageOverrides, SQLBuildStorage):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


class SqliteEventStorage(SqliteStorageOverrides, SQLEventStorage):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


class SqliteSpaceStorage(SqliteStorageOverrides, SQLSpaceStorage):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


class SqliteStepRunStorage(SqliteStorageOverrides, SQLStepRunStorage):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


class SqliteTargetRunStorage(SqliteStorageOverrides, SQLTargetRunStorage):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


class SqliteNodeFailureStorage(SqliteStorageOverrides, SQLNodeFailureStorage):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


class SqliteSpaceUserStorage(SqliteStorageOverrides, SQLSpaceUserStorage):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
