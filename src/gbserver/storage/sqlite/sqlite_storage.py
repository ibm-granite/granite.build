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
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar, Union

from filelock import FileLock
from pydantic import BaseModel
from sqlalchemy import Integer, text

from gbcommon.types.constants import get_gb_home_dir
from gbserver.storage.sql.artifact_registry import SQLArtifactRegistry
from gbserver.storage.sql.build_storage import SQLBuildStorage
from gbserver.storage.sql.event_storage import SQLEventStorage
from gbserver.storage.sql.kv_pair_storage import SQLKeyValuePairStorage
from gbserver.storage.sql.lineage_job_storage import SQLLineageJobStorage
from gbserver.storage.sql.lineage_row_storage import SQLLineageRowStorage
from gbserver.storage.sql.node_failure_storage import SQLNodeFailureStorage
from gbserver.storage.sql.space_storage import SQLSpaceStorage
from gbserver.storage.sql.space_user_storage import SQLSpaceUserStorage
from gbserver.storage.sql.sql_storage import retry_read
from gbserver.storage.sql.steprun_storage import SQLStepRunStorage
from gbserver.storage.sql.target_run_storage import SQLTargetRunStorage
from gbserver.storage.storage import BASE_ITEM_TYPE, IItemStorage, QueryControl
from gbserver.storage.stored_target_run import StoredTargetRun

# Legacy location, retained for the one-time standalone startup migration.
LEGACY_LLMB_DIR_NAME = ".llmb"
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
            # One FileLock per path, shared by every storage in the process, so
            # its (per-thread) re-entrancy holds across storages too: two
            # instances on the same .lck would otherwise block each other.
            self._db_file_lock = FileLock(lock_file, is_singleton=True)
        db_url = f"sqlite:///{str(self._db_path.absolute())}"
        db_obfuscated_url = db_url
        connect_args = {}  # type: ignore[var-annotated]
        return db_schema, db_url, db_obfuscated_url, connect_args

    def _get_db_file_path(self) -> Path:
        # Stored under the shared GB home dir (default ~/.granite.build); the
        # standalone startup migration copies any legacy ~/.llmb db here once.
        gb_home_dir = get_gb_home_dir()
        if not os.path.exists(gb_home_dir):
            try:
                os.makedirs(gb_home_dir)
            except FileExistsError:
                pass  # Something else got in ahead of us
        db_file = os.path.join(gb_home_dir, SQLITE_DB_FILE_NAME)
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
            return super().add(item)  # type: ignore[misc]

    def get_by_uuid(
        self, uuids: Optional[Union[str, list[str]]]
    ) -> Union[BASE_ITEM_TYPE, list[BASE_ITEM_TYPE]]:
        with self._db_file_lock:
            return super().get_by_uuid(uuids)  # type: ignore[misc]

    def get_by_where(
        self,
        where: Optional[Union[str, dict]] = None,
        query_control: Optional[QueryControl] = None,
    ) -> list[BASE_ITEM_TYPE]:
        with self._db_file_lock:
            return super().get_by_where(where, query_control)  # type: ignore[misc]

    def _ensure_table(self) -> bool:
        return _locked_per_attempt(self, super()._ensure_table)  # type: ignore[misc]

    def delete_table(self):
        with self._db_file_lock:
            super().delete_table()

    def delete(self, uuids: Union[str, list[str]]) -> None:
        with self._db_file_lock:
            super().delete(uuids)  # type: ignore[misc]

    def update_fields(self, *args, **kwargs):
        method = super().update_fields  # type: ignore[misc]
        return _locked_per_attempt(
            self, method, *args, retrying=method.retry.wraps, **kwargs
        )

    def update(
        self,
        item: BASE_ITEM_TYPE,
        update_updated_time: bool = True,
        create_if_not_exist: bool = True,
    ) -> None:
        with self._db_file_lock:
            super().update(item, update_updated_time, create_if_not_exist)  # type: ignore[misc]


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


# Indexes on expressions using date functions need SQLite 3.20; an older library
# opening a file that holds one fails on its schema, so it is never created there.
_EXPRESSION_INDEX_MIN_SQLITE = (3, 20, 0)


class SqliteTargetRunStorage(SqliteStorageOverrides, SQLTargetRunStorage):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._finished_key_indexed = False

    def get_successful_finished_after(
        self,
        cutoff_utc: Optional[datetime],
        after: Optional[tuple[Any, str]],
        page_size: int,
    ) -> list[tuple[Any, StoredTargetRun]]:
        if sqlite3.sqlite_version_info < _EXPRESSION_INDEX_MIN_SQLITE:
            raise NotImplementedError(
                f"SQLite {sqlite3.sqlite_version} cannot index the completion time"
            )
        if not self._finished_key_indexed:
            self._create_finished_key_index()
        return _locked_per_attempt(
            self,
            super().get_successful_finished_after,
            cutoff_utc,
            after,
            page_size,
        )

    def _create_finished_key_index(self) -> None:
        """Build the keyset read's expression index, once per process.

        A failure is logged and the read goes on unindexed (slow, still correct);
        it is retried by the next process.
        """
        self._finished_key_indexed = True
        try:
            with self._db_file_lock:  # pylint: disable=protected-access
                if not self._ensure_table_without_retry():
                    return
                assert self._engine is not None
                with self._engine.begin() as connection:
                    connection.execute(text(self._finished_key_index_ddl()))
        except Exception as e:  # pylint: disable=broad-except
            self.logger.warning(
                f"Could not create the completion-time index on {self.table_name}: {e}"
            )

    def get_successful_finished_since(
        self, cutoff_utc: datetime, page_index: int, page_size: int
    ) -> list[StoredTargetRun]:
        return _locked_per_attempt(
            self,
            super().get_successful_finished_since,
            cutoff_utc,
            page_index,
            page_size,
        )


def _locked_per_attempt(storage, method, *args, retrying=None, **kwargs):
    """Call a retried SQL method holding the file lock per attempt only.

    Locking around the decorated method would hold the process-wide lock through
    every backoff (minutes), stalling all other SQLite reads and writes; this
    retries the undecorated function, releasing the lock between attempts.
    ``retrying`` is the retry decorator to apply, ``retry_read`` by default; a
    method with its own policy passes that (``method.retry.wraps``).
    """
    inner = getattr(method, "__wrapped__", None)
    if inner is None:
        with storage._db_file_lock:  # pylint: disable=protected-access
            return method(*args, **kwargs)

    @(retrying or retry_read)
    def attempt():
        with storage._db_file_lock:  # pylint: disable=protected-access
            return inner(storage, *args, **kwargs)

    return attempt()


class SqliteLineageRowStorage(SqliteStorageOverrides, SQLLineageRowStorage):
    """Sqlite storage for lineage rows.

    The walk's own SQL (grouped edges, self-loops, the jobs touching an artifact)
    opens its sessions directly rather than through ``get_by_where``, so it takes
    the file lock here, like every other read; otherwise a read during a long
    indexer write sees ``database is locked``. The lock is taken per retry attempt
    (``_locked_per_attempt``), and those methods call the non-retrying
    ``_ensure_table_without_retry`` so no backoff sleeps under it. The lock is one
    ``FileLock`` per path for the whole process and re-entrant per thread, so
    methods that call one another under it do not deadlock.
    """

    def count_jobs_touching(self, *args, **kwargs) -> int:
        return _locked_per_attempt(self, super().count_jobs_touching, *args, **kwargs)

    def get_job_ids_touching(self, *args, **kwargs) -> list[str]:
        return _locked_per_attempt(self, super().get_job_ids_touching, *args, **kwargs)

    def grouped_self_loops(self, *args, **kwargs) -> list:
        return _locked_per_attempt(self, super().grouped_self_loops, *args, **kwargs)

    def grouped_edges(self, *args, **kwargs) -> list:
        return _locked_per_attempt(self, super().grouped_edges, *args, **kwargs)

    def _hop_rows(self, *args, **kwargs) -> list:
        return _locked_per_attempt(self, super()._hop_rows, *args, **kwargs)

    def delete_all_rows(self) -> int:
        with self._db_file_lock:
            return super().delete_all_rows()

    def add_replacing(self, *args, **kwargs) -> None:
        with self._db_file_lock:
            super().add_replacing(*args, **kwargs)


class SqliteLineageJobStorage(SqliteStorageOverrides, SQLLineageJobStorage):
    """Sqlite storage for lineage job records."""

    def _tag_candidates(self, *args, **kwargs) -> list:
        return _locked_per_attempt(self, super()._tag_candidates, *args, **kwargs)

    def get_recorded_after(self, *args, **kwargs) -> list:
        return _locked_per_attempt(self, super().get_recorded_after, *args, **kwargs)


class SqliteNodeFailureStorage(SqliteStorageOverrides, SQLNodeFailureStorage):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


class SqliteSpaceUserStorage(SqliteStorageOverrides, SQLSpaceUserStorage):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


class SqliteKeyValuePairStorage(SqliteStorageOverrides, SQLKeyValuePairStorage):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
