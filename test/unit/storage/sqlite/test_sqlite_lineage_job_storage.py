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

"""``gb_lineage_job`` (SqliteLineageJobStorage) tests against SQLite.

The backend-agnostic bodies live in test/libgbtest/storage/lineage_job_storage.py.
What is here is only what SQLite can prove: that the declared indexes and the
unique on ``job_id`` actually exist (``__create_unique_indexes`` only *warns* on
failure, so a clean log is not evidence), read from ``sqlite_master`` and
``PRAGMA table_info``.
"""

import os
import sqlite3

import pytest
from libgbtest.storage.lineage_job_storage import BaseLineageJobStorageTest, job

from gbserver.storage.sqlite.sqlite_storage import SqliteLineageJobStorage
from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory


@pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
class TestSqliteLineageJobStorage(BaseLineageJobStorageTest):

    @classmethod
    def _get_storage_factory(cls):
        return SqliteStorageFactory()

    def _get_tested_storage(self) -> SqliteLineageJobStorage:
        storage = super()._get_tested_storage()
        assert isinstance(storage, SqliteLineageJobStorage)
        return storage

    def _query_schema(self, sql: str, parameters: tuple = ()) -> list:
        storage = self._get_tested_storage()
        connection = sqlite3.connect(str(storage._get_db_file_path()))
        try:
            return connection.execute(sql, parameters).fetchall()
        finally:
            connection.close()

    def _index_statements(self) -> list:
        return self._query_schema(
            "SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name=?",
            (self._get_tested_storage().table_name,),
        )

    def _column_types(self) -> dict:
        table = self._get_tested_storage().table_name
        return {
            name: kind
            for _, name, kind, *_ in self._query_schema(f"PRAGMA table_info('{table}')")
        }

    # Schema: the declared indexes and unique must actually exist in the table.
    # ``__create_unique_indexes`` only warns on failure and runs only at
    # table-creation time, so these assert the index is really there rather than
    # trusting a clean log.

    @pytest.mark.parametrize("column", ["job_id", "space_name", "owner", "started_at"])
    def test_schema_column_is_indexed(self, column):
        self._get_tested_storage().add(job())
        statements = " ".join(sql or "" for _, sql in self._index_statements())
        assert f"({column})" in statements, f"{column} is not indexed: {statements}"

    def test_schema_unique_on_job_id(self):
        """It can only be created with the table, never added later."""
        self._get_tested_storage().add(job())
        statements = [sql or "" for _, sql in self._index_statements()]
        unique = [s for s in statements if "UNIQUE" in s and "job_id" in s]
        assert unique, statements
        assert "(job_id)" in unique[0]

    def test_schema_every_promoted_column_is_text(self):
        """get_by_where only builds an IN clause for string columns; anything else
        silently degrades to ``column == [list]`` -- a predicate that returns
        plausible but wrong rows with no error.

        So the invariant is that *every* promoted column is text, which makes that
        failure unreachable rather than merely avoided by convention.
        """
        self._get_tested_storage().add(job())
        columns = self._column_types()
        for column in (
            "job_id",
            "job_namespace",
            "space_name",
            "owner",
            "status",
            "started_at",
        ):
            assert columns[column].startswith("VARCHAR"), (column, columns[column])

        non_text = {
            name: kind
            for name, kind in columns.items()
            # "json" is the blob; it is never a query predicate. (There is no
            # autoincrement "index" column -- uuid is the primary key.)
            if name != "json" and not kind.startswith(("VARCHAR", "TEXT"))
        }
        # The joined tag list has no natural width, so it is unbounded text: on
        # Postgres a VARCHAR(256) rejected realistic tag sets and lost the record.
        assert columns["tags"] == "TEXT", columns["tags"]
        assert not non_text, f"a non-text promoted column is a latent bug: {non_text}"

    def test_schema_started_at_is_text(self):
        """Ordering is lexicographic over the source's own spelling. Converting the
        column to DateTime would also break the all-text IN invariant above.
        """
        self._get_tested_storage().add(job())
        assert self._column_types()["started_at"].startswith("VARCHAR")

    def test_schema_tags_comma_joined_and_sorted(self):
        """The gb_builds/gb_artifacts form, which the %like% filter relies on."""
        storage = self._get_tested_storage()
        storage.add(job(job_id="tagged", tags=["b=2", "a=1"]))
        (stored,) = self._query_schema(
            f"SELECT tags FROM {storage.table_name} WHERE job_id = 'tagged'",  # nosec
        )
        assert stored[0] == "a=1,b=2"
        # The item keeps its list; only the column is joined.
        assert storage.get_job("tagged").tags == ["b=2", "a=1"]
