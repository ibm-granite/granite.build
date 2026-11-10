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

"""``gb_lineage_index`` (SqliteLineageRowStorage) tests against SQLite.

The backend-agnostic bodies live in test/libgbtest/storage/lineage_index_storage.py.
What is here is only what SQLite can prove: that the declared indexes and the
composite unique actually exist, read from ``sqlite_master`` and ``PRAGMA
table_info``.
"""

import os
import sqlite3

import pytest
from libgbtest.storage.lineage_index_storage import BaseLineageIndexStorageTest, row

from gbserver.storage.sqlite.sqlite_storage import SqliteLineageRowStorage
from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory
from gbserver.storage.stored_lineage_row import MAX_LINEAGE_URI_LENGTH


@pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
class TestSqliteLineageIndexStorage(BaseLineageIndexStorageTest):

    @classmethod
    def _get_storage_factory(cls):
        return SqliteStorageFactory()

    def _get_tested_storage(self) -> SqliteLineageRowStorage:
        storage = super()._get_tested_storage()
        assert isinstance(storage, SqliteLineageRowStorage)
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

    @pytest.mark.parametrize("column", ["job_id", "job_store", "recorded_at"])
    def test_schema_column_is_indexed(self, column):
        self._get_tested_storage().add(row())
        statements = " ".join(sql or "" for _, sql in self._index_statements())
        assert f"({column})" in statements, f"{column} is not indexed: {statements}"

    @pytest.mark.parametrize("columns", ["(input, output)", "(output, input)"])
    def test_schema_edge_composite_index(self, columns):
        """Both directions of the walk, and the per-edge GROUP BY, are index-only."""
        self._get_tested_storage().add(row())
        statements = [sql or "" for _, sql in self._index_statements()]
        assert any(columns in s and "UNIQUE" not in s for s in statements), statements

    def test_schema_composite_indexes_idempotent(self):
        """A schema adjust on an existing table re-runs creation without error."""
        storage = self._get_tested_storage()
        storage.add(row())
        before = sorted(name for name, _ in self._index_statements())
        storage._create_or_adjust_schema_item_dict(
            storage._convert_item_to_row_dict(storage._get_sample_item())
        )
        after = sorted(name for name, _ in self._index_statements())
        assert before == after

    def test_schema_composite_unique(self):
        """It can only be created with the table, never added later."""
        self._get_tested_storage().add(row())
        statements = [sql or "" for _, sql in self._index_statements()]
        unique = [s for s in statements if "UNIQUE" in s and "job_id" in s]
        assert unique, statements
        assert "(job_id, input, output)" in unique[0]

    def test_schema_every_promoted_column_is_text(self):
        """get_by_where only builds an IN clause for string columns; anything else
        silently degrades to ``column == [list]`` -- a predicate that returns
        plausible but wrong rows with no error.

        So the invariant is stronger than "the batched ones are text": *every*
        promoted column is text, which makes that failure unreachable rather than
        merely avoided by convention: there is no non-text promoted column left for
        it to happen to.
        """
        self._get_tested_storage().add(row())
        columns = self._column_types()
        # The producer is on the job record, not on every row.
        assert "source_system" not in columns
        # build_id and target_run_uuid stay in the blob - they're GB-specific
        assert "build_id" not in columns
        assert "target_run_uuid" not in columns

        for column in ("input", "output", "job_id", "job_store"):
            assert columns[column].startswith("VARCHAR"), (column, columns[column])

        non_text = {
            name: kind
            for name, kind in columns.items()
            # "json" is the blob; it is never a query predicate. (There is no
            # autoincrement "index" column -- uuid is the primary key.)
            if name != "json" and not kind.startswith("VARCHAR")
        }
        assert not non_text, f"a non-text promoted column is a latent bug: {non_text}"

    def test_schema_uri_column_width(self):
        """The endpoints are 512, matching MAX_LINEAGE_URI_LENGTH.

        A truncated URI would merge two distinct artifacts sharing a prefix, so the
        normalizer drops anything longer rather than letting the column cut it.
        """
        self._get_tested_storage().add(row())
        columns = self._column_types()
        for column in ("input", "output"):
            assert columns[column] == f"VARCHAR({MAX_LINEAGE_URI_LENGTH})"
