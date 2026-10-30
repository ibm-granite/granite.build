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

"""``gb_lineage_index`` (SQLLineageRowStorage) tests against PostgreSQL.

The backend-agnostic bodies live in test/libgbtest/storage/lineage_index_storage.py;
re-running them here is what proves the two backends agree.
``SqliteLineageRowStorage`` is ``SqliteStorageOverrides + SQLLineageRowStorage``, so
everything the overrides touch -- connection specs, DDL dialect, index
introspection -- is untested on postgres until it runs here.

What only postgres can prove, and so is written here:

- The schema assertions. The SQLite tests read ``sqlite_master`` and
  ``PRAGMA table_info``; neither exists here, so indexes come from ``pg_indexes``
  and column types from ``information_schema.columns``.
- **Index name truncation.** ``__get_unique_index_ddl`` hashes any index name
  over 63 characters into ``uq_<sha1[:7]>_<suffix>`` because that is postgres's
  identifier limit. SQLite has no such limit, so the truncation branch is only
  ever reached here -- and a table_name_prefix in a real deployment is what pushes
  a name over it.
- **The ``""`` terminal sentinel, not NULL,** under a dialect strict about
  ``NULL != NULL``.

The bodies run here against the local test PostgreSQL and, re-exported by
test/integration/ibm/storage/sql/test_sql_lineage_index_storage.py, against the hosted one.
"""

from typing import Optional

import pytest
from libgbtest.local_postgres import LOCAL_POSTGRES_TEST_MARKS
from libgbtest.storage.lineage_index_storage import BaseLineageIndexStorageTest, row
from libgbtest.storage.postgres_storage import PostgresStorageTestMixin
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from gbserver.storage.sql.lineage_row_storage import SQLLineageRowStorage
from gbserver.storage.stored_lineage_row import TERMINAL

pytestmark = LOCAL_POSTGRES_TEST_MARKS


class SQLLineageIndexStorageTests(
    PostgresStorageTestMixin, BaseLineageIndexStorageTest
):
    """``gb_lineage_index`` tests, shared with the IBM copy (see TestSQLLineageIndexStorage below)."""

    def _get_tested_storage(self) -> SQLLineageRowStorage:
        storage = super()._get_tested_storage()
        assert isinstance(storage, SQLLineageRowStorage)
        return storage

    def _schema(self) -> Optional[str]:
        """The configured GBSERVER_SQL_SCHEMA, or None for the connection's default."""
        return self._get_tested_storage()._db_schema or None

    def _qualified_table(self) -> str:
        table = self._get_tested_storage().table_name
        schema = self._schema()
        return f"{schema}.{table}" if schema else table

    def _query(self, sql: str, parameters: Optional[dict] = None) -> list:
        engine = self._get_tested_storage()._engine
        assert engine is not None
        with engine.connect() as connection:
            return list(connection.execute(text(sql), parameters or {}).fetchall())

    def _index_definitions(self, table: Optional[str] = None) -> list[str]:
        rows = self._query(
            "SELECT indexdef FROM pg_indexes WHERE tablename = :table "
            "AND schemaname = coalesce(:schema, current_schema())",
            {
                "table": table or self._get_tested_storage().table_name,
                "schema": self._schema(),
            },
        )
        return [definition for (definition,) in rows]

    # Schema: the declared indexes and unique must exist in the real postgres table.

    @pytest.mark.parametrize("column", ["job_id", "job_store", "recorded_at"])
    def test_schema_column_is_indexed(self, column):
        self._get_tested_storage().add(row())
        definitions = " ".join(self._index_definitions())
        assert f"({column})" in definitions, f"{column} is not indexed: {definitions}"

    @pytest.mark.parametrize("columns", ["(input, output)", "(output, input)"])
    def test_schema_edge_composite_index(self, columns):
        """The hops use these; they replaced single-column input/output indexes."""
        self._get_tested_storage().add(row())
        definitions = self._index_definitions()
        assert any(
            columns in d and "UNIQUE" not in d.upper() for d in definitions
        ), definitions

    def test_schema_invalid_composite_index_rebuilt(self):
        """An interrupted CONCURRENTLY build leaves an INVALID index; it is redone."""
        storage = self._get_tested_storage()
        storage.add(row())
        validity = (
            "SELECT ic.relname, i.indisvalid FROM pg_index i "
            "JOIN pg_class ic ON ic.oid = i.indexrelid "
            "JOIN pg_class t ON t.oid = i.indrelid "
            r"WHERE t.relname = :table AND ic.relname LIKE 'ix\_%\_input\_output'"
        )
        ((is_superuser,),) = self._query(
            "SELECT rolsuper FROM pg_roles WHERE rolname = current_user"
        )
        if not is_superuser:
            pytest.skip("marking an index INVALID needs a superuser")
        ((name, valid),) = self._query(validity, {"table": storage.table_name})
        assert valid

        # What an interrupted build leaves behind (needs a superuser).
        with storage._engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE pg_index SET indisvalid = false WHERE indexrelid = "
                    "(SELECT oid FROM pg_class WHERE relname = :name)"
                ),
                {"name": name},
            )
        storage._BaseSQLItemStorage__create_composite_indexes()
        assert self._query(validity, {"table": storage.table_name}) == [(name, True)]

    def test_schema_composite_unique(self):
        """It can only be created with the table, never added later."""
        self._get_tested_storage().add(row())
        definitions = self._index_definitions()
        assert any(
            "UNIQUE" in d.upper() and "(job_id, input, output)" in d
            for d in definitions
        ), definitions

    def test_schema_traversal_columns_are_text(self):
        """``get_by_where`` only builds an ``IN`` clause for string columns."""
        self._get_tested_storage().add(row())
        rows = self._query(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = :table "
            "AND table_schema = coalesce(:schema, current_schema())",
            {
                "table": self._get_tested_storage().table_name,
                "schema": self._schema(),
            },
        )
        columns = dict(rows)
        for column in ("input", "output", "job_id"):
            assert columns[column] == "character varying", (column, columns[column])
        # Every promoted column is text, so there is no non-string type to assert.
        # get_by_where builds an IN clause only for string columns and silently
        # degrades otherwise, which is what that invariant protects against.
        assert "derivable" not in columns
        assert "build_id" not in columns
        assert "target_run_uuid" not in columns
        # uuid is the primary key; the autoincrement index column was dropped.
        assert "index" not in columns

    def test_schema_long_table_name_unique_index(self):
        """A unique index name over postgres's 63-char identifier limit is hashed.

        ``__get_unique_index_ddl`` falls back to ``uq_<sha1[:7]>_<suffix>`` when the
        natural name would be too long. SQLite has no identifier limit, so this
        branch is unreachable there -- yet it is exactly what a long
        ``table_name_prefix`` in a real deployment triggers, and an over-long name
        would make the table creation itself fail.
        """
        # Long enough that "<table>_job_id_input_output" exceeds 63 chars. Built on
        # the prefixed table name so the base class's teardown never sees it, and
        # dropped here.
        table = (self._get_tested_storage().table_name + "_" + "x" * 63)[:63]
        storage = self._get_storage_factory().create_lineage_row_storage(
            table_name=table
        )
        try:
            storage.add(row(job_id="J", input="a", output="b"))

            # The guard still holds under the hashed name.
            with pytest.raises(IntegrityError):
                storage.add(row(job_id="J", input="a", output="b"))

            definitions = self._index_definitions(table)
            unique = [
                d
                for d in definitions
                if "UNIQUE" in d.upper() and "(job_id, input, output)" in d
            ]
            assert unique, definitions
            for definition in definitions:
                # Every emitted identifier must be within postgres's limit.
                name = definition.split(" ON ")[0].split()[-1].strip('"')
                assert len(name) <= 63, name
        finally:
            storage.delete_table()

    # The "" sentinel, not NULL, is what protects terminal rows. This is the one
    # place the choice is dialect-sensitive: in SQL NULL never equals NULL, so a
    # NULL endpoint slips past a unique index.

    def test_terminal_creation_rows_of_two_jobs_coexist(self):
        storage = self._get_tested_storage()
        storage.add(row(job_id="C1", input=TERMINAL, output="x"))
        storage.add(row(job_id="C2", input=TERMINAL, output="x"))
        assert len(storage.get_rows_by_output(["x"])) == 2

    def test_terminal_stored_as_empty_string(self):
        storage = self._get_tested_storage()
        storage.add(row(job_id="C", input=TERMINAL, output="x"))
        nulls = self._query(
            f"SELECT count(*) FROM {self._qualified_table()} "
            "WHERE input IS NULL OR output IS NULL"
        )
        assert nulls == [(0,)]


class TestSQLLineageIndexStorage(SQLLineageIndexStorageTests):
    """``gb_lineage_index`` tests against the local test PostgreSQL."""
