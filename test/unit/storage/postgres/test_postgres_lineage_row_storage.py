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

"""Storage-level tests for lineage rows against a real SQL (postgres) backend.

The behavioural half of this file is deliberately a subclass of the SQLite tests
in ``test/unit/storage/sqlite/test_sqlite_lineage_row_storage.py``: those bodies
are backend-agnostic, and re-running them here is what proves the two backends
agree. ``SqliteLineageRowStorage`` is ``SqliteStorageOverrides +
SQLLineageRowStorage``, so everything the overrides touch -- connection specs,
DDL dialect, index introspection -- is untested on postgres until it runs here.

What only postgres can prove, and so is written fresh below:

- The schema assertions. The SQLite tests read ``sqlite_master`` and
  ``PRAGMA table_info``; neither exists here, so indexes come from ``pg_indexes``
  and column types from ``information_schema.columns``.
- **Index name truncation.** ``__get_unique_index_ddl`` hashes any index name
  over 63 characters into ``uq_<sha1[:7]>_<suffix>`` because that is postgres's
  identifier limit (``sql_storage.py:388-393``). SQLite has no such limit, so the
  truncation branch is only ever reached here -- and a table_name_prefix in a real
  deployment is what pushes a name over it.
- **The composite unique under a real dialect.** The unique index is created with
  the table and can never be added later, so a postgres deployment that gets this
  wrong is not repairable in place.

The bodies run here against the local test PostgreSQL and, re-exported by
test/integration/ibm/storage/sql/test_sql_lineage_row_storage.py, against the hosted one.
"""

import uuid as uuid_module
from typing import Optional

import pytest

# Imported for its test bodies, not to be collected twice -- same aliasing trick
# the sqlite suite uses in the other direction.
import unit.storage.sqlite.test_sqlite_lineage_row_storage as HIDE_FROM_PYTEST
from libgbtest.local_postgres import LOCAL_POSTGRES_TEST_MARKS
from sqlalchemy.exc import IntegrityError

from gbserver.storage.sql.storage_factory import SQLStorageFactory
from gbserver.storage.stored_lineage_row import TERMINAL, StoredLineageRow

pytestmark = LOCAL_POSTGRES_TEST_MARKS

row = HIDE_FROM_PYTEST.row


@pytest.fixture(name="storage")
def sql_storage_fixture():
    """A lineage row storage on a postgres table unique to this test.

    Dropped afterwards: unlike the SQLite suite, which throws away a temp
    GB_HOME_DIR, these tables land in a shared database and would otherwise
    accumulate. The drop also matters for correctness of a re-run -- the unique
    index exists only if the table is created fresh, so a leftover table would
    quietly test without it.
    """
    table = f"t_lin_{uuid_module.uuid4().hex[:8]}"
    storage = SQLStorageFactory().create_lineage_row_storage(table_name=table)
    # Force the lazy create so schema introspection has something to read.
    storage.add(row(job_id=f"seed_{table}"))
    yield storage
    try:
        storage.delete_table()
    except Exception:  # pragma: no cover - best-effort cleanup
        pass


def _schema(storage):
    """The configured GBSERVER_SQL_SCHEMA, or None for the connection's default."""
    return storage._db_schema or None


def _qualified(storage) -> str:
    schema = _schema(storage)
    return f"{schema}.{storage.table_name}" if schema else storage.table_name


def _index_definitions(table, engine, schema) -> list[str]:
    from sqlalchemy import text

    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT indexdef FROM pg_indexes WHERE tablename = :table "
                "AND schemaname = coalesce(:schema, current_schema())"
            ),
            {"table": table, "schema": schema},
        ).fetchall()
    return [definition for (definition,) in rows]


def _seeded(storage) -> int:
    """Rows the fixture itself wrote, to be discounted from whole-table counts."""
    return len(storage.get_rows_by_job(f"seed_{storage.table_name}"))


# ---------------------------------------------------------------------------
# Behaviour: the SQLite bodies, re-run against postgres.
# ---------------------------------------------------------------------------


class TestSQLHops(HIDE_FROM_PYTEST.TestHops):
    """Indexed, batched traversal queries under the postgres dialect."""


class TestSQLJobGrouping(HIDE_FROM_PYTEST.TestJobGrouping):
    pass


class TestSQLDedupSupport(HIDE_FROM_PYTEST.TestDedupSupport):
    pass


class TestSQLUniqueConstraint(HIDE_FROM_PYTEST.TestUniqueConstraint):
    """The same idempotency guard, enforced by postgres rather than SQLite.

    Worth re-running rather than trusting: SQLite raises ``IntegrityError`` from
    its own unique index, while here the constraint is DDL this codebase emits as
    a string. A malformed ``CREATE UNIQUE INDEX`` would leave re-ingest silently
    duplicating, and only a real postgres round trip shows that.
    """


class TestSQLRoundTrip(HIDE_FROM_PYTEST.TestRoundTrip):
    pass


class TestSQLUriIsTheIdentity(HIDE_FROM_PYTEST.TestUriIsTheIdentity):
    pass


class TestSQLWalkAgainstRealStorage(HIDE_FROM_PYTEST.TestWalkAgainstRealStorage):
    """The traversal over a real postgres table.

    This is the one that matters most for the read API: ``walk_lineage`` issues
    one ``WHERE input IN (frontier)`` per level, and ``get_by_where`` only builds
    an ``IN`` for string columns. On a non-string column it degrades to
    ``column == [list]`` instead of raising, so a dialect mismatch here shows up
    as an empty graph rather than an error.
    """

    # The inherited body is named ..._in_sqlite; here the chain is stored in
    # postgres. Re-point the name so the report does not claim the wrong backend.
    test_walks_a_chain_stored_in_sqlite = None

    def test_walks_a_chain_stored_in_postgres(self, storage):
        from gbserver.lineage.walk import Direction, walk_lineage

        storage.add(row(job_id="J1", input="a", output="b"))
        storage.add(row(job_id="J2", input="b", output="c"))

        graph = walk_lineage(storage, ["a"], Direction.DESCENDANTS)
        assert graph.depths == {"a": 0, "b": 1, "c": 2}

        back = walk_lineage(storage, ["c"], Direction.ANCESTORS)
        assert back.depths == {"c": 0, "b": 1, "a": 2}


class TestPostgresSchema:
    """The declared indexes and unique must exist in the real postgres table."""

    @staticmethod
    def _query(storage, sql: str, parameters: Optional[dict] = None) -> list:
        from sqlalchemy import text

        with storage._engine.connect() as connection:
            return list(connection.execute(text(sql), parameters or {}).fetchall())

    @classmethod
    def _index_definitions(cls, storage) -> list[str]:
        return _index_definitions(storage.table_name, storage._engine, _schema(storage))

    @pytest.mark.parametrize("column", ["job_id", "job_store", "recorded_at"])
    def test_column_is_indexed(self, storage, column):
        definitions = " ".join(self._index_definitions(storage))
        assert f"({column})" in definitions, f"{column} is not indexed: {definitions}"

    @pytest.mark.parametrize("columns", ["(input, output)", "(output, input)"])
    def test_composite_hop_index_exists(self, storage, columns):
        """The hops use these; they replaced single-column input/output indexes."""
        definitions = self._index_definitions(storage)
        assert any(
            columns in d and "UNIQUE" not in d.upper() for d in definitions
        ), definitions

    def test_invalid_composite_index_is_rebuilt(self, storage):
        """An interrupted CONCURRENTLY build leaves an INVALID index; it is redone."""
        validity = (
            "SELECT ic.relname, i.indisvalid FROM pg_index i "
            "JOIN pg_class ic ON ic.oid = i.indexrelid "
            "JOIN pg_class t ON t.oid = i.indrelid "
            r"WHERE t.relname = :table AND ic.relname LIKE 'ix\_%\_input\_output'"
        )
        ((is_superuser,),) = self._query(
            storage, "SELECT rolsuper FROM pg_roles WHERE rolname = current_user"
        )
        if not is_superuser:
            pytest.skip("marking an index INVALID needs a superuser")
        ((name, valid),) = self._query(storage, validity, {"table": storage.table_name})
        assert valid
        from sqlalchemy import text

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
        assert self._query(storage, validity, {"table": storage.table_name}) == [
            (name, True)
        ]

    def test_composite_unique_exists(self, storage):
        """It can only be created with the table, never added later."""
        definitions = self._index_definitions(storage)
        assert any(
            "UNIQUE" in d.upper() and "(job_id, input, output)" in d
            for d in definitions
        ), definitions

    def test_traversal_columns_are_text(self, storage):
        """``get_by_where`` only builds an ``IN`` clause for string columns."""
        rows = self._query(
            storage,
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = :table "
            "AND table_schema = coalesce(:schema, current_schema())",
            {"table": storage.table_name, "schema": _schema(storage)},
        )
        columns = {name: kind for name, kind in rows}
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


class TestPostgresIndexNameLimit:
    """A unique index name over postgres's 63-char identifier limit is hashed.

    ``__get_unique_index_ddl`` falls back to ``uq_<sha1[:7]>_<suffix>`` when the
    natural name would be too long. SQLite has no identifier limit, so this branch
    is unreachable there -- yet it is exactly what a long ``table_name_prefix`` in
    a real deployment triggers, and an over-long name would make the table
    creation itself fail.
    """

    def test_long_table_name_still_gets_its_unique_index(self):
        # Long enough that "<table>_job_id_input_output" exceeds 63 chars.
        table = "t_lin_" + ("x" * 48) + uuid_module.uuid4().hex[:6]
        assert len(table) > 45
        storage = SQLStorageFactory().create_lineage_row_storage(table_name=table)
        try:
            storage.add(row(job_id="J", input="a", output="b"))

            # The guard still holds under the hashed name.
            with pytest.raises(IntegrityError):
                storage.add(row(job_id="J", input="a", output="b"))

            definitions = _index_definitions(table, storage._engine, _schema(storage))
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
            try:
                storage.delete_table()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass


class TestPostgresTerminalSentinel:
    """The ``""`` sentinel, not NULL, is what protects terminal rows.

    This is the one place the choice is dialect-sensitive: in SQL ``NULL`` never
    equals ``NULL``, so a NULL endpoint slips past a unique index. Postgres is
    strict about that in a way worth pinning separately from SQLite.
    """

    def test_creation_rows_of_two_jobs_coexist(self, storage):
        storage.add(row(job_id="C1", input=TERMINAL, output="x"))
        storage.add(row(job_id="C2", input=TERMINAL, output="x"))
        assert len(storage.get_rows_by_output(["x"])) == 2

    def test_duplicate_creation_row_is_rejected(self, storage):
        storage.add(row(job_id="C", input=TERMINAL, output="x"))
        with pytest.raises(IntegrityError):
            storage.add(row(job_id="C", input=TERMINAL, output="x"))

    def test_sentinel_is_stored_as_empty_string_not_null(self, storage):
        from sqlalchemy import text

        storage.add(row(job_id="C", input=TERMINAL, output="x"))
        with storage._engine.connect() as connection:
            nulls = connection.execute(
                text(
                    f"SELECT count(*) FROM {_qualified(storage)} "
                    "WHERE input IS NULL OR output IS NULL"
                )
            ).scalar()
        assert nulls == 0
