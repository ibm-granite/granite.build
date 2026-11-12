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

"""``gb_lineage_job`` (SQLLineageJobStorage) tests against PostgreSQL.

The backend-agnostic bodies live in test/libgbtest/storage/lineage_job_storage.py.
What is here is only what postgres can prove: the widths SQLite does not enforce.
SQLite ignores VARCHAR lengths, so a value too long for a column is stored there
and rejected by postgres -- where the job record is the only record of the
execution under the ``db`` provider, so a rejected write loses it for good.

The bodies run here against the local test PostgreSQL and, re-exported by
test/integration/ibm/storage/sql/test_sql_lineage_job_storage.py, against the hosted one.
"""

import uuid as uuid_module

from libgbtest.local_postgres import LOCAL_POSTGRES_TEST_MARKS
from libgbtest.storage.lineage_job_storage import BaseLineageJobStorageTest
from libgbtest.storage.postgres_storage import PostgresStorageTestMixin
from sqlalchemy import text

from gbserver.lineage.merge import ADDED, REJECTED, upsert_job
from gbserver.storage.stored_lineage_job import StoredLineageJob
from gbserver.storage.stored_lineage_row import MAX_JOB_ID_LENGTH

pytestmark = LOCAL_POSTGRES_TEST_MARKS

# The tags a real target carries, joined: ~300 chars, past the old VARCHAR(256).
_REALISTIC_TAGS = [
    f"build_id={uuid_module.uuid4()}",
    f"target_run_uuid={uuid_module.uuid4()}",
    f"target_id={uuid_module.uuid4()}",
    "username=someone.with.a.long.name@example.com",
    "space_name=a-fairly-long-space-name",
    f"output_id={uuid_module.uuid4()}",
    "team=data-platform",
    "purpose=evaluation-sweep",
]


class SQLLineageJobStorageTests(PostgresStorageTestMixin, BaseLineageJobStorageTest):
    """``gb_lineage_job`` tests, shared with the IBM copy (see TestSQLLineageJobStorage below)."""

    def _tags_type(self, storage) -> str:
        with storage._engine.connect() as connection:
            return connection.execute(
                text(
                    "SELECT data_type FROM information_schema.columns "
                    "WHERE table_name = :table AND column_name = 'tags' "
                    "AND table_schema = coalesce(:schema, current_schema())"
                ),
                {"table": storage.table_name, "schema": storage._db_schema or None},
            ).scalar()

    def test_schema_tags_realistic_set(self):
        storage = self._get_tested_storage()
        assert len(",".join(_REALISTIC_TAGS)) > 256
        assert upsert_job(
            storage, StoredLineageJob(job_id="J", tags=_REALISTIC_TAGS)
        ) == (ADDED)
        stored = storage.get_job("J")
        assert stored is not None
        assert sorted(stored.tags) == sorted(_REALISTIC_TAGS)

    def test_schema_tags_unbounded_text(self):
        storage = self._get_tested_storage()
        storage.add(StoredLineageJob(job_id="seed"))
        assert self._tags_type(storage) == "text"

    def test_schema_tags_varchar_widened(self):
        """A table from before ``tags`` was Text keeps VARCHAR(256) unless the schema
        adjust widens it; this is the upgrade path, not the fresh-table one."""
        storage = self._get_tested_storage()
        storage.add(StoredLineageJob(job_id="seed"))
        table = (
            f"{storage._db_schema}.{storage.table_name}"
            if storage._db_schema
            else storage.table_name
        )
        with storage._engine.connect() as connection:
            connection.execute(
                text(f"ALTER TABLE {table} ALTER COLUMN tags TYPE VARCHAR(256)")
            )
            connection.commit()
        assert self._tags_type(storage) == "character varying"

        reopened = self._get_storage_factory().create_lineage_job_storage(
            table_name=storage.table_name
        )
        assert upsert_job(
            reopened, StoredLineageJob(job_id="J", tags=_REALISTIC_TAGS)
        ) == (ADDED)
        assert self._tags_type(reopened) == "text"

    def test_upsert_job_rejects_over_long_job_id(self):
        storage = self._get_tested_storage()
        job_id = "x" * (MAX_JOB_ID_LENGTH + 1)
        assert upsert_job(storage, StoredLineageJob(job_id=job_id)) == REJECTED
        assert storage.get_job(job_id) is None


class TestSQLLineageJobStorage(SQLLineageJobStorageTests):
    """``gb_lineage_job`` tests against the local test PostgreSQL."""
