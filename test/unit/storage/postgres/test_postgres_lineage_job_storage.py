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

"""``gb_lineage_job`` against postgres: the widths SQLite does not enforce.

SQLite ignores VARCHAR lengths, so a value too long for a column is stored there
and rejected by postgres -- where the job record is the only record of the
execution under the ``db`` provider, so a rejected write loses it for good.

The bodies run here against the local test PostgreSQL and, re-exported by
test/integration/ibm/storage/sql/test_sql_lineage_job_storage.py, against the hosted one.
"""

import uuid as uuid_module

import pytest
from libgbtest.local_postgres import LOCAL_POSTGRES_TEST_MARKS

from gbserver.lineage.merge import ADDED, REJECTED, upsert_job
from gbserver.storage.sql.storage_factory import SQLStorageFactory
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


@pytest.fixture(name="storage")
def sql_storage_fixture():
    table = f"t_ljob_{uuid_module.uuid4().hex[:8]}"
    storage = SQLStorageFactory().create_lineage_job_storage(table_name=table)
    yield storage
    try:
        storage.delete_table()
    except Exception:  # pragma: no cover - best-effort cleanup
        pass


def test_a_realistic_tag_set_is_stored(storage):
    assert len(",".join(_REALISTIC_TAGS)) > 256
    assert upsert_job(storage, StoredLineageJob(job_id="J", tags=_REALISTIC_TAGS)) == (
        ADDED
    )
    stored = storage.get_job("J")
    assert stored is not None
    assert sorted(stored.tags) == sorted(_REALISTIC_TAGS)


def test_tags_column_is_unbounded_text(storage):
    from sqlalchemy import text

    storage.add(StoredLineageJob(job_id="seed"))
    schema = storage._db_schema or None
    with storage._engine.connect() as connection:
        kind = connection.execute(
            text(
                "SELECT data_type FROM information_schema.columns "
                "WHERE table_name = :table AND column_name = 'tags' "
                "AND table_schema = coalesce(:schema, current_schema())"
            ),
            {"table": storage.table_name, "schema": schema},
        ).scalar()
    assert kind == "text"


def _tags_type(storage) -> str:
    from sqlalchemy import text

    with storage._engine.connect() as connection:
        return connection.execute(
            text(
                "SELECT data_type FROM information_schema.columns "
                "WHERE table_name = :table AND column_name = 'tags' "
                "AND table_schema = coalesce(:schema, current_schema())"
            ),
            {"table": storage.table_name, "schema": storage._db_schema or None},
        ).scalar()


def test_an_existing_varchar_tags_column_is_widened(storage):
    """A table from before ``tags`` was Text keeps VARCHAR(256) unless the schema
    adjust widens it; this is the upgrade path, not the fresh-table one."""
    from sqlalchemy import text

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
    assert _tags_type(storage) == "character varying"

    reopened = SQLStorageFactory().create_lineage_job_storage(
        table_name=storage.table_name
    )
    assert upsert_job(reopened, StoredLineageJob(job_id="J", tags=_REALISTIC_TAGS)) == (
        ADDED
    )
    assert _tags_type(reopened) == "text"


def test_an_over_long_job_id_is_rejected_before_the_db(storage):
    job_id = "x" * (MAX_JOB_ID_LENGTH + 1)
    assert upsert_job(storage, StoredLineageJob(job_id=job_id)) == REJECTED
    assert storage.get_job(job_id) is None
