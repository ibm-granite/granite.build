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

"""``get_successful_finished_after`` (the lineage indexer's keyset read) on postgres.

The key is the ``finished_at`` timestamp column; a successful row without one
(written before the column existed) cannot be placed and makes the read fall back.
"""

import uuid as uuid_module
from datetime import datetime, timedelta, timezone

import pytest
from libgbtest.local_postgres import LOCAL_POSTGRES_TEST_MARKS
from sqlalchemy import text

from gbserver.storage.sql.storage_factory import SQLStorageFactory
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.status import Status

pytestmark = LOCAL_POSTGRES_TEST_MARKS

CUTOFF = datetime(2026, 9, 25, 18, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(name="storage")
def storage_fixture():
    table = f"t_tgt_{uuid_module.uuid4().hex[:8]}"
    storage = SQLStorageFactory().create_target_storage(table_name=table)
    yield storage
    try:
        storage.delete_table()
    except Exception:  # pragma: no cover - best-effort cleanup
        pass


def _add(storage, name, finished_at, status=Status.SUCCESS):
    target = StoredTargetRun(
        build_id="b1",
        environment_uri="env",
        name=name,
        status=status,
        finished_at=finished_at,
    )
    storage.add(target)
    return target.uuid


def _names(storage, cutoff=None, page_size=2):
    names, after = [], None
    while True:
        page = storage.get_successful_finished_after(cutoff, after, page_size)
        names.extend(t.name for _, t in page)
        if len(page) < page_size:
            return names
        key, last = page[-1]
        after = (key, last.uuid)


def test_keyset_read_follows_the_instant_across_offsets(storage):
    minus3 = timezone(timedelta(hours=-3))
    _add(storage, "second", datetime(2026, 9, 25, 16, 0, tzinfo=minus3))  # 19:00Z
    _add(storage, "first", datetime(2026, 9, 25, 18, 30, tzinfo=timezone.utc))
    _add(storage, "third", datetime(2026, 9, 25, 21, 0, tzinfo=timezone.utc))
    _add(storage, "before", CUTOFF - timedelta(seconds=1))
    _add(storage, "failed", CUTOFF, status=Status.FAILED)
    assert _names(storage, cutoff=CUTOFF) == ["first", "second", "third"]


def test_a_successful_row_without_the_column_falls_back(storage):
    uuid = _add(storage, "legacy", CUTOFF)
    table = (
        f"{storage._db_schema}.{storage.table_name}"
        if storage._db_schema
        else storage.table_name
    )
    with storage._engine.begin() as conn:  # pylint: disable=protected-access
        conn.execute(
            text(f"UPDATE {table} SET finished_at = NULL WHERE uuid = :u"), {"u": uuid}
        )
    with pytest.raises(NotImplementedError):
        storage.get_successful_finished_after(None, None, 10)
