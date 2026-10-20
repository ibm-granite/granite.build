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

"""The lineage storages' own SQL takes the file lock and retries, like get_by_where.

Without both, a graph read during a long indexer write raised ``database is
locked`` straight to the API.
"""

import os
import uuid as uuid_module

import pytest
from sqlalchemy.exc import OperationalError

from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory
from gbserver.storage.stored_lineage_row import StoredLineageRow

pytestmark = pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)


@pytest.fixture(name="rows")
def rows_fixture():
    table = f"t_lock_{uuid_module.uuid4().hex[:8]}"
    storage = SqliteStorageFactory().create_lineage_row_storage(table_name=table)
    storage.add(StoredLineageRow(job_id="J1", input="a", output="b"))
    yield storage
    storage.delete_table()


@pytest.fixture(name="no_wait")
def no_wait_fixture(monkeypatch):
    """Skip the retry's backoff so a retried test runs in milliseconds."""
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda _seconds: None)


class _CountingLock:
    def __init__(self, lock):
        self.lock = lock
        self.entered = 0

    def __enter__(self):
        self.entered += 1
        return self.lock.__enter__()

    def __exit__(self, *exc):
        return self.lock.__exit__(*exc)


@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.grouped_edges(["a"], "downstream"),
        lambda s: s.grouped_self_loops(["a"]),
        lambda s: s.count_jobs_touching("a"),
        lambda s: s.get_job_ids_touching("a", 10, 0),
        lambda s: s.get_rows_by_input(["a"], self_loops=False),
    ],
    ids=["edges", "self_loops", "count", "job_ids", "no_self_loops"],
)
def test_the_walk_sql_holds_the_file_lock(rows, call):
    lock = _CountingLock(rows._db_file_lock)  # pylint: disable=protected-access
    object.__setattr__(rows, "_db_file_lock", lock)
    call(rows)
    assert lock.entered >= 1


def test_a_locked_database_is_retried(rows, no_wait, monkeypatch):
    real = type(rows)._get_session_without_retry
    failures = iter([OperationalError("SELECT", {}, Exception("database is locked"))])

    def flaky(self):
        failure = next(failures, None)
        if failure is not None:
            raise failure
        return real(self)

    monkeypatch.setattr(type(rows), "_get_session_without_retry", flaky)
    assert [e.output for e in rows.grouped_edges(["a"], "downstream")] == ["b"]


def test_a_bad_argument_is_not_retried(rows, no_wait):
    with pytest.raises(ValueError):
        rows.grouped_edges(["a"], "sideways")


def test_tag_search_holds_the_file_lock():
    jobs = SqliteStorageFactory().create_lineage_job_storage(
        table_name=f"t_lockjob_{uuid_module.uuid4().hex[:8]}"
    )
    lock = _CountingLock(jobs._db_file_lock)  # pylint: disable=protected-access
    object.__setattr__(jobs, "_db_file_lock", lock)
    jobs.search_by_tags(["x=1"])
    assert lock.entered >= 1


def test_the_lock_is_released_between_retry_attempts(monkeypatch):
    """A retried read must not hold the process-wide lock through its backoff."""
    import threading

    from gbserver.storage.sql import sql_storage
    from gbserver.storage.sqlite import sqlite_storage

    lock = threading.RLock()
    free_during_backoff = []

    def probe_sleep(_seconds):
        # Another thread can take the lock while the read waits to retry.
        def take():
            got = lock.acquire(timeout=1)
            free_during_backoff.append(got)
            if got:
                lock.release()

        thread = threading.Thread(target=take)
        thread.start()
        thread.join()

    from tenacity import retry, retry_if_exception, stop_after_attempt

    monkeypatch.setattr(
        sqlite_storage,
        "retry_read",
        retry(
            sleep=probe_sleep,
            stop=stop_after_attempt(3),
            retry=retry_if_exception(sql_storage._is_retryable_read_error),
            reraise=True,
        ),
    )

    class _Storage:
        _db_file_lock = lock
        calls = 0

        @sql_storage.retry_read
        def read(self):
            self.calls += 1
            if self.calls == 1:
                raise OperationalError("SELECT", {}, Exception("database is locked"))
            return "ok"

    storage = _Storage()
    assert sqlite_storage._locked_per_attempt(storage, storage.read) == "ok"
    assert storage.calls == 2
    assert free_during_backoff == [True]


def test_a_deterministic_read_error_is_not_retried():
    from gbserver.storage.sql import sql_storage

    calls = []

    @sql_storage.retry_read
    def read():
        calls.append(1)
        raise OperationalError("SELECT", {}, Exception("no such column: tags"))

    with pytest.raises(OperationalError):
        read()
    assert len(calls) == 1


def test_update_fields_releases_the_lock_between_retry_attempts(monkeypatch):
    """Every SQLite storage's update_fields: no lock held through a backoff."""
    import threading

    from filelock import Timeout

    from gbserver.storage.stored_build import StoredBuild

    builds = SqliteStorageFactory().create_build_storage(
        table_name=f"t_lockupd_{uuid_module.uuid4().hex[:8]}"
    )
    try:
        uuid = builds.add(
            StoredBuild(name="b", space_name="s", source_uri="file://b", username="u")
        )
        lock = builds._db_file_lock  # pylint: disable=protected-access
        free_during_backoff = []

        def probe_sleep(_seconds):
            def take():
                try:
                    lock.acquire(timeout=1)
                except Timeout:
                    free_during_backoff.append(False)
                    return
                free_during_backoff.append(True)
                lock.release()

            thread = threading.Thread(target=take)
            thread.start()
            thread.join()

        monkeypatch.setattr("tenacity.nap.time.sleep", probe_sleep)
        real = type(builds)._get_session_without_retry
        failures = iter(
            [OperationalError("UPDATE", {}, Exception("database is locked"))]
        )

        def flaky(self):
            failure = next(failures, None)
            if failure is not None:
                raise failure
            return real(self)

        monkeypatch.setattr(type(builds), "_get_session_without_retry", flaky)
        assert builds.update_fields(uuid, {"name": "renamed"}).name == "renamed"
        assert free_during_backoff == [True]
    finally:
        builds.delete_table()
