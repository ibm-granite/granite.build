"""``get_successful_finished_since`` on a real SQLite table.

The ``finished_at`` column holds two spellings, one of them local wall-clock with
its offset dropped, so the filter reads the JSON blob's offset-carrying value. These
rows are rewritten to every shape a deployment has been seen to hold, on a table
unique to the test.
"""

import os
import uuid as uuid_module
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text

from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.status import Status

pytestmark = pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)

CUTOFF = datetime(2026, 9, 25, 18, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(name="targets")
def targets_fixture():
    table = f"t_targets_{uuid_module.uuid4().hex[:8]}"
    storage = SqliteStorageFactory().create_target_storage(table_name=table)
    yield storage, table
    storage.delete_table()


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


def _rewrite(storage, table, uuid, column=None, json_value=None, json_null=False):
    """Set the raw column text and/or the JSON blob's ``finished_at``."""
    with storage._engine.begin() as conn:  # pylint: disable=protected-access
        if column is not None:
            conn.execute(
                text(f"UPDATE {table} SET finished_at = :v WHERE uuid = :u"),
                {"v": column, "u": uuid},
            )
        if json_null:
            conn.execute(
                text(
                    f"UPDATE {table} SET json = json_set(json, '$.finished_at', "
                    "json('null')) WHERE uuid = :u"
                ),
                {"u": uuid},
            )
        elif json_value is not None:
            conn.execute(
                text(
                    f"UPDATE {table} SET json = json_set(json, '$.finished_at', :v) "
                    "WHERE uuid = :u"
                ),
                {"v": json_value, "u": uuid},
            )


def _names(storage):
    return {t.name for t in storage.get_successful_finished_since(CUTOFF, 0, 100)}


def test_offsets_in_the_json_decide_not_the_column_text(targets):
    storage, table = targets
    cases = {
        # name: (column text as stored, json value, expected selected)
        "west_after": ("2026-09-25 15:30:00.000000", "2026-09-25T15:30:00-03:00", True),
        "west_before": (
            "2026-09-25 14:30:00.000000",
            "2026-09-25T14:30:00-03:00",
            False,
        ),
        "east_after": ("2026-09-25T20:30:00.000Z", "2026-09-25T20:30:00+02:00", True),
        "east_before": ("2026-09-25T19:30:00.000Z", "2026-09-25T19:30:00+02:00", False),
        "utc_after": ("2026-09-25T18:00:01.000Z", "2026-09-25T18:00:01Z", True),
        "utc_before": ("2026-09-25T17:59:59.000Z", "2026-09-25T17:59:59Z", False),
        "at_cutoff": ("2026-09-25 15:00:00.000000", "2026-09-25T15:00:00-03:00", True),
    }
    for name, (column, json_value, _) in cases.items():
        uuid = _add(storage, name, CUTOFF)
        _rewrite(storage, table, uuid, column=column, json_value=json_value)
    expected = {name for name, (_, _, keep) in cases.items() if keep}
    assert _names(storage) == expected


def test_a_null_json_time_still_reaches_the_caller(targets):
    storage, table = targets
    uuid = _add(storage, "unfinished", CUTOFF - timedelta(days=30))
    _rewrite(storage, table, uuid, json_null=True)
    assert _names(storage) == {"unfinished"}


def test_only_successful_targets_newest_first_and_paged(targets):
    storage, _ = targets
    _add(storage, "failed", CUTOFF + timedelta(hours=1), status=Status.FAILED)
    for i in range(3):
        _add(storage, f"t{i}", CUTOFF + timedelta(hours=i))
    first = storage.get_successful_finished_since(CUTOFF, 0, 2)
    second = storage.get_successful_finished_since(CUTOFF, 1, 2)
    assert [t.name for t in first] == ["t2", "t1"]
    assert [t.name for t in second] == ["t0"]


def test_a_json_time_without_an_offset_is_left_to_the_caller(targets):
    """julianday would read it as UTC, Python as local: SQL must not decide it."""
    storage, table = targets
    uuid = _add(storage, "naive", CUTOFF)
    # Well before the cutoff in any reading, yet still returned for Python to judge.
    _rewrite(
        storage,
        table,
        uuid,
        column="2026-09-20 10:00:00.000000",
        json_value="2026-09-20T10:00:00",
    )
    assert _names(storage) == {"naive"}


def test_the_scan_reads_the_status_finished_at_index(targets):
    """The indexer's every-scan query is served by ``(status, finished_at)``.

    Only an index is added: this test writes its own rows, never a real target.
    """
    storage, table = targets
    _add(storage, "t1", CUTOFF + timedelta(minutes=1))
    with storage._engine.connect() as conn:  # pylint: disable=protected-access
        indexes = {
            row[1] for row in conn.execute(text(f"PRAGMA index_list('{table}')"))
        }
        plan = " ".join(
            str(row[-1])
            for row in conn.execute(
                text(
                    f"EXPLAIN QUERY PLAN SELECT uuid FROM {table} "
                    "WHERE status = 'SUCCESS' ORDER BY finished_at DESC LIMIT 10"
                )
            )
        )
    assert f"ix_{table}_status_finished_at" in indexes
    assert f"ix_{table}_status_finished_at" in plan
    assert "TEMP B-TREE" not in plan


# -- get_successful_finished_after: the keyset read ----------------------------


def _keyset_names(storage, cutoff=None, page_size=2):
    """Every page of the keyset read, followed to the end."""
    names, after = [], None
    while True:
        page = storage.get_successful_finished_after(cutoff, after, page_size)
        names.extend(t.name for _, t in page)
        if len(page) < page_size:
            return names
        key, last = page[-1]
        after = (key, last.uuid)


def test_keyset_read_follows_the_instant_across_offsets(targets):
    storage, _ = targets
    minus3 = timezone(timedelta(hours=-3))
    _add(storage, "second", datetime(2026, 9, 25, 16, 0, tzinfo=minus3))  # 19:00Z
    _add(storage, "first", datetime(2026, 9, 25, 18, 30, tzinfo=timezone.utc))
    _add(storage, "third", datetime(2026, 9, 25, 21, 0, tzinfo=timezone.utc))
    _add(storage, "failed", CUTOFF, status=Status.FAILED)
    assert _keyset_names(storage) == ["first", "second", "third"]


def test_keyset_read_keeps_the_cutoff_and_drops_what_is_before(targets):
    storage, _ = targets
    _add(storage, "before", CUTOFF - timedelta(seconds=1))
    _add(storage, "at", CUTOFF)
    _add(storage, "after", CUTOFF + timedelta(hours=1))
    assert _keyset_names(storage, cutoff=CUTOFF) == ["at", "after"]


def test_keyset_read_seeks_on_the_expression_index(targets):
    storage, table = targets
    _add(storage, "t", CUTOFF)
    storage.get_successful_finished_after(CUTOFF, None, 10)
    sql = (
        f"EXPLAIN QUERY PLAN SELECT uuid FROM {table} WHERE status = 'SUCCESS' "
        f"AND {storage._SQLITE_FINISHED_KEY} >= julianday('2026-01-01') "
        f"ORDER BY {storage._SQLITE_FINISHED_KEY}, uuid LIMIT 10"
    )
    with storage._engine.connect() as conn:  # pylint: disable=protected-access
        plan = " ".join(str(row[-1]) for row in conn.execute(text(sql)))
    assert f"ix_{table}_status_finished_key" in plan, plan
    assert "TEMP B-TREE" not in plan, plan


def test_a_row_the_key_cannot_place_falls_back(targets):
    """An offset-less JSON time reads as UTC in SQL but local in Python."""
    storage, table = targets
    uuid = _add(storage, "legacy", CUTOFF)
    _rewrite(storage, table, uuid, json_value="2026-09-25T18:00:00")
    with pytest.raises(NotImplementedError):
        storage.get_successful_finished_after(None, None, 10)


def test_a_malformed_blob_still_writes_once_indexed(targets):
    """json_valid in the indexed expression: a bad blob gives a null key, not an
    error on every write to the table."""
    storage, table = targets
    uuid = _add(storage, "t", CUTOFF)
    storage.get_successful_finished_after(None, None, 10)  # builds the index
    with storage._engine.begin() as conn:  # pylint: disable=protected-access
        conn.execute(
            text(f"UPDATE {table} SET json = '{{' WHERE uuid = :u"), {"u": uuid}
        )
    _add(storage, "next", CUTOFF)


def test_a_malformed_blob_is_passed_over_not_a_stop(targets):
    """A blob SQLite cannot parse must not fail every scan: it is read as no value
    and skipped, and the rest are still paged by keyset."""
    storage, table = targets
    bad = _add(storage, "bad", CUTOFF)
    _add(storage, "good", CUTOFF + timedelta(hours=1))
    with storage._engine.begin() as conn:  # pylint: disable=protected-access
        conn.execute(
            text(f"UPDATE {table} SET json = '{{' WHERE uuid = :u"), {"u": bad}
        )
    assert _keyset_names(storage) == ["good"]
    assert _names(storage) == {"good"}


def test_an_unplaceable_row_written_later_still_falls_back(targets):
    """The probe runs on every scan's first page, not once per process."""
    storage, table = targets
    _add(storage, "t", CUTOFF)
    assert _keyset_names(storage, cutoff=CUTOFF) == ["t"]
    uuid = _add(storage, "legacy", CUTOFF + timedelta(hours=1))
    _rewrite(storage, table, uuid, json_value="2026-09-25T19:00:00")
    with pytest.raises(NotImplementedError):
        storage.get_successful_finished_after(CUTOFF, None, 10)


def test_an_offset_less_row_far_behind_the_cutoff_does_not_stop_the_keyset(targets):
    storage, table = targets
    uuid = _add(storage, "old", CUTOFF - timedelta(days=3))
    _rewrite(storage, table, uuid, json_value="2026-09-22T18:00:00")
    _add(storage, "new", CUTOFF + timedelta(hours=1))
    assert _keyset_names(storage, cutoff=CUTOFF) == ["new"]
