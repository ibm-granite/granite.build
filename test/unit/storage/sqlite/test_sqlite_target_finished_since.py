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
