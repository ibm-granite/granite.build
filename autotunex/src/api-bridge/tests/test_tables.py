# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the api-bridge Core metadata and portable column types."""

from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine, insert, select

from api_bridge.tables import UtcDateTime, Uuid36Str, configurations, metadata, users


def _sqlite_engine():
    eng = create_engine("sqlite://")
    metadata.create_all(eng)
    return eng


def test_metadata_create_all_builds_every_table():
    _sqlite_engine()
    names = set(metadata.tables)
    assert {
        "users",
        "configurations",
        "datasets",
        "jobs",
        "trials",
        "results",
        "log_entries",
        "gb_tasks",
    } <= names


def test_configurations_has_no_artifact_columns():
    assert "artifact_id" not in configurations.c
    assert "artifact_url" not in configurations.c


def test_jobs_has_no_precision_column():
    from api_bridge.tables import jobs

    assert "precision" not in jobs.c


def test_json_column_roundtrips_as_dict():
    eng = _sqlite_engine()
    payload = {"lr": {"min": 1e-5, "max": 1e-3}, "n": [1, 2, 3]}
    with eng.begin() as conn:
        conn.execute(
            insert(users).values(
                id="00000000-0000-0000-0000-000000000009",
                email="a@example.com",
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )
        conn.execute(
            insert(configurations).values(
                id="11111111-1111-1111-1111-111111111111",
                user_id="00000000-0000-0000-0000-000000000009",
                name="c1",
                tuner_type="bayesian",
                config_data=payload,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )
    with eng.connect() as conn:
        row = conn.execute(select(configurations)).mappings().first()
    assert row["config_data"] == payload  # a dict, not a JSON string


def test_uuid36str_binds_uuid_as_string():
    import uuid

    dec = Uuid36Str()
    u = uuid.uuid4()
    assert dec.process_bind_param(u, None) == str(u)
    assert dec.process_bind_param(None, None) is None
    assert dec.process_result_value(str(u), None) == str(u)


def test_utcdatetime_rejects_naive_and_normalizes():
    dec = UtcDateTime()
    with pytest.raises(TypeError):
        dec.process_bind_param(datetime(2026, 1, 1, 0, 0, 0), None)
    aware = datetime(2026, 1, 1, tzinfo=UTC)
    assert dec.process_bind_param(aware, None) == aware
    assert dec.process_result_value("0000-00-00 00:00:00", None) is None  # zero-date -> None


def test_enum_columns_are_native_enums_with_live_type_names():
    """Regression (Postgres): status/type are SQLAlchemy ``Enum`` with the live
    native type names, not ``String``.

    Declaring them ``String`` made psycopg bind writes as ``$1::VARCHAR``, which
    Postgres refuses to assign to its ``run_status``/``gb_task_type`` enum columns
    (``DatatypeMismatch``). The SQLite suite could not catch it — SQLite has no
    enum type — so this pins the column types directly.
    """
    from sqlalchemy import Enum

    from api_bridge.tables import gb_tasks, jobs, trials

    for col in (jobs.c.status, trials.c.status, gb_tasks.c.status):
        assert isinstance(col.type, Enum)
        assert col.type.name == "run_status"
    assert isinstance(gb_tasks.c.type.type, Enum)
    assert gb_tasks.c.type.type.name == "gb_task_type"


def test_enum_write_is_not_cast_to_varchar_on_postgres():
    """The status assignment must not render a ``::VARCHAR`` cast under psycopg.

    Compiled against the psycopg dialect (compile-only; needs no live DB) — this
    is the exact render that failed in production (``SET status=$1::VARCHAR``).
    """
    from sqlalchemy import update
    from sqlalchemy.dialects.postgresql import psycopg as pg

    from api_bridge.tables import jobs

    sql = str(update(jobs).values(status="COMPLETED").compile(dialect=pg.dialect()))
    assert "::VARCHAR" not in sql.upper()
