"""Column types must round-trip identically on every supported backend.

``UtcDateTime`` already has coverage through the API tests; these cover the two
types added for the real schema.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from sqlalchemy import Text
from sqlalchemy.dialects import mysql, postgresql, sqlite

from autotunex.db.types import MEDIUM_TEXT, UtcDateTime, Uuid36


def test_uuid36_renders_a_36_character_column_not_char_32() -> None:
    rendered = Uuid36().compile(dialect=mysql.dialect())

    assert rendered == "VARCHAR(36)"


def test_uuid36_stores_a_dashed_string() -> None:
    value = UUID("0b1e7c4a-1111-2222-3333-444455556666")

    stored = Uuid36().process_bind_param(value, sqlite.dialect())

    assert stored == "0b1e7c4a-1111-2222-3333-444455556666"


def test_uuid36_reads_back_a_uuid_object() -> None:
    loaded = Uuid36().process_result_value("0b1e7c4a-1111-2222-3333-444455556666", sqlite.dialect())

    assert loaded == UUID("0b1e7c4a-1111-2222-3333-444455556666")


def test_uuid36_round_trips() -> None:
    value = uuid4()
    column_type = Uuid36()

    stored = column_type.process_bind_param(value, sqlite.dialect())

    assert column_type.process_result_value(stored, sqlite.dialect()) == value


def test_uuid36_passes_none_through() -> None:
    assert Uuid36().process_bind_param(None, sqlite.dialect()) is None
    assert Uuid36().process_result_value(None, sqlite.dialect()) is None


def test_uuid36_reads_an_empty_string_as_none() -> None:
    """The live database holds ``''`` in nullable id columns.

    Found in ``datasets.artifact_id``: clients that send an empty string rather
    than omitting the field leave a zero-length value behind. ``UUID('')`` raises,
    which took down a whole page of results over two rows, so an empty string is
    read as "absent" — which is what it means.
    """
    assert Uuid36().process_result_value("", sqlite.dialect()) is None


def test_uuid36_still_raises_on_a_genuinely_corrupt_value() -> None:
    """Only the empty string is tolerated; real corruption must not be hidden."""
    with pytest.raises(ValueError, match="badly formed hexadecimal UUID string"):
        Uuid36().process_result_value("not-a-uuid", sqlite.dialect())


def test_utc_datetime_reads_a_mysql_zero_date_as_none() -> None:
    """MySQL's zero date reads back as a raw string, not a datetime.

    ``'0000-00-00 00:00:00'`` has no valid year/month/day, so ``asyncmy`` cannot
    build a ``datetime`` and returns the string unchanged. A ``trials``/``results``
    row the tuning pipeline wrote without a timestamp holds exactly this value; the
    old code called ``.tzinfo`` on it and 500-ed the whole job-detail response.
    Treated as absent (``None``), mirroring ``Uuid36``'s empty-string handling.
    """
    assert UtcDateTime().process_result_value("0000-00-00 00:00:00", mysql.dialect()) is None


def test_utc_datetime_labels_a_naive_value_as_utc() -> None:
    """The normal read path: a naive value from the driver is labelled UTC."""
    loaded = UtcDateTime().process_result_value(datetime(2026, 8, 18, 13, 15), sqlite.dialect())

    assert loaded == datetime(2026, 8, 18, 13, 15, tzinfo=UTC)


def test_utc_datetime_passes_none_through() -> None:
    assert UtcDateTime().process_result_value(None, sqlite.dialect()) is None


def test_medium_text_renders_mediumtext_only_on_mysql() -> None:
    # postgresql.dialect() is untyped in SQLAlchemy's stubs, unlike the mysql and
    # sqlite equivalents, so mypy strict needs the one suppression.
    postgres = postgresql.dialect()  # type: ignore[no-untyped-call]

    assert MEDIUM_TEXT.compile(dialect=mysql.dialect()) == "MEDIUMTEXT"
    assert MEDIUM_TEXT.compile(dialect=postgres) == "TEXT"
    assert MEDIUM_TEXT.compile(dialect=sqlite.dialect()) == "TEXT"


def test_medium_text_is_a_text_type_so_python_sees_str() -> None:
    assert isinstance(MEDIUM_TEXT, Text)
