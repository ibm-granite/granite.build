# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Custom column types."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import DateTime, String, Text
from sqlalchemy.dialects.mysql import MEDIUMTEXT
from sqlalchemy.engine import Dialect
from sqlalchemy.types import TypeDecorator


class UtcDateTime(TypeDecorator[datetime]):
    """A timestamp that is always timezone-aware UTC in Python.

    PostgreSQL stores offsets; SQLite does not, so a naive value comes back from
    SQLite and the API would then serialize timestamps with no offset — clients
    could not tell which zone they were in. This normalizes both directions so
    the two backends behave identically:

    - on write, values must be aware and are converted to UTC
    - on read, naive values are labelled UTC
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        """Reject naive datetimes and normalize to UTC before storing."""
        if value is None:
            return None
        if value.tzinfo is None:
            raise TypeError("naive datetime rejected; use datetime.now(UTC)")
        return value.astimezone(UTC)

    def process_result_value(
        self, value: datetime | str | None, dialect: Dialect
    ) -> datetime | None:
        """Return an aware UTC datetime, or ``None`` for a value the driver couldn't parse.

        ``asyncmy`` cannot build a ``datetime`` from MySQL's zero date
        ``'0000-00-00 00:00:00'`` (there is no year, month, or day 0), so it hands the
        raw string straight through. ``trials``/``results`` rows the tuning pipeline
        inserts without a timestamp hold exactly that: the live columns are
        ``DATETIME NOT NULL`` with no default, so a lax ``sql_mode`` stores the zero
        date instead of rejecting the insert. Reading one such row must not 500 the
        whole page, so a non-``datetime`` is treated as absent (``None``) — the same
        tolerance :class:`Uuid36` gives the empty string.
        """
        if value is None:
            return None
        if isinstance(value, str):
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class Uuid36(TypeDecorator[UUID]):
    """A UUID stored as a 36-character dashed string.

    ``resources/autotunex_schema.sql`` declares these ids as ``VARCHAR(36)`` /
    ``CHAR(36)`` holding the dashed form MySQL's ``UUID()`` produces.
    SQLAlchemy's built-in :class:`sqlalchemy.Uuid` renders ``CHAR(32)`` and
    strips the dashes on non-native backends, so it would silently fail to
    mirror the live schema — and would not match the values already in it.

    Empty strings read back as ``None``. The live database contains them: nullable
    id columns written by clients that send ``""`` rather than omitting the field
    hold a zero-length value that is not a UUID and never was. Raising there would
    take down an entire page of results over one malformed row in one column, so
    the read path treats ``""`` as "absent", which is what it means. Writes are
    unaffected — this layer never produces an empty string.
    """

    impl = String(36)
    cache_ok = True

    def process_bind_param(self, value: UUID | None, dialect: Dialect) -> str | None:
        """Store the dashed string form."""
        if value is None:
            return None
        return str(value)

    def process_result_value(self, value: str | None, dialect: Dialect) -> UUID | None:
        """Parse the stored string back into a :class:`~uuid.UUID`.

        Raises:
            ValueError: the column holds a non-empty value that is not a UUID.
                Deliberately not swallowed — that indicates real data corruption
                rather than the benign empty-string case, and silently returning
                ``None`` would hide it.
        """
        if value is None or value == "":
            return None
        return UUID(value)


MEDIUM_TEXT = Text().with_variant(MEDIUMTEXT(), "mysql")
"""``TEXT`` everywhere, ``MEDIUMTEXT`` on MySQL.

Mirrors ``log_entries.message``. MySQL's ``TEXT`` caps at 64 KiB, which would
truncate a long training log; ``MEDIUMTEXT`` gives 16 MiB. SQLite and Postgres
have no such limit, so plain ``TEXT`` is correct there.

A module-level constant rather than a class because :meth:`with_variant` returns
an already-configured instance — there is nothing to subclass. Share it freely:
SQLAlchemy type instances are immutable once built.
"""
