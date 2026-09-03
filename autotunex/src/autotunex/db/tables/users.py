# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""The ``users`` table.

Mirrors ``resources/autotunex_schema.sql`` apart from ``last_login_at``, which
the schema file does not have — one of the tracked ORM-vs-schema divergences
listed under CLAUDE.md open decision 6, added by Alembic revision
``a3c71d94e5b2``. The schema's ``DEFAULT (uuid())`` is a MySQL server-side
default with no SQLite equivalent, so ids are generated in Python — which also
means a new row's id is known before the flush.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from autotunex.db.base import Base
from autotunex.db.tables._helpers import utcnow
from autotunex.db.types import UtcDateTime, Uuid36

if TYPE_CHECKING:
    from autotunex.db.tables.configurations import ConfigurationTable
    from autotunex.db.tables.datasets import DatasetTable
    from autotunex.db.tables.jobs import JobTable


class UserTable(Base):
    """A person who owns configurations, datasets and jobs.

    ``email`` is ``UNIQUE``. Note that MySQL's default collation is
    case-insensitive, so ``A@x.com`` collides with ``a@x.com`` there but not on
    SQLite or Postgres — see ``docs/schema-review.md`` §E "Cross-dialect
    portability", item 1. That divergence is why
    :meth:`~autotunex.db.repositories.sqlalchemy.SqlAlchemyUserRepository.get_by_email`
    folds case on both sides, and why it has to treat several matching rows as an
    error rather than a choice.

    ``last_login_at`` is deliberately its own column rather than a reading of
    ``updated_at``. The two answer different questions: ``updated_at`` is when
    the *row* last changed, and it changes for reasons that are not logins —
    an admin editing a role through ``PATCH /users/{id}`` being the one that
    actually happens. Reusing it made every such edit render as that user's
    last login in the Users table, which is the bug this column exists to fix.
    Nullable because some rows genuinely have no login behind them — one the
    tuning pipeline created and never authenticated as, for instance; the API
    surfaces that honestly as ``null`` rather than inventing a timestamp. Rows
    that *did* have one kept it: the revision backfills from ``updated_at``, where
    the pre-refactor service used to write logins, for every row whose
    ``updated_at`` had moved past its ``created_at``. Only
    :meth:`~autotunex.db.repositories.sqlalchemy.SqlAlchemyUserRepository.touch_login`
    and ``provision`` ever write it afterwards.

    The three child relationships need an explicit ``primaryjoin`` because the
    child tables declare ``user_id`` as ``VARCHAR(255)`` while ``users.id`` is
    ``VARCHAR(36)`` (item C1), so SQLAlchemy cannot infer the join from the
    foreign key's types. They are ``viewonly=True``: the read path never writes
    through them, and the database's own ``ON DELETE CASCADE`` remains the thing
    that deletes children.
    """

    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(Uuid36, primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    role: Mapped[str | None] = mapped_column(String(50), default="user")
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, onupdate=utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(UtcDateTime, nullable=True, default=None)

    configurations: Mapped[list[ConfigurationTable]] = relationship(
        "ConfigurationTable",
        back_populates="user",
        primaryjoin="foreign(ConfigurationTable.user_id) == UserTable.id",
        viewonly=True,
    )
    datasets: Mapped[list[DatasetTable]] = relationship(
        "DatasetTable",
        back_populates="user",
        primaryjoin="foreign(DatasetTable.user_id) == UserTable.id",
        viewonly=True,
    )
    jobs: Mapped[list[JobTable]] = relationship(
        "JobTable",
        primaryjoin="foreign(JobTable.user_id) == UserTable.id",
        viewonly=True,
    )
