# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""The ``log_entries`` table."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from autotunex.db.base import Base
from autotunex.db.types import MEDIUM_TEXT, Uuid36

if TYPE_CHECKING:
    from autotunex.db.tables.jobs import JobTable


class LogEntryTable(Base):
    """One log line emitted during a job or trial.

    Two mirrored defects, both in ``docs/schema-review.md``: ``trial_id`` is
    ``CHAR(36)`` while ``trials.id`` is ``VARCHAR(10)``, so it carries no foreign
    key and can never match (item A2); and ``timestamp`` is a bare ``DATETIME``
    with no timezone, unlike every other timestamp in the schema (item C4). The
    latter is why this column uses plain ``DateTime`` rather than
    :class:`autotunex.db.types.UtcDateTime` — mirroring means mirroring the flaw.

    ``trial_id`` is mapped as a plain string, not :class:`~autotunex.db.types.Uuid36`,
    even though the underlying column is wide enough to hold a dashed UUID: the
    values the pipeline actually writes there are ``trials.id``-shaped short codes
    (``VARCHAR(10)``, see :class:`~autotunex.db.tables.trials.TrialTable`), the
    same convention :class:`~autotunex.db.tables.results.ResultTable.trial_id`
    already follows. ``Uuid36`` would raise ``ValueError`` reading back anything
    that is not a well-formed UUID — i.e. every real row — so it is wrong here
    despite the column's width. The log endpoints match it by plain string
    equality accordingly (item A2: no FK, so nothing enforces the shape either
    way).
    """

    __tablename__ = "log_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[UUID] = mapped_column(
        Uuid36, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False
    )
    trial_id: Mapped[str | None] = mapped_column(String(36), default=None)
    level: Mapped[str | None] = mapped_column(String(50))
    filename: Mapped[str | None] = mapped_column(String(255))
    message: Mapped[str | None] = mapped_column(MEDIUM_TEXT)
    iteration: Mapped[int | None] = mapped_column(Integer, default=None)
    epoch: Mapped[float | None] = mapped_column(Float, default=None)
    timestamp: Mapped[datetime | None] = mapped_column(DateTime)

    job: Mapped[JobTable] = relationship("JobTable", back_populates="log_entries")
