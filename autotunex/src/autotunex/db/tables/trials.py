# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""The ``trials`` table."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import JSON, Enum, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from autotunex.db.base import Base
from autotunex.db.tables._helpers import utcnow
from autotunex.db.types import UtcDateTime, Uuid36
from autotunex.models.status import RunStatus

if TYPE_CHECKING:
    from autotunex.db.tables.jobs import JobTable
    from autotunex.db.tables.results import ResultTable


class TrialTable(Base):
    """One training run inside a job, testing one parameter assignment.

    ``id`` is ``VARCHAR(16)`` in the schema — a short opaque code assigned by the
    tuning pipeline, not a UUID, and not generated here. ``docs/schema-review.md``
    item A2 notes that ``log_entries.trial_id`` is ``CHAR(36)`` and therefore
    cannot reference it; that mismatch is mirrored rather than fixed.

    ``Enum(RunStatus, name="run_status")`` appears in three tables with the same
    ``name=`` on purpose: on Postgres SQLAlchemy then creates one shared named
    type and Alembic emits a single ``CREATE TYPE``.
    """

    __tablename__ = "trials"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    job_id: Mapped[UUID] = mapped_column(
        Uuid36, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[RunStatus] = mapped_column(
        Enum(RunStatus, name="run_status"), nullable=False, default=RunStatus.PENDING
    )
    config: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, onupdate=utcnow)

    job: Mapped[JobTable] = relationship("JobTable", back_populates="trials")
    result: Mapped[ResultTable | None] = relationship(
        "ResultTable", back_populates="trial", uselist=False
    )
