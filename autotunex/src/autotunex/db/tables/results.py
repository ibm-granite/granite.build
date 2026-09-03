# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""The ``results`` table."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from autotunex.db.base import Base
from autotunex.db.tables._helpers import utcnow
from autotunex.db.types import UtcDateTime, Uuid36

if TYPE_CHECKING:
    from autotunex.db.tables.jobs import JobTable
    from autotunex.db.tables.trials import TrialTable


class ResultTable(Base):
    """The metrics one trial reported.

    ``trial_id`` is ``UNIQUE``, so this is one-to-one with ``trials`` despite
    looking like a one-to-many. ``job_id`` is therefore derivable from the trial
    and is a denormalization the schema keeps — see ``docs/schema-review.md``
    item D2.
    """

    __tablename__ = "results"

    id: Mapped[UUID] = mapped_column(Uuid36, primary_key=True, default=uuid4)
    job_id: Mapped[UUID] = mapped_column(
        Uuid36, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False
    )
    trial_id: Mapped[str] = mapped_column(
        String(16), ForeignKey("trials.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    metric: Mapped[str] = mapped_column(String(255), nullable=False)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, onupdate=utcnow)

    job: Mapped[JobTable] = relationship("JobTable", back_populates="results")
    trial: Mapped[TrialTable] = relationship("TrialTable", back_populates="result")
