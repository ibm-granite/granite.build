# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""The ``training_metrics`` table — per-step training metrics time series."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import JSON, Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from autotunex.db.base import Base
from autotunex.db.tables._helpers import utcnow
from autotunex.db.types import UtcDateTime, Uuid36

if TYPE_CHECKING:
    from autotunex.db.tables.jobs import JobTable


class TrainingMetricTable(Base):
    """One per-step metrics row emitted by a training run.

    ``trial_id`` is a soft reference (no FK): the final-training run may have no
    ``trials`` row, mirroring ``log_entries.trial_id``. It is covered by the
    composite index below rather than a single-column index of its own.

    Both indexes trail ``id``, not ``global_step``, because every read is the
    keyset page ``WHERE job_id = ? [AND trial_id = ?] AND id > ? ORDER BY id``
    (see ``SqlAlchemyTrainingMetricsRepository.metrics_page``). Trailing
    ``global_step`` instead would let the engine filter but still force a sort
    for the ordering — and on an append-only per-step table that grows without
    bound, the sort is the part that hurts.
    """

    __tablename__ = "training_metrics"
    __table_args__ = (
        Index("ix_training_metrics_job_keyset", "job_id", "id"),
        Index("ix_training_metrics_job_trial_keyset", "job_id", "trial_id", "id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[UUID] = mapped_column(
        Uuid36, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False
    )
    trial_id: Mapped[str | None] = mapped_column(String(16), default=None)
    global_step: Mapped[int] = mapped_column(Integer, nullable=False)
    epoch: Mapped[float | None] = mapped_column(Float, default=None)
    loss: Mapped[float | None] = mapped_column(Float, default=None)
    grad_norm: Mapped[float | None] = mapped_column(Float, default=None)
    learning_rate: Mapped[float | None] = mapped_column(Float, default=None)
    split: Mapped[str] = mapped_column(String(16), nullable=False, default="train")
    extra: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)

    job: Mapped[JobTable] = relationship("JobTable")
