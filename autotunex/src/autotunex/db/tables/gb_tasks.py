# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""The ``gb_tasks`` table."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, Enum, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from autotunex.db.base import Base
from autotunex.db.types import Uuid36
from autotunex.models.status import GbTaskType, RunStatus

if TYPE_CHECKING:
    from autotunex.db.tables.jobs import JobTable


class GbTaskTable(Base):
    """A build or deployment task attached to a job.

    ``started_at`` and ``updated_at`` are ``VARCHAR(255)`` in the schema, not
    timestamps, and are mirrored as strings — the API does not pretend otherwise.
    ``docs/schema-review.md`` item B5 recommends converting them. There are also
    no ``created_at``/``updated_at`` timestamp columns of the usual shape here.
    """

    __tablename__ = "gb_tasks"

    id: Mapped[UUID] = mapped_column(Uuid36, primary_key=True, default=uuid4)
    job_id: Mapped[UUID] = mapped_column(
        Uuid36, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False
    )
    build_id: Mapped[UUID | None] = mapped_column(Uuid36, default=None)
    status: Mapped[RunStatus] = mapped_column(
        Enum(RunStatus, name="run_status"), nullable=False, default=RunStatus.PENDING
    )
    type: Mapped[GbTaskType] = mapped_column(Enum(GbTaskType, name="gb_task_type"), nullable=False)
    pr_url: Mapped[str | None] = mapped_column(Text, default=None)
    artifact_id: Mapped[UUID | None] = mapped_column(Uuid36, default=None)
    artifact_uri: Mapped[str | None] = mapped_column(Text, default=None)
    build_status: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    started_at: Mapped[str | None] = mapped_column(String(255), default=None)
    updated_at: Mapped[str | None] = mapped_column(String(255), default=None)
    rits_url: Mapped[str | None] = mapped_column(Text, default=None)

    job: Mapped[JobTable] = relationship("JobTable", back_populates="tasks")
