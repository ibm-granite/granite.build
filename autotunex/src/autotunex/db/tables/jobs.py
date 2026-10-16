# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""The ``jobs`` table — the unit of API interaction.

Mirrors ``resources/autotunex_schema.sql``, with one deliberate deviation:
``precision`` is absent. That column was ``VARCHAR(50) NOT NULL`` with no
default, so requiring it forced every writer to supply a value. Its values
live in ``config_snapshot['precision']`` instead, backfilled by the migration
that dropped the column.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, Boolean, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from autotunex.db.base import Base
from autotunex.db.tables._helpers import utcnow
from autotunex.db.types import UtcDateTime, Uuid36
from autotunex.models.status import RunStatus

if TYPE_CHECKING:
    from autotunex.db.tables.configurations import ConfigurationTable
    from autotunex.db.tables.datasets import DatasetTable
    from autotunex.db.tables.gb_tasks import GbTaskTable
    from autotunex.db.tables.log_entries import LogEntryTable
    from autotunex.db.tables.results import ResultTable
    from autotunex.db.tables.trials import TrialTable
    from autotunex.db.tables.users import UserTable


class JobTable(Base):
    """One optimization run.

    ``status`` is declared with SQLAlchemy's ``Enum`` over :class:`RunStatus`,
    which persists the member *name* — so the column holds ``PENDING``, matching
    the schema's ``ENUM``, while the API emits ``pending``. The ``name=`` is
    required for Postgres, where Alembic needs a named type to create.

    ``user``, ``configuration`` and ``dataset`` are ``viewonly=True``: the read
    path never writes them, and the ``user_id`` type mismatch (item C1) makes
    write-side management ambiguous. ``joinedload`` still works on them.
    """

    __tablename__ = "jobs"

    id: Mapped[UUID] = mapped_column(Uuid36, primary_key=True, default=uuid4)
    user_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[RunStatus] = mapped_column(
        Enum(RunStatus, name="run_status"), nullable=False, default=RunStatus.PENDING
    )
    seed: Mapped[int | None] = mapped_column(Integer, default=42)
    config_id: Mapped[UUID] = mapped_column(
        Uuid36, ForeignKey("configurations.id", ondelete="RESTRICT"), nullable=False
    )
    config_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    dataset_id: Mapped[UUID] = mapped_column(
        Uuid36, ForeignKey("datasets.id", ondelete="RESTRICT"), nullable=False
    )
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    model_source: Mapped[str] = mapped_column(
        String(50), nullable=False, server_default="huggingface"
    )
    experiment_name: Mapped[str] = mapped_column(String(255), nullable=False)
    tuning_type: Mapped[str | None] = mapped_column(String(100), default=None)
    reward_function_code: Mapped[str | None] = mapped_column(Text, default=None)
    reward_function_name: Mapped[str | None] = mapped_column(String(255), default=None)
    ray_address: Mapped[str | None] = mapped_column(String(50), default=None)
    cleanup: Mapped[bool | None] = mapped_column(Boolean, default=True)
    autotune: Mapped[bool | None] = mapped_column(Boolean, default=True)
    # Written by the tuning pipeline, not by this service: an object for some
    # writers, a bare list of file descriptors for the publish step. The union
    # is the column's real shape — see JobRead.output_artifacts.
    output_artifacts: Mapped[dict[str, Any] | list[Any] | None] = mapped_column(JSON, default=None)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, onupdate=utcnow)

    user: Mapped[UserTable] = relationship("UserTable", viewonly=True)
    configuration: Mapped[ConfigurationTable] = relationship("ConfigurationTable", viewonly=True)
    dataset: Mapped[DatasetTable] = relationship("DatasetTable", viewonly=True)
    trials: Mapped[list[TrialTable]] = relationship(
        "TrialTable", back_populates="job", cascade="all, delete-orphan"
    )
    tasks: Mapped[list[GbTaskTable]] = relationship(
        "GbTaskTable", back_populates="job", cascade="all, delete-orphan"
    )
    results: Mapped[list[ResultTable]] = relationship(
        "ResultTable", back_populates="job", cascade="all, delete-orphan"
    )
    log_entries: Mapped[list[LogEntryTable]] = relationship(
        "LogEntryTable", back_populates="job", cascade="all, delete-orphan"
    )
