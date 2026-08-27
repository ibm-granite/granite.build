# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""The ``datasets`` table."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import (
    Computed,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    column,
    literal_column,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql.elements import ColumnElement

from autotunex.db.base import Base
from autotunex.db.tables._helpers import utcnow
from autotunex.db.types import UtcDateTime, Uuid36
from autotunex.models.status import DatasetStatus

if TYPE_CHECKING:
    from autotunex.db.tables.users import UserTable


def _suffixed(name: str, suffix: str) -> ColumnElement[str]:
    """Return a ``CONCAT(name, suffix)`` expression that compiles on all backends.

    SQLAlchemy's concatenation operator renders ``concat()`` on MySQL and ``||``
    on SQLite and Postgres, so one expression mirrors the schema's
    ``GENERATED ALWAYS AS (CONCAT(name, '_train')) STORED`` everywhere — no
    dialect branch needed.
    """
    return column(name, String).concat(literal_column(f"'{suffix}'"))


class DatasetTable(Base):
    """A training and validation dataset pair.

    ``train_file`` and ``validation_file`` are stored generated columns derived
    from ``name``, which makes them non-writable — see ``docs/schema-review.md``
    item D5 on why deriving a filename from a mutable display name is fragile.
    """

    __tablename__ = "datasets"
    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_datasets_user_name"),)

    id: Mapped[UUID] = mapped_column(Uuid36, primary_key=True, default=uuid4)
    user_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    train_file: Mapped[str] = mapped_column(
        String(255), Computed(_suffixed("name", "_train"), persisted=True)
    )
    train_records: Mapped[int | None] = mapped_column(Integer, default=None)
    train_file_size: Mapped[int | None] = mapped_column(Integer, default=None)
    validation_file: Mapped[str] = mapped_column(
        String(255), Computed(_suffixed("name", "_validation"), persisted=True)
    )
    validation_records: Mapped[int | None] = mapped_column(Integer, default=None)
    validation_file_size: Mapped[int | None] = mapped_column(Integer, default=None)
    data_format: Mapped[str] = mapped_column(String(10), nullable=False, server_default="jsonl")
    artifact_id: Mapped[UUID | None] = mapped_column(Uuid36, default=None)
    artifact_url: Mapped[str | None] = mapped_column(Text, default=None)
    status: Mapped[DatasetStatus] = mapped_column(
        String(20), nullable=False, server_default=DatasetStatus.EMPTY.value
    )
    status_detail: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, onupdate=utcnow)

    user: Mapped[UserTable] = relationship("UserTable", back_populates="datasets", viewonly=True)
