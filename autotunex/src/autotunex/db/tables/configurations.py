# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""The ``configurations`` table."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from autotunex.db.base import Base
from autotunex.db.tables._helpers import utcnow
from autotunex.db.types import UtcDateTime, Uuid36

if TYPE_CHECKING:
    from autotunex.db.tables.users import UserTable


class ConfigurationTable(Base):
    """A reusable tuning configuration.

    ``user_id`` is ``VARCHAR(255)`` in the schema even though ``users.id`` is
    ``VARCHAR(36)`` — mirrored as-is; see ``docs/schema-review.md`` item C1.
    ``config_data`` is a schema-less ``JSON`` blob of tuning settings, in the
    shape the tuning pipeline writes (``tune_config`` / ``tuners_config`` /
    ``training_config`` / ``tuners_rl_config`` / ``training_rl_config``). The
    configuration endpoints require only that it be a non-empty object; the
    structure within it is the tuning core's contract, not the API's.
    """

    __tablename__ = "configurations"
    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_configurations_user_name"),)

    id: Mapped[UUID] = mapped_column(Uuid36, primary_key=True, default=uuid4)
    user_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    tuner_type: Mapped[str | None] = mapped_column(String(50))
    rl_tuner_type: Mapped[str | None] = mapped_column(String(50), default=None)
    config_data: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, onupdate=utcnow)

    user: Mapped[UserTable] = relationship(
        "UserTable", back_populates="configurations", viewonly=True
    )
