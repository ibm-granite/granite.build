# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Log schemas returned by the DB-backed log endpoints.

``LogEntryRead`` is one ``log_entries`` row. ``LogPage`` is one keyset page of
lines, newest first — deliberately distinct from :class:`Page` because logs are
an append stream read backward by cursor, not a stable offset-addressable
collection.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class LogEntryRead(BaseModel):
    """One ``log_entries`` row as returned by the DB-backed log endpoints.

    ``job_id`` and ``trial_id`` are intentionally omitted — the endpoint path
    already fixes both. ``timestamp`` is the column's naive, unzoned ``DATETIME``
    (schema defect C4), surfaced as stored.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    level: str | None = None
    filename: str | None = None
    message: str | None = None
    iteration: int | None = None
    epoch: float | None = None
    timestamp: datetime | None = None


class LogPage(BaseModel):
    """One keyset page of log lines, newest first.

    ``next_before_id`` is the ``before_id`` to request for the next (older) page —
    the id of the last row returned — or ``None`` when ``has_more`` is false.
    """

    logs: list[LogEntryRead]
    has_more: bool
    next_before_id: int | None = None
