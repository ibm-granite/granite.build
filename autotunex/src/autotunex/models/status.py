# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Status vocabularies shared by jobs, trials and build tasks.

``jobs``, ``trials`` and ``gb_tasks`` each declare the identical six-value
``ENUM`` in ``resources/autotunex_schema.sql``, so one Python enum serves all
three.

The casing is deliberate and load-bearing. SQLAlchemy's ``Enum`` persists a
member's *name*, while Pydantic serializes its *value*. Declaring
``PENDING = "pending"`` therefore satisfies both the database's ``ENUM('PENDING',
...)`` and the API's lowercase contract with no mapping layer in between. Do not
"tidy" the casing.
"""

from __future__ import annotations

from enum import StrEnum


class RunStatus(StrEnum):
    """Lifecycle states of a job, a trial, or a build task."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    TERMINATED = "terminated"
    ERROR = "error"
    COMPLETED = "completed"


TERMINAL_RUN_STATUSES: frozenset[RunStatus] = frozenset(
    {RunStatus.COMPLETED, RunStatus.ERROR, RunStatus.TERMINATED}
)
"""States with no outgoing transitions."""


class GbTaskType(StrEnum):
    """Kind of build task attached to a job.

    Values are uppercase because the schema's ``ENUM`` is uppercase; here name
    and value coincide, unlike :class:`RunStatus`.
    """

    RITS = "RITS"
    TUNING = "TUNING"
    DOWNLOAD = "DOWNLOAD"


class DatasetStatus(StrEnum):
    """Lifecycle of a dataset's uploaded file.

    Stored in the ``datasets.status`` ``VARCHAR(20)`` column (not a DB ``ENUM``),
    so — unlike :class:`RunStatus`, which SQLAlchemy persists by *name* — name and
    value need not diverge here; the lowercase value is what lands in the column.
    The row's ``status`` is also the cross-replica coordination point for the
    "already uploading" guard, so it is a durable DB value, not in-process state.
    """

    EMPTY = "empty"
    UPLOADING = "uploading"
    READY = "ready"
    ERROR = "error"
