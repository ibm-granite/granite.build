# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""Synchronous SQLAlchemy Core metadata for the api-bridge.

Defines the subset of the AutoTuneX schema the bridge reads and writes, as Core
``Table`` objects plus two portable ``TypeDecorator``s. Letting SQLAlchemy own
the SQL is what makes the same ``Database`` methods run on SQLite, MySQL, and
PostgreSQL: it quotes identifiers per dialect, (de)serializes JSON, and wraps
driver errors uniformly.

Independent of ``src/autotunex`` by design (nothing is imported from it); the
``UtcDateTime`` decorator is copied from ``src/autotunex/db/types.py``.

Deliberate schema choices, matching what the bridge actually reads/writes rather
than the main ORM:
- ``configurations`` has NO ``artifact_id``/``artifact_url`` (unused; removed).
- ``jobs`` has NO ``precision`` (dropped from the live schema).
- ``datasets.train_file``/``validation_file`` are MySQL ``GENERATED`` columns:
  declared read-only here (never written); ``NULL`` on a SQLite table.
- Enum columns (status/type) use SQLAlchemy ``Enum`` with the live Postgres type
  names (``run_status``/``gb_task_type``). Plain ``String`` here would make psycopg
  bind writes as ``$1::VARCHAR``, which Postgres refuses to assign to an enum
  column; the native ``Enum`` binds as the enum instead. Degrades to ``VARCHAR`` on
  SQLite and MySQL ``ENUM`` on MySQL, so the string round-trip is unchanged there.
- ``NOT NULL`` columns the bridge omits on insert carry a ``server_default`` so a
  table created from this metadata (SQLite tests) accepts those inserts.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.engine import Dialect
from sqlalchemy.types import TypeDecorator


class Uuid36Str(TypeDecorator[str]):
    """A UUID stored as a 36-character string, bound as ``str`` and returned as ``str``.

    Binding ``str(value)`` keeps a ``uuid.UUID`` from reaching psycopg/pysqlite as
    a native UUID a ``VARCHAR``/``CHAR(36)`` column might reject, and returning the
    stored string unchanged preserves the bridge's historical string-id shape
    (``pymysql`` returns ``str`` for ``VARCHAR``).
    """

    impl = String(36)
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Dialect) -> str | None:
        if value is None:
            return None
        return str(value)

    def process_result_value(self, value: str | None, dialect: Dialect) -> str | None:
        return value


class UtcDateTime(TypeDecorator[datetime]):
    """Timezone-aware UTC datetime. Copied from ``src/autotunex/db/types.py``.

    On write, values must be aware and are converted to UTC. On read, naive
    values are labelled UTC and an unparseable string (MySQL's zero date) becomes
    ``None`` rather than crashing the request.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise TypeError("naive datetime rejected; use datetime.now(UTC)")
        return value.astimezone(UTC)

    def process_result_value(self, value: Any, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, str):
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


def _run_status() -> Enum:
    """The six-value run-status enum, matching the live Postgres ``run_status`` type.

    The labels are the UPPERCASE member *names* SQLAlchemy's ``Enum`` persists for
    the main service's ``RunStatus`` (a ``StrEnum`` whose values are lowercase),
    which is what the live ``run_status`` type actually carries and what the bridge
    writes (``JobStatus``/``TrialStatus`` ``.value`` are uppercase). Using ``Enum``
    rather than ``String`` makes a write bind as ``$1::run_status`` instead of
    ``$1::VARCHAR`` — Postgres refuses the latter against an enum column. A fresh
    instance per column mirrors the main ORM. Degrades to ``VARCHAR`` on SQLite.
    """
    return Enum(
        "PENDING",
        "RUNNING",
        "PAUSED",
        "TERMINATED",
        "ERROR",
        "COMPLETED",
        name="run_status",
        create_constraint=False,
    )


def _gb_task_type() -> Enum:
    """The build-task-type enum, matching the live Postgres ``gb_task_type`` type."""
    return Enum("RITS", "TUNING", "DOWNLOAD", name="gb_task_type", create_constraint=False)


metadata = MetaData()

users = Table(
    "users",
    metadata,
    Column("id", Uuid36Str, primary_key=True),
    Column("email", String(255), nullable=False, unique=True),
    Column("role", String(50), server_default="user"),
    Column("created_at", UtcDateTime),
    Column("updated_at", UtcDateTime),
    # Declared, never written here. The main service stamps it on authentication
    # (Alembic revision ``a3c71d94e5b2``); a pipeline-created user has not logged
    # in, so leaving it NULL is correct. It is declared anyway because
    # ``metadata.create_all()`` is how this subproject's tests build their schema,
    # and a table created without the column would diverge from the one the main
    # service migrates — the same two-declarations-in-step rule
    # ``training_metrics`` carries.
    Column("last_login_at", UtcDateTime),
)

configurations = Table(
    "configurations",
    metadata,
    Column("id", Uuid36Str, primary_key=True),
    Column("user_id", Uuid36Str, ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("name", String(255), nullable=False),
    Column("tuner_type", String(50)),
    Column("rl_tuner_type", String(50)),
    Column("config_data", JSON),
    Column("created_at", UtcDateTime),
    Column("updated_at", UtcDateTime),
    UniqueConstraint("user_id", "name", name="uq_configurations_user_name"),
)

datasets = Table(
    "datasets",
    metadata,
    Column("id", Uuid36Str, primary_key=True),
    Column("user_id", Uuid36Str, ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("name", String(255), nullable=False),
    Column("description", Text),
    # Read-only in this layer: MySQL GENERATED columns; never written by the bridge.
    Column("train_file", String(255)),
    Column("train_records", Integer),
    Column("train_file_size", Integer),
    Column("validation_file", String(255)),
    Column("validation_records", Integer),
    Column("validation_file_size", Integer),
    Column("data_format", String(10), server_default="jsonl"),
    Column("artifact_id", Uuid36Str),
    Column("artifact_url", Text),
    # Mirrors datasets.status VARCHAR(20) in the live schema. The bridge writes
    # 'ready' when it attaches an artifact (update_dataset_metadata); a plain
    # insert takes the 'empty' server-default, matching the main service's ORM.
    Column("status", String(20), server_default="empty", nullable=False),
    Column("created_at", UtcDateTime),
    Column("updated_at", UtcDateTime),
    UniqueConstraint("user_id", "name", name="uq_datasets_user_name"),
)

jobs = Table(
    "jobs",
    metadata,
    Column("id", Uuid36Str, primary_key=True),
    Column("user_id", Uuid36Str, ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("status", _run_status(), server_default="PENDING", nullable=False),
    Column("seed", Integer, server_default="42"),
    Column("config_id", Uuid36Str, ForeignKey("configurations.id"), nullable=False),
    Column("config_snapshot", JSON),
    Column("dataset_id", Uuid36Str, ForeignKey("datasets.id"), nullable=False),
    Column("model", String(255), nullable=False),
    Column("model_source", String(50), server_default="huggingface", nullable=False),
    Column("experiment_name", String(255), nullable=False),
    Column("tuning_type", String(100)),
    Column("ray_address", String(50)),
    Column("cleanup", Boolean),
    Column("autotune", Boolean),
    Column("output_artifacts", JSON),
    Column("created_at", UtcDateTime),
    Column("updated_at", UtcDateTime),
)

trials = Table(
    "trials",
    metadata,
    Column("id", String(16), primary_key=True),
    Column("job_id", Uuid36Str, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False),
    Column("status", _run_status(), server_default="PENDING", nullable=False),
    Column("config", JSON),
    Column("created_at", UtcDateTime),
    Column("updated_at", UtcDateTime),
)

results = Table(
    "results",
    metadata,
    Column("id", Uuid36Str, primary_key=True),
    Column("job_id", Uuid36Str, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False),
    Column("trial_id", String(16), nullable=False, unique=True),
    Column("metric", String(255), nullable=False),
    Column("metrics", JSON),
    Column("created_at", UtcDateTime),
    Column("updated_at", UtcDateTime),
)

log_entries = Table(
    "log_entries",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("job_id", Uuid36Str, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False),
    Column("trial_id", String(36)),
    Column("level", String(50)),
    Column("filename", String(255)),
    Column("message", Text),
    Column("iteration", Integer),
    Column("epoch", Float),
    Column("timestamp", DateTime),  # bare DATETIME: client-supplied, mirrors the schema flaw
)

training_metrics = Table(
    "training_metrics",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("job_id", Uuid36Str, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False),
    # No index=True: trial_id is covered by the composite (job_id, trial_id, id)
    # keyset index, and declaring one here would make create_all() build an index
    # that Alembic never creates — metadata drift against the main service's ORM.
    Column("trial_id", String(16)),
    Column("global_step", Integer, nullable=False),
    Column("epoch", Float),
    Column("loss", Float),
    Column("grad_norm", Float),
    Column("learning_rate", Float),
    Column("split", String(16), nullable=False),
    Column("extra", JSON),
    Column("created_at", UtcDateTime),
)

gb_tasks = Table(
    "gb_tasks",
    metadata,
    Column("id", Uuid36Str, primary_key=True),
    Column("job_id", Uuid36Str, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False),
    Column("build_id", Uuid36Str),
    Column("status", _run_status(), server_default="PENDING", nullable=False),
    Column("type", _gb_task_type(), nullable=False),
    Column("pr_url", Text),
    Column("artifact_id", Uuid36Str),
    Column("artifact_uri", Text),
    Column("build_status", JSON),
    Column("started_at", String(255)),  # bridge writes ISO strings
    Column("updated_at", String(255)),
    Column("rits_url", Text),
)
