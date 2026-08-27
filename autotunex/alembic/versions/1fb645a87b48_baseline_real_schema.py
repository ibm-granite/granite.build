"""Baseline: the real AutoTuneX schema.

Creates every table as ``resources/autotunex_schema.sql`` declares it, including
``jobs.precision`` — the next revision is what drops that.

**Against the live MySQL database this revision must NOT be executed.** The
tables already exist and ``upgrade()`` would fail. Adopt the schema instead::

    alembic stamp 1fb645a87b48

Against SQLite and Postgres, ``upgrade()`` runs normally and builds the schema,
which is what tests and CI use.

The schema file's ``SET GLOBAL time_zone = '+00:00'`` is deliberately not
reproduced: it needs ``SYSTEM_VARIABLES_ADMIN`` and would affect every other
database on the server. Instead, MySQL driver support sets the session
``time_zone`` to ``+00:00`` via the engine's ``connect_args``, achieving the
same effect per-connection without needing elevated privileges or affecting other
databases on the server.

Revision ID: 1fb645a87b48
Revises:
Create Date: 2026-07-29 15:49:32.976091

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

import autotunex.db.types

# revision identifiers, used by Alembic.
revision: str = "1fb645a87b48"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _generated(suffix: str) -> sa.Computed:
    """Return the schema's ``CONCAT(name, suffix)`` generated-column expression.

    Autogenerate hard-coded SQLite's ``name || '_train'``, which is invalid DDL on
    MySQL. Building the expression with SQLAlchemy's concatenation operator
    instead lets each dialect render its own correct form — ``concat()`` on MySQL,
    ``||`` on SQLite and Postgres — from one definition, exactly as
    ``db/tables/datasets.py`` does.
    """
    return sa.Computed(
        sa.column("name", sa.String).concat(sa.literal_column(f"'{suffix}'")),
        persisted=True,
    )


def upgrade() -> None:
    """Create the eight tables, then the system user every job graph needs."""
    op.create_table(
        "users",
        sa.Column("id", autotunex.db.types.Uuid36(length=36), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=50), nullable=True),
        sa.Column("created_at", autotunex.db.types.UtcDateTime(timezone=True), nullable=False),
        sa.Column("updated_at", autotunex.db.types.UtcDateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
    )
    op.create_table(
        "configurations",
        sa.Column("id", autotunex.db.types.Uuid36(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("tuner_type", sa.String(length=50), nullable=True),
        sa.Column("rl_tuner_type", sa.String(length=50), nullable=True),
        sa.Column("config_data", sa.JSON(), nullable=True),
        sa.Column("created_at", autotunex.db.types.UtcDateTime(timezone=True), nullable=False),
        sa.Column("updated_at", autotunex.db.types.UtcDateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "name", name="uq_configurations_user_name"),
    )
    op.create_table(
        "datasets",
        sa.Column("id", autotunex.db.types.Uuid36(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("train_file", sa.String(length=255), _generated("_train"), nullable=False),
        sa.Column("train_records", sa.Integer(), nullable=True),
        sa.Column("train_file_size", sa.Integer(), nullable=True),
        sa.Column(
            "validation_file", sa.String(length=255), _generated("_validation"), nullable=False
        ),
        sa.Column("validation_records", sa.Integer(), nullable=True),
        sa.Column("validation_file_size", sa.Integer(), nullable=True),
        sa.Column("data_format", sa.String(length=10), server_default="jsonl", nullable=False),
        sa.Column("artifact_id", autotunex.db.types.Uuid36(length=36), nullable=True),
        sa.Column("artifact_url", sa.Text(), nullable=True),
        sa.Column("created_at", autotunex.db.types.UtcDateTime(timezone=True), nullable=False),
        sa.Column("updated_at", autotunex.db.types.UtcDateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "name", name="uq_datasets_user_name"),
    )
    op.create_table(
        "jobs",
        sa.Column("id", autotunex.db.types.Uuid36(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "PENDING",
                "RUNNING",
                "PAUSED",
                "TERMINATED",
                "ERROR",
                "COMPLETED",
                name="run_status",
            ),
            nullable=False,
        ),
        sa.Column("seed", sa.Integer(), nullable=True),
        sa.Column("config_id", autotunex.db.types.Uuid36(length=36), nullable=False),
        sa.Column("config_snapshot", sa.JSON(), nullable=True),
        sa.Column("dataset_id", autotunex.db.types.Uuid36(length=36), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column(
            "model_source", sa.String(length=50), server_default="huggingface", nullable=False
        ),
        sa.Column("experiment_name", sa.String(length=255), nullable=False),
        sa.Column("tuning_type", sa.String(length=100), nullable=True),
        # No server_default here on purpose: production's `precision` column has
        # none. The ORM carries one only so tests need not supply a value during
        # the single task before this column is dropped.
        sa.Column("precision", sa.String(length=50), nullable=False),
        sa.Column("ray_address", sa.String(length=50), nullable=True),
        sa.Column("cleanup", sa.Boolean(), nullable=True),
        sa.Column("autotune", sa.Boolean(), nullable=True),
        sa.Column("output_artifacts", sa.JSON(), nullable=True),
        sa.Column("created_at", autotunex.db.types.UtcDateTime(timezone=True), nullable=False),
        sa.Column("updated_at", autotunex.db.types.UtcDateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["config_id"], ["configurations.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "gb_tasks",
        sa.Column("id", autotunex.db.types.Uuid36(length=36), nullable=False),
        sa.Column("job_id", autotunex.db.types.Uuid36(length=36), nullable=False),
        sa.Column("build_id", autotunex.db.types.Uuid36(length=36), nullable=True),
        sa.Column(
            "status",
            sa.Enum(
                "PENDING",
                "RUNNING",
                "PAUSED",
                "TERMINATED",
                "ERROR",
                "COMPLETED",
                name="run_status",
            ),
            nullable=False,
        ),
        sa.Column(
            "type", sa.Enum("RITS", "TUNING", "DOWNLOAD", name="gb_task_type"), nullable=False
        ),
        sa.Column("pr_url", sa.Text(), nullable=True),
        sa.Column("artifact_id", autotunex.db.types.Uuid36(length=36), nullable=True),
        sa.Column("artifact_uri", sa.Text(), nullable=True),
        sa.Column("build_status", sa.JSON(), nullable=True),
        sa.Column("started_at", sa.String(length=255), nullable=True),
        sa.Column("updated_at", sa.String(length=255), nullable=True),
        sa.Column("rits_url", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "log_entries",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_id", autotunex.db.types.Uuid36(length=36), nullable=False),
        sa.Column("trial_id", autotunex.db.types.Uuid36(length=36), nullable=True),
        sa.Column("level", sa.String(length=50), nullable=True),
        sa.Column("filename", sa.String(length=255), nullable=True),
        sa.Column("message", sa.Text().with_variant(mysql.MEDIUMTEXT(), "mysql"), nullable=True),
        sa.Column("iteration", sa.Integer(), nullable=True),
        sa.Column("epoch", sa.Float(), nullable=True),
        sa.Column("timestamp", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "trials",
        sa.Column("id", sa.String(length=10), nullable=False),
        sa.Column("job_id", autotunex.db.types.Uuid36(length=36), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "PENDING",
                "RUNNING",
                "PAUSED",
                "TERMINATED",
                "ERROR",
                "COMPLETED",
                name="run_status",
            ),
            nullable=False,
        ),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.Column("created_at", autotunex.db.types.UtcDateTime(timezone=True), nullable=False),
        sa.Column("updated_at", autotunex.db.types.UtcDateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "results",
        sa.Column("id", autotunex.db.types.Uuid36(length=36), nullable=False),
        sa.Column("job_id", autotunex.db.types.Uuid36(length=36), nullable=False),
        sa.Column("trial_id", sa.String(length=10), nullable=False),
        sa.Column("metric", sa.String(length=255), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("created_at", autotunex.db.types.UtcDateTime(timezone=True), nullable=False),
        sa.Column("updated_at", autotunex.db.types.UtcDateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["trial_id"], ["trials.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("trial_id"),
    )

    # The system user. Every configuration, dataset and job is owned by a user,
    # so a fresh database needs at least one before anything can be inserted.
    # The all-zero id is a fixed, recognizable sentinel rather than a random one
    # so that downgrade() and any fixture can refer to it without a lookup.
    op.execute(
        sa.text(
            "INSERT INTO users (id, email, role, created_at, updated_at) "
            "VALUES (:id, :email, 'system', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        ).bindparams(id="00000000-0000-0000-0000-000000000000", email="system@ibm.com")
    )


def downgrade() -> None:
    """Drop the tables child-first, then the two enum types Postgres created.

    ``op.drop_table`` removes the seed user with ``users``, so it needs no
    separate ``DELETE``. The enum drops are Postgres-only: MySQL spells enums
    inline on the column and SQLite has no enum type at all, so on those two
    backends the types never existed independently of the tables.
    """
    op.drop_table("results")
    op.drop_table("trials")
    op.drop_table("log_entries")
    op.drop_table("gb_tasks")
    op.drop_table("jobs")
    op.drop_table("datasets")
    op.drop_table("configurations")
    op.drop_table("users")

    if op.get_bind().dialect.name == "postgresql":
        sa.Enum(name="run_status").drop(op.get_bind(), checkfirst=True)
        sa.Enum(name="gb_task_type").drop(op.get_bind(), checkfirst=True)
