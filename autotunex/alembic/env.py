"""Alembic migration environment.

The database URL is taken from application settings (AUTOTUNEX_DATABASE_URL or
.env), not from alembic.ini, so migrations always target the same database the
app does.
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from autotunex.core.config import get_settings
from autotunex.db.base import Base
from autotunex.db.session import build_connect_args, build_ssl_context
from autotunex.db import tables  # noqa: F401  (registers tables on Base.metadata)

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", get_settings().database_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting to a database."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations against an open connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        # Required for SQLite, harmless on PostgreSQL: emits ALTER as
        # create-copy-drop so migrations stay portable across both.
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine and run migrations through it.

    The engine is built the same way the running app builds its own
    (``db.session``): with the MySQL session-timezone ``init_command`` and a TLS
    context resolved from ``database_ssl_mode``/``database_ssl_ca``. Managed MySQL
    (IBM Cloud) refuses non-TLS auth, so without this ``alembic upgrade`` fails to
    connect at all. ``NullPool`` keeps the one-shot migration process from leaving
    pooled connections open.
    """
    settings = get_settings()
    ssl_context = build_ssl_context(settings.database_ssl_ca, settings.database_ssl_mode)
    connectable = create_async_engine(
        settings.database_url,
        poolclass=pool.NullPool,
        connect_args=build_connect_args(settings.database_url, ssl_context),
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations with a live database connection."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
