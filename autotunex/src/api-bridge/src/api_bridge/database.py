# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""Synchronous SQLAlchemy database layer for the api-bridge.

This module is the ONLY place in the api-bridge that talks to the database. Every
service and route calls ``Database.<method>(...)`` and nothing else, so the
connection/query mechanism lives entirely here.

The api-bridge is a synchronous service, so this uses a synchronous SQLAlchemy
**Core** layer — ``create_engine`` plus the table metadata in ``tables.py``,
executed via Core constructs (no raw ``text()`` SQL) — not the async ORM engine
of the main ``src/autotunex`` service. It supports SQLite, MySQL, and PostgreSQL,
reading a single ``AUTOTUNEX_DATABASE_URL`` connection string (the same variable
the main service uses); an async driver such as ``asyncmy``, ``asyncpg``, or
``aiosqlite`` is coerced to its matching synchronous driver here — ``pymysql``,
``psycopg``, or stdlib ``sqlite3`` respectively — so one connection string
configures both services. In-memory SQLite (``sqlite://`` / ``sqlite:///:memory:``)
uses SQLAlchemy's ``StaticPool`` so the schema created at startup survives across
connections. Optional TLS is configured with ``AUTOTUNEX_DATABASE_SSL_CA`` /
``AUTOTUNEX_DATABASE_SSL_MODE``, mirroring the main service's semantics.

The two services stay independent: nothing is imported from ``src/autotunex`` —
the small amount of shared logic (URL driver coercion, the TLS-context builder) is
copied, not imported. See
``docs/superpowers/specs/2026-08-20-api-bridge-multi-dialect-design.md``.
"""

import logging
import os
import ssl
import uuid
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from dotenv import load_dotenv
from fastapi import HTTPException
from sqlalchemy import create_engine, event, func, insert, inspect, select, update
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.pool import StaticPool

from api_bridge import model as bridge_models
from api_bridge import tables
from api_bridge.utils import SYSTEM_USER, get_utc_timestamp, utc_now_string

logger = logging.getLogger(__name__)

load_dotenv()

_DISABLE = "disable"
_REQUIRE = "require"
_VERIFY = "verify"


_SYNC_DRIVERS = {
    "mysql": "mysql+pymysql",
    "postgresql": "postgresql+psycopg",
    "sqlite": "sqlite",
}


def _to_sync_url(raw: str) -> str:
    """Coerce a database URL to the matching **synchronous** driver.

    The bridge shares the main service's ``AUTOTUNEX_DATABASE_URL``, which names an
    async driver (``asyncmy``/``asyncpg``/``aiosqlite``). A synchronous engine
    cannot use those, so the driver is rewritten to the sync equivalent for the
    URL's backend (MySQL→pymysql, PostgreSQL→psycopg 3, SQLite→pysqlite). Parsing
    goes through ``make_url`` so passwords/query parameters survive. An unknown
    backend passes through unchanged and fails later with a clear dialect error.
    """
    url = make_url(raw)
    driver = _SYNC_DRIVERS.get(url.get_backend_name())
    if driver is not None:
        url = url.set(drivername=driver)
    return url.render_as_string(hide_password=False)


def _build_pool_kwargs(sync_url: str) -> dict[str, Any]:
    """Return engine pool kwargs, gating the QueuePool knobs off SQLite.

    ``pool_pre_ping`` is understood by every pool. ``pool_size``/``max_overflow``
    are passed only for server databases: SQLite's pool raises ``TypeError`` at
    engine construction when handed them. Mirrors
    ``src/autotunex/db/session.build_pool_kwargs``.
    """
    kwargs: dict[str, Any] = {"pool_pre_ping": True}
    if make_url(sync_url).get_backend_name() != "sqlite":
        kwargs["pool_size"] = 10
        kwargs["max_overflow"] = 5
    return kwargs


def _build_ssl_context(ca_path: str | None, mode: str | None) -> ssl.SSLContext | None:
    """Return a TLS context for the database according to ``mode``, or ``None``.

    Copied in behavior from ``src/autotunex/db/session.build_ssl_context`` (the
    two services are independent, so this is duplicated rather than imported):

    - ``"disable"`` — no TLS at all (local MySQL). Returns ``None``.
    - ``"require"`` — encrypt without authenticating the server: no CA needed.
      Stops a passive eavesdropper but not an active man-in-the-middle; dev only.
    - ``"verify"`` — encrypt and verify the server against ``ca_path`` with
      ``check_hostname`` on and ``CERT_REQUIRED``. ``VERIFY_X509_STRICT`` is
      cleared so IBM Cloud Databases for MySQL — whose CA marks ``basicConstraints``
      ``CA:TRUE`` but not critical, which OpenSSL 3.x's strict mode rejects —
      validates while chain and hostname verification stay intact.

    ``mode`` of ``None`` (or empty) derives from ``ca_path``: ``"verify"`` when a
    CA is set, ``"disable"`` otherwise.
    """
    effective = mode or (_VERIFY if ca_path else _DISABLE)

    if effective == _DISABLE:
        return None
    if effective == _REQUIRE:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context
    if effective == _VERIFY:
        if ca_path is None:
            raise ValueError(
                "AUTOTUNEX_DATABASE_SSL_MODE='verify' requires AUTOTUNEX_DATABASE_SSL_CA"
            )
        context = ssl.create_default_context(cafile=ca_path)
        context.verify_flags &= ~ssl.VERIFY_X509_STRICT
        return context
    raise ValueError(f"unknown AUTOTUNEX_DATABASE_SSL_MODE: {mode!r}")


def _build_connect_args(sync_url: str, ssl_context: ssl.SSLContext | None) -> dict[str, Any]:
    """Return driver connection arguments for ``sync_url``.

    For MySQL, pin the session time zone to UTC (``init_command``) so ``TIMESTAMP``
    columns are not shifted by the server's local zone on read — the per-connection
    equivalent of the schema's historical ``SET GLOBAL time_zone = '+00:00'``, and
    what the main service does. ``pymysql`` accepts both ``init_command`` and a
    prebuilt ``ssl.SSLContext`` for its ``ssl`` argument.
    """
    if make_url(sync_url).get_backend_name() == "mysql":
        args: dict[str, Any] = {"init_command": "SET time_zone = '+00:00'"}
        if ssl_context is not None:
            args["ssl"] = ssl_context
        return args
    return {}


REQUIRED_TABLES = {
    "configurations": ["id", "name", "config_data"],
    "log_entries": ["id", "job_id", "level", "filename", "message", "timestamp"],
    "jobs": [
        "id",
        "seed",
        "config_id",
        "dataset",
        "model",
        "task_type",
        "experiment_name",
        "tuning_type",
        "ray_address",
        "cleanup",
        "autotune",
    ],
}


class Database:
    def __init__(self, engine: Engine | None = None) -> None:
        if engine is not None:
            self._engine = engine
            return

        raw_url = os.getenv("AUTOTUNEX_DATABASE_URL", "").strip()
        if not raw_url:
            raise RuntimeError(
                "AUTOTUNEX_DATABASE_URL is not set. Export a SQLAlchemy connection string, "
                "e.g. AUTOTUNEX_DATABASE_URL=mysql+asyncmy://user:password@host:3306/schema "
                "(or a postgresql+asyncpg://… or sqlite:///./file.db URL). An async driver is "
                "coerced to its synchronous equivalent automatically."
            )

        sync_url = _to_sync_url(raw_url)
        ca_path = os.getenv("AUTOTUNEX_DATABASE_SSL_CA") or None
        ssl_mode = (os.getenv("AUTOTUNEX_DATABASE_SSL_MODE") or "").strip().lower() or None
        ssl_context = _build_ssl_context(ca_path, ssl_mode)

        url_obj = make_url(sync_url)
        connect_args = _build_connect_args(sync_url, ssl_context)
        engine_kwargs = _build_pool_kwargs(sync_url)
        if url_obj.get_backend_name() == "sqlite":
            # Share one in-memory DB across the pool, and allow cross-thread use
            # (the sync service is served by a threadpool for sync routes).
            connect_args = {**connect_args, "check_same_thread": False}
            if url_obj.database in (None, "", ":memory:"):
                engine_kwargs["poolclass"] = StaticPool

        # The engine is lazy: create_engine opens no connection here. A bad
        # credential/host/schema surfaces on the first real connection, which
        # test_db_connection_and_structure() (called from the server's lifespan)
        # forces at startup — so the server still refuses to serve a bad database.
        self._engine = create_engine(sync_url, connect_args=connect_args, **engine_kwargs)

        if url_obj.get_backend_name() == "sqlite":
            # Match MySQL/Postgres: enforce foreign keys so a bad FK raises
            # IntegrityError (relied on by insert_job -> 404). Off by default in SQLite.
            @event.listens_for(self._engine, "connect")
            def _enable_sqlite_fks(dbapi_conn, _record):
                dbapi_conn.execute("PRAGMA foreign_keys=ON")

    def test_db_connection_and_structure(self):
        """Verify database connection and required tables exist.

        Uses SQLAlchemy's inspect() to check for table existence in a dialect-independent
        way (works on SQLite, MySQL, PostgreSQL). Raises HTTPException(500) if any
        required table is missing or if a connection error occurs.
        """
        try:
            inspector = inspect(self._engine)
            for table in REQUIRED_TABLES:
                if not inspector.has_table(table):
                    raise HTTPException(status_code=500, detail=f"Table '{table}' is missing.")
            logger.info("Database connection and tables verified successfully.")
        except SQLAlchemyError as e:
            logger.error("Database connection or table check failed.")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    def insert_logs(self, buffer: list) -> bool:
        """
        Insert a batch of log entries into the database.

        Args:
            buffer (list): A list of log entry dictionaries.

        Returns:
            bool: True if the logs were inserted successfully, False otherwise.
        """
        if not buffer:
            return True
        try:
            rows = [
                {
                    "job_id": entry["job_id"],
                    "trial_id": entry["trial_id"],
                    "level": entry["level"],
                    "filename": entry["filename"],
                    "message": entry["message"],
                    "iteration": entry["iteration"],
                    "epoch": entry["epoch"],
                    "timestamp": entry["timestamp"],
                }
                for entry in buffer
            ]
            with self._engine.begin() as connection:
                connection.execute(insert(tables.log_entries), rows)
            return True
        except Exception as e:
            logger.error(f"Error inserting logs: {e!s}")
            return False

    def update_job_status(self, id: str, status: bridge_models.JobStatus) -> str:
        with self._engine.begin() as connection:
            connection.execute(
                update(tables.jobs).where(tables.jobs.c.id == id).values(status=status.value)
            )
        return True

    def insert_trial(self, data: bridge_models.Trial) -> str:
        config_copy = data["config"].copy() if isinstance(data["config"], dict) else {}
        if (
            isinstance(config_copy.get("tune_config"), dict)
            and "search_alg" in config_copy["tune_config"]
        ):
            config_copy["tune_config"]["search_alg"] = config_copy["tune_config"][
                "search_alg"
            ].__class__.__name__
        # trials.created_at/updated_at are DATETIME NOT NULL with no default in the
        # live schema, so an insert that omits them lands MySQL's zero date
        # ('0000-00-00 00:00:00') under a lax sql_mode — which the async api-server
        # then cannot read back (asyncmy returns the raw string for it). Set them
        # explicitly as a datetime object; pymysql renders it as a valid DATETIME
        # literal, whereas an ISO string would carry a "+00:00" offset MySQL rejects.
        now = datetime.now(UTC)
        with self._engine.begin() as connection:
            connection.execute(
                insert(tables.trials).values(
                    id=data["id"],
                    job_id=str(data["job_id"]),
                    status=data["status"].value,
                    config=config_copy,
                    created_at=now,
                    updated_at=now,
                )
            )
        return data["id"]

    def update_trial_status(self, trial_id: str, status: bridge_models.TrialStatus) -> bool:
        with self._engine.begin() as connection:
            connection.execute(
                update(tables.trials)
                .where(tables.trials.c.id == trial_id)
                .values(status=status.value)
            )
        return True

    def insert_result(self, metadata: bridge_models.Result) -> bridge_models.Result:
        try:
            result_id = uuid.uuid4()
            now = datetime.now(UTC)
            with self._engine.begin() as connection:
                connection.execute(
                    insert(tables.results).values(
                        id=result_id,
                        job_id=metadata["job_id"],
                        trial_id=metadata["trial_id"],
                        metric=metadata["metric"],
                        metrics=metadata["metrics"],
                        created_at=now,
                        updated_at=now,
                    )
                )
            return metadata
        except SQLAlchemyError as e:
            raise HTTPException(status_code=500, detail=f"Internal Server error: \n{e}")

    def update_all_trial_status(self, job_id: str, status: bridge_models.TrialStatus) -> bool:
        with self._engine.begin() as connection:
            connection.execute(
                update(tables.trials)
                .where(tables.trials.c.job_id == job_id)
                .values(status=status.value)
            )
        return True

    # ------------------------------------------------------------------
    # User CRUD (from api/services/db_service.py)
    # ------------------------------------------------------------------

    def insert_user(self, email: str) -> str:
        user_id = uuid.uuid4()
        # users.created_at/updated_at are DATETIME NOT NULL with no default (see
        # insert_trial), so set them explicitly to avoid MySQL's zero date.
        now = datetime.now(UTC)
        with self._engine.begin() as connection:
            connection.execute(
                insert(tables.users).values(id=user_id, email=email, created_at=now, updated_at=now)
            )
        return user_id

    def update_user(self, user) -> str:
        with self._engine.begin() as connection:
            connection.execute(
                update(tables.users)
                .where(tables.users.c.id == str(user.id))
                .values(email=user.email, role=user.role, updated_at=user.updated_at)
            )
        return user.id

    def get_user(self, email: str):
        with self._engine.connect() as connection:
            row = (
                connection.execute(
                    select(tables.users).where(func.lower(tables.users.c.email) == email.lower())
                )
                .mappings()
                .first()
            )
        result = dict(row) if row is not None else None
        if result is not None:
            result["created_at"] = get_utc_timestamp(result.get("created_at"))
            result["updated_at"] = get_utc_timestamp(result.get("updated_at"))
        return result

    # ------------------------------------------------------------------
    # Configuration CRUD (from api/services/db_service.py)
    # ------------------------------------------------------------------

    def insert_configuration(self, config) -> str:
        config_id = uuid.uuid4()
        # created_at/updated_at are DATETIME NOT NULL with no default (see
        # insert_trial), so set them explicitly to avoid MySQL's zero date.
        now = datetime.now(UTC)
        with self._engine.begin() as connection:
            connection.execute(
                insert(tables.configurations).values(
                    id=config_id,
                    user_id=config.user_id,
                    name=config.name,
                    tuner_type=config.tuner_type,
                    config_data=config.config_data,
                    created_at=now,
                    updated_at=now,
                )
            )
        return config_id

    def update_configuration(self, config) -> str:
        with self._engine.begin() as connection:
            connection.execute(
                update(tables.configurations)
                .where(tables.configurations.c.id == str(config.id))
                .values(
                    user_id=config.user_id,
                    name=config.name,
                    tuner_type=config.tuner_type,
                    config_data=config.config_data,
                )
            )
        return config.id

    def get_configs(self, user_id: str, ids: list = None) -> list:
        c = tables.configurations
        with self._engine.connect() as connection:
            if ids is None or len(ids) == 0:
                stmt = select(c).where((c.c.user_id == user_id) | (c.c.user_id == SYSTEM_USER))
            else:
                stmt = select(c).where(c.c.id.in_(list(ids)))
            rows = connection.execute(stmt).mappings().all()
            results = [dict(row) for row in rows]
            for result in results:
                jobs = (
                    connection.execute(
                        select(tables.jobs).where(
                            (tables.jobs.c.config_id == result["id"])
                            & (tables.jobs.c.user_id == user_id)
                        )
                    )
                    .mappings()
                    .all()
                )
                result["created_at"] = get_utc_timestamp(result.get("created_at"))
                result["updated_at"] = get_utc_timestamp(result.get("updated_at"))
                result["associated_jobs"] = [dict(job) for job in jobs]
        return results

    def get_config(self, config_id: str, user_id: str = None):
        c = tables.configurations
        with self._engine.connect() as connection:
            stmt = select(c).where(c.c.id == config_id)
            if user_id:
                stmt = stmt.where((c.c.user_id == user_id) | (c.c.user_id == SYSTEM_USER))
            row = connection.execute(stmt).mappings().first()
        if row is None:
            raise HTTPException(status_code=404, detail="Invalid config_id")
        result = dict(row)
        result["created_at"] = get_utc_timestamp(result.get("created_at"))
        result["updated_at"] = get_utc_timestamp(result.get("updated_at"))
        return result

    def get_config_by_name(self, config_name: str):
        c = tables.configurations
        with self._engine.connect() as connection:
            row = connection.execute(select(c).where(c.c.name == config_name)).mappings().first()
        result = dict(row) if row is not None else None
        if result is not None:
            result["created_at"] = get_utc_timestamp(result.get("created_at"))
            result["updated_at"] = get_utc_timestamp(result.get("updated_at"))
        return result

    def get_config_by_name_and_user(self, config_name: str, user_id: str):
        c = tables.configurations
        with self._engine.connect() as connection:
            row = (
                connection.execute(
                    select(c).where((c.c.name == config_name) & (c.c.user_id == user_id))
                )
                .mappings()
                .first()
            )
        result = dict(row) if row is not None else None
        if result is not None:
            result["created_at"] = get_utc_timestamp(result.get("created_at"))
            result["updated_at"] = get_utc_timestamp(result.get("updated_at"))
        return result

    # ------------------------------------------------------------------
    # Job CRUD (from api/services/db_service.py)
    # ------------------------------------------------------------------

    def insert_job(self, config) -> str:
        try:
            job_id = str(config.id) if config.id else str(uuid.uuid4())
            # created_at/updated_at are DATETIME NOT NULL with no default (see
            # insert_trial), so set them explicitly to avoid MySQL's zero date.
            # Jobs created via the main service's SQLAlchemy path get these for
            # free; a job created here would otherwise land the zero date.
            now = datetime.now(UTC)
            with self._engine.begin() as connection:
                connection.execute(
                    insert(tables.jobs).values(
                        id=job_id,
                        user_id=config.user_id,
                        status=config.status.value,
                        seed=config.seed,
                        config_id=config.config_id,
                        dataset_id=config.dataset_id,
                        model=config.model,
                        experiment_name=config.experiment_name,
                        tuning_type=config.tuning_type,
                        ray_address=config.ray_address,
                        cleanup=config.cleanup,
                        autotune=config.autotune,
                        created_at=now,
                        updated_at=now,
                    )
                )
            return job_id
        except IntegrityError:
            raise HTTPException(status_code=404, detail="Config_id doesn't exist")
        except SQLAlchemyError as e:
            logger.error(e)
            raise HTTPException(status_code=500, detail="Internal Server error")

    def get_job_by_id(self, job_id: str):
        """Return a single job row by id, or None if not found."""
        with self._engine.connect() as connection:
            row = (
                connection.execute(select(tables.jobs).where(tables.jobs.c.id == job_id))
                .mappings()
                .first()
            )
        return dict(row) if row is not None else None

    def insert_gb_task(self, job_id, build_id, task_type="TUNING") -> str:
        task_id = str(uuid.uuid4())
        now = utc_now_string()
        with self._engine.begin() as connection:
            connection.execute(
                insert(tables.gb_tasks).values(
                    id=task_id,
                    job_id=str(job_id),
                    build_id=build_id,
                    type=task_type,
                    started_at=now,
                    updated_at=now,
                )
            )
        return task_id

    # ------------------------------------------------------------------
    # Dataset CRUD (from api/services/db_service.py)
    # ------------------------------------------------------------------

    def get_dataset_by_name_and_user(self, dataset_name: str, user_id: str):
        d = tables.datasets
        with self._engine.connect() as connection:
            row = (
                connection.execute(
                    select(d).where((d.c.name == dataset_name) & (d.c.user_id == user_id))
                )
                .mappings()
                .first()
            )
        result = dict(row) if row is not None else None
        if result is not None:
            result["created_at"] = get_utc_timestamp(result.get("created_at"))
            result["updated_at"] = get_utc_timestamp(result.get("updated_at"))
        return result

    def insert_dataset(self, dataset):
        try:
            dataset_id = uuid.uuid4()
            # created_at/updated_at are DATETIME NOT NULL with no default (see
            # insert_trial), so set them explicitly to avoid MySQL's zero date.
            now = datetime.now(UTC)
            with self._engine.begin() as connection:
                connection.execute(
                    insert(tables.datasets).values(
                        id=dataset_id,
                        user_id=dataset.user_id,
                        name=dataset.name,
                        description=dataset.description,
                        created_at=now,
                        updated_at=now,
                    )
                )
            dataset.id = dataset_id
            return dataset
        except IntegrityError:
            raise HTTPException(status_code=400, detail="Dataset name must be unique")
        except SQLAlchemyError as e:
            raise HTTPException(status_code=500, detail=f"Internal Server error: \n{e}")

    def update_dataset(self, dataset):
        try:
            with self._engine.begin() as connection:
                connection.execute(
                    update(tables.datasets)
                    .where(tables.datasets.c.id == str(dataset.id))
                    .values(name=dataset.name, description=dataset.description)
                )
            return dataset
        except IntegrityError:
            raise HTTPException(status_code=400, detail="Dataset name must be unique")
        except SQLAlchemyError as e:
            raise HTTPException(status_code=500, detail=f"Internal Server error: \n{e}")

    def update_dataset_metadata(self, id: str, user_id: str, metadata: dict[str, Any]):
        try:
            with self._engine.begin() as connection:
                connection.execute(
                    update(tables.datasets)
                    .where(tables.datasets.c.id == id)
                    .values(
                        train_records=metadata["train_records"],
                        train_file_size=metadata["train_file_size"],
                        validation_records=metadata["validation_records"],
                        validation_file_size=metadata["validation_file_size"],
                        artifact_id=metadata["artifact_id"],
                        artifact_url=metadata["artifact_url"],
                        # Attaching an artifact means the data now exists, so mark
                        # the dataset 'ready' — the main service only previews
                        # ready-or-artifact-backed datasets, and this write path
                        # (the only one that attaches artifacts) never set status.
                        status="ready",
                    )
                )
            result = self.get_dataset(dataset_id=id, user_id=user_id)
            return result
        except SQLAlchemyError as e:
            raise HTTPException(status_code=500, detail=f"Internal Server error: \n{e}")

    def check_dataset_exists(self, id: str) -> bool:
        with self._engine.connect() as connection:
            row = connection.execute(
                select(tables.datasets.c.id).where(tables.datasets.c.id == id)
            ).first()
        return row is not None

    def get_dataset(self, dataset_id: str, user_id: str | None = None):
        d = tables.datasets
        is_id = True
        try:
            UUID(dataset_id)
        except ValueError:
            is_id = False
        key_col = d.c.id if is_id else d.c.name
        stmt = select(d).where(key_col == dataset_id)
        if user_id is not None:
            stmt = stmt.where((d.c.user_id == user_id) | (d.c.user_id == SYSTEM_USER))
        with self._engine.connect() as connection:
            row = connection.execute(stmt).mappings().first()
        result = dict(row) if row is not None else None
        if result is not None:
            result["created_at"] = get_utc_timestamp(result.get("created_at"))
            result["updated_at"] = get_utc_timestamp(result.get("updated_at"))
        return result

    def get_datasets(self, user_id: str, ids: list = None) -> list:
        d = tables.datasets
        with self._engine.connect() as connection:
            if ids is None or len(ids) == 0:
                stmt = select(d).where((d.c.user_id == user_id) | (d.c.user_id == SYSTEM_USER))
            else:
                stmt = select(d).where(d.c.id.in_(list(ids)))
            rows = connection.execute(stmt).mappings().all()
            results = [dict(row) for row in rows]
            for result in results:
                jobs = (
                    connection.execute(
                        select(tables.jobs).where(
                            (tables.jobs.c.dataset_id == result["id"])
                            & (tables.jobs.c.user_id == user_id)
                        )
                    )
                    .mappings()
                    .all()
                )
                result["associated_jobs"] = [dict(job) for job in jobs]
                result["created_at"] = get_utc_timestamp(result.get("created_at"))
                result["updated_at"] = get_utc_timestamp(result.get("updated_at"))
            return results
