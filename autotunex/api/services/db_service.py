# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

import os
import models as api
import pymysql
import aiomysql
import uuid
from uuid import UUID
import json
import threading
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import HTTPException
import constants
import logging
from typing import Dict, Any, Optional
from utils import (
    is_gb_enabled,
    get_utc_timestamp,
    extract_artifact_identifier,
    build_dmf_url,
)
from dbutils.pooled_db import PooledDB

logger = logging.getLogger(__name__)

load_dotenv()

DB_HOST = os.environ.get("DB_HOST", "")
DB_PORT = os.environ.get("DB_PORT", "")
DB_USER = os.environ.get("DB_USER", "")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
# Accept either DB_SCHEMA or DB_NAME so the api server and api-bridge can share a
# single deployment secret for the database name (the bridge uses the same
# fallback in api-bridge/database.py). DB_SCHEMA wins when both are set, keeping
# existing DB_SCHEMA-based deployments unchanged.
DB_SCHEMA = os.getenv("DB_SCHEMA") or os.getenv("DB_NAME") or ""


def _ssl_verify_identity() -> bool:
    """
    Whether to verify the MySQL server's TLS identity.

    Defaults to True, preserving behavior for existing (e.g. IBM Cloud) MySQL
    deployments that present a verifiable certificate. Set
    DB_SSL_VERIFY_IDENTITY=false when connecting to a MySQL that only offers a
    self-signed certificate (such as the stock `mysql` container in the local
    docker-compose stack): TLS is still negotiated — required by
    caching_sha2_password — but the server identity is not verified.
    """
    return os.getenv("DB_SSL_VERIFY_IDENTITY", "true").strip().lower() not in (
        "false",
        "0",
        "no",
    )


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

# ---------------------------------------------------------------------------
# Module-level pool singletons
# ---------------------------------------------------------------------------
_async_pool: Optional[aiomysql.Pool] = None
_sync_pool: Optional[PooledDB] = None
_pool_lock = threading.Lock()


def _build_ssl_context():
    """
    Build an ssl.SSLContext for aiomysql.
    Always returns a context because IBM Cloud MySQL uses caching_sha2_password
    which requires an encrypted connection for the auth handshake.
    If DB_KEY points to a cert file, it is loaded into the context.
    """
    import ssl as _ssl

    ctx = _ssl.create_default_context()
    # When identity verification is disabled (e.g. a self-signed MySQL in the
    # local docker-compose stack), keep TLS on but skip cert/hostname checks so
    # the caching_sha2_password handshake still succeeds. The default (verify on)
    # leaves create_default_context()'s verification intact.
    if not _ssl_verify_identity():
        ctx.check_hostname = False
        ctx.verify_mode = _ssl.CERT_NONE
    ssl_key = os.getenv("DB_KEY")
    if ssl_key and os.path.isfile(ssl_key):
        ctx.load_cert_chain(certfile=ssl_key)
    return ctx


def _sync_ssl_kwargs() -> dict:
    """
    SSL connect kwargs for the sync (pymysql) pools.

    When identity verification is on (default), preserve the original
    ssl_key + ssl_verify_identity=True behavior. When it is off, pass an
    explicit non-verifying SSLContext: PyMySQL only negotiates TLS if a truthy
    ssl_* arg is present, so ssl_verify_identity=False alone would leave a
    plaintext connection that caching_sha2_password rejects.
    """
    ssl_key = os.getenv("DB_KEY")
    if _ssl_verify_identity():
        kwargs: dict = {"ssl_verify_identity": True}
        if ssl_key:
            kwargs["ssl_key"] = ssl_key
        return kwargs
    return {"ssl": _build_ssl_context()}


async def init_pool(
    min_size: int = None,
    max_size: int = None,
):
    global _async_pool, _sync_pool

    min_size = min_size or int(os.getenv("DB_POOL_MIN_SIZE", "5"))
    max_size = max_size or int(os.getenv("DB_POOL_MAX_SIZE", "30"))

    port = int(DB_PORT) if DB_PORT else 3306

    # Build SSL context for aiomysql (always enabled for caching_sha2_password)
    ssl_ctx = _build_ssl_context()

    pool_recycle = int(os.getenv("DB_POOL_RECYCLE", "60"))

    # Async pool (aiomysql)
    _async_pool = await aiomysql.create_pool(
        host=DB_HOST,
        port=port,
        user=DB_USER,
        password=DB_PASSWORD,
        db=DB_SCHEMA,
        minsize=min_size,
        maxsize=max_size,
        pool_recycle=pool_recycle,
        autocommit=False,
        connect_timeout=10,
        cursorclass=aiomysql.DictCursor,
        ssl=ssl_ctx,
    )

    # Sync pool (DBUtils + pymysql) for thread contexts
    with _pool_lock:
        if _sync_pool is None:
            _sync_pool = PooledDB(
                creator=pymysql,
                mincached=min_size,
                maxcached=max_size,
                maxconnections=max_size,
                blocking=True,
                maxshared=0,
                ping=1,
                host=DB_HOST,
                port=port,
                user=DB_USER,
                password=DB_PASSWORD,
                database=DB_SCHEMA,
                cursorclass=pymysql.cursors.DictCursor,
                **_sync_ssl_kwargs(),
            )

    logger.info(
        f"Database pools initialized (async: {min_size}-{max_size}, sync: {min_size}-{max_size})"
    )


async def shutdown_pool():
    global _async_pool, _sync_pool
    if _async_pool is not None:
        _async_pool.close()
        await _async_pool.wait_closed()
        _async_pool = None
    with _pool_lock:
        if _sync_pool is not None:
            _sync_pool.close()
            _sync_pool = None
    logger.info("Database pools closed")


def _get_sync_pool() -> PooledDB:
    global _sync_pool
    if _sync_pool is None:
        with _pool_lock:
            if _sync_pool is None:
                port = int(DB_PORT) if DB_PORT else 3306
                _sync_pool = PooledDB(
                    creator=pymysql,
                    mincached=2,
                    maxcached=5,
                    maxconnections=10,
                    blocking=True,
                    maxshared=0,
                    ping=1,
                    host=DB_HOST,
                    port=port,
                    user=DB_USER,
                    password=DB_PASSWORD,
                    database=DB_SCHEMA,
                    cursorclass=pymysql.cursors.DictCursor,
                    **_sync_ssl_kwargs(),
                )
    return _sync_pool


def _get_async_pool() -> aiomysql.Pool:
    if _async_pool is None:
        raise RuntimeError(
            "Async pool not initialized. Call await init_pool() at app startup."
        )
    return _async_pool


@asynccontextmanager
async def _acquire():
    """
    Acquire a connection from the async pool with automatic stale-connection
    recovery.  If the pooled connection was killed server-side (IBM Cloud
    MySQL aggressively closes idle connections), we detect it via ping,
    discard it, and acquire a fresh one.
    """
    pool = _get_async_pool()
    conn = await pool.acquire()
    try:
        await conn.ping(reconnect=True)
    except Exception:
        conn.close()
        conn = await pool.acquire()
    # End any stale implicit transaction so the next query gets a fresh
    # REPEATABLE-READ snapshot.  Without this, a reused pooled connection
    # can read from a snapshot that predates a concurrent commit (e.g. the
    # INSERT from POST /dataset), causing spurious 404s on the next SELECT.
    await conn.commit()
    try:
        yield conn
    finally:
        pool.release(conn)


# ---------------------------------------------------------------------------
# Helper: group LEFT JOIN rows by a primary key
# ---------------------------------------------------------------------------
def _group_rows(rows, pk, job_fields, config_transform=None):
    grouped = {}
    for row in rows:
        key = row[pk]
        if key not in grouped:
            entity = {k: v for k, v in row.items() if k not in job_fields}
            if config_transform:
                config_transform(entity)
            entity["created_at"] = get_utc_timestamp(entity.get("created_at"))
            entity["updated_at"] = get_utc_timestamp(entity.get("updated_at"))
            entity["associated_jobs"] = []
            grouped[key] = entity
        if row.get("job_id"):
            job = {
                f.replace("job_", "", 1) if f.startswith("job_") else f: row[f]
                for f in job_fields
                if f in row
            }
            job["id"] = row["job_id"]
            grouped[key]["associated_jobs"].append(job)
    return list(grouped.values())


JOB_JOIN_FIELDS = [
    "job_id",
    "job_status",
    "job_experiment_name",
    "job_model",
    "job_tuning_type",
    "job_created_at",
    "job_updated_at",
    "job_user_id",
    "job_seed",
    "job_config_id",
    "job_dataset_id",
]


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------
class Database:
    def __init__(self):
        if not DB_HOST:
            raise RuntimeError(
                "Invalid DB_HOST, please make sure you export DB_HOST=your-db-host"
            )
        if not DB_PORT:
            raise RuntimeError(
                "Invalid DB_PORT, please make sure you export DB_PORT=your-db-port"
            )
        if not DB_USER:
            raise RuntimeError(
                "Invalid DB_USER, please make sure you export DB_USER=your-db-user"
            )
        if not DB_PASSWORD:
            raise RuntimeError(
                "Invalid DB_PASSWORD, please make sure you export DB_PASSWORD=your-db-password"
            )
        if not DB_SCHEMA:
            raise RuntimeError(
                "Invalid DB_SCHEMA, please make sure you export DB_SCHEMA=your-db-schema "
                "(DB_NAME is also accepted)"
            )

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------
    def _get_sync_connection(self):
        return _get_sync_pool().connection()

    # ------------------------------------------------------------------
    # ASYNC METHODS (primary path)
    # ------------------------------------------------------------------

    # ---- DB health ----
    async def test_db_connection_and_structure(self):
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    for table, columns in REQUIRED_TABLES.items():
                        await cursor.execute(f"SHOW TABLES LIKE '{table}';")
                        result = await cursor.fetchone()
                        if not result:
                            raise HTTPException(
                                status_code=500,
                                detail=f"Table '{table}' is missing.",
                            )
            logger.info("Database connection and table verification successful.")
        except aiomysql.MySQLError as e:
            logger.error("Database connection or table check failed.")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    # ---- Users ----
    async def insert_user(self, email: str) -> str:
        user_id = uuid.uuid4()
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = "INSERT INTO `users` (`id`, `email`) VALUES (%s, %s)"
                    await cursor.execute(sql, (str(user_id), email))
                await conn.commit()
            return str(user_id)
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in insert_user: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def touch_user_login(self, email: str) -> None:
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = "UPDATE `users` SET `updated_at`=NOW() WHERE `email`=%s"
                    await cursor.execute(sql, (email,))
                await conn.commit()
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in touch_user_login: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def update_user(self, user: api.User) -> str:
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = "UPDATE `users` SET `email`=%s, `role`=%s, `updated_at`=%s WHERE `id`=%s"
                    await cursor.execute(
                        sql, (user.email, user.role, user.updated_at, user.id)
                    )
                await conn.commit()
            return user.id
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in update_user: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def get_user(self, email: str) -> api.User:
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = "SELECT * FROM `users` WHERE `email` = %s"
                    await cursor.execute(sql, (email,))
                    result = await cursor.fetchone()
                    if result is not None:
                        result["created_at"] = get_utc_timestamp(
                            result.get("created_at")
                        )
                        result["updated_at"] = get_utc_timestamp(
                            result.get("updated_at")
                        )
                    return result
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in get_user: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def get_user_metadata(self, id: str) -> api.UserMetadata:
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    metadata = {}
                    sql = "SELECT COUNT(*) FROM `jobs` WHERE `user_id` = %s"
                    await cursor.execute(sql, (id,))
                    job_count = await cursor.fetchone()
                    metadata["number_of_jobs"] = (
                        job_count["COUNT(*)"] if job_count else 0
                    )

                    sql = "SELECT COUNT(*) FROM `configurations` WHERE `user_id` = %s OR `user_id` = %s"
                    await cursor.execute(sql, (id, constants.SYSTEM_USER))
                    config_count = await cursor.fetchone()
                    metadata["number_of_configurations"] = (
                        config_count["COUNT(*)"] if config_count else 0
                    )

                    sql = "SELECT COUNT(*) FROM `datasets` WHERE `user_id` = %s OR `user_id` = %s"
                    await cursor.execute(sql, (id, constants.SYSTEM_USER))
                    dataset_count = await cursor.fetchone()
                    metadata["number_of_datasets"] = (
                        dataset_count["COUNT(*)"] if dataset_count else 0
                    )
                    return metadata
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in get_user_metadata: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def get_user_by_id(self, id: str) -> api.User:
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = "SELECT * FROM `users` WHERE `id` = %s"
                    await cursor.execute(sql, (id,))
                    return await cursor.fetchone()
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in get_user_by_id: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def get_user_detail(self, id: str) -> api.User:
        """Single connection, multiple queries — eliminates N+1 connection overhead."""
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    # User
                    await cursor.execute("SELECT * FROM `users` WHERE `id` = %s", (id,))
                    result = await cursor.fetchone()
                    if result is None:
                        return result

                    # Jobs (use the materialized view)
                    await cursor.execute(
                        "SELECT * FROM `autotunex_jobs` WHERE `task_type`='TUNING' AND `user_id`=%s",
                        (id,),
                    )
                    jobs = await cursor.fetchall()
                    for job in jobs:
                        job["created_at"] = get_utc_timestamp(job.get("created_at"))
                        job["updated_at"] = get_utc_timestamp(job.get("updated_at"))
                        if is_gb_enabled():
                            job.pop("output_artifacts", None)
                            job.pop("build_status", None)
                            if job.get("task_started_at") and job.get(
                                "task_updated_at"
                            ):
                                job["updated_at"] = job.get("task_updated_at")
                    result["jobs"] = jobs

                    # Configs
                    await cursor.execute(
                        "SELECT * FROM `configurations` WHERE `user_id`=%s OR `user_id`=%s",
                        (id, constants.SYSTEM_USER),
                    )
                    configs = await cursor.fetchall()
                    for c in configs:
                        c["config_data"] = json.loads(c["config_data"])
                        c["created_at"] = get_utc_timestamp(c.get("created_at"))
                        c["updated_at"] = get_utc_timestamp(c.get("updated_at"))
                    result["configs"] = configs

                    # Datasets
                    await cursor.execute(
                        "SELECT * FROM `datasets` WHERE `user_id`=%s OR `user_id`=%s",
                        (id, constants.SYSTEM_USER),
                    )
                    datasets = await cursor.fetchall()
                    for d in datasets:
                        d["created_at"] = get_utc_timestamp(d.get("created_at"))
                        d["updated_at"] = get_utc_timestamp(d.get("updated_at"))
                    result["datasets"] = datasets
                    return result
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in get_user_detail: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def get_users(self) -> list[api.User]:
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute("SELECT * FROM `users`")
                    results = await cursor.fetchall()
                    for result in results:
                        result["created_at"] = get_utc_timestamp(
                            result.get("created_at")
                        )
                        result["updated_at"] = get_utc_timestamp(
                            result.get("updated_at")
                        )
                    return results
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in get_users: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    # ---- Configurations ----
    async def insert_configuration(self, config: api.Configuration) -> str:
        config_id = uuid.uuid4()
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = "INSERT INTO `configurations` (`id`, `user_id`, `name`, `tuner_type`, `rl_tuner_type`, `config_data`) VALUES (%s, %s, %s, %s, %s, %s)"
                    await cursor.execute(
                        sql,
                        (
                            str(config_id),
                            config.user_id,
                            config.name,
                            config.tuner_type,
                            config.rl_tuner_type,
                            json.dumps(config.config_data),
                        ),
                    )
                await conn.commit()
            return str(config_id)
        except aiomysql.MySQLError as e:
            if getattr(e, "args", [None])[0] == 1062:
                raise HTTPException(
                    status_code=409,
                    detail=f"A configuration named '{config.name}' already exists for this user.",
                )
            logger.error(f"Database error in insert_configuration: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def update_configuration(self, config: api.Configuration) -> str:
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = "UPDATE `configurations` SET `name`=%s, `tuner_type`=%s, `rl_tuner_type`=%s, `config_data`=%s WHERE `id`=%s"
                    await cursor.execute(
                        sql,
                        (
                            config.name,
                            config.tuner_type,
                            config.rl_tuner_type,
                            json.dumps(config.config_data),
                            config.id,
                        ),
                    )
                await conn.commit()
            return config.id
        except aiomysql.MySQLError as e:
            if getattr(e, "args", [None])[0] == 1062:
                raise HTTPException(
                    status_code=409,
                    detail=f"A configuration named '{config.name}' already exists for this user.",
                )
            logger.error(f"Database error in update_configuration: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def get_configs(
        self, user_id: str, ids: list[str] = None
    ) -> list[api.Config]:
        """N+1 fix: LEFT JOIN jobs instead of per-config loop query."""
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    job_select = (
                        ", j.id AS job_id, j.status AS job_status, j.experiment_name AS job_experiment_name"
                        ", j.model AS job_model, j.tuning_type AS job_tuning_type"
                        ", j.created_at AS job_created_at, j.updated_at AS job_updated_at"
                        ", j.user_id AS job_user_id, j.seed AS job_seed"
                        ", j.config_id AS job_config_id, j.dataset_id AS job_dataset_id"
                    )
                    job_join = (
                        "LEFT JOIN `jobs` j ON j.config_id = c.id AND j.user_id = %s"
                    )

                    if ids is None or len(ids) == 0:
                        sql = f"SELECT c.*{job_select} FROM `configurations` c {job_join} WHERE (c.`user_id`=%s OR c.`user_id`=%s)"
                        await cursor.execute(
                            sql, (user_id, user_id, constants.SYSTEM_USER)
                        )
                    elif len(ids) == 1:
                        sql = f"SELECT c.*{job_select} FROM `configurations` c {job_join} WHERE c.`id` = %s"
                        await cursor.execute(sql, (user_id, ids[0]))
                    else:
                        in_fmt = ",".join(["%s"] * len(ids))
                        sql = f"SELECT c.*{job_select} FROM `configurations` c {job_join} WHERE c.`id` IN ({in_fmt})"
                        await cursor.execute(sql, (user_id, *ids))

                    rows = await cursor.fetchall()

                    def transform(entity):
                        entity["config_data"] = json.loads(entity["config_data"])

                    return _group_rows(
                        rows, "id", JOB_JOIN_FIELDS, config_transform=transform
                    )
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in get_configs: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def get_config(
        self, config_id: str, user_id: str = None
    ) -> api.Configuration:
        """N+1 fix: JOIN jobs in single query."""
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    if user_id:
                        sql = (
                            "SELECT c.*, j.id AS job_id, j.status AS job_status, j.experiment_name AS job_experiment_name"
                            ", j.model AS job_model, j.tuning_type AS job_tuning_type"
                            ", j.created_at AS job_created_at, j.updated_at AS job_updated_at"
                            ", j.user_id AS job_user_id, j.seed AS job_seed"
                            ", j.config_id AS job_config_id, j.dataset_id AS job_dataset_id"
                            " FROM `configurations` c"
                            " LEFT JOIN `jobs` j ON j.config_id = c.id AND j.user_id = %s"
                            " WHERE c.id = %s AND (c.user_id = %s OR c.user_id = %s)"
                        )
                        await cursor.execute(
                            sql, (user_id, config_id, user_id, constants.SYSTEM_USER)
                        )
                    else:
                        sql = (
                            "SELECT c.*, j.id AS job_id, j.status AS job_status, j.experiment_name AS job_experiment_name"
                            ", j.model AS job_model, j.tuning_type AS job_tuning_type"
                            ", j.created_at AS job_created_at, j.updated_at AS job_updated_at"
                            ", j.user_id AS job_user_id, j.seed AS job_seed"
                            ", j.config_id AS job_config_id, j.dataset_id AS job_dataset_id"
                            " FROM `configurations` c"
                            " LEFT JOIN `jobs` j ON j.config_id = c.id"
                            " WHERE c.id = %s"
                        )
                        await cursor.execute(sql, (config_id,))

                    rows = await cursor.fetchall()
                    if not rows:
                        raise HTTPException(status_code=404, detail="Invalid config_id")

                    # Build config from first row
                    first = rows[0]
                    result = {
                        k: v for k, v in first.items() if k not in JOB_JOIN_FIELDS
                    }
                    result["config_data"] = json.loads(result["config_data"])
                    result["created_at"] = get_utc_timestamp(result.get("created_at"))
                    result["updated_at"] = get_utc_timestamp(result.get("updated_at"))
                    result["associated_jobs"] = []
                    for row in rows:
                        if row.get("job_id"):
                            job = {
                                f.replace("job_", "", 1)
                                if f.startswith("job_")
                                else f: row[f]
                                for f in JOB_JOIN_FIELDS
                                if f in row
                            }
                            job["id"] = row["job_id"]
                            result["associated_jobs"].append(job)
                    return result
        except HTTPException:
            raise
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in get_config: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def get_config_by_name_and_user(self, config_name: str, user_id: str):
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = "SELECT * FROM `configurations` WHERE `name` = %s AND `user_id` = %s"
                    await cursor.execute(sql, (config_name, user_id))
                    result = await cursor.fetchone()
                    if result is not None:
                        result["config_data"] = json.loads(result["config_data"])
                        result["created_at"] = get_utc_timestamp(
                            result.get("created_at")
                        )
                        result["updated_at"] = get_utc_timestamp(
                            result.get("updated_at")
                        )
            return result
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in get_config_by_name_and_user: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def delete_config(self, config_id: str, user_id: str) -> bool:
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = "DELETE FROM `configurations` WHERE `id` = %s AND `user_id` = %s"
                    await cursor.execute(sql, (config_id, user_id))
                await conn.commit()
            return True
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in delete_config: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    # ---- Datasets ----
    async def get_dataset_by_name_and_user(self, dataset_name: str, user_id: str):
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = (
                        "SELECT * FROM `datasets` WHERE `name` = %s AND `user_id` = %s"
                    )
                    await cursor.execute(sql, (dataset_name, user_id))
                    result = await cursor.fetchone()
                    if result is not None:
                        result["created_at"] = get_utc_timestamp(
                            result.get("created_at")
                        )
                        result["updated_at"] = get_utc_timestamp(
                            result.get("updated_at")
                        )
            return result
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in get_dataset_by_name_and_user: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def insert_dataset(self, dataset: api.DatasetInfo) -> api.DatasetInfo:
        dataset_id = uuid.uuid4()
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = "INSERT INTO `datasets` (`id`, `user_id`, `name`, `description`) VALUES (%s, %s, %s, %s)"
                    await cursor.execute(
                        sql,
                        (
                            str(dataset_id),
                            dataset.user_id,
                            dataset.name,
                            dataset.description,
                        ),
                    )
                await conn.commit()
                dataset.id = dataset_id
            return dataset
        except aiomysql.MySQLError as e:
            if getattr(e, "args", [None])[0] == 1062:
                raise HTTPException(
                    status_code=409,
                    detail=f"A dataset named '{dataset.name}' already exists for this user.",
                )
            raise HTTPException(status_code=500, detail=f"Internal Server error: \n{e}")

    async def update_dataset(self, dataset: api.DatasetInfo) -> api.DatasetInfo:
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = "UPDATE `datasets` SET `name`=%s, `description`=%s WHERE `id`=%s"
                    await cursor.execute(
                        sql, (dataset.name, dataset.description, dataset.id)
                    )
                await conn.commit()
            return dataset
        except aiomysql.MySQLError as e:
            if getattr(e, "args", [None])[0] == 1062:
                raise HTTPException(
                    status_code=409,
                    detail=f"A dataset named '{dataset.name}' already exists for this user.",
                )
            raise HTTPException(status_code=500, detail=f"Internal Server error: \n{e}")

    async def update_dataset_metadata(
        self, id: str, user_id: str, metadata: Dict[str, Any]
    ) -> api.DatasetInfo:
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = "UPDATE `datasets` SET `train_records`=%s, `train_file_size`=%s, `validation_records`=%s, `validation_file_size`=%s, `data_format`=%s, `artifact_id`=%s, `artifact_url`=%s WHERE `id`=%s"
                    await cursor.execute(
                        sql,
                        (
                            metadata["train_records"],
                            metadata["train_file_size"],
                            metadata["validation_records"],
                            metadata["validation_file_size"],
                            metadata.get("data_format", "jsonl"),
                            metadata["artifact_id"],
                            metadata["artifact_url"],
                            id,
                        ),
                    )
                await conn.commit()
            result = await self.get_dataset(dataset_id=id, user_id=user_id)
            return result
        except HTTPException:
            raise
        except aiomysql.MySQLError as e:
            raise HTTPException(status_code=500, detail=f"Internal Server error: \n{e}")

    async def check_dataset_exists(self, id: str) -> bool:
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = "SELECT EXISTS(SELECT * FROM `datasets` WHERE `id` = %s) AS dataset_exists"
                    await cursor.execute(sql, (id,))
                    result = await cursor.fetchone()
            return result["dataset_exists"]
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in check_dataset_exists: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def get_dataset(
        self, dataset_id: str, user_id: Optional[str] = None
    ) -> api.DatasetInfo:
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    is_id = True
                    try:
                        UUID(dataset_id)
                    except ValueError:
                        is_id = False
                    if user_id is not None:
                        if is_id:
                            sql = "SELECT * FROM `datasets` WHERE `id`=%s AND (`user_id`=%s OR `user_id`=%s)"
                        else:
                            sql = "SELECT * FROM `datasets` WHERE `name`=%s AND (`user_id`=%s OR `user_id`=%s)"
                        await cursor.execute(
                            sql, (dataset_id, user_id, constants.SYSTEM_USER)
                        )
                    else:
                        if is_id:
                            sql = "SELECT * FROM `datasets` WHERE `id`=%s"
                        else:
                            sql = "SELECT * FROM `datasets` WHERE `name`=%s"
                        await cursor.execute(sql, (dataset_id,))
                    result = await cursor.fetchone()
                    if result is not None:
                        result["created_at"] = get_utc_timestamp(
                            result.get("created_at")
                        )
                        result["updated_at"] = get_utc_timestamp(
                            result.get("updated_at")
                        )
                    return result
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in get_dataset: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def get_datasets(
        self, user_id: str, ids: list[str] = None
    ) -> list[api.DatasetInfo]:
        """N+1 fix: LEFT JOIN jobs instead of per-dataset loop query."""
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    job_select = (
                        ", j.id AS job_id, j.status AS job_status, j.experiment_name AS job_experiment_name"
                        ", j.model AS job_model, j.tuning_type AS job_tuning_type"
                        ", j.created_at AS job_created_at, j.updated_at AS job_updated_at"
                        ", j.user_id AS job_user_id, j.seed AS job_seed"
                        ", j.config_id AS job_config_id, j.dataset_id AS job_dataset_id"
                    )
                    job_join = (
                        "LEFT JOIN `jobs` j ON j.dataset_id = d.id AND j.user_id = %s"
                    )

                    if ids is None or len(ids) == 0:
                        sql = f"SELECT d.*{job_select} FROM `datasets` d {job_join} WHERE (d.`user_id` = %s OR d.`user_id` = %s)"
                        await cursor.execute(
                            sql, (user_id, user_id, constants.SYSTEM_USER)
                        )
                    elif len(ids) == 1:
                        sql = f"SELECT d.*{job_select} FROM `datasets` d {job_join} WHERE d.`id` = %s"
                        await cursor.execute(sql, (user_id, ids[0]))
                    else:
                        in_fmt = ",".join(["%s"] * len(ids))
                        sql = f"SELECT d.*{job_select} FROM `datasets` d {job_join} WHERE d.`id` IN ({in_fmt})"
                        await cursor.execute(sql, (user_id, *ids))

                    rows = await cursor.fetchall()
                    return _group_rows(rows, "id", JOB_JOIN_FIELDS)
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in get_datasets: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def delete_dataset(self, job_id: str, user_id: str) -> bool:
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = "DELETE FROM `datasets` WHERE `id` = %s AND `user_id` = %s"
                    await cursor.execute(sql, (job_id, user_id))
                await conn.commit()
            return True
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in delete_dataset: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    # ---- Jobs ----
    async def insert_job(
        self, config: api.TuningConfig, config_snapshot: dict = None
    ) -> str:
        job_id = uuid.uuid4()
        precision = "bf16"
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = "INSERT INTO `jobs` (`id`, `user_id`, `status`, `seed`, `config_id`, `config_snapshot`, `dataset_id`, `model`, `model_source`, `experiment_name`, `tuning_type`, `precision`, `ray_address`, `cleanup`, `autotune`) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                    await cursor.execute(
                        sql,
                        (
                            str(job_id),
                            config.user_id,
                            config.status.value,
                            config.seed,
                            config.config_id,
                            json.dumps(config_snapshot) if config_snapshot else None,
                            config.dataset_id,
                            config.model,
                            config.model_source.value,
                            config.experiment_name,
                            config.tuning_type,
                            precision,
                            config.ray_address,
                            config.cleanup,
                            config.autotune,
                        ),
                    )
                await conn.commit()
            return str(job_id)
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in insert_job: {e}")
            if "Cannot add or update a child row" in str(e):
                raise HTTPException(status_code=404, detail="Config_id doesn't exist")
            raise HTTPException(status_code=500, detail="Internal Server error")

    async def update_job_status(self, id: str, status: api.JobStatus) -> str:
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = "UPDATE `jobs` SET `status`=%s WHERE `id`=%s"
                    await cursor.execute(sql, (status.value, id))
                await conn.commit()
            return True
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in update_job_status: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def get_job(
        self,
        id: str,
        user_id: str,
        include_logs: bool = True,
        log_limit: int = 1000,
        all_logs: bool = False,
    ) -> api.JobResponse:
        """N+1 fix: JOIN config+dataset, then fetch logs as a second query."""
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = (
                        "SELECT j.*,"
                        " COALESCE(JSON_UNQUOTE(JSON_EXTRACT(j.config_snapshot, '$.name')), c.name) AS config_name,"
                        " COALESCE(NULLIF(JSON_UNQUOTE(JSON_EXTRACT(j.config_snapshot, '$.rl_tuner_type')), 'null'), c.rl_tuner_type) AS rl_tuner_type,"
                        " d.name AS dataset_name"
                        " FROM `jobs` j"
                        " LEFT JOIN `configurations` c ON c.id = j.config_id"
                        " LEFT JOIN `datasets` d ON d.id = j.dataset_id"
                        " WHERE j.id = %s AND j.user_id = %s"
                    )
                    await cursor.execute(sql, (id, user_id))
                    result = await cursor.fetchone()
                    if not result:
                        raise HTTPException(status_code=404, detail="Invalid job_id")

                    if result.get("config_snapshot") and isinstance(
                        result["config_snapshot"], str
                    ):
                        result["config_snapshot"] = json.loads(
                            result["config_snapshot"]
                        )

                    result["dataset"] = result.pop("dataset_name", None)
                    result["created_at"] = get_utc_timestamp(result.get("created_at"))
                    result["updated_at"] = get_utc_timestamp(result.get("updated_at"))

                    # Fetch logs (second query, same connection)
                    if include_logs:
                        sql_logs = "SELECT * FROM `log_entries` WHERE `job_id`=%s AND `trial_id` IS NULL ORDER BY `timestamp` DESC"
                        if all_logs:
                            await cursor.execute(sql_logs, (id,))
                        else:
                            sql_logs += " LIMIT %s"
                            await cursor.execute(sql_logs, (id, log_limit))
                        logs = await cursor.fetchall()
                        for log in logs:
                            log.pop("trial_id", None)
                            log.pop("epoch", None)
                            log.pop("iteration", None)
                            log.pop("job_id", None)
                            log["timestamp"] = get_utc_timestamp(log.get("timestamp"))
                        result["logs"] = logs
                    else:
                        result["logs"] = []
                    return result
        except HTTPException:
            raise
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in get_job: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def get_job_config_snapshot(self, job_id: str, user_id: str) -> dict:
        """Return the config snapshot for a job, with an is_stale flag."""
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = (
                        "SELECT j.config_snapshot, j.config_id, j.created_at AS job_created_at,"
                        " c.updated_at AS config_updated_at, c.name, c.tuner_type,"
                        " c.rl_tuner_type, c.config_data"
                        " FROM `jobs` j"
                        " LEFT JOIN `configurations` c ON c.id = j.config_id"
                        " WHERE j.id = %s AND j.user_id = %s"
                    )
                    await cursor.execute(sql, (job_id, user_id))
                    row = await cursor.fetchone()
                    if not row:
                        raise HTTPException(status_code=404, detail="Invalid job_id")

                    is_stale = False
                    if row.get("job_created_at") and row.get("config_updated_at"):
                        is_stale = row["config_updated_at"] > row["job_created_at"]

                    snapshot = row.get("config_snapshot")
                    if snapshot:
                        if isinstance(snapshot, str):
                            snapshot = json.loads(snapshot)
                        return {
                            "name": snapshot.get("name"),
                            "tuner_type": snapshot.get("tuner_type"),
                            "rl_tuner_type": snapshot.get("rl_tuner_type"),
                            "config_data": snapshot.get("config_data"),
                            "is_stale": is_stale,
                        }

                    # Fallback for jobs created before snapshotting was added
                    config_data = row.get("config_data")
                    if isinstance(config_data, str):
                        config_data = json.loads(config_data)
                    return {
                        "name": row.get("name"),
                        "tuner_type": row.get("tuner_type"),
                        "rl_tuner_type": row.get("rl_tuner_type"),
                        "config_data": config_data,
                        "is_stale": is_stale,
                    }
        except HTTPException:
            raise
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in get_job_config_snapshot: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def get_job_by_id(self, id: str) -> api.JobResponse:
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = "SELECT * FROM `jobs` WHERE `id`=%s"
                    await cursor.execute(sql, (id,))
                    return await cursor.fetchone()
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in get_job_by_id: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def get_jobs(self, user_id: str = None) -> list[api.JobResponse]:
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    if user_id:
                        sql = "SELECT * FROM `autotunex_jobs` WHERE `task_type`='TUNING' AND `user_id`=%s"
                        await cursor.execute(sql, (user_id,))
                    else:
                        sql = (
                            "SELECT * FROM `autotunex_jobs` WHERE `task_type`='TUNING'"
                        )
                        await cursor.execute(sql)
                    results = await cursor.fetchall()
                    for result in results:
                        result["created_at"] = get_utc_timestamp(
                            result.get("created_at")
                        )
                        result["updated_at"] = get_utc_timestamp(
                            result.get("updated_at")
                        )
                        if is_gb_enabled():
                            result.pop("output_artifacts", None)
                            result.pop("build_status", None)
                            if result.get("task_started_at") and result.get(
                                "task_updated_at"
                            ):
                                result["updated_at"] = result.get("task_updated_at")
                    return results
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in get_jobs: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def get_job_stats(self, user_id: str) -> dict:
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = "SELECT `status`, COUNT(*) AS `count` FROM `jobs` WHERE `user_id`=%s GROUP BY `status`"
                    await cursor.execute(sql, (user_id,))
                    rows = await cursor.fetchall()
                    stats = {status.value.lower(): 0 for status in api.JobStatus}
                    for row in rows:
                        stats[row["status"].lower()] = row["count"]
                    stats["total"] = sum(stats.values())
                    return stats
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in get_job_stats: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def delete_job(self, job_id: str, user_id) -> bool:
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = "DELETE FROM `jobs` WHERE `id` = %s AND `user_id` = %s"
                    await cursor.execute(sql, (job_id, user_id))
                await conn.commit()
            return True
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in delete_job: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def push_job_artifacts(
        self, job_id: str, output_artifacts: Dict[str, Any]
    ) -> bool:
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = "UPDATE `jobs` SET `output_artifacts`=%s WHERE `id`=%s"
                    await cursor.execute(sql, (json.dumps(output_artifacts), job_id))
                await conn.commit()
            return True
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in push_job_artifacts: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def get_running_jobs(self) -> list[api.JobResponse]:
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = (
                        "SELECT * FROM `jobs` WHERE `status` IN ('RUNNING', 'PENDING')"
                    )
                    await cursor.execute(sql)
                    return await cursor.fetchall()
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in get_running_jobs: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    # ---- Logging ----
    async def create_logging_table(self) -> bool:
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(
                        """
                        CREATE TABLE IF NOT EXISTS log_entries (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            job_id CHAR(36) NOT NULL,
                            level VARCHAR(50),
                            filename VARCHAR(255),
                            message TEXT,
                            timestamp DATETIME,
                            FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
                        );
                        """
                    )
                await conn.commit()
            return True
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in create_logging_table: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def insert_logs(self, buffer: list) -> bool:
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.executemany(
                        """
                        INSERT INTO log_entries (
                            job_id, trial_id, level, filename, message, iteration, epoch, timestamp
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        [
                            (
                                entry["job_id"],
                                entry["trial_id"],
                                entry["level"],
                                entry["filename"],
                                entry["message"],
                                entry["iteration"],
                                entry["epoch"],
                                entry["timestamp"],
                            )
                            for entry in buffer
                        ],
                    )
                await conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error inserting logs: {str(e)}")
            return False

    async def get_job_logs(self, id: str) -> list[api.LogEntry]:
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = "SELECT * FROM `log_entries` WHERE `job_id`=%s AND `trial_id` IS NULL ORDER BY `timestamp` DESC"
                    await cursor.execute(sql, (id,))
                    results = await cursor.fetchall()
                    for result in results:
                        result.pop("trial_id", None)
                        result.pop("epoch", None)
                        result.pop("iteration", None)
                        result["timestamp"] = get_utc_timestamp(result.get("timestamp"))
            return results
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in get_job_logs: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def get_logs_page(
        self, job_id: str, before_id: int = 0, limit: int = 50
    ) -> dict:
        """Fetch a page of log entries in descending order for scroll pagination."""
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    if before_id > 0:
                        sql = (
                            "SELECT `id`, `level`, `filename`, `message`, `timestamp` "
                            "FROM `log_entries` "
                            "WHERE `job_id`=%s AND `trial_id` IS NULL AND `id` < %s "
                            "ORDER BY `id` DESC LIMIT %s"
                        )
                        await cursor.execute(sql, (job_id, before_id, limit + 1))
                    else:
                        sql = (
                            "SELECT `id`, `level`, `filename`, `message`, `timestamp` "
                            "FROM `log_entries` "
                            "WHERE `job_id`=%s AND `trial_id` IS NULL "
                            "ORDER BY `id` DESC LIMIT %s"
                        )
                        await cursor.execute(sql, (job_id, limit + 1))
                    results = await cursor.fetchall()
                    has_more = len(results) > limit
                    if has_more:
                        results = results[:limit]
                    for row in results:
                        row["timestamp"] = get_utc_timestamp(row.get("timestamp"))
                    return {"logs": results, "has_more": has_more}
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in get_logs_page: {e}")
            return {"logs": [], "has_more": False}

    async def get_trial_logs_page(
        self, trial_id: str, before_id: int = 0, limit: int = 50
    ) -> dict:
        """Fetch a page of trial log entries in descending order for scroll pagination."""
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    if before_id > 0:
                        sql = (
                            "SELECT `id`, `level`, `filename`, `message`, `timestamp` "
                            "FROM `log_entries` "
                            "WHERE `trial_id`=%s AND `id` < %s "
                            "ORDER BY `id` DESC LIMIT %s"
                        )
                        await cursor.execute(sql, (trial_id, before_id, limit + 1))
                    else:
                        sql = (
                            "SELECT `id`, `level`, `filename`, `message`, `timestamp` "
                            "FROM `log_entries` "
                            "WHERE `trial_id`=%s "
                            "ORDER BY `id` DESC LIMIT %s"
                        )
                        await cursor.execute(sql, (trial_id, limit + 1))
                    results = await cursor.fetchall()
                    has_more = len(results) > limit
                    if has_more:
                        results = results[:limit]
                    for row in results:
                        row["timestamp"] = get_utc_timestamp(row.get("timestamp"))
                    return {"logs": results, "has_more": has_more}
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in get_trial_logs_page: {e}")
            return {"logs": [], "has_more": False}

    # ---- Trials ----
    async def insert_trial(self, data: api.Trial) -> str:
        config_copy = data["config"].copy() if isinstance(data["config"], dict) else {}
        if (
            isinstance(config_copy.get("tune_config"), dict)
            and "search_alg" in config_copy["tune_config"]
        ):
            config_copy["tune_config"]["search_alg"] = config_copy["tune_config"][
                "search_alg"
            ].__class__.__name__
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = "INSERT INTO `trials` (`id`, `job_id`, `status`, `config`) VALUES (%s, %s, %s, %s)"
                    await cursor.execute(
                        sql,
                        (
                            data["id"],
                            data["job_id"],
                            data["status"].value,
                            json.dumps(config_copy),
                        ),
                    )
                await conn.commit()
            return data["id"]
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in insert_trial: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def update_trial_status(self, trial_id: str, status: api.TrialStatus) -> str:
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = "UPDATE `trials` SET `status`=%s WHERE `id`=%s"
                    await cursor.execute(sql, (status.value, trial_id))
                await conn.commit()
            return True
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in update_trial_status: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def update_all_trial_status(
        self, job_id: str, status: api.TrialStatus
    ) -> str:
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = "UPDATE `trials` SET `status`=%s WHERE `job_id`=%s"
                    await cursor.execute(sql, (status.value, job_id))
                await conn.commit()
            return True
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in update_all_trial_status: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def get_trials_by_job_id(self, job_id: str) -> list[api.Trial]:
        """N+1 fix: LEFT JOIN results instead of per-trial query."""
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = (
                        "SELECT t.*, r.id AS result_id, r.metric AS result_metric, r.metrics AS result_metrics"
                        ", r.created_at AS result_created_at, r.updated_at AS result_updated_at"
                        " FROM `trials` t"
                        " LEFT JOIN `results` r ON r.trial_id = t.id"
                        " WHERE t.job_id = %s"
                    )
                    await cursor.execute(sql, (job_id,))
                    rows = await cursor.fetchall()
                    if not rows:
                        raise HTTPException(status_code=404, detail="Invalid job_id")

                    results = []
                    seen = set()
                    for row in rows:
                        tid = row["id"]
                        if tid not in seen:
                            seen.add(tid)
                            trial = {
                                "id": row["id"],
                                "job_id": row["job_id"],
                                "status": row["status"],
                                "config": json.loads(row["config"]),
                                "created_at": get_utc_timestamp(row.get("created_at")),
                                "updated_at": get_utc_timestamp(row.get("updated_at")),
                                "logs": [],
                            }
                            if row.get("result_id"):
                                trial["score"] = {
                                    "id": row["result_id"],
                                    "metric": row["result_metric"],
                                    "metrics": json.loads(row["result_metrics"]),
                                    "created_at": get_utc_timestamp(
                                        row.get("result_created_at")
                                    ),
                                    "updated_at": get_utc_timestamp(
                                        row.get("result_updated_at")
                                    ),
                                }
                            else:
                                trial["score"] = {}
                            results.append(trial)
                    return results
        except HTTPException:
            raise
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in get_trials_by_job_id: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def get_trials_logs_by_job_id(self, job_id: str) -> list[api.LogEntry]:
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = "SELECT * FROM `log_entries` WHERE `trial_id` IS NOT NULL AND `job_id`=%s ORDER BY `trial_id`, `timestamp` DESC"
                    await cursor.execute(sql, (job_id,))
                    results = await cursor.fetchall()
                    for result in results:
                        result["timestamp"] = get_utc_timestamp(result["timestamp"])
            return results
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in get_trials_logs_by_job_id: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def get_trials_logs_by_id(self, trial_id: str) -> list[api.LogEntry]:
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = "SELECT * FROM `log_entries` WHERE `trial_id`=%s"
                    await cursor.execute(sql, (trial_id,))
                    results = await cursor.fetchall()
                    for result in results:
                        result["timestamp"] = get_utc_timestamp(result["timestamp"])
            return results
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in get_trials_logs_by_id: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    # ---- Results ----
    async def insert_result(self, metadata: api.Result) -> api.Result:
        result_id = uuid.uuid4()
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = "INSERT INTO `results` (`id`, `job_id`, `trial_id`, `metric`, `metrics`) VALUES (%s, %s, %s, %s, %s)"
                    await cursor.execute(
                        sql,
                        (
                            str(result_id),
                            metadata["job_id"],
                            metadata["trial_id"],
                            metadata["metric"],
                            json.dumps(metadata["metrics"]),
                        ),
                    )
                await conn.commit()
            return metadata
        except aiomysql.MySQLError as e:
            raise HTTPException(status_code=500, detail=f"Internal Server error: \n{e}")

    async def get_results_by_job_id(self, job_id: str):
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = "SELECT * FROM `results` WHERE `job_id`=%s"
                    await cursor.execute(sql, (job_id,))
                    results = await cursor.fetchall()
                    if results is not None:
                        for result in results:
                            result["metrics"] = json.loads(result["metrics"])
                            result["created_at"] = get_utc_timestamp(
                                result.get("created_at")
                            )
                            result["updated_at"] = get_utc_timestamp(
                                result.get("updated_at")
                            )
                    else:
                        raise HTTPException(status_code=404, detail="Invalid job_id")
            return results
        except HTTPException:
            raise
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in get_results_by_job_id: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def get_result_by_trial_id(self, trial_id: str):
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = "SELECT * FROM `results` WHERE `trial_id`=%s"
                    await cursor.execute(sql, (trial_id,))
                    result = await cursor.fetchone()
                    if result is not None:
                        result["metrics"] = json.loads(result["metrics"])
                        result["created_at"] = get_utc_timestamp(
                            result.get("created_at")
                        )
                        result["updated_at"] = get_utc_timestamp(
                            result.get("updated_at")
                        )
                    else:
                        result = {}
            return result
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in get_result_by_trial_id: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    # ---- Tasks ----
    async def insert_task(self, task: api.Task) -> str:
        task_id = task.id if task.id else str(uuid.uuid4())
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = "INSERT INTO `gb_tasks` (`id`, `job_id`, `type`, `pr_url`, `started_at`, `updated_at`) VALUES (%s, %s, %s, %s, %s, %s)"
                    await cursor.execute(
                        sql,
                        (
                            task_id,
                            task.job_id,
                            task.type,
                            task.pr_url,
                            task.started_at,
                            task.updated_at,
                        ),
                    )
                await conn.commit()
            return task_id
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in insert_task: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def update_task(self, task: api.Task) -> bool:
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = "UPDATE `gb_tasks` SET `build_id`=%s, `status`=%s, `pr_url`=%s, `artifact_id`=%s, `artifact_uri`=%s, `build_status`=%s, `started_at`=%s, `updated_at`=%s, `rits_url`=%s WHERE `id`=%s"
                    await cursor.execute(
                        sql,
                        (
                            task.build_id,
                            task.status,
                            task.pr_url,
                            task.artifact_id,
                            task.artifact_uri,
                            json.dumps(task.build_status),
                            task.started_at,
                            task.updated_at,
                            task.rits_url,
                            task.id,
                        ),
                    )
                await conn.commit()
            return True
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in update_task: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def get_task_by_job_id(self, job_id: str, type: api.TaskType):
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = "SELECT * FROM `gb_tasks` WHERE `job_id`=%s AND `type`=%s ORDER BY `updated_at` DESC"
                    await cursor.execute(sql, (job_id, type))
                    result = await cursor.fetchone()
                    try:
                        if result and result.get("build_status"):
                            result["build_status"] = json.loads(result["build_status"])
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.error(f"error occurred in get_task_by_job_id: {e}")
                        result["build_status"] = None
            return result
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in get_task_by_job_id: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def get_job_id_by_build_id(self, build_id: str) -> str:
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = "SELECT `job_id` FROM `gb_tasks` WHERE `build_id`=%s ORDER BY `updated_at` DESC LIMIT 1"
                    await cursor.execute(sql, (build_id,))
                    result = await cursor.fetchone()
                    if not result:
                        raise HTTPException(status_code=404, detail="Invalid build_id")
                    return result["job_id"]
        except HTTPException:
            raise
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in get_job_id_by_build_id: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def get_task(self, id: str) -> api.Task:
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = "SELECT * FROM `gb_tasks` WHERE `id` = %s"
                    await cursor.execute(sql, (id,))
                    result = await cursor.fetchone()
                    if result and result.get("build_status"):
                        try:
                            result["build_status"] = json.loads(result["build_status"])
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.error(f"error occurred in get_task: {e}")
                            result["build_status"] = None
                    return result
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in get_task: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def get_tasks(self, job_id: str) -> list[api.Task]:
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = "SELECT * FROM `gb_tasks` WHERE `job_id` = %s"
                    await cursor.execute(sql, (job_id,))
                    results = await cursor.fetchall()
                    for result in results:
                        if result and result.get("build_status"):
                            try:
                                result["build_status"] = json.loads(
                                    result["build_status"]
                                )
                            except (json.JSONDecodeError, TypeError) as e:
                                logger.error(f"error occurred in get_tasks: {e}")
                                result["build_status"] = None
                    return results
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in get_tasks: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def get_pending_tasks(self) -> list[api.Task]:
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = "SELECT * FROM `gb_tasks` WHERE `status` IN ('RUNNING', 'PENDING')"
                    await cursor.execute(sql)
                    results = await cursor.fetchall()
                    for result in results:
                        if result and result.get("build_status"):
                            try:
                                result["build_status"] = json.loads(
                                    result["build_status"]
                                )
                            except (json.JSONDecodeError, TypeError) as e:
                                logger.error(
                                    f"error occurred in get_pending_tasks: {e}"
                                )
                                result["build_status"] = None
                    return results
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in get_pending_tasks: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    async def get_expired_download_tasks(self, max_age_minutes: int = 60) -> list:
        """Get DOWNLOAD tasks older than max_age_minutes."""
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = """
                        SELECT * FROM `gb_tasks`
                        WHERE `type` = 'DOWNLOAD'
                        AND `updated_at` < DATE_FORMAT(
                            DATE_SUB(UTC_TIMESTAMP(), INTERVAL %s MINUTE),
                            '%%Y-%%m-%%dT%%H:%%i:%%s'
                        )
                    """
                    await cursor.execute(sql, (max_age_minutes,))
                    results = await cursor.fetchall()
                    for result in results:
                        if result and result.get("build_status"):
                            try:
                                result["build_status"] = json.loads(
                                    result["build_status"]
                                )
                            except (json.JSONDecodeError, TypeError):
                                result["build_status"] = None
                    return results
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in get_expired_download_tasks: {e}")
            return []

    async def delete_task(self, task_id: str) -> bool:
        """Delete a task by ID."""
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = "DELETE FROM `gb_tasks` WHERE `id` = %s"
                    await cursor.execute(sql, (task_id,))
                await conn.commit()
            return True
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in delete_task: {e}")
            return False

    # ---- Published models ----
    async def get_gb_published_models(self, user_id: str = None) -> list[api.ModelInfo]:
        try:
            async with _acquire() as conn:
                async with conn.cursor() as cursor:
                    base_sql = (
                        "SELECT jobs.id AS id, gb_tasks.id AS task_id, jobs.user_id,"
                        " (SELECT email FROM users WHERE id = jobs.user_id) AS user,"
                        " jobs.status, jobs.model AS base_model, jobs.experiment_name,"
                        " jobs.output_artifacts, gb_tasks.artifact_uri, jobs.created_at,"
                        " gb_tasks.updated_at"
                        " FROM jobs INNER JOIN gb_tasks ON jobs.id = gb_tasks.job_id"
                        ' WHERE jobs.status = "COMPLETED" AND gb_tasks.type = "TUNING"'
                        " AND gb_tasks.artifact_uri IS NOT NULL"
                    )
                    if user_id is not None:
                        sql = (
                            base_sql
                            + " AND jobs.user_id = %s ORDER BY gb_tasks.updated_at DESC"
                        )
                        await cursor.execute(sql, (user_id,))
                    else:
                        sql = base_sql + " ORDER BY gb_tasks.updated_at DESC"
                        await cursor.execute(sql)

                    results = await cursor.fetchall()
                    for result in results:
                        result["model_label"] = result.get("experiment_name")
                        result["product_name"] = "autotunex"
                        result["open"] = True
                        result["revision"] = result.get("id")
                        if result.get("artifact_uri") is not None:
                            model_id, revision = extract_artifact_identifier(
                                result.get("artifact_uri")
                            )
                            if model_id is not None:
                                result["model_id"] = model_id
                                result["dmf_url"] = build_dmf_url(
                                    model_id, revision or result["revision"]
                                )
                    return results
        except aiomysql.MySQLError as e:
            logger.error(f"Database error in get_gb_published_models: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    # ------------------------------------------------------------------
    # SYNC METHODS (for thread contexts: Ray callbacks, logging handler, LocalRunner)
    # Uses DBUtils.PooledDB with pymysql
    # ------------------------------------------------------------------

    def test_db_connection_and_structure_sync(self):
        connection = None
        try:
            connection = self._get_sync_connection()
            with connection.cursor() as cursor:
                for table, columns in REQUIRED_TABLES.items():
                    cursor.execute(f"SHOW TABLES LIKE '{table}';")
                    result = cursor.fetchone()
                    if not result:
                        raise HTTPException(
                            status_code=500, detail=f"Table '{table}' is missing."
                        )
            logger.info("Database connection and table verification successful (sync).")
        except pymysql.MySQLError as e:
            logger.error("Database connection or table check failed.")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")
        finally:
            if connection:
                connection.close()

    def create_logging_table_sync(self) -> bool:
        conn = self._get_sync_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS log_entries (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        job_id CHAR(36) NOT NULL,
                        level VARCHAR(50),
                        filename VARCHAR(255),
                        message TEXT,
                        timestamp DATETIME,
                        FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
                    );
                    """
                )
            conn.commit()
            return True
        except pymysql.MySQLError as e:
            logger.error(f"Database error in create_logging_table_sync: {e}")
            return False
        finally:
            conn.close()

    def insert_logs_sync(self, buffer: list) -> bool:
        conn = self._get_sync_connection()
        try:
            with conn.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO log_entries (
                        job_id, trial_id, level, filename, message, iteration, epoch, timestamp
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            entry["job_id"],
                            entry["trial_id"],
                            entry["level"],
                            entry["filename"],
                            entry["message"],
                            entry["iteration"],
                            entry["epoch"],
                            entry["timestamp"],
                        )
                        for entry in buffer
                    ],
                )
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error inserting logs (sync): {str(e)}")
            return False
        finally:
            conn.close()

    def insert_trial_sync(self, data: api.Trial) -> str:
        config_copy = data["config"].copy() if isinstance(data["config"], dict) else {}
        if (
            isinstance(config_copy.get("tune_config"), dict)
            and "search_alg" in config_copy["tune_config"]
        ):
            config_copy["tune_config"]["search_alg"] = config_copy["tune_config"][
                "search_alg"
            ].__class__.__name__
        conn = self._get_sync_connection()
        try:
            with conn.cursor() as cursor:
                sql = "INSERT INTO `trials` (`id`, `job_id`, `status`, `config`) VALUES (%s, %s, %s, %s)"
                cursor.execute(
                    sql,
                    (
                        data["id"],
                        data["job_id"],
                        data["status"].value,
                        json.dumps(config_copy),
                    ),
                )
            conn.commit()
            return data["id"]
        except pymysql.MySQLError as e:
            logger.error(f"Database error in insert_trial_sync: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")
        finally:
            conn.close()

    def update_trial_status_sync(self, trial_id: str, status: api.TrialStatus) -> str:
        conn = self._get_sync_connection()
        try:
            with conn.cursor() as cursor:
                sql = "UPDATE `trials` SET `status`=%s WHERE `id`=%s"
                cursor.execute(sql, (status.value, trial_id))
            conn.commit()
            return True
        except pymysql.MySQLError as e:
            logger.error(f"Database error in update_trial_status_sync: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")
        finally:
            conn.close()

    def insert_result_sync(self, metadata: api.Result) -> api.Result:
        result_id = uuid.uuid4()
        conn = self._get_sync_connection()
        try:
            with conn.cursor() as cursor:
                sql = "INSERT INTO `results` (`id`, `job_id`, `trial_id`, `metric`, `metrics`) VALUES (%s, %s, %s, %s, %s)"
                cursor.execute(
                    sql,
                    (
                        str(result_id),
                        metadata["job_id"],
                        metadata["trial_id"],
                        metadata["metric"],
                        json.dumps(metadata["metrics"]),
                    ),
                )
            conn.commit()
            return metadata
        except pymysql.MySQLError as e:
            raise HTTPException(status_code=500, detail=f"Internal Server error: \n{e}")
        finally:
            conn.close()

    def get_config_sync(self, config_id: str, user_id: str = None) -> api.Configuration:
        conn = self._get_sync_connection()
        try:
            with conn.cursor() as cursor:
                if user_id:
                    sql = "SELECT * FROM `configurations` WHERE `id` = %s AND (`user_id` = %s OR `user_id` = %s)"
                    cursor.execute(sql, (config_id, user_id, constants.SYSTEM_USER))
                else:
                    sql = "SELECT * FROM `configurations` WHERE `id` = %s"
                    cursor.execute(sql, (config_id,))
                result = cursor.fetchone()
                if result is not None:
                    result["config_data"] = json.loads(result["config_data"])
                    result["created_at"] = get_utc_timestamp(result.get("created_at"))
                    result["updated_at"] = get_utc_timestamp(result.get("updated_at"))
                    # Fetch associated jobs
                    sql_jobs = "SELECT * FROM `jobs` WHERE `config_id` = %s"
                    params = [result["id"]]
                    if user_id:
                        sql_jobs += " AND `user_id` = %s"
                        params.append(user_id)
                    cursor.execute(sql_jobs, tuple(params))
                    result["associated_jobs"] = cursor.fetchall()
                else:
                    raise HTTPException(status_code=404, detail="Invalid config_id")
            return result
        except HTTPException:
            raise
        except pymysql.MySQLError as e:
            logger.error(f"Database error in get_config_sync: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")
        finally:
            conn.close()

    def get_dataset_sync(
        self, dataset_id: str, user_id: Optional[str] = None
    ) -> api.DatasetInfo:
        conn = self._get_sync_connection()
        try:
            with conn.cursor() as cursor:
                is_id = True
                try:
                    UUID(dataset_id)
                except ValueError:
                    is_id = False
                if user_id is not None:
                    if is_id:
                        sql = "SELECT * FROM `datasets` WHERE `id`=%s AND (`user_id`=%s OR `user_id`=%s)"
                    else:
                        sql = "SELECT * FROM `datasets` WHERE `name`=%s AND (`user_id`=%s OR `user_id`=%s)"
                    cursor.execute(sql, (dataset_id, user_id, constants.SYSTEM_USER))
                else:
                    if is_id:
                        sql = "SELECT * FROM `datasets` WHERE `id`=%s"
                    else:
                        sql = "SELECT * FROM `datasets` WHERE `name`=%s"
                    cursor.execute(sql, (dataset_id,))
                result = cursor.fetchone()
                if result is not None:
                    result["created_at"] = get_utc_timestamp(result.get("created_at"))
                    result["updated_at"] = get_utc_timestamp(result.get("updated_at"))
                return result
        except pymysql.MySQLError as e:
            logger.error(f"Database error in get_dataset_sync: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")
        finally:
            conn.close()

    def update_job_status_sync(self, id: str, status: api.JobStatus) -> str:
        conn = self._get_sync_connection()
        try:
            with conn.cursor() as cursor:
                sql = "UPDATE `jobs` SET `status`=%s WHERE `id`=%s"
                cursor.execute(sql, (status.value, id))
            conn.commit()
            return True
        except pymysql.MySQLError as e:
            logger.error(f"Database error in update_job_status_sync: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")
        finally:
            conn.close()

    def get_task_by_job_id_sync(self, job_id: str, type: api.TaskType):
        conn = self._get_sync_connection()
        try:
            with conn.cursor() as cursor:
                sql = "SELECT * FROM `gb_tasks` WHERE `job_id`=%s AND `type`=%s ORDER BY `updated_at` DESC"
                cursor.execute(sql, (job_id, type))
                result = cursor.fetchone()
                try:
                    if result and result.get("build_status"):
                        result["build_status"] = json.loads(result["build_status"])
                except (json.JSONDecodeError, TypeError) as e:
                    logger.error(f"error occurred in get_task_by_job_id_sync: {e}")
                    result["build_status"] = None
            return result
        except pymysql.MySQLError as e:
            logger.error(f"Database error in get_task_by_job_id_sync: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")
        finally:
            conn.close()

    def update_task_sync(self, task: api.Task) -> bool:
        conn = self._get_sync_connection()
        try:
            with conn.cursor() as cursor:
                sql = "UPDATE `gb_tasks` SET `build_id`=%s, `status`=%s, `pr_url`=%s, `artifact_id`=%s, `artifact_uri`=%s, `build_status`=%s, `started_at`=%s, `updated_at`=%s, `rits_url`=%s WHERE `id`=%s"
                cursor.execute(
                    sql,
                    (
                        task.build_id,
                        task.status,
                        task.pr_url,
                        task.artifact_id,
                        task.artifact_uri,
                        json.dumps(task.build_status),
                        task.started_at,
                        task.updated_at,
                        task.rits_url,
                        task.id,
                    ),
                )
            conn.commit()
            return True
        except pymysql.MySQLError as e:
            logger.error(f"Database error in update_task_sync: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")
        finally:
            conn.close()

    def insert_task_sync(self, task: api.Task) -> str:
        task_id = task.id if task.id else str(uuid.uuid4())
        conn = self._get_sync_connection()
        try:
            with conn.cursor() as cursor:
                sql = "INSERT INTO `gb_tasks` (`id`, `job_id`, `type`, `pr_url`, `started_at`, `updated_at`) VALUES (%s, %s, %s, %s, %s, %s)"
                cursor.execute(
                    sql,
                    (
                        task_id,
                        task.job_id,
                        task.type,
                        task.pr_url,
                        task.started_at,
                        task.updated_at,
                    ),
                )
            conn.commit()
            return task_id
        except pymysql.MySQLError as e:
            logger.error(f"Database error in insert_task_sync: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")
        finally:
            conn.close()
