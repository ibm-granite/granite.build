# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import uuid
from typing import Any, Dict, Optional
from uuid import UUID

import pymysql as db
from dbutils.pooled_db import PooledDB
from dotenv import load_dotenv
from fastapi import HTTPException

import model as bridge_models
from utils import SYSTEM_USER, get_utc_timestamp, utc_now_string

logger = logging.getLogger(__name__)

load_dotenv()

DB_HOST = os.environ["DB_HOST"] if "DB_HOST" in os.environ else ""
DB_PORT = os.environ["DB_PORT"] if "DB_PORT" in os.environ else ""
DB_USER = os.environ["DB_USER"] if "DB_USER" in os.environ else ""
DB_PASSWORD = os.environ["DB_PASSWORD"] if "DB_PASSWORD" in os.environ else ""
# Accept either DB_SCHEMA (bridge convention) or DB_NAME (api-server convention)
# so the two services can share a single deployment secret.
DB_SCHEMA = os.getenv("DB_SCHEMA") or os.getenv("DB_NAME") or ""


def _ssl_verify_identity() -> bool:
    """
    Whether to verify the MySQL server's TLS identity.

    Defaults to True, preserving behavior for existing (e.g. IBM Cloud) MySQL
    deployments that present a verifiable certificate. Set
    DB_SSL_VERIFY_IDENTITY=false when connecting to a MySQL that only offers a
    self-signed certificate (such as the stock `mysql` container in the local
    docker-compose stack): TLS is still negotiated — required by
    caching_sha2_password — but the server identity is not verified. Mirrors the
    api server's flag in api/services/db_service.py.
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

        self.host = DB_HOST
        self.port = int(DB_PORT)
        self.user = DB_USER
        self.password = DB_PASSWORD
        self.schema = DB_SCHEMA

        ssl_key = os.getenv("DB_KEY")
        connect_kwargs = {"cursorclass": db.cursors.DictCursor}
        if _ssl_verify_identity():
            # Verify the server's TLS identity (existing IBM Cloud MySQL behavior).
            connect_kwargs["ssl_verify_identity"] = True
            if ssl_key:
                connect_kwargs["ssl_key"] = ssl_key
        else:
            # Identity verification disabled (e.g. a self-signed MySQL in the local
            # docker-compose stack). Pass an explicit non-verifying SSL context so
            # TLS is still negotiated — passing only ssl_verify_identity=False would
            # leave PyMySQL on a plaintext connection, which caching_sha2_password
            # rejects.
            import ssl as _ssl

            ssl_ctx = _ssl.create_default_context()
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = _ssl.CERT_NONE
            if ssl_key and os.path.isfile(ssl_key):
                ssl_ctx.load_cert_chain(certfile=ssl_key)
            connect_kwargs["ssl"] = ssl_ctx

        try:
            self._pool = PooledDB(
                creator=db,
                mincached=2,
                maxcached=10,
                maxconnections=10,
                blocking=True,
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.schema,
                **connect_kwargs,
            )
        except db.MySQLError as e:
            # PooledDB(mincached=2) opens real connections here, so credential,
            # grant, TLS, or host/schema problems surface at startup. Surface a
            # readable message instead of the misleading "IndexError: pop from
            # empty list" that dbutils raises from its empty idle-cache path.
            logger.error(
                "Failed to connect to MySQL at %s:%s as user '%s' (schema '%s'): %s",
                self.host,
                self.port,
                self.user,
                self.schema,
                e,
            )
            raise RuntimeError(
                f"Database connection failed for user '{self.user}'@'{self.host}:{self.port}' "
                f"on schema '{self.schema}'. Check DB_USER/DB_PASSWORD/DB_SCHEMA (or DB_NAME), "
                f"that the user is granted access from this host, and that TLS/DB_KEY settings "
                f"match the server. Underlying error: {e}"
            ) from e

    def _connect(self):
        return self._pool.connection()

    def test_db_connection_and_structure(self):
        connection = None
        # Attempt to connect to the database
        try:
            connection = self._connect()
            logger.info("Database connection successful.")

            # Verify tables and columns
            with connection.cursor() as cursor:
                for table, columns in REQUIRED_TABLES.items():
                    # Check if the table exists
                    cursor.execute(f"SHOW TABLES LIKE '{table}';")
                    result = cursor.fetchone()
                    if not result:
                        raise HTTPException(
                            status_code=500, detail=f"Table '{table}' is missing."
                        )

                    # # Check if each expected column exists in the table
                    # cursor.execute(f"SHOW COLUMNS FROM `{table}`;")
                    # existing_columns = {row[0] for row in cursor.fetchall()}
                    # missing_columns = set(columns) - existing_columns
                    # if missing_columns:
                    #     raise HTTPException(
                    #         status_code=500,
                    #         detail=f"Table '{table}' is missing columns: {', '.join(missing_columns)}",
                    #     )

            logger.info("Database table verified successfully.")

        except db.MySQLError as e:
            logger.error("Database connection or table check failed.")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

        finally:
            if connection:
                connection.close()

    def insert_logs(self, buffer: list) -> bool:
        """
        Insert a batch of log entries into the database.

        Args:
            buffer (list): A list of log entry dictionaries.

        Returns:
            bool: True if the logs were inserted successfully, False otherwise.
        """
        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.executemany(
                        """
                        INSERT INTO log_entries (job_id, trial_id, level, filename, message, iteration, epoch,
                                                 timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
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
                connection.commit()
            return True
        except Exception as e:
            logger.error(f"Error inserting logs: {str(e)}")
            return False

    def update_job_status(self, id: str, status: bridge_models.JobStatus) -> str:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                sql = "UPDATE `jobs` SET `status`=%s WHERE `id`=%s"
                cursor.execute(
                    sql,
                    (status.value, id),
                )
            connection.commit()
        return True

    def insert_trial(self, data: bridge_models.Trial) -> str:
        config_copy = data["config"].copy() if isinstance(data["config"], dict) else {}
        # Remove or convert non-serializable objects
        if (
            isinstance(config_copy.get("tune_config"), dict)
            and "search_alg" in config_copy["tune_config"]
        ):
            # # Option 1: Remove the search algorithm
            # del config_copy["tune_config"]["search_alg"]

            # Option 2: Or store just the name/type of the search algorithm
            config_copy["tune_config"]["search_alg"] = config_copy["tune_config"][
                "search_alg"
            ].__class__.__name__
        with self._connect() as connection:
            with connection.cursor() as cursor:
                sql = "INSERT into `trials` (`id`, `job_id`, `status`, `config`) VALUES (%s, %s, %s, %s)"
                cursor.execute(
                    sql,
                    (
                        data["id"],
                        data["job_id"],
                        data["status"].value,
                        json.dumps(config_copy),
                    ),
                )
            connection.commit()
        return data["id"]

    def update_trial_status(
        self, trial_id: str, status: bridge_models.TrialStatus
    ) -> bool:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                sql = "UPDATE `trials` SET `status`=%s WHERE `id`=%s"
                cursor.execute(
                    sql,
                    (status.value, trial_id),
                )
            connection.commit()
        return True

    def insert_result(self, metadata: bridge_models.Result) -> bridge_models.Result:
        try:
            result_id = uuid.uuid4()
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    sql = "INSERT into `results` (`id`,`job_id`, `trial_id`, `metric`, `metrics`) VALUES (%s, %s, %s, %s, %s)"
                    cursor.execute(
                        sql,
                        (
                            result_id,
                            metadata["job_id"],
                            metadata["trial_id"],
                            metadata["metric"],
                            json.dumps(metadata["metrics"]),
                        ),
                    )
                connection.commit()
            return metadata
        except db.MySQLError as e:
            raise HTTPException(status_code=500, detail=f"Internal Server error: \n{e}")

    def update_all_trial_status(
        self, job_id: str, status: bridge_models.TrialStatus
    ) -> bool:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                sql = "UPDATE `trials` SET `status`=%s WHERE `job_id`=%s"
                cursor.execute(
                    sql,
                    (status.value, job_id),
                )
            connection.commit()
        return True

    # ------------------------------------------------------------------
    # User CRUD (from api/services/db_service.py)
    # ------------------------------------------------------------------

    def insert_user(self, email: str) -> str:
        user_id = uuid.uuid4()
        with self._connect() as connection:
            with connection.cursor() as cursor:
                sql = "INSERT into `users` (`id`, `email`) VALUES (%s, %s)"
                cursor.execute(sql, (user_id, email))
            connection.commit()
        return user_id

    def update_user(self, user) -> str:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                sql = "UPDATE `users` SET `email`=%s, `role`=%s,`updated_at`=%s WHERE `id`=%s"
                cursor.execute(sql, (user.email, user.role, user.updated_at, user.id))
            connection.commit()
        return user.id

    def get_user(self, email: str):
        with self._connect() as connection:
            with connection.cursor() as cursor:
                sql = "SELECT * FROM `users` a where LOWER(a.email) = LOWER(%s)"
                cursor.execute(sql, email)
                result = cursor.fetchone()
                if result is not None:
                    result["created_at"] = get_utc_timestamp(result.get("created_at"))
                    result["updated_at"] = get_utc_timestamp(result.get("updated_at"))
                return result

    # ------------------------------------------------------------------
    # Configuration CRUD (from api/services/db_service.py)
    # ------------------------------------------------------------------

    def insert_configuration(self, config) -> str:
        config_id = uuid.uuid4()
        with self._connect() as connection:
            with connection.cursor() as cursor:
                sql = "INSERT into `configurations` (`id`, `user_id`, `name`, `tuner_type`, `artifact_id`, `artifact_url`, `config_data`) VALUES (%s, %s, %s, %s, %s, %s, %s)"
                cursor.execute(
                    sql,
                    (
                        config_id,
                        config.user_id,
                        config.name,
                        config.tuner_type,
                        config.artifact_id,
                        config.artifact_url,
                        json.dumps(config.config_data),
                    ),
                )
            connection.commit()
        return config_id

    def update_configuration(self, config) -> str:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                sql = "UPDATE `configurations` SET `user_id`=%s, `name`=%s, `tuner_type`=%s, `artifact_id`=%s, `artifact_url`=%s, `config_data`=%s WHERE `id`=%s"
                cursor.execute(
                    sql,
                    (
                        config.user_id,
                        config.name,
                        config.tuner_type,
                        config.artifact_id,
                        config.artifact_url,
                        json.dumps(config.config_data),
                        config.id,
                    ),
                )
            connection.commit()
        return config.id

    def get_configs(self, user_id: str, ids: list = None) -> list:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                if ids is None or len(ids) == 0:
                    sql = "SELECT * FROM `configurations` WHERE (`user_id`=%s OR `user_id`=%s)"
                    cursor.execute(sql, (user_id, SYSTEM_USER))
                elif len(ids) == 1:
                    sql = "SELECT * FROM `configurations` a where a.id = %s"
                    cursor.execute(sql, (ids[0]))
                else:
                    in_fmt_str = ",".join(["%s"] * len(ids))
                    sql = (
                        "SELECT * FROM `configurations` a where a.id in (%s)"
                        % in_fmt_str
                    )
                    cursor.execute(sql, tuple(ids))
                results = cursor.fetchall()
                for result in results:
                    result["config_data"] = json.loads(result["config_data"])
                    sql = "SELECT * from `jobs` a where a.config_id = %s and a.user_id = %s"
                    cursor.execute(sql, (result["id"], user_id))
                    jobs = cursor.fetchall()
                    result["created_at"] = get_utc_timestamp(result.get("created_at"))
                    result["updated_at"] = get_utc_timestamp(result.get("updated_at"))
                    result["associated_jobs"] = jobs
                return results

    def get_config(self, config_id: str, user_id: str = None):
        with self._connect() as connection:
            with connection.cursor() as cursor:
                if user_id:
                    sql = "SELECT * FROM `configurations` a WHERE a.id = %s AND (a.user_id = %s OR a.user_id = %s)"
                    cursor.execute(sql, (config_id, user_id, SYSTEM_USER))
                else:
                    sql = "SELECT * FROM `configurations` a WHERE a.id = %s"
                    cursor.execute(sql, config_id)
                result = cursor.fetchone()
                if result is not None:
                    result["config_data"] = json.loads(result["config_data"])
                    result["created_at"] = get_utc_timestamp(result.get("created_at"))
                    result["updated_at"] = get_utc_timestamp(result.get("updated_at"))
                else:
                    raise HTTPException(status_code=404, detail="Invalid config_id")
        return result

    def get_config_by_name(self, config_name: str):
        with self._connect() as connection:
            with connection.cursor() as cursor:
                sql = "SELECT * FROM `configurations` a where a.name = %s"
                cursor.execute(sql, config_name)
                result = cursor.fetchone()
                if result is not None:
                    result["config_data"] = json.loads(result["config_data"])
                    result["created_at"] = get_utc_timestamp(result.get("created_at"))
                    result["updated_at"] = get_utc_timestamp(result.get("updated_at"))
        return result

    def get_config_by_name_and_user(self, config_name: str, user_id: str):
        with self._connect() as connection:
            with connection.cursor() as cursor:
                sql = "SELECT * FROM `configurations` WHERE `name` = %s AND `user_id` = %s"
                logger.debug(
                    f"Executing SQL: {sql} with config_name={config_name} and user_id={user_id}"
                )
                cursor.execute(sql, (config_name, user_id))
                result = cursor.fetchone()
                if result is not None:
                    result["config_data"] = json.loads(result["config_data"])
                    result["created_at"] = get_utc_timestamp(result.get("created_at"))
                    result["updated_at"] = get_utc_timestamp(result.get("updated_at"))
        return result

    # ------------------------------------------------------------------
    # Job CRUD (from api/services/db_service.py)
    # ------------------------------------------------------------------

    def insert_job(self, config) -> str:
        try:
            job_id = config.id if config.id else uuid.uuid4()
            precision = "bf16"
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    sql = "INSERT into `jobs` (`id`, `user_id`, `status`, `seed`,  `config_id`, `dataset_id`, `model`,`experiment_name`, `tuning_type`, `precision`, `ray_address`, `cleanup`, `autotune`) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                    cursor.execute(
                        sql,
                        (
                            job_id,
                            config.user_id,
                            config.status.value,
                            config.seed,
                            config.config_id,
                            config.dataset_id,
                            config.model,
                            config.experiment_name,
                            config.tuning_type,
                            precision,
                            config.ray_address,
                            config.cleanup,
                            config.autotune,
                        ),
                    )
                connection.commit()
            return job_id
        except db.MySQLError as e:
            logger.error(e)
            if "Cannot add or update a child row" in str(e):
                raise HTTPException(status_code=404, detail="Config_id doesn't exist")
            else:
                raise HTTPException(status_code=500, detail="Internal Server error")

    def get_job_by_id(self, job_id: str):
        """Return a single job row by id, or None if not found."""
        with self._connect() as connection:
            with connection.cursor() as cursor:
                sql = "SELECT * FROM `jobs` WHERE `id` = %s"
                cursor.execute(sql, (job_id,))
                return cursor.fetchone()

    def insert_gb_task(self, job_id, build_id, task_type="TUNING") -> str:
        task_id = uuid.uuid4()
        now = utc_now_string()
        with self._connect() as connection:
            with connection.cursor() as cursor:
                sql = "INSERT into `gb_tasks` (`id`, `job_id`, `build_id`, `type`, `started_at`, `updated_at`) VALUES (%s, %s, %s, %s, %s, %s)"
                cursor.execute(
                    sql,
                    (task_id, str(job_id), build_id, task_type, now, now),
                )
            connection.commit()
        return task_id

    # ------------------------------------------------------------------
    # Dataset CRUD (from api/services/db_service.py)
    # ------------------------------------------------------------------

    def get_dataset_by_name_and_user(self, dataset_name: str, user_id: str):
        with self._connect() as connection:
            with connection.cursor() as cursor:
                sql = "SELECT * FROM `datasets` WHERE `name` = %s AND `user_id` = %s"
                cursor.execute(sql, (dataset_name, user_id))
                result = cursor.fetchone()
                if result is not None:
                    result["created_at"] = get_utc_timestamp(result.get("created_at"))
                    result["updated_at"] = get_utc_timestamp(result.get("updated_at"))
        return result

    def insert_dataset(self, dataset):
        try:
            dataset_id = uuid.uuid4()
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    sql = "INSERT into `datasets` (`id`, `user_id`, `name`, `description`) VALUES (%s, %s, %s, %s)"
                    cursor.execute(
                        sql,
                        (
                            dataset_id,
                            dataset.user_id,
                            dataset.name,
                            dataset.description,
                        ),
                    )
                connection.commit()
                dataset.id = dataset_id
            return dataset
        except db.MySQLError as e:
            if f"Duplicate entry '{dataset.name}' for key 'datasets.name'" in str(e):
                raise HTTPException(
                    status_code=400, detail="Dataset name must be unique"
                )
            else:
                raise HTTPException(
                    status_code=500, detail=f"Internal Server error: \n{e}"
                )

    def update_dataset(self, dataset):
        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    sql = "UPDATE `datasets` SET `name`=%s, `description`=%s WHERE `id`=%s"
                    cursor.execute(sql, (dataset.name, dataset.description, dataset.id))
                connection.commit()
            return dataset
        except db.MySQLError as e:
            if f"Duplicate entry '{dataset.name}' for key 'datasets.name'" in str(e):
                raise HTTPException(
                    status_code=400, detail="Dataset name must be unique"
                )
            else:
                raise HTTPException(
                    status_code=500, detail=f"Internal Server error: \n{e}"
                )

    def update_dataset_metadata(self, id: str, user_id: str, metadata: Dict[str, Any]):
        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    sql = "UPDATE `datasets` SET `train_records`=%s, `train_file_size`=%s, `validation_records`=%s,`validation_file_size`=%s, `artifact_id`=%s,`artifact_url`=%s  WHERE `id`=%s"
                    cursor.execute(
                        sql,
                        (
                            metadata["train_records"],
                            metadata["train_file_size"],
                            metadata["validation_records"],
                            metadata["validation_file_size"],
                            metadata["artifact_id"],
                            metadata["artifact_url"],
                            id,
                        ),
                    )
                connection.commit()
                result = self.get_dataset(dataset_id=id, user_id=user_id)
            return result
        except db.MySQLError as e:
            raise HTTPException(status_code=500, detail=f"Internal Server error: \n{e}")

    def check_dataset_exists(self, id: str) -> bool:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                sql = "SELECT EXISTS(SELECT * FROM `datasets` WHERE `id` = %s) AS dataset_exists;"
                cursor.execute(sql, id)
                result = cursor.fetchone()
            connection.commit()
        return result["dataset_exists"]

    def get_dataset(self, dataset_id: str, user_id: Optional[str] = None):
        with self._connect() as connection:
            with connection.cursor() as cursor:
                is_id = True
                try:
                    UUID(dataset_id)
                except ValueError:
                    is_id = False
                if user_id is not None:
                    if is_id:
                        sql = "SELECT * from datasets where `id`=%s AND (`user_id`=%s OR `user_id`=%s)"
                    else:
                        sql = "SELECT * from datasets where `name`=%s AND (`user_id`=%s OR `user_id`=%s)"
                    cursor.execute(sql, (dataset_id, user_id, SYSTEM_USER))
                    result = cursor.fetchone()
                else:
                    if is_id:
                        sql = "SELECT * from datasets where `id`=%s"
                    else:
                        sql = "SELECT * from datasets where `name`=%s"
                    cursor.execute(sql, dataset_id)
                    result = cursor.fetchone()

                if result is not None:
                    result["created_at"] = get_utc_timestamp(result.get("created_at"))
                    result["updated_at"] = get_utc_timestamp(result.get("updated_at"))
                return result

    def get_datasets(self, user_id: str, ids: list = None) -> list:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                if ids is None or len(ids) == 0:
                    sql = "SELECT * FROM `datasets` where (`user_id` = %s OR `user_id` = %s)"
                    cursor.execute(sql, (user_id, SYSTEM_USER))
                elif len(ids) == 1:
                    sql = "SELECT * FROM `datasets` a where a.id = %s"
                    cursor.execute(sql, (ids[0]))
                else:
                    in_fmt_str = ",".join(["%s"] * len(ids))
                    sql = "SELECT * FROM `datasets` a where a.id in (%s)" % in_fmt_str
                    cursor.execute(sql, tuple(ids))
                results = cursor.fetchall()
                for result in results:
                    sql = "SELECT * from `jobs` a where a.dataset_id = %s and a.user_id = %s"
                    cursor.execute(sql, (result["id"], user_id))
                    jobs = cursor.fetchall()
                    result["associated_jobs"] = jobs
                    result["created_at"] = get_utc_timestamp(result.get("created_at"))
                    result["updated_at"] = get_utc_timestamp(result.get("updated_at"))
                return results
