#!/usr/bin/env python3

# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The local test PostgreSQL: its connection env and the skip that gates its tests.

``make postgres-setup`` (scripts/postgres/setup-postgres.sh) starts a local server
and writes its GBSERVER_SQL_* settings, plus GBTEST_LOCAL_POSTGRES=true, to
``$GBTEST_POSTGRES_DIR/postgres.env`` (default <repo>/.cache/gbtest-postgres/).
Importing this module loads that file in GB_ENVIRONMENT=STANDALONE runs.

gbserver.types.constants reads GBSERVER_SQL_* once, at import, so this module must
be imported before gbserver is: test/conftest.py imports it first. It deliberately
imports nothing from gbserver (gbcommon, which doesn't import gbserver, is fine).

Usage:
    from libgbtest.local_postgres import requires_local_postgres

    @requires_local_postgres
    class TestSQLSomething: ...
"""

import importlib.util
import os
import socket
import warnings
from pathlib import Path
from typing import Optional

import pytest
from libgbtest.mode import is_live  # imports only os

# gbcommon only: it doesn't import gbserver, so GBSERVER_SQL_* aren't read yet.
from gbcommon.types.gbenvconfig import is_standalone

_DEFAULT_POSTGRES_DIR = (
    Path(__file__).resolve().parents[2] / ".cache" / "gbtest-postgres"
)


def load_local_postgres_env() -> Optional[Path]:
    """In STANDALONE runs, load the local test PostgreSQL's env file, if it exists.

    The file's values replace any already in the environment, so a partial set of
    hosted-database settings can't mix with the local ones. That points every
    SQL-storage user in the run (subprocesses included) at the local server, so
    set GBTEST_LOCAL_POSTGRES=false to keep the environment's GBSERVER_SQL_*
    instead. (That doesn't enable the hosted-PostgreSQL tests, which skip in any
    STANDALONE run; see requires_hosted_postgres.)

    Returns:
        The file loaded, or None if not STANDALONE, opted out, or there is no env file.

    Warns:
        UserWarning: if the file, which holds the password, is readable or
            writable by group or others (setup writes it 0600).
    """
    if not is_standalone():
        return None
    if os.getenv("GBTEST_LOCAL_POSTGRES", "").lower() == "false":
        return None
    pg_dir = os.getenv("GBTEST_POSTGRES_DIR")
    env_file = (Path(pg_dir) if pg_dir else _DEFAULT_POSTGRES_DIR) / "postgres.env"
    if not env_file.is_file():
        return None
    if env_file.stat().st_mode & 0o077:
        warnings.warn(
            f"{env_file} holds the local test PostgreSQL password but is accessible "
            f"to group/others (mode {env_file.stat().st_mode & 0o777:o}); "
            f"run `chmod 600 {env_file}`",
            stacklevel=2,
        )
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()
    return env_file


LOADED_ENV_FILE: Optional[Path] = load_local_postgres_env()
"""The env file loaded on import, or None."""

GBTEST_LOCAL_POSTGRES = os.getenv("GBTEST_LOCAL_POSTGRES", "").lower() == "true"
"""True when the local test PostgreSQL's connection env is in effect."""

HAS_PSYCOPG = importlib.util.find_spec("psycopg") is not None
"""True when the PostgreSQL driver (psycopg 3, the ``thirdparty`` extra) is installed.

Without it the SQL storage retries the failing driver import instead of erroring,
so a run would look hung rather than fail.
"""


def _local_postgres_skip_reason() -> Optional[str]:
    """Why the local-PostgreSQL tests can't run, or None if they can.

    Checks, in order: the local server's env is loaded, the driver is installed,
    and the server accepts TCP connections. The last catches a stale env file
    whose server is gone (stopped container, reboot) so the tests skip instead
    of failing to connect.

    Returns:
        A skip reason, or None when the tests can run.
    """
    if not GBTEST_LOCAL_POSTGRES:
        return (
            "local test PostgreSQL not configured: run `make postgres-setup` and run "
            "with GB_ENVIRONMENT=STANDALONE (e.g. `make extended-tests`)"
        )
    if not HAS_PSYCOPG:
        return (
            "psycopg (PostgreSQL driver) not installed: install the `thirdparty` "
            "extra, e.g. `make g4os-skypilot-venv` or `make venv`"
        )
    host = os.getenv("GBSERVER_SQL_HOST", "")
    port = os.getenv("GBSERVER_SQL_PORT", "")
    try:
        with socket.create_connection((host, int(port)), timeout=1):
            return None
    except (OSError, ValueError):
        return (
            f"local test PostgreSQL not reachable at {host}:{port} (stale "
            f"{LOADED_ENV_FILE}?): run `make postgres-setup`, or "
            "`make postgres-teardown` to remove the env file"
        )


LOCAL_POSTGRES_SKIP_REASON: Optional[str] = _local_postgres_skip_reason()
"""Why the local-PostgreSQL tests skip, or None if they run (computed on import)."""

requires_local_postgres = pytest.mark.skipif(
    LOCAL_POSTGRES_SKIP_REASON is not None,
    reason=LOCAL_POSTGRES_SKIP_REASON or "",
)
"""Skip for SQL storage tests that run against the local test PostgreSQL."""


def _hosted_postgres_skip_reason() -> Optional[str]:
    """Why the hosted-PostgreSQL tests can't run, or None if they can.

    A STANDALONE run can't target the hosted database: GBSERVER_SQL_SCHEMA defaults
    to the STANDALONE config's "standalone" schema, which the hosted database
    doesn't have, and when ``make postgres-setup`` has run, GBSERVER_SQL_* point at
    localhost. With storage mocked (mock mode), conftest swaps the SQL factory for
    SQLite, so the tests would pass without touching PostgreSQL; SQLite has its own
    tests in test/unit/storage/sqlite.

    Returns:
        A skip reason, or None when the tests can run.
    """
    if is_standalone():
        return (
            'GB_ENVIRONMENT=STANDALONE: its SQL schema ("standalone") doesn\'t exist '
            "in the hosted database, and the local test PostgreSQL's settings may be "
            "loaded; run these with GB_ENVIRONMENT=STAGING or DEV"
        )
    if not is_live("storage"):
        return (
            "storage is mocked (GBTEST_MODE=mock), which would run these hosted-"
            "PostgreSQL tests on SQLite; run with GBTEST_MODE=live or "
            "GBTEST_LIVE_STORAGE=true"
        )
    return None


HOSTED_POSTGRES_SKIP_REASON: Optional[str] = _hosted_postgres_skip_reason()
"""Why the hosted-PostgreSQL tests skip, or None if they run (computed on import)."""

requires_hosted_postgres = pytest.mark.skipif(
    HOSTED_POSTGRES_SKIP_REASON is not None,
    reason=HOSTED_POSTGRES_SKIP_REASON or "",
)
"""Skip for SQL storage tests meant for the hosted PostgreSQL, unless they would
really run against it (see _hosted_postgres_skip_reason)."""

LOCAL_POSTGRES_TEST_MARKS = [
    # Needs the psycopg driver from the `thirdparty` extra.
    pytest.mark.thirdparty,
    # Real-infra test: extended suite only (what libgbtest.constants'
    # extended_testing_only applies; not imported here, as it imports gbserver).
    pytest.mark.extended,
    requires_local_postgres,
    # Keep the real SQL factory even in mock mode, where conftest would otherwise
    # swap in SQLite.
    pytest.mark.live("storage"),
]
"""``pytestmark`` for SQL storage test modules that run against the local test
PostgreSQL (test/unit/storage/postgres)."""

HOSTED_POSTGRES_TEST_MARKS = [
    pytest.mark.ibm,
    requires_hosted_postgres,
    pytest.mark.skipif(
        os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
        reason="Don't want to run this in CICD.",
    ),
    # Keep the real SQL factory (GBTEST_LIVE_STORAGE=true in mock mode), where
    # conftest would otherwise swap in SQLite.
    pytest.mark.live("storage"),
]
"""``pytestmark`` for SQL storage test modules that run against the hosted PostgreSQL
(test/integration/ibm/storage/sql)."""
