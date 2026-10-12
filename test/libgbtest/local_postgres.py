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
from pathlib import Path
from typing import Optional

import pytest

# gbcommon only: it doesn't import gbserver, so GBSERVER_SQL_* aren't read yet.
from gbcommon.types.gbenvconfig import is_standalone

_DEFAULT_POSTGRES_DIR = (
    Path(__file__).resolve().parents[2] / ".cache" / "gbtest-postgres"
)


def load_local_postgres_env() -> Optional[Path]:
    """In STANDALONE runs, load the local test PostgreSQL's env file, if it exists.

    The file's values replace any already in the environment, so a partial set of
    hosted-database settings can't mix with the local ones. Set
    GBTEST_LOCAL_POSTGRES=false to keep the hosted database's settings in a
    STANDALONE run (e.g. to run test/integration/ibm/storage/sql).

    Returns:
        The file loaded, or None if not STANDALONE, opted out, or there is no env file.
    """
    if not is_standalone():
        return None
    if os.getenv("GBTEST_LOCAL_POSTGRES", "").lower() == "false":
        return None
    pg_dir = os.getenv("GBTEST_POSTGRES_DIR")
    env_file = (Path(pg_dir) if pg_dir else _DEFAULT_POSTGRES_DIR) / "postgres.env"
    if not env_file.is_file():
        return None
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

requires_local_postgres = pytest.mark.skipif(
    not (GBTEST_LOCAL_POSTGRES and HAS_PSYCOPG),
    reason=(
        "local test PostgreSQL not configured: run `make postgres-setup` and run "
        "with GB_ENVIRONMENT=STANDALONE (e.g. `make extended-tests`)"
        if not GBTEST_LOCAL_POSTGRES
        else "psycopg (PostgreSQL driver) not installed: install the `thirdparty` "
        "extra, e.g. `make g4os-skypilot-venv` or `make venv`"
    ),
)
"""Skip for SQL storage tests that run against the local test PostgreSQL."""

requires_hosted_postgres = pytest.mark.skipif(
    GBTEST_LOCAL_POSTGRES,
    reason="the local test PostgreSQL's settings are loaded (STANDALONE run with "
    ".cache/gbtest-postgres/postgres.env present), so GBSERVER_SQL_* point at "
    "localhost, not the hosted database; set GBTEST_LOCAL_POSTGRES=false to use "
    "the hosted database's settings",
)
"""Skip for SQL storage tests meant for the hosted PostgreSQL, while the local one's
settings are in effect (GBSERVER_SQL_* are read once per process, so they can't be
switched per test)."""
