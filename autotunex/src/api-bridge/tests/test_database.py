# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the api-bridge database configuration helpers.

These cover the pure functions only — URL driver coercion and TLS-context
construction — which need no live MySQL. The query methods require a real
database and are exercised by integration/smoke testing, not here.
"""

import ssl
from unittest.mock import patch

import pytest
from sqlalchemy.engine import make_url

from api_bridge.database import _build_ssl_context, _to_sync_url

# --- _to_sync_url ----------------------------------------------------------


def test_to_sync_url_coerces_asyncmy_to_pymysql():
    result = make_url(_to_sync_url("mysql+asyncmy://u:p@host:3306/db"))

    assert result.drivername == "mysql+pymysql"


def test_to_sync_url_coerces_aiomysql_to_pymysql():
    result = make_url(_to_sync_url("mysql+aiomysql://u:p@host:3306/db"))

    assert result.drivername == "mysql+pymysql"


def test_to_sync_url_coerces_bare_mysql_to_pymysql():
    result = make_url(_to_sync_url("mysql://u:p@host:3306/db"))

    assert result.drivername == "mysql+pymysql"


def test_to_sync_url_leaves_pymysql_unchanged():
    result = make_url(_to_sync_url("mysql+pymysql://u:p@host:3306/db"))

    assert result.drivername == "mysql+pymysql"


def test_to_sync_url_preserves_connection_components():
    # Password with URL-special characters, percent-encoded in the source URL.
    raw = "mysql+asyncmy://user:p%40ss%3Aw%2Frd@db.example.com:3307/autotune?charset=utf8mb4"

    result = make_url(_to_sync_url(raw))

    assert result.drivername == "mysql+pymysql"
    assert result.username == "user"
    assert result.password == "p@ss:w/rd"  # decoded, preserved through the round-trip
    assert result.host == "db.example.com"
    assert result.port == 3307
    assert result.database == "autotune"
    assert result.query.get("charset") == "utf8mb4"


# --- _build_ssl_context ----------------------------------------------------


def test_build_ssl_context_disable_returns_none():
    assert _build_ssl_context(None, "disable") is None


def test_build_ssl_context_derives_disable_when_no_ca():
    assert _build_ssl_context(None, None) is None


def test_build_ssl_context_require_encrypts_without_verifying():
    ctx = _build_ssl_context(None, "require")

    assert ctx is not None
    assert ctx.check_hostname is False
    assert ctx.verify_mode is ssl.CERT_NONE


def test_build_ssl_context_verify_without_ca_raises():
    with pytest.raises(ValueError):
        _build_ssl_context(None, "verify")


def test_build_ssl_context_verify_with_ca_clears_x509_strict():
    # Avoid needing a real CA file: hand back a fresh context and assert the
    # VERIFY_X509_STRICT flag (which the IBM Cloud CA needs cleared) is cleared.
    real_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    real_ctx.verify_flags |= ssl.VERIFY_X509_STRICT

    with patch("ssl.create_default_context", return_value=real_ctx) as create_ctx:
        ctx = _build_ssl_context("/path/to/ca.pem", "verify")

    create_ctx.assert_called_once_with(cafile="/path/to/ca.pem")
    assert ctx.verify_mode is ssl.CERT_REQUIRED
    assert not (ctx.verify_flags & ssl.VERIFY_X509_STRICT)


def test_build_ssl_context_derives_verify_when_ca_present():
    real_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

    with patch("ssl.create_default_context", return_value=real_ctx) as create_ctx:
        ctx = _build_ssl_context("/path/to/ca.pem", None)

    create_ctx.assert_called_once_with(cafile="/path/to/ca.pem")
    assert ctx is real_ctx


def test_build_ssl_context_unknown_mode_raises():
    with pytest.raises(ValueError):
        _build_ssl_context(None, "bogus")


# --- _to_sync_url multi-dialect coercion -----------------------------------


def test_to_sync_url_coerces_asyncpg_to_psycopg():
    from api_bridge.database import _to_sync_url

    assert make_url(_to_sync_url("postgresql+asyncpg://u:p@h:5432/db")).drivername == (
        "postgresql+psycopg"
    )


def test_to_sync_url_coerces_bare_postgresql_to_psycopg():
    from api_bridge.database import _to_sync_url

    assert make_url(_to_sync_url("postgresql://u:p@h:5432/db")).drivername == "postgresql+psycopg"


def test_to_sync_url_coerces_aiosqlite_to_pysqlite():
    from api_bridge.database import _to_sync_url

    assert make_url(_to_sync_url("sqlite+aiosqlite:///./x.db")).drivername == "sqlite"


def test_to_sync_url_leaves_bare_sqlite_unchanged():
    from api_bridge.database import _to_sync_url

    assert make_url(_to_sync_url("sqlite://")).drivername == "sqlite"


# --- _build_pool_kwargs gating ---------------------------------------------


def test_build_pool_kwargs_omits_queue_knobs_for_sqlite():
    from api_bridge.database import _build_pool_kwargs

    kwargs = _build_pool_kwargs("sqlite://")
    assert "pool_size" not in kwargs
    assert "max_overflow" not in kwargs
    assert kwargs["pool_pre_ping"] is True


def test_build_pool_kwargs_includes_queue_knobs_for_server_dbs():
    from api_bridge.database import _build_pool_kwargs

    kwargs = _build_pool_kwargs("mysql+pymysql://u:p@h/db")
    assert kwargs["pool_size"] == 10
    assert kwargs["max_overflow"] == 5


def test_database_accepts_injected_engine():
    from sqlalchemy import create_engine

    from api_bridge.database import Database
    from api_bridge.tables import metadata

    eng = create_engine("sqlite://")
    metadata.create_all(eng)
    db = Database(engine=eng)
    assert db._engine is eng
