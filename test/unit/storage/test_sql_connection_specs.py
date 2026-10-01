"""BaseSQLItemStorage._get_connection_specs() pins the Postgres driver to psycopg 3."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy.engine import make_url

from gbserver.storage.sql import sql_storage
from gbserver.storage.sql.sql_storage import BaseSQLItemStorage


def _specs(monkeypatch, scheme, cert_file=None):
    monkeypatch.setattr(sql_storage, "GBSERVER_SQL_SCHEME", scheme)
    monkeypatch.setattr(sql_storage, "get_ssl_cert_file", lambda _logger: cert_file)
    return BaseSQLItemStorage._get_connection_specs(SimpleNamespace(logger=MagicMock()))


def test_bare_postgresql_pins_psycopg3(monkeypatch):
    _, db_url, obfuscated_db_url, _ = _specs(monkeypatch, "postgresql")
    assert make_url(db_url).get_dialect().driver == "psycopg"
    assert obfuscated_db_url.startswith("postgresql+psycopg://")


@pytest.mark.parametrize("scheme", ["postgresql+psycopg2", "mysql+pymysql"])
def test_explicit_scheme_is_kept(monkeypatch, scheme):
    _, db_url, _, _ = _specs(monkeypatch, scheme)
    assert db_url.startswith(f"{scheme}://")


def test_cert_file_sets_verify_full(monkeypatch):
    _, db_url, _, connect_args = _specs(monkeypatch, "postgresql", "ca.cert")
    assert db_url.endswith("?sslmode=verify-full")
    assert connect_args == {"sslrootcert": "ca.cert"}
