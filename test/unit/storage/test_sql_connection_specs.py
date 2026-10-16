"""The SQL store's Postgres connection names the psycopg 3 driver explicitly."""

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy.engine import make_url

from gbserver.storage.sql import sql_storage
from gbserver.storage.sql.sql_storage import BaseSQLItemStorage
from gbserver.types import constants


def _specs(monkeypatch, scheme, cert_file=None):
    monkeypatch.setattr(sql_storage, "GBSERVER_SQL_SCHEME", scheme)
    monkeypatch.setattr(sql_storage, "get_ssl_cert_file", lambda _logger: cert_file)
    return BaseSQLItemStorage._get_connection_specs(SimpleNamespace(logger=MagicMock()))


def test_default_scheme_names_psycopg3(monkeypatch):
    monkeypatch.delenv(constants.ENV_VAR_GBSERVER_SQL_SCHEME, raising=False)
    try:
        importlib.reload(constants)
        assert constants.GBSERVER_SQL_SCHEME == "postgresql+psycopg"
        _, db_url, _, _ = _specs(monkeypatch, constants.GBSERVER_SQL_SCHEME)
        assert make_url(db_url).get_dialect().driver == "psycopg"
    finally:
        monkeypatch.undo()
        importlib.reload(constants)


@pytest.mark.parametrize("scheme", ["postgresql", "mysql+pymysql"])
def test_scheme_is_used_as_given(monkeypatch, scheme):
    _, db_url, obfuscated_db_url, _ = _specs(monkeypatch, scheme)
    assert db_url.startswith(f"{scheme}://")
    assert obfuscated_db_url.startswith(f"{scheme}://")


def test_cert_file_sets_verify_full(monkeypatch):
    _, db_url, _, connect_args = _specs(monkeypatch, "postgresql+psycopg", "ca.cert")
    assert db_url.endswith("?sslmode=verify-full")
    assert connect_args == {"sslrootcert": "ca.cert"}
