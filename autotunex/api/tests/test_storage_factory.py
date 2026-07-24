# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch
from services.storage import get_storage_backend
from services.storage.local_backend import LocalStorageBackend


def test_factory_returns_local_when_gb_disabled():
    assert isinstance(get_storage_backend(gb_enabled=False), LocalStorageBackend)


def test_factory_defaults_to_is_gb_enabled():
    with patch("services.storage.is_gb_enabled", return_value=False):
        assert isinstance(get_storage_backend(), LocalStorageBackend)


def test_factory_returns_gb_when_gb_enabled(monkeypatch):
    # GBStorageBackend's real import chain (gb_service -> db_service -> pymysql/
    # gbcli) is not importable in this environment, and the factory imports it
    # lazily. Inject a fake gb_backend module so we can verify the gb_enabled
    # branch returns whatever GBStorageBackend() yields, without the heavy deps.
    import sys
    import types

    fake_mod = types.ModuleType("services.storage.gb_backend")

    class _FakeGB:
        pass

    fake_mod.GBStorageBackend = _FakeGB
    # Remove any already-imported real module first so the lazy import re-fires.
    monkeypatch.delitem(sys.modules, "services.storage.gb_backend", raising=False)
    monkeypatch.setitem(sys.modules, "services.storage.gb_backend", fake_mod)

    backend = get_storage_backend(gb_enabled=True)
    assert isinstance(backend, _FakeGB)
