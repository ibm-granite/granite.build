"""Test that gbserver core modules can be imported without lakehouse/dmf installed."""

import importlib
import sys
import unittest.mock as mock

import pytest

# All lakehouse submodules that might be imported
_LH_MODULES = {
    "lakehouse": None,
    "lakehouse.api": None,
    "lakehouse.assets": None,
    "lakehouse.assets.dataset": None,
    "lakehouse.assets.fileset": None,
    "lakehouse.assets.model": None,
    "lakehouse.assets.table": None,
    "lakehouse.core": None,
}


class TestOptionalLakehouse:
    """Verify gbserver starts without the lakehouse library."""

    def test_storage_singleton_imports_without_lakehouse(self):
        """The storage singleton should import even if lakehouse is absent."""
        with mock.patch.dict(sys.modules, _LH_MODULES):
            # Force reimport
            mod = importlib.import_module("gbserver.storage.singleton_storage")
            importlib.reload(mod)
            assert hasattr(mod, "get_storage_factory")

    def test_sqlite_storage_factory_without_lakehouse(self):
        """SQLite storage factory should work without lakehouse."""
        with mock.patch.dict(sys.modules, _LH_MODULES):
            from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory

            factory = SqliteStorageFactory()
            assert factory is not None

    def test_lhstore_raises_clear_error_without_lakehouse(self):
        """Attempting to use lhstore without lakehouse should give a clear error."""
        with mock.patch.dict(sys.modules, _LH_MODULES):
            mod = importlib.import_module("gbserver.asset.lhstore")
            importlib.reload(mod)
            # Import should succeed (lazy), but instantiation should raise
            with pytest.raises(ImportError, match="lakehouse"):
                mod.Lhstore("lh://test/namespace/tables/test_table")

    def test_lineage_module_imports_without_lakehouse(self):
        """The lineage API module should import even if lakehouse is absent."""
        with mock.patch.dict(sys.modules, _LH_MODULES):
            mod = importlib.import_module("gbserver.api.lineage")
            importlib.reload(mod)
            assert hasattr(mod, "lineage_api")

    def test_lakehouse_utils_raises_clear_error_without_lakehouse(self):
        """lakehouse_utils should import but raise on use without lakehouse."""
        with mock.patch.dict(sys.modules, _LH_MODULES):
            mod = importlib.import_module("gbserver.utils.lakehouse_utils")
            importlib.reload(mod)
            with pytest.raises(ImportError, match="lakehouse"):
                mod.create_lakehouse_iceberg()
