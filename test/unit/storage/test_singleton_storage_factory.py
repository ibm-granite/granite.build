"""Swapping the storage factory drops admin storage cached from the old one."""

import pytest

from gbserver.storage import singleton_storage
from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory


class _TaggedFactory(SqliteStorageFactory):
    """Records the space storage it builds, to tell which factory built it."""

    def __init__(self):
        super().__init__()
        self.built = []

    def create_space_storage(self, table_name=None):
        storage = super().create_space_storage(table_name=table_name)
        self.built.append(storage)
        return storage


@pytest.fixture
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("GB_HOME_DIR", str(tmp_path))
    saved = singleton_storage.get_storage_factory()
    yield
    singleton_storage.set_storage_factory(saved)


def test_set_storage_factory_drops_cached_storage(isolated_storage):
    old, new = _TaggedFactory(), _TaggedFactory()
    singleton_storage.set_storage_factory(old)
    assert singleton_storage.get_admin_storage().space_storage in old.built

    singleton_storage.set_storage_factory(new)

    assert singleton_storage.get_admin_storage().space_storage in new.built
