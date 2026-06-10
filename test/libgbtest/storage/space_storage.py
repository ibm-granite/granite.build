from libgbtest.storage.storage import (
    AbstractExistingDataReadTest,
    AbstractStorageTest,
    AbstractStorageTestSupport,
)

from gbserver.storage import singleton_storage
from gbserver.storage.storage import BaseItemStorage
from gbserver.storage.stored_space import StoredSpace


class SpaceStorageTestSupport(AbstractStorageTestSupport):
    def __init__(self):
        super().__init__(sort_column="name")

    def _get_test_item(self, index):
        obj = StoredSpace(
            name=f"foo{index}",
            git_repo_uri=f"http://foo.bar/{index}",
            lakehouse_namespace=f"lhnamespace{index}",
        )
        return obj


class BaseSpaceStorageTest(AbstractStorageTest):

    @classmethod
    def _get_test_config(cls) -> AbstractStorageTestSupport:
        return SpaceStorageTestSupport()

    def _get_tested_storage(self) -> BaseItemStorage:
        return self.storage.space_storage

    def test_get_by_where_dict_multimatch(self):
        """Override the super since `name` is unique and the parent multimatch
        test reuses the same `name` for two rows, which would collide."""
        pass

    def test_uniqueness_enforcement(self):
        # Only `name` is unique.  `git_repo_uri` is intentionally NOT unique
        # so multiple alias rows (e.g. legacy `standalone` + current `public`)
        # can share a single URI.
        self._duplication_test_helper(["name"])

    def test_get_by_name(self):
        storage = self._get_tested_storage()
        item1 = self._get_test_item(1)
        item2 = self._get_test_item(2)
        storage.add([item1, item2])
        item = storage.get_by_name(item1.name)
        assert item is not None, f"Did not find item by name={item1.name}"


class BaseLegacyStoredSpaceTest(AbstractExistingDataReadTest):

    def _get_tested_readonly_storage(
        self, storage: singleton_storage.SingletonAdminStorage
    ):
        return storage.space_storage
