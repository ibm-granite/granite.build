from gbserver_test.storage.storage import (
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
        """Override the super since this class does not support multiple item match search because both fields (name and uri) are forced to be unique."""
        pass

    def test_uniqueness_enforcement(self):
        # Make sure we can't use duplicate space names or git URIs
        self._duplication_test_helper(["git_repo_uri"])
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
