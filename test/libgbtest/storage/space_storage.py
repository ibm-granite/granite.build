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

    def test_hf_default_resource_group_id_round_trips(self):
        """hf_default_resource_group_id is stored in the json column and round-trips.

        It also defaults to None for rows created without it (old rows).
        """
        storage = self._get_tested_storage()
        with_rg = StoredSpace(
            name="rg_space",
            git_repo_uri="http://foo.bar/rg",
            lakehouse_namespace="lhnamespace_rg",
            hf_default_resource_group_id="rg-abc-123",
        )
        without_rg = StoredSpace(
            name="no_rg_space",
            git_repo_uri="http://foo.bar/no_rg",
            lakehouse_namespace="lhnamespace_no_rg",
        )
        storage.add([with_rg, without_rg])

        reloaded = storage.get_by_name("rg_space")
        assert reloaded is not None
        assert reloaded.hf_default_resource_group_id == "rg-abc-123"

        reloaded_none = storage.get_by_name("no_rg_space")
        assert reloaded_none is not None
        assert reloaded_none.hf_default_resource_group_id is None

        # Update the null-id row (mirrors the helper's write-back) and confirm
        # the change persists.
        reloaded_none.hf_default_resource_group_id = "written-back"
        storage.update(reloaded_none)
        assert (
            storage.get_by_name("no_rg_space").hf_default_resource_group_id
            == "written-back"
        )


class BaseLegacyStoredSpaceTest(AbstractExistingDataReadTest):

    def _get_tested_readonly_storage(
        self, storage: singleton_storage.SingletonAdminStorage
    ):
        return storage.space_storage
