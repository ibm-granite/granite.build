from gbserver_test.storage.storage import (
    AbstractExistingDataReadTest,
    AbstractStorageTest,
    AbstractStorageTestSupport,
)

from gbserver.storage import singleton_storage
from gbserver.storage.storage import BaseItemStorage
from gbserver.storage.stored_target_run import StoredTargetRun


class TargetStorageTestSupport(AbstractStorageTestSupport):

    def __init__(self):
        super().__init__(sort_column="name")

    def _get_test_item(self, index):
        obj = StoredTargetRun(
            name=f"mytarget{index}",
            build_id=f"buildid{index}",
            environment_uri=f"https://env{index}",
            started_at="2020-01-02T00:00:00.000Z",
            finished_at="2021-01-02T00:00:00.000Z",
        )
        return obj


class BaseTargetStorageTest(AbstractStorageTest):

    @classmethod
    def _get_test_config(cls) -> AbstractStorageTestSupport:
        return TargetStorageTestSupport()

    def _get_tested_storage(self) -> BaseItemStorage:
        return self.storage.target_storage


class BaseLegacyStoredTargetTest(AbstractExistingDataReadTest):

    def _get_tested_readonly_storage(
        self, storage: singleton_storage.SingletonAdminStorage
    ):
        return storage.target_storage
