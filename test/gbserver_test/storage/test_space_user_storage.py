from gbserver_test.storage.storage import (
    AbstractStorageTest,
    AbstractStorageTestSupport,
)

from gbserver.storage import singleton_storage
from gbserver.storage.storage import BaseItemStorage
from gbserver.storage.stored_space_user import StoredSpaceUser


class SpaceUserStorageTestSupport(AbstractStorageTestSupport):
    def __init__(self):
        super().__init__(sort_column="username")

    def _get_test_item(self, index):
        return StoredSpaceUser(
            space_name=f"space{index}",
            username=f"user{index}",
            role="member",
        )


class BaseSpaceUserStorageTest(AbstractStorageTest):

    @classmethod
    def _get_test_config(cls) -> AbstractStorageTestSupport:
        return SpaceUserStorageTestSupport()

    def _get_tested_storage(self) -> BaseItemStorage:
        return self.storage.space_user_storage

    def test_uniqueness_enforcement(self):
        # The (space_name, username) pair must be unique
        self._duplication_test_helper(["space_name", "username"])

    def test_get_by_space(self):
        storage = self._get_tested_storage()
        item1 = StoredSpaceUser(space_name="myspace", username="alice", role="admin")
        item2 = StoredSpaceUser(space_name="myspace", username="bob", role="member")
        item3 = StoredSpaceUser(space_name="otherspace", username="alice", role="member")
        storage.add([item1, item2, item3])

        results = storage.get_by_space("myspace")
        assert len(results) == 2
        usernames = {r.username for r in results}
        assert usernames == {"alice", "bob"}

    def test_get_by_username(self):
        storage = self._get_tested_storage()
        item1 = StoredSpaceUser(space_name="space_a", username="carol", role="admin")
        item2 = StoredSpaceUser(space_name="space_b", username="carol", role="member")
        item3 = StoredSpaceUser(space_name="space_a", username="dave", role="member")
        storage.add([item1, item2, item3])

        results = storage.get_by_username("carol")
        assert len(results) == 2
        spaces = {r.space_name for r in results}
        assert spaces == {"space_a", "space_b"}

    def test_get_by_space_and_username(self):
        storage = self._get_tested_storage()
        item = StoredSpaceUser(space_name="proj", username="eve", role="admin")
        storage.add(item)

        found = storage.get_by_space_and_username("proj", "eve")
        assert found is not None
        assert found.role == "admin"

        not_found = storage.get_by_space_and_username("proj", "nobody")
        assert not_found is None
