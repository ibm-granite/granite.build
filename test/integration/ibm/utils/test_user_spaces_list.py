import pytest

pytestmark = pytest.mark.ibm

from libgbtest.storage.space_storage import SpaceStorageTestSupport
from libgbtest.utils import AbstractSingletonStorageUsingTest

from gbserver.spaces.space_access_manager import set_space_access_manager
from gbserver.spaces.storage_space_access_manager import StorageSpaceAccessManager
from gbserver.spaces.user_spaces_list import user_spaces_list
from gbserver.storage import singleton_storage
from gbserver.storage.stored_space import StoredSpace
from gbserver.storage.stored_space_user import StoredSpaceUser
from gbserver.types.constants import PUBLIC_SPACE_NAME

_FAKE_EMAIL = "test@example.com"

_PUBLIC_SPACE = StoredSpace(
    name=PUBLIC_SPACE_NAME,
    git_repo_uri="http://foo.bar/public",
    lakehouse_namespace="lhnamespace_public",
)


class TestUserSpacesList(AbstractSingletonStorageUsingTest):

    def _get_storage_to_clear(self):
        return [
            self.storage.space_storage,
            self.storage.space_user_storage,
        ]

    def test_user_spaces_list(self):
        # Use StorageSpaceAccessManager so the test exercises gb_space_users storage
        set_space_access_manager(StorageSpaceAccessManager())

        # Set up the rest api methods to use our test storage
        ssts = SpaceStorageTestSupport()
        storage = singleton_storage.get_admin_storage().space_storage
        space_user_storage = singleton_storage.get_admin_storage().space_user_storage

        item0 = ssts._get_test_item(0)
        item1 = ssts._get_test_item(1)
        item2 = ssts._get_test_item(2)
        item3 = ssts._get_test_item(3)
        storage.add([item0, item1])

        # Get all and expect none (no memberships yet, and no public space in storage)
        result = user_spaces_list(_FAKE_EMAIL)

        assert len(result) == 0

        storage.add([item2, item3])
        # Add memberships: testuser is admin of item2's space, member of item3's space
        space_user_storage.add(
            [
                StoredSpaceUser(
                    space_name=item2.name, username=_FAKE_EMAIL, role="admin"
                ),
                StoredSpaceUser(
                    space_name=item3.name, username=_FAKE_EMAIL, role="member"
                ),
            ]
        )

        # Get all and expect the 2 from the ones just added
        result = user_spaces_list(_FAKE_EMAIL)

        is_admin_space = 0
        is_not_admin_space = 0
        for space in result:
            if space["is_admin"]:
                is_admin_space += 1
            else:
                is_not_admin_space += 1
        assert len(result) == 2
        assert is_admin_space == 1
        assert is_not_admin_space == 1

    def test_public_space_included_when_no_memberships(self):
        """User with no memberships should still get the public space."""
        set_space_access_manager(StorageSpaceAccessManager())
        storage = singleton_storage.get_admin_storage().space_storage

        # Add the public space to storage
        storage.add([_PUBLIC_SPACE])

        result = user_spaces_list(_FAKE_EMAIL)

        assert len(result) == 1
        assert result[0]["name"] == PUBLIC_SPACE_NAME
        assert result[0]["is_admin"] is False

    def test_public_space_no_duplicate_when_admin(self):
        """User who is admin of public space should not get a duplicate non-admin entry."""
        set_space_access_manager(StorageSpaceAccessManager())
        storage = singleton_storage.get_admin_storage().space_storage
        space_user_storage = singleton_storage.get_admin_storage().space_user_storage

        storage.add([_PUBLIC_SPACE])
        space_user_storage.add(
            [
                StoredSpaceUser(
                    space_name=PUBLIC_SPACE_NAME,
                    username=_FAKE_EMAIL,
                    role="admin",
                ),
            ]
        )

        result = user_spaces_list(_FAKE_EMAIL)

        public_entries = [s for s in result if s["name"] == PUBLIC_SPACE_NAME]
        assert len(public_entries) == 1
        assert public_entries[0]["is_admin"] is True

    def test_public_space_no_duplicate_when_member(self):
        """User who is member of public space should not get a duplicate entry."""
        set_space_access_manager(StorageSpaceAccessManager())
        storage = singleton_storage.get_admin_storage().space_storage
        space_user_storage = singleton_storage.get_admin_storage().space_user_storage

        storage.add([_PUBLIC_SPACE])
        space_user_storage.add(
            [
                StoredSpaceUser(
                    space_name=PUBLIC_SPACE_NAME,
                    username=_FAKE_EMAIL,
                    role="member",
                ),
            ]
        )

        result = user_spaces_list(_FAKE_EMAIL)

        public_entries = [s for s in result if s["name"] == PUBLIC_SPACE_NAME]
        assert len(public_entries) == 1
        assert public_entries[0]["is_admin"] is False
