import os

import pytest
from gbserver_test.test_utils import AbstractSingletonStorageUsingTest

from gbserver.spaces.storage_space_access_manager import StorageSpaceAccessManager
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_space import StoredSpace
from gbserver.storage.stored_space_user import StoredSpaceUser
from gbserver.types.constants import PUBLIC_SPACE_NAME

_FAKE_EMAIL = "alice@example.com"

_SPACE_A = StoredSpace(
    name="space_a",
    git_repo_uri="https://github.com/org/space_a",
    lakehouse_namespace="ns_a",
)
_SPACE_B = StoredSpace(
    name="space_b",
    git_repo_uri="https://github.com/org/space_b",
    lakehouse_namespace="ns_b",
)
_PUBLIC_SPACE = StoredSpace(
    name=PUBLIC_SPACE_NAME,
    git_repo_uri="https://github.com/org/public",
    lakehouse_namespace="ns_public",
)


@pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
class TestStorageSpaceAccessManager(AbstractSingletonStorageUsingTest):

    @classmethod
    def _get_storage_factory(cls):
        from gbserver.storage.sql.storage_factory import SQLStorageFactory

        return SQLStorageFactory()

    def _get_storage_to_clear(self):
        return [
            self.storage.build_storage,
            self.storage.space_storage,
            self.storage.space_user_storage,
        ]

    def _setup_spaces_and_manager(self):
        """Add test spaces and return a StorageSpaceAccessManager."""
        self.storage.space_storage.add([_SPACE_A, _SPACE_B])
        return StorageSpaceAccessManager()

    def test_get_user_spaces_with_access_no_memberships(self):
        manager = self._setup_spaces_and_manager()
        result = manager.get_user_spaces_with_access(_FAKE_EMAIL)
        assert result == []

    def test_get_user_spaces_with_access_member_role(self):
        manager = self._setup_spaces_and_manager()
        self.storage.space_user_storage.add(
            StoredSpaceUser(space_name="space_a", username=_FAKE_EMAIL, role="member")
        )

        result = manager.get_user_spaces_with_access(_FAKE_EMAIL)
        assert len(result) == 1
        assert result[0].space.name == "space_a"
        assert result[0].is_admin is False

    def test_get_user_spaces_with_access_admin_role(self):
        manager = self._setup_spaces_and_manager()
        self.storage.space_user_storage.add(
            [
                StoredSpaceUser(
                    space_name="space_a", username=_FAKE_EMAIL, role="admin"
                ),
                StoredSpaceUser(
                    space_name="space_b", username=_FAKE_EMAIL, role="member"
                ),
            ]
        )

        result = manager.get_user_spaces_with_access(_FAKE_EMAIL)
        assert len(result) == 2
        by_name = {r.space.name: r for r in result}
        assert by_name["space_a"].is_admin is True
        assert by_name["space_b"].is_admin is False

    def test_get_user_spaces_with_access_unknown_email(self):
        manager = self._setup_spaces_and_manager()
        result = manager.get_user_spaces_with_access("unknown@example.com")
        assert result == []

    def test_is_space_admin_true(self):
        manager = self._setup_spaces_and_manager()
        self.storage.space_user_storage.add(
            StoredSpaceUser(space_name="space_a", username=_FAKE_EMAIL, role="admin")
        )
        assert manager.is_space_admin(_FAKE_EMAIL, "space_a") is True

    def test_is_space_admin_false_when_member(self):
        manager = self._setup_spaces_and_manager()
        self.storage.space_user_storage.add(
            StoredSpaceUser(space_name="space_a", username=_FAKE_EMAIL, role="member")
        )
        assert manager.is_space_admin(_FAKE_EMAIL, "space_a") is False

    def test_is_space_admin_false_when_no_membership(self):
        manager = self._setup_spaces_and_manager()
        assert manager.is_space_admin(_FAKE_EMAIL, "space_a") is False

    def test_has_space_access_true_for_member(self):
        manager = self._setup_spaces_and_manager()
        self.storage.space_user_storage.add(
            StoredSpaceUser(space_name="space_a", username=_FAKE_EMAIL, role="member")
        )
        assert manager.has_space_access(_FAKE_EMAIL, "space_a") is True

    def test_has_space_access_true_for_admin(self):
        manager = self._setup_spaces_and_manager()
        self.storage.space_user_storage.add(
            StoredSpaceUser(space_name="space_a", username=_FAKE_EMAIL, role="admin")
        )
        assert manager.has_space_access(_FAKE_EMAIL, "space_a") is True

    def test_has_space_access_false_when_no_membership(self):
        manager = self._setup_spaces_and_manager()
        assert manager.has_space_access(_FAKE_EMAIL, "space_a") is False

    def test_has_build_access_true(self):
        manager = self._setup_spaces_and_manager()
        build = StoredBuild(
            name="mybuild",
            space_name="space_a",
            source_uri="https://github.com/org/repo",
            username="alice",
        )
        self.storage.build_storage.add(build)
        self.storage.space_user_storage.add(
            StoredSpaceUser(space_name="space_a", username=_FAKE_EMAIL, role="member")
        )
        result = manager.has_build_access(_FAKE_EMAIL, build.uuid)
        assert result is True

    def test_has_build_access_false_no_membership(self):
        manager = self._setup_spaces_and_manager()
        build = StoredBuild(
            name="mybuild",
            space_name="space_a",
            source_uri="https://github.com/org/repo",
            username="alice",
        )
        self.storage.build_storage.add(build)
        result = manager.has_build_access(_FAKE_EMAIL, build.uuid)
        assert result is False

    def test_has_build_access_404_for_unknown_build(self):
        manager = self._setup_spaces_and_manager()
        from fastapi.responses import JSONResponse

        result = manager.has_build_access(_FAKE_EMAIL, "nonexistent-uuid")
        assert isinstance(result, JSONResponse)
        assert result.status_code == 404

    def test_has_space_access_true_for_public_space_without_membership(self):
        manager = self._setup_spaces_and_manager()
        assert manager.has_space_access(_FAKE_EMAIL, PUBLIC_SPACE_NAME) is True

    def test_has_build_access_true_for_public_space_without_membership(self):
        manager = self._setup_spaces_and_manager()
        self.storage.space_storage.add(_PUBLIC_SPACE)
        build = StoredBuild(
            name="public-build",
            space_name=PUBLIC_SPACE_NAME,
            source_uri="https://github.com/org/repo",
            username="bob",
        )
        self.storage.build_storage.add(build)
        result = manager.has_build_access(_FAKE_EMAIL, build.uuid)
        assert result is True

    def test_get_user_spaces_includes_public_space(self):
        manager = self._setup_spaces_and_manager()
        self.storage.space_storage.add(_PUBLIC_SPACE)
        result = manager.get_user_spaces_with_access(_FAKE_EMAIL)
        names = [s.space.name for s in result]
        assert PUBLIC_SPACE_NAME in names
        public_entry = next(s for s in result if s.space.name == PUBLIC_SPACE_NAME)
        assert public_entry.is_admin is False
