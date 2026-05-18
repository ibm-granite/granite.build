from unittest.mock import patch

import pytest

pytestmark = pytest.mark.ibm

from lib.api.utils import AbstractAPITest
from lib.storage.space_storage import SpaceStorageTestSupport

import gbserver.api.spaces as spaces_api

# REST API
from gbserver.api.auth import get_gh_user
from gbserver.spaces.space_access_manager import set_space_access_manager
from gbserver.spaces.storage_space_access_manager import StorageSpaceAccessManager
from gbserver.storage import singleton_storage
from gbserver.storage.stored_space import StoredSpace
from gbserver.storage.stored_space_user import StoredSpaceUser
from gbserver.types.constants import GBSERVER_GITHUB_TOKEN

base_url = "api/v1/spaces"


class TestSpacesAPI(AbstractAPITest):

    def test_get_spaces(self):
        # Set up the rest api methods to use our test storage
        ssts = SpaceStorageTestSupport()
        storage = singleton_storage.get_admin_storage().space_storage
        client = self.get_test_client()

        # Get all and expect only the pre-registered public test space
        response = client.get(f"{base_url}")
        assert response.status_code == 200
        resp_json = response.json()
        resp: spaces_api.ListSpacesResponse = (
            spaces_api.ListSpacesResponse.model_validate(resp_json)
        )
        spaces = resp.spaces
        assert len(spaces) == 1

        item0 = ssts._get_test_item(0)
        item1 = ssts._get_test_item(1)
        storage.add([item0, item1])

        # Get all  and expect the 2 just added plus the preregistered public test space
        response = client.get(f"{base_url}")
        assert response.status_code == 200
        resp_json = response.json()
        resp: spaces_api.ListSpacesResponse = (
            spaces_api.ListSpacesResponse.model_validate(resp_json)
        )
        spaces = resp.spaces
        assert len(spaces) == 3

        # Now search for a specific space
        response = client.get(f"{base_url}?name={item0.name}")
        assert response.status_code == 200
        resp_json = response.json()
        resp: spaces_api.ListSpacesResponse = (
            spaces_api.ListSpacesResponse.model_validate(resp_json)
        )
        spaces = resp.spaces
        assert len(spaces) == 1


_TEST_SPACE = StoredSpace(
    name="testspace",
    git_repo_uri="https://github.com/org/testspace",
    lakehouse_namespace="ns_test",
)


class TestSpaceMembersAPI(AbstractAPITest):
    """Tests for the space member management endpoints (list, add, delete)."""

    def _get_storage_to_clear(self):
        return [
            self.storage.space_storage,
            self.storage.space_user_storage,
        ]

    def _get_test_user_email(self) -> str:
        """Get the email of the user associated with the default test token."""
        user, _ = get_gh_user(GBSERVER_GITHUB_TOKEN)
        assert (
            user is not None
        ), "Could not resolve test user from GBSERVER_GITHUB_TOKEN"
        return user.email

    def _setup_storage_mode(self):
        """Switch to StorageSpaceAccessManager and set up test data.

        Returns the test user email used for space user records.
        """
        set_space_access_manager(StorageSpaceAccessManager())
        self.storage.space_storage.add([_TEST_SPACE])
        email = self._get_test_user_email()
        # Make the test user a super-admin (admin of "public" space)
        self.storage.space_user_storage.add(
            StoredSpaceUser(space_name="public", username=email, role="admin")
        )
        return email

    def test_list_members_empty(self):
        with patch.dict(
            "gbserver.types.constants.GB_ENVIRONMENT_CONFIG.feature_flags",
            {"lakehouse_space_membership": False},
        ):
            self._setup_storage_mode()
            client = self.get_test_client()
            response = client.get(f"{base_url}/{_TEST_SPACE.name}/members")
            assert response.status_code == 200
            data = response.json()
            assert data["members"] == []

    def test_list_members_with_data(self):
        with patch.dict(
            "gbserver.types.constants.GB_ENVIRONMENT_CONFIG.feature_flags",
            {"lakehouse_space_membership": False},
        ):
            self._setup_storage_mode()
            self.storage.space_user_storage.add(
                [
                    StoredSpaceUser(
                        space_name=_TEST_SPACE.name,
                        username="alice@example.com",
                        role="admin",
                    ),
                    StoredSpaceUser(
                        space_name=_TEST_SPACE.name,
                        username="bob@example.com",
                        role="member",
                    ),
                ]
            )
            client = self.get_test_client()
            response = client.get(f"{base_url}/{_TEST_SPACE.name}/members")
            assert response.status_code == 200
            data = response.json()
            assert len(data["members"]) == 2
            usernames = {m["username"] for m in data["members"]}
            assert usernames == {"alice@example.com", "bob@example.com"}

    def test_add_member(self):
        with patch.dict(
            "gbserver.types.constants.GB_ENVIRONMENT_CONFIG.feature_flags",
            {"lakehouse_space_membership": False},
        ):
            self._setup_storage_mode()
            client = self.get_test_client()
            response = client.post(
                f"{base_url}/{_TEST_SPACE.name}/members",
                json={"username": "newuser@example.com", "role": "member"},
            )
            assert response.status_code == 201
            data = response.json()
            assert data["member"]["username"] == "newuser@example.com"
            assert data["member"]["role"] == "member"
            assert data["member"]["space_name"] == _TEST_SPACE.name

    def test_add_member_admin_role(self):
        with patch.dict(
            "gbserver.types.constants.GB_ENVIRONMENT_CONFIG.feature_flags",
            {"lakehouse_space_membership": False},
        ):
            self._setup_storage_mode()
            client = self.get_test_client()
            response = client.post(
                f"{base_url}/{_TEST_SPACE.name}/members",
                json={"username": "admin@example.com", "role": "admin"},
            )
            assert response.status_code == 201
            data = response.json()
            assert data["member"]["role"] == "admin"

    def test_add_member_conflict(self):
        with patch.dict(
            "gbserver.types.constants.GB_ENVIRONMENT_CONFIG.feature_flags",
            {"lakehouse_space_membership": False},
        ):
            self._setup_storage_mode()
            self.storage.space_user_storage.add(
                StoredSpaceUser(
                    space_name=_TEST_SPACE.name,
                    username="existing@example.com",
                    role="member",
                )
            )
            client = self.get_test_client()
            response = client.post(
                f"{base_url}/{_TEST_SPACE.name}/members",
                json={"username": "existing@example.com", "role": "admin"},
            )
            assert response.status_code == 409

    def test_delete_member(self):
        with patch.dict(
            "gbserver.types.constants.GB_ENVIRONMENT_CONFIG.feature_flags",
            {"lakehouse_space_membership": False},
        ):
            self._setup_storage_mode()
            self.storage.space_user_storage.add(
                StoredSpaceUser(
                    space_name=_TEST_SPACE.name,
                    username="toremove@example.com",
                    role="member",
                )
            )
            client = self.get_test_client()
            response = client.delete(
                f"{base_url}/{_TEST_SPACE.name}/members/toremove@example.com"
            )
            assert response.status_code == 200
            assert response.json() == {"result": "success"}
            # Verify the member is gone
            response = client.get(f"{base_url}/{_TEST_SPACE.name}/members")
            assert response.status_code == 200
            assert len(response.json()["members"]) == 0

    def test_delete_member_not_found(self):
        with patch.dict(
            "gbserver.types.constants.GB_ENVIRONMENT_CONFIG.feature_flags",
            {"lakehouse_space_membership": False},
        ):
            self._setup_storage_mode()
            client = self.get_test_client()
            response = client.delete(
                f"{base_url}/{_TEST_SPACE.name}/members/nonexistent@example.com"
            )
            assert response.status_code == 404

    def test_update_member_role(self):
        with patch.dict(
            "gbserver.types.constants.GB_ENVIRONMENT_CONFIG.feature_flags",
            {"lakehouse_space_membership": False},
        ):
            self._setup_storage_mode()
            self.storage.space_user_storage.add(
                StoredSpaceUser(
                    space_name=_TEST_SPACE.name,
                    username="user@example.com",
                    role="member",
                )
            )
            client = self.get_test_client()
            response = client.patch(
                f"{base_url}/{_TEST_SPACE.name}/members/user@example.com",
                json={"role": "admin"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["member"]["username"] == "user@example.com"
            assert data["member"]["role"] == "admin"

    def test_update_member_not_found(self):
        with patch.dict(
            "gbserver.types.constants.GB_ENVIRONMENT_CONFIG.feature_flags",
            {"lakehouse_space_membership": False},
        ):
            self._setup_storage_mode()
            client = self.get_test_client()
            response = client.patch(
                f"{base_url}/{_TEST_SPACE.name}/members/nonexistent@example.com",
                json={"role": "admin"},
            )
            assert response.status_code == 404

    def test_space_not_found(self):
        with patch.dict(
            "gbserver.types.constants.GB_ENVIRONMENT_CONFIG.feature_flags",
            {"lakehouse_space_membership": False},
        ):
            set_space_access_manager(StorageSpaceAccessManager())
            email = self._get_test_user_email()
            self.storage.space_user_storage.add(
                StoredSpaceUser(space_name="public", username=email, role="admin")
            )
            client = self.get_test_client()
            response = client.get(f"{base_url}/nonexistent/members")
            assert response.status_code == 404

    def test_unauthorized_user(self):
        """Endpoints return 401 when user is not a space admin or super-admin."""
        with patch.dict(
            "gbserver.types.constants.GB_ENVIRONMENT_CONFIG.feature_flags",
            {"lakehouse_space_membership": False},
        ):
            set_space_access_manager(StorageSpaceAccessManager())
            self.storage.space_storage.add([_TEST_SPACE])
            # Do NOT add the test user as admin of testspace or public
            client = self.get_test_client()
            response = client.get(f"{base_url}/{_TEST_SPACE.name}/members")
            assert response.status_code == 401
