import os

from libgbtest.storage.storage import (
    AbstractExistingDataReadTest,
    AbstractStorageTest,
    AbstractStorageTestSupport,
)

from gbserver.api.auth import get_gh_user
from gbserver.storage import singleton_storage
from gbserver.storage.storage import BaseItemStorage
from gbserver.storage.stored_build import StoredBuild
from gbserver.types.constants import ENV_VAR_DEFAULT_GITHUB_TOKEN


class BuildStorageTestSupport(AbstractStorageTestSupport):

    def __init__(self):
        super().__init__(sort_column="name")

    def _get_build_user(self) -> str:
        """Return the username to use in test builds. Subclasses can override to avoid GitHub API calls."""
        token = os.getenv(ENV_VAR_DEFAULT_GITHUB_TOKEN, None)
        if token is None:
            token = os.getenv("GITHUB_TOKEN", None)
            # IF we dont' get this, we can make authenticated build cancellations.
            assert (
                token is not None
            ), "Need a github token to create the correct username for the build"
        user, error = get_gh_user(token)
        assert user is not None, f"Could not get user login for token {error}"
        return user.login

    def _get_test_item(self, index):
        username = self._get_build_user()
        obj = StoredBuild(
            name=f"bname{index}",
            space_name=f"myspace{index}",
            source_uri=f"https://git.ibm.com/{index}",
            username=username,
            build_archive="",
            description=f"description{index}",
            targets=[f"foo{index}"],
            tags=[f"tag{index}", f"tag{index+1000}"],
        )
        return obj


class BaseBuildStorageTest(AbstractStorageTest):

    @classmethod
    def _get_test_config(cls) -> AbstractStorageTestSupport:
        return BuildStorageTestSupport()

    def _get_tested_storage(self) -> BaseItemStorage:
        return self.storage.build_storage


class BaseLegacyStoredBuildTest(AbstractExistingDataReadTest):

    def _get_tested_readonly_storage(
        self, storage: singleton_storage.SingletonAdminStorage
    ):
        return storage.build_storage
