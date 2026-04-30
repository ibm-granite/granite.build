import os
import threading
import time
from datetime import timedelta
from pathlib import Path

import pytest
from gbserver_test.test_utils import AbstractSingletonStorageUsingPreloadedSpaceTest

from gbserver.github.githubmanager import GitHubManager
from gbserver.storage.stored_build import StoredBuild
from gbserver.types.constants import GBSERVER_GITHUB_TOKEN
from gbserver.types.status import Status
from gbserver.utils.logger import get_logger
from gbserver.utils.utils import get_time

logger = get_logger(__name__)


@pytest.mark.skip(reason="pr-watcher being deprecated")
class TestGithubManager(AbstractSingletonStorageUsingPreloadedSpaceTest):

    def setup_method(self, method):
        super().setup_method(method)  # call the super class to give us the tables in self.storage

    def test_builds(self):
        # src_file_dir = os.path.abspath(os.path.dirname(__file__))
        # test_data_dir = src_file_dir.replace("test", "test-data", 1)

        token = GBSERVER_GITHUB_TOKEN
        # domain = DEFAULT_GH_DOMAIN
        # owner = "granite-dot-build"
        # repo = "gb-test"                    # TODO we need an empty repo to start with
        # We need to disable input validation since we're picking a random PR from the past and may not need it.
        # test_data_path = Path(test_data_dir).resolve()
        # config_path = test_data_path / "test_1_config" / "pr-watcher-config.yaml"
        # assert config_path.is_file(), f"the path {config_path} is not a file"
        # # logger.info(f"Testing with\n .   domain:{domain}\n .   owner:{owner}\n .   repo:{repo}")
        # logger.info(f"Testing with\n .   config at path:{config_path}")

        # TODO: Publish the build to github

        # TODO: Merge the PR so it is picked up by the manager below

        expected_start_builds = self.__run_githubmanager(
            token=token, expected_start_builds=0, wait_for_new_builds=4
        )

        # Start a new GHM that will start with some builds already moved to the StoredBuilds table.
        self.__run_githubmanager(
            token=token,
            expected_start_builds=expected_start_builds,
            wait_for_new_builds=4,
        )

    def __run_githubmanager(self, token: str, expected_start_builds: int, wait_for_new_builds: int):
        __tracebackhide__ = True  # Hide the token on stack traces.
        assert expected_start_builds >= 0, "Mis-used test"
        assert wait_for_new_builds > 0, "Mis-used test"

        stored_builds = self.storage.build_storage.get_by_uuid(None)
        assert isinstance(stored_builds, list)
        assert (
            len(stored_builds) == expected_start_builds
        ), f"That were expected to be {expected_start_builds} builds in storage."

        # Watch for the PR and copy to StoredBuildStorage
        ghm = GitHubManager(token=token)  # , config_path=config_path)
        ghm.created_after = get_time() - timedelta(days=5)
        ghm.config.validate_inputs_are_registered = (
            False  # Since we're picking an arbitrary PR from the past which may be invalid.
        )
        # Run this in a separate thread.  Also, uses the same singleton test storage
        ghm_thread = threading.Thread(target=ghm.start_and_wait, args=())
        ghm_thread.start()

        # Start looking for the StoredBuild for the PR from above
        max_seconds = 600
        start_seconds = time.time()
        elapsed_seconds = 0
        found_build = False
        total_builds_to_wait_for = expected_start_builds + wait_for_new_builds
        while elapsed_seconds < max_seconds:
            stored_builds = self.storage.build_storage.get_by_uuid(None)
            assert isinstance(stored_builds, list)
            if len(stored_builds) >= total_builds_to_wait_for:  # Wait for a few
                # For now assume the repo has many prs that need to be merged into our test storage
                pr_urls = []
                for stored_build in stored_builds:
                    assert isinstance(stored_build, StoredBuild)
                    assert (
                        not stored_build.source_uri in pr_urls
                    ), f"PR {stored_build.source_uri} stored as a build more than once."
                    pr_urls.append(stored_build.source_uri)
                    assert stored_build.status == Status.PENDING

                found_build = True
                break
            time.sleep(1)
            elapsed_seconds = time.time() - start_seconds

        # Clean up before assertions
        ghm.stop()
        ghm_thread.join()

        assert found_build, f"Did not see the build in build storage after {max_seconds} seconds."
        stored_builds = self.storage.build_storage.get_by_uuid(None)
        return len(stored_builds)
