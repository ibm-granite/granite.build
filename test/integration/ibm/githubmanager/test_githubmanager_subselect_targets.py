import os
import threading
import time

import pytest
from libgbtest.buildrunner.utils import pre_register_input_artifacts

pytestmark = pytest.mark.ibm
from libgbtest.utils import AbstractSingletonStorageUsingPreloadedSpaceTest

from gbserver.github.githubmanager import GitHubManager
from gbserver.storage.stored_build import StoredBuild
from gbserver.types.constants import (
    GB_ENVIRONMENT,
    PUBLIC_SPACE_GIT_URI,
    PUBLIC_SPACE_NAME,
)
from gbserver.types.status import Status
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


@pytest.mark.skip(reason="pr-watcher being deprecated")
class TestGithubManagerSubSelectTargets(
    AbstractSingletonStorageUsingPreloadedSpaceTest
):

    def test_build_subselect_targets(self):
        """
        We use the https://github.ibm.com/granite-dot-build/gb-test/pull/1238 PR to test migration/validation to StoredBuild
        along with a specific target of 'tunedmodel'.
        """
        # src_file_dir = os.path.abspath(os.path.dirname(__file__))
        # test_data_dir = src_file_dir.replace("test", "test-data", 1)

        token = os.environ.get("GITHUB_TOKEN")
        assert token is not None, "GITHUB_TOKEN must be set to run this test"
        # domain = DEFAULT_GH_DOMAIN
        # owner = "granite-dot-build"
        # repo = "gb-test"                    # TODO we need an empty repo to start with
        # test_data_path = Path(test_data_dir).resolve()
        # config_path = test_data_path / "test_1_config" / "pr-watcher-config.yaml"
        # assert config_path.is_file(), f"the path {config_path} is not a file"
        # # logger.info(f"Testing with\n .   domain:{domain}\n .   owner:{owner}\n .   repo:{repo}")
        # logger.info(f"Testing with\n .   config at path:{config_path}")

        # TODO: Publish the build to github

        # TODO: Merge the PR so it is picked up by the manager below

        stored_builds = self.storage.build_storage.get_by_uuid(None)
        assert isinstance(stored_builds, list)
        assert (
            len(stored_builds) == 0
        ), "What?!  There should be no builds to start with!"

        # We use PRs https://github.ibm.com/granite-dot-build/gb-test/pull/1238 (STAGING) and
        # https://github.ibm.com/granite-dot-build/gbspace-public-dev/pull/379 (DEV) for this test.
        pr = (
            "1238"
            if GB_ENVIRONMENT == "STAGING"
            else "381" if GB_ENVIRONMENT == "DEV" else None
        )
        assert pr is not None, "Tests are not running with expected environment"
        source_uri = PUBLIC_SPACE_GIT_URI + "/pull/" + pr
        # The test requires the following inputs to be registered ahead of time in order to pass build.yaml validation.
        inputs = [
            "lh://prod/granite_dot_build.public/models/model_shared/granite-2b-base/20250319T181102",
            "lh://prod/granite_dot_build.public/tables/all_data_large_dedup_regression_test",
        ]
        pre_register_input_artifacts(
            self.storage.artifact_registry, PUBLIC_SPACE_NAME, inputs
        )

        # Watch for the PR and copy to StoredBuildStorage
        ghm = GitHubManager(token=token)
        ghm._skip_all_but_prs = [source_uri]

        # Run this in a separate thread.  Also, uses the same singleton test storage
        ghm_thread = threading.Thread(target=ghm.start_and_wait, args=())
        ghm_thread.start()

        # Start looking for the StoredBuild for the PR from above
        max_seconds = 300
        start_seconds = time.time()
        elapsed_seconds = 0
        found_build = False
        try:
            while elapsed_seconds < max_seconds:
                stored_builds = self.storage.build_storage.get_by_where(
                    {"source_uri": source_uri}
                )
                if len(stored_builds) > 0:
                    assert len(stored_builds) == 1, "Got more than 1 build"
                    stored_build = stored_builds[0]
                    assert isinstance(stored_build, StoredBuild)
                    assert stored_build.status == Status.PENDING
                    assert stored_build.source_uri == source_uri
                    assert (
                        stored_build.targets is not None
                    ), "the list of targets is None"
                    assert (
                        len(stored_build.targets) == 1
                    ), "the number of targets is wrong"
                    assert (
                        stored_build.targets[0] == "tunedmodel"
                    ), "the target name is wrong"
                    stored_build.source_uri
                    found_build = True
                    break
                time.sleep(5)
                elapsed_seconds = time.time() - start_seconds
        finally:
            # Clean up before assertions
            ghm.stop()
            ghm_thread.join()

        assert (
            found_build
        ), f"Did not see the build in build storage after {max_seconds} seconds."
