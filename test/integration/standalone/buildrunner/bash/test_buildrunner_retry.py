#!/usr/bin/env python3

# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build-level retry with target reuse, in the local Bash environment.

Mirrors the former K8s ``TestBuildRunnerRetry`` but runs entirely in-process via
the Bash environment, so no cluster/Lakehouse is required and the test runs in
the standard PR suite.

Flow:
  1. Run a build whose single target succeeds (``command: exit 0``) to SUCCESS.
  2. Mark that build FAILED (leaving its target/steps SUCCESS so they can be
     reused), as BuildRunner only retries FAILED builds.
  3. Re-run via a BuildRunner: it auto-creates a linked retry build, and because
     the target already succeeded earlier in the same retry chain, the retry
     SKIPS it (``skipped_for_prerun_target_id`` points back to the original
     target).  The retry build then completes SUCCESS.

The retry chain linkage (``retry_of_build_id`` / ``retry_build_id`` /
``retry_count``) and the target-skip are verified across gb_builds and
gb_targets.
"""

from time import sleep, time
from typing import Optional, Self

import pytest
from libgbtest.buildrunner.buildtest import (
    AbstractBuildTest,
    BuildTestSpecification,
    ClassTestedEnum,
    get_test_data_dir_for,
)
from libgbtest.buildrunner.utils import ExceptionRaisingThread
from libgbtest.constants import GBTEST_USER_NAME

from gbserver.buildrunner.buildrunner import BuildRunner
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.status import Status
from gbserver.utils.logger import get_logger

pytestmark = pytest.mark.standalone

logger = get_logger(__name__)


@pytest.mark.xdist_group(name="buildrunner_bash_retry")
class TestBuildRunnerRetryBash(AbstractBuildTest):
    """Verifies build-level retry and target reuse in the local Bash environment."""

    def setup_method(self, method):
        # Run in-process via the local Bash environment — no cluster login.
        self.run_locally = True
        super().setup_method(method)

    def _get_spec(self) -> BuildTestSpecification:
        return BuildTestSpecification.from_yaml(
            get_test_data_dir_for(__file__) / "retry" / "buildtest.yaml"
        )

    # ------------------------------------------------------------------
    # Helpers for locating the auto-created retry build
    # ------------------------------------------------------------------

    def _get_retried_build_id(self, original_build_id: str) -> Optional[str]:
        """Return the uuid of the retry build (the one that is not the original)."""
        builds = self.storage.build_storage.get_by_uuid(None)
        if builds is None or len(builds) <= 1:
            return None  # Not found yet.
        index = 1 if builds[0].uuid == original_build_id else 0
        return builds[index].uuid

    def _has_retried_build(self, original_build_id: str) -> bool:
        return self._get_retried_build_id(original_build_id) is not None

    def _wait_for(
        self, fn, args: tuple, wait_condition: str, failure_msg: str, timeout_seconds
    ):
        """Poll ``fn(*args)`` until it is truthy or ``timeout_seconds`` elapses."""
        sleep_time = 5 if timeout_seconds >= 5 else timeout_seconds / 10
        time_waited = 0.0
        start_time = time()
        while time_waited <= timeout_seconds:
            if fn(*args):
                break
            time_waited = time() - start_time
            logger.info(f"Waited {time_waited} second: {wait_condition}")
            sleep(sleep_time)
        assert time_waited <= timeout_seconds, failure_msg
        logger.info(f"Done waiting for {wait_condition}")

    def _wait_for_second_build(self, original_build_id: str, timeout_seconds):
        self._wait_for(
            fn=self._has_retried_build,
            args=(original_build_id,),
            wait_condition="2nd (retry) build to appear",
            failure_msg=f"Did not find retried build for original build id {original_build_id}",
            timeout_seconds=timeout_seconds,
        )

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_buildrunner_retry_skips_succeeded_target(self: Self):
        spec = self._get_spec()
        space = self._check_and_setup_space(spec)
        timeout_seconds = spec.timeout_minutes * 60

        # --- Phase 1: run the build to SUCCESS ---
        original_build = StoredBuild.create(
            name="test-retry",
            space_name=space.name,
            source_uri="",
            username=GBTEST_USER_NAME,
            build_yaml_path=spec.build_yaml,
            status=Status.PENDING,
        )
        original_id = original_build.uuid
        self._run_build_test_build(
            stored_build=original_build,
            tested_class=ClassTestedEnum.TEST_BUILDRUNNER,
            test_cancel=False,
            expected_status=Status.SUCCESS,
            timeout_seconds=timeout_seconds,
            space_uri=spec.space_uri,
        )

        # --- Phase 2: mark the successful build FAILED so BuildRunner will retry it ---
        # Targets/steps/artifacts are intentionally left SUCCESS so they can be
        # reused (skipped) in the retry.
        original_stored = self.storage.build_storage.get_by_uuid(original_id)
        assert isinstance(original_stored, StoredBuild)
        original_stored.status = Status.FAILED
        self.storage.build_storage.update(original_stored)

        # --- Phase 3: re-run on the FAILED build; BuildRunner auto-creates a retry ---
        runner2 = BuildRunner(
            original_stored, space_uri=spec.space_uri, create_pr=False
        )
        runner_thread = ExceptionRaisingThread(
            name="Run retry build", target=runner2.start_and_wait, args=()
        )
        runner_thread.start()
        try:
            self._wait_for_second_build(original_id, timeout_seconds)
            retry_id = self._get_retried_build_id(original_id)
            assert retry_id is not None, "Did not find retry build"
            self._wait_for_build_status(retry_id, [Status.SUCCESS], timeout_seconds)
        finally:
            runner_thread.join(timeout=60)

        # --- gb_builds: verify retry linkage ---
        original = self.storage.build_storage.get_by_uuid(original_id)
        assert isinstance(original, StoredBuild)
        assert original.status == Status.FAILED, self._failed_build_msg(
            original_id, f"Original build status: {original.status}"
        )
        assert original.retry_build_id == retry_id, self._failed_build_msg(
            original_id, "Original build should point to retry"
        )
        assert original.retry_of_build_id is None, self._failed_build_msg(
            original_id, "Original build should not have a retry_of_build_id"
        )

        retry = self.storage.build_storage.get_by_uuid(retry_id)
        assert isinstance(retry, StoredBuild)
        assert retry.retry_of_build_id == original_id, self._failed_build_msg(
            retry_id, "Retry build should point back to original"
        )
        assert retry.retry_count == 1, self._failed_build_msg(
            retry_id, f"Expected retry_count=1, got {retry.retry_count}"
        )
        assert retry.retry_build_id is None, self._failed_build_msg(
            retry_id, "Retry build should not itself have been retried"
        )

        # --- gb_targets: every original target was skipped in the retry ---
        original_targets = self.storage.target_storage.get_by_where(
            {"build_id": original_id}
        )
        assert len(original_targets) > 0, self._failed_build_msg(
            original_id, "Expected targets in original build"
        )
        for original_target in original_targets:
            assert isinstance(original_target, StoredTargetRun)
            retry_targets = self.storage.target_storage.get_by_where(
                {"build_id": retry_id, "name": original_target.name}
            )
            assert len(retry_targets) == 1, self._failed_build_msg(
                retry_id,
                f"Expected exactly one retry target named '{original_target.name}'",
            )
            retry_target = retry_targets[0]
            assert isinstance(retry_target, StoredTargetRun)
            assert (
                retry_target.skipped_for_prerun_target_id == original_target.uuid
            ), self._failed_build_msg(
                retry_id,
                f"Retry target '{original_target.name}' skipped_for_prerun_target_id "
                f"({retry_target.skipped_for_prerun_target_id}) does not point to the "
                f"original target ({original_target.uuid})",
            )
