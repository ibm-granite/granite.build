from time import sleep, time
from typing import Optional, Self

import pytest
from libgbtest.buildrunner.buildtest import (
    AbstractBuildTest,
    BuildTestSpecification,
    ClassTestedEnum,
    ExpectedTarget,
    get_test_data_dir_for,
)
from libgbtest.buildrunner.utils import ExceptionRaisingThread
from libgbtest.constants import (
    GBTEST_SPACE_NAME,
    GBTEST_USER_NAME,
    extended_testing_only,
)

from gbserver.buildrunner.buildrunner import BuildRunner
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.status import Status
from gbserver.utils.logger import get_logger

pytestmark = pytest.mark.ibm


_RETRY_CPU_BUILD_YAML = get_test_data_dir_for(__file__) / "retry" / "cpu" / "build.yaml"


_RETRY_CPU_TEST_SPEC = BuildTestSpecification(
    build_yaml=str(_RETRY_CPU_BUILD_YAML),
    expected_status=Status.SUCCESS,
    target_expectations=[
        ExpectedTarget(
            target_name="download_file",
            step_count=5,
            input_artifact_count=1,
            output_artifact_count=1,
            jobstats_count=3,
        ),
    ],
    simulate_step_failure=False,
)

# Same build config but download_file is expected to be skipped on the retry because
# it already succeeded in the original run that is in the same retry chain.
_RETRY_CPU_SKIPPED_TARGET_SPEC = BuildTestSpecification(
    build_yaml=str(_RETRY_CPU_BUILD_YAML),
    expected_status=Status.SUCCESS,
    target_expectations=[
        ExpectedTarget(
            target_name="download_file",
            step_count=5,
            input_artifact_count=1,
            output_artifact_count=1,
            jobstats_count=3,
        ),
    ],
    simulate_step_failure=False,
    skip_target_names=["download_file"],
)


logger = get_logger(__name__)


@extended_testing_only
@pytest.mark.xdist_group(name="buildtest_cpu")
class TestBuildRunnerRetry(AbstractBuildTest):
    """
    Verifies that a FAILED build with max_retries > 0 is retried by BuildRunner
    and validates the full storage state across gb_builds, gb_targets, gb_steps,
    gb_artifacts, and gb_events.
    Although this is using the K8s environment, this test should really be independent
    of the environment.
    """

    def _get_retried_build_id(self, original_build_id: str) -> Optional[str]:
        builds = self.storage.build_storage.get_by_uuid(None)
        if len(builds) <= 1:
            return None  # Not found yet.
        if builds[0].uuid == original_build_id:
            index = 1
        else:
            index = 0
        return builds[index].uuid

    def _has_retried_build(self, original_build_id) -> bool:
        return self._get_retried_build_id(original_build_id) != None

    def _wait_for(
        self, fn, args: tuple, wait_condition: str, failure_msg: str, timeout_seconds
    ):
        sleep_time = 5
        if timeout_seconds < sleep_time:
            sleep_time = timeout_seconds / 10
        time_waited = 0
        start_time = time()
        while time_waited <= timeout_seconds:
            now = time()
            result = fn(*args)
            if result:
                break
            time_waited = now - start_time
            logger.info(f"Waited {time_waited} second: {wait_condition}")
            sleep(sleep_time)

        assert time_waited <= timeout_seconds, failure_msg
        logger.info(f"Done waiting for {wait_condition}")

    def _wait_for_second_build(self, original_build_id, timeout_seconds):
        self._wait_for(
            fn=self._has_retried_build,
            args=(original_build_id,),
            wait_condition="2nd build to appear",
            failure_msg=f"Did not find retried build for original build id {original_build_id}",
            timeout_seconds=timeout_seconds,
        )

    def test_buildrunner_retry(self: Self):
        self.class_tested = ClassTestedEnum.TEST_BUILDRUNNER
        space = self.storage.space_storage.get_by_name(name=GBTEST_SPACE_NAME)
        assert space is not None, f"Could not find space '{GBTEST_SPACE_NAME}'"
        timeout_seconds = _RETRY_CPU_TEST_SPEC.timeout_minutes * 60

        # --- Phase 1: run the build to SUCCESS ---
        original_build = StoredBuild.create(
            name="test-retry",
            space_name=space.name,
            source_uri="",
            username=GBTEST_USER_NAME,
            build_yaml_path=_RETRY_CPU_TEST_SPEC.build_yaml,
            status=Status.PENDING,
        )
        original_id = original_build.uuid
        self._run_build_test_build(
            stored_build=original_build,
            tested_class=self.class_tested,
            test_cancel=False,
            expected_status=Status.SUCCESS,
            timeout_seconds=timeout_seconds,
        )
        self._verify_build_results(
            build_ids=[original_id],
            test_spec=_RETRY_CPU_TEST_SPEC,
            tested_class=self.class_tested,
            test_cancel=False,
        )

        # --- Phase 2: mark the successful build FAILED so BuildRunner will retry it ---
        # Note that we are NOT marking the build's targets or steps or artifacts as FAILED
        # so that they can be reused in the retry.
        assert self.storage is not None
        original_stored = self.storage.build_storage.get_by_uuid(original_id)
        assert isinstance(original_stored, StoredBuild)
        original_stored.status = Status.FAILED
        self.storage.build_storage.update(original_stored)

        # --- Phase 3: re-run on the FAILED build; BuildRunner auto-creates a retry ---
        # Because retry_of_build_id is set on the new build, __is_target_already_run
        # searches the retry chain and skips targets that succeeded in Phase 1.
        runner2 = BuildRunner(original_stored)
        runner_thread = ExceptionRaisingThread(
            name="Run retry build",
            target=runner2.start_and_wait,
            args=(),
        )
        runner_thread.start()

        retry_thread = ExceptionRaisingThread(
            name="Wait for retry build creation",
            target=self._wait_for_second_build,
            args=(original_id, timeout_seconds),
        )
        retry_thread.start()
        retry_thread.join()
        retry_id = self._get_retried_build_id(original_id)
        assert retry_id is not None, "Did not find retry build"

        self._wait_for_build_status(retry_id, [Status.SUCCESS], timeout_seconds)
        runner_thread.join()

        # --- gb_builds, gb_targets, gb_steps, gb_artifacts ---
        # Targets should be skipped because they already succeeded in Phase 1.
        self._verify_finished_build_expectations(
            retry_id, _RETRY_CPU_SKIPPED_TARGET_SPEC
        )

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
        assert retry.source_uri != "", self._failed_build_msg(
            retry_id, "Retry build should have a new PR source_uri"
        )
        assert retry.source_uri != original.source_uri, self._failed_build_msg(
            retry_id, "Retry build should have a different source_uri from the original"
        )

        # Verify that every target from Phase 1 was skipped in the retry by checking that
        # each retry target's skipped_for_prerun_target_id points to the original target.
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
                f"({retry_target.skipped_for_prerun_target_id}) does not point to the original "
                f"target ({original_target.uuid})",
            )

        # --- gb_events ---
        retry_events = self.storage.event_storage.get_sorted_build_events(
            build_id=retry_id
        )
        assert len(retry_events) > 0, self._failed_build_msg(
            retry_id, "Expected events for retry build"
        )
