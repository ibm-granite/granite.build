import os
import tempfile
from pathlib import Path
from time import sleep, time
from typing import Optional, Self

import pytest
from gbserver_test.buildwatcher.buildtest import (
    AbstractBuildTest,
    BuildTestSpecification,
    ClassTestedEnum,
    ExpectedTarget,
)
from gbserver_test.buildwatcher.utils import ExceptionRaisingThread
from gbserver_test.constants import (
    GBTEST_SPACE_NAME,
    GBTEST_USER_NAME,
    extended_testing_only,
)

from gbserver.buildwatcher.buildrunner import BuildRunner
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.status import Status
from gbserver.utils.logger import get_logger

_src_file_dir = os.path.abspath(os.path.dirname(__file__))
_test_data_dir = _src_file_dir.replace("test", "test-data", 1)


_RETRY_CPU_TEST_SPEC = BuildTestSpecification(
    build_yaml=os.path.join(_test_data_dir, "retry/cpu/build.yaml"),
    expected_status=Status.SUCCESS,
    target_expections=[
        ExpectedTarget(
            target_name="download_file",
            step_count=5,
            input_artifact_count=1,
            output_artifact_count=1,
            jobstats_count=3,
        ),
    ],
    simulate_failure=False,
)

# Same build config but download_file is expected to be skipped on the retry because
# it already succeeded in the original run that is in the same retry chain.
_RETRY_CPU_SKIPPED_TARGET_SPEC = BuildTestSpecification(
    build_yaml=os.path.join(_test_data_dir, "retry/cpu/build.yaml"),
    expected_status=Status.SUCCESS,
    target_expections=[
        ExpectedTarget(
            target_name="download_file",
            step_count=5,
            input_artifact_count=1,
            output_artifact_count=1,
            jobstats_count=3,
        ),
    ],
    simulate_failure=False,
    skip_target_names=["download_file"],
)


def _make_stored_build_with_config(
    build_config_yaml: str,
    status: Status = Status.FAILED,
    retry_count: int = 0,
    retry_of_build_id: str = None,
) -> StoredBuild:
    """Create a StoredBuild whose build_archive encodes the given build.yaml content."""
    with tempfile.TemporaryDirectory() as tmp:
        build_dir = Path(tmp)
        (build_dir / "build.yaml").write_text(build_config_yaml)
        build = StoredBuild.create(
            name="test-build",
            space_name="test-space",
            source_uri="",
            username="test-user",
            build_yaml_path=build_dir / "build.yaml",
            status=status,
        )
        build.retry_count = retry_count
        build.retry_of_build_id = retry_of_build_id
        return build


_BUILD_YAML_NO_RETRY = """\
llm.build:
  name: test
  targets:
    mytarget:
      environment_uri: space://environments/cpu
      steps:
        - step_uri: space://steps/download
"""

_BUILD_YAML_MAX_RETRIES_2 = """\
llm.build:
  name: test
  retries:
    max_retries: 2
  targets:
    mytarget:
      environment_uri: space://environments/cpu
      steps:
        - step_uri: space://steps/download
"""
logger = get_logger(__name__)


class TestShouldRetry:
    """Unit tests for the BuildRunner._should_retry() instance method."""

    def _runner(self) -> BuildRunner:
        """Create a minimal BuildRunner instance without triggering __init__."""
        return object.__new__(BuildRunner)

    def test_non_failed_build_not_retried(self):
        runner = self._runner()
        for st in (
            Status.SUCCESS,
            Status.PENDING,
            Status.RUNNING,
            Status.CANCELLED,
            Status.INVALID,
        ):
            build = _make_stored_build_with_config(_BUILD_YAML_MAX_RETRIES_2, status=st)
            assert not runner._should_retry(build), f"Expected no retry for status {st}"

    def test_failed_no_max_retries(self):
        build = _make_stored_build_with_config(_BUILD_YAML_NO_RETRY, status=Status.FAILED)
        assert not self._runner()._should_retry(build)

    def test_failed_with_max_retries_first_attempt(self):
        build = _make_stored_build_with_config(
            _BUILD_YAML_MAX_RETRIES_2, status=Status.FAILED, retry_count=0
        )
        assert self._runner()._should_retry(build)

    def test_failed_with_max_retries_second_attempt(self):
        build = _make_stored_build_with_config(
            _BUILD_YAML_MAX_RETRIES_2, status=Status.FAILED, retry_count=1
        )
        assert self._runner()._should_retry(build)

    def test_failed_exhausted_retries(self):
        build = _make_stored_build_with_config(
            _BUILD_YAML_MAX_RETRIES_2, status=Status.FAILED, retry_count=2
        )
        assert not self._runner()._should_retry(build)

    def test_retry_build_preserves_lineage(self):
        """The retry_of_build_id field should point to the original build UUID."""
        original_id = "original-build-uuid"
        build = _make_stored_build_with_config(
            _BUILD_YAML_MAX_RETRIES_2,
            status=Status.FAILED,
            retry_count=1,
            retry_of_build_id=original_id,
        )
        assert build.retry_of_build_id == original_id
        assert build.retry_count == 1
        assert self._runner()._should_retry(build)


@extended_testing_only
@pytest.mark.xdist_group(name="buildtest_cpu")
class TestBuildRunnerRetry(AbstractBuildTest):
    """
    Verifies that a FAILED build with max_retries > 0 is retried by BuildRunner
    and validates the full storage state across gb_builds, gb_targets, gb_steps,
    gb_artifacts, and gb_events.
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

    def _wait_for(self, fn, args: tuple, wait_condition: str, failure_msg: str, timeout_seconds):
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
        self._verify_finished_build_expectations(retry_id, _RETRY_CPU_SKIPPED_TARGET_SPEC)

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
        original_targets = self.storage.target_storage.get_by_where({"build_id": original_id})
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
        retry_events = self.storage.event_storage.get_sorted_build_events(build_id=retry_id)
        assert len(retry_events) > 0, self._failed_build_msg(
            retry_id, "Expected events for retry build"
        )
