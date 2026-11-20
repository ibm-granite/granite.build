# SPDX-License-Identifier: Apache-2.0

"""mergekit on slurm: an in-place build retry reuses the merge an earlier attempt finished.

The target merges, then a byoc step fails on the first attempt only. The build retries
in place (same build id, new target and step runs). On the second attempt the merge
finds its own finished output beside a record with the same retry key and input hash,
and registers it (`reused_earlier_attempt` metadata) instead of refusing the existing
output; the byoc step then succeeds, and so does the build.

The same pattern as test/integration/standalone/buildrunner/bash/test_buildrunner_retry.py:
the build is driven with BuildRunner directly and FAILED is waited through as a
transient status, since the YAML build-test runner treats a first failure as final and
stops the build before it can retry. Gated on a reachable Docker SLURM cluster (`make
test-setup`); extended suite only. Fixtures live in the test-data/ dir mirroring this
file (get_test_data_dir_for).
"""

from typing import Self

import pytest
import yaml
from integration.environment.test_skypilot_slurm_e2e import _slurm_cluster_reachable
from integration.ibm.buildrunner.skypilot.slurm_local.cluster import remove_from_shared
from libgbtest.buildrunner.buildtest import (
    AbstractBuildTest,
    BuildTestSpecification,
    get_test_data_dir_for,
)
from libgbtest.buildrunner.retry_wait import wait_for_last_attempt
from libgbtest.buildrunner.utils import ExceptionRaisingThread
from libgbtest.constants import GBTEST_USER_NAME, extended_testing_only
from libgbtest.shared_paths import step_output_dir

from gbserver.buildrunner.buildrunner import BuildRunner
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.status import Status

pytestmark = pytest.mark.skypilot_integration


#: This test's fixture (also read for the output path to clear).
_BUILD_YAML = get_test_data_dir_for(__file__) / "build.yaml"


@extended_testing_only
@pytest.mark.skipif(
    not _slurm_cluster_reachable(),
    reason="Docker SLURM cluster not reachable (run: make test-setup)",
)
class TestSkypilotSlurmMergekitRetry(AbstractBuildTest):
    """mergekit step, an in-place build retry on slurm."""

    def setup_method(self, method):
        """Run the BuildRunner in-process, as the YAML build tests do."""
        self.run_locally = True
        super().setup_method(method)

    @pytest.fixture(autouse=True)
    def _fresh_output(self):
        """Clear this test's ``/shared/mergekit-test/<test>`` dir before it runs.

        It holds the model, its retry record and the byoc attempt markers; a re-run
        would otherwise be refused. The path comes from build.yaml.
        """
        remove_from_shared(
            step_output_dir(_BUILD_YAML, "merge", 0, "mergekit_config", "output_path")
        )

    def _store_build(self: Self, spec: BuildTestSpecification) -> StoredBuild:
        """Create the stored build the runner drives, in this test's space."""
        space = self._check_and_setup_space(spec)
        stored_build = StoredBuild.create(
            name="test-mergekit-retry",
            space_name=space.name,
            source_uri="",
            username=GBTEST_USER_NAME,
            build_yaml_path=spec.build_yaml,
            status=Status.PENDING,
        )
        self.storage.build_storage.add(stored_build)
        return stored_build

    def _run_to_the_last_attempt(
        self: Self, stored_build: StoredBuild, spec: BuildTestSpecification
    ) -> StoredBuild:
        """Run the build in-process until it succeeds or its last attempt ends.

        Waits with ``wait_for_last_attempt``, so a failure is reported with its real
        status, not as a timeout.
        """
        build_id = stored_build.uuid
        runner = BuildRunner(stored_build, space_uri=spec.space_uri, create_pr=False)
        runner_thread = ExceptionRaisingThread(
            name="mergekit in-place retry build",
            target=runner.start_and_wait,
            args=(),
        )
        runner_thread.start()
        build_yaml = yaml.safe_load(_BUILD_YAML.read_text(encoding="utf-8"))
        build = None
        try:
            build = wait_for_last_attempt(
                lambda: self.storage.build_storage.get_by_uuid(build_id),
                runner_thread.is_alive,
                build_yaml["granite.build"]["retries"]["max_retries"],
                spec.timeout_minutes * 60,
            )
        finally:
            if build is None:
                # Timed out (or the wait raised): stop the build, so it doesn't go on
                # writing to /shared while a re-run's setup clears it. Not after a
                # result: the runner may still be finishing, and a stop then could
                # mark a finished build CANCELLED.
                runner.stop()
            runner_thread.join(timeout=120)
        if build is None:
            pytest.fail(
                self._failed_build_msg(
                    build_id, f"build not done after {spec.timeout_minutes} minutes"
                )
            )
        assert isinstance(build, StoredBuild)
        return build

    def _check_target_runs(self: Self, build_id: str) -> dict:
        """One FAILED and one SUCCESS target run; returns them by status."""
        runs = self.storage.target_storage.get_by_where({"build_id": build_id})
        assert all(isinstance(r, StoredTargetRun) for r in runs)
        by_status = {r.status: r for r in runs}
        assert len(runs) == 2 and set(by_status) == {
            Status.FAILED,
            Status.SUCCESS,
        }, self._failed_build_msg(
            build_id,
            f"expected one FAILED and one SUCCESS target run, got "
            f"{[(r.name, r.status) for r in runs]}",
        )
        return by_status

    @pytest.mark.timeout(2100)  # above buildtest.yaml's timeout_minutes (30)
    def test_build_retry_reuses_the_finished_merge(self: Self):
        """Attempt 1 merges and then fails; attempt 2 reuses the merge and succeeds."""
        spec = BuildTestSpecification.from_yaml(
            get_test_data_dir_for(__file__) / "buildtest.yaml"
        )
        stored_build = self._store_build(spec)
        build_id = stored_build.uuid

        build = self._run_to_the_last_attempt(stored_build, spec)

        assert build.status == Status.SUCCESS, self._failed_build_msg(
            build_id, f"build should end SUCCESS after the retry, got {build.status}"
        )
        assert build.retry_count == 1, self._failed_build_msg(
            build_id, f"expected exactly one in-place retry, got {build.retry_count}"
        )
        by_status = self._check_target_runs(build_id)

        expected = next(t for t in spec.target_expectations if t.target_name == "merge")
        # The SUCCESS run: all steps succeeded, the model is registered, and its merge
        # step reports reused_earlier_attempt (so it reused the output, not merged again).
        self._verify_target_and_steps(
            build_id, by_status[Status.SUCCESS], [Status.SUCCESS], expected
        )
        self._verify_target_failure_count(build_id, "merge", expected)
