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
from integration.environment.test_skypilot_slurm_e2e import _slurm_cluster_reachable
from libgbtest.buildrunner.buildtest import (
    AbstractBuildTest,
    BuildTestSpecification,
    get_test_data_dir_for,
)
from libgbtest.buildrunner.utils import ExceptionRaisingThread
from libgbtest.constants import GBTEST_USER_NAME, extended_testing_only

from gbserver.buildrunner.buildrunner import BuildRunner
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.status import Status

pytestmark = pytest.mark.skypilot_integration


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

    @pytest.mark.timeout(2100)  # above buildtest.yaml's timeout_minutes (30)
    def test_build_retry_reuses_the_finished_merge(self: Self):
        """Attempt 1 merges and then fails; attempt 2 reuses the merge and succeeds."""
        spec = BuildTestSpecification.from_yaml(
            get_test_data_dir_for(__file__) / "buildtest.yaml"
        )
        space = self._check_and_setup_space(spec)
        stored_build = StoredBuild.create(
            name="test-mergekit-retry",
            space_name=space.name,
            source_uri="",
            username=GBTEST_USER_NAME,
            build_yaml_path=spec.build_yaml,
            status=Status.PENDING,
        )
        build_id = stored_build.uuid
        self.storage.build_storage.add(stored_build)

        runner = BuildRunner(stored_build, space_uri=spec.space_uri, create_pr=False)
        runner_thread = ExceptionRaisingThread(
            name="mergekit in-place retry build",
            target=runner.start_and_wait,
            args=(),
        )
        runner_thread.start()
        try:
            self._wait_for_build_status(
                build_id,
                [Status.SUCCESS],
                spec.timeout_minutes * 60,
                transient_statuses=[Status.FAILED],
            )
        finally:
            runner_thread.join(timeout=60)

        build = self.storage.build_storage.get_by_uuid(build_id)
        assert isinstance(build, StoredBuild)
        assert build.status == Status.SUCCESS, self._failed_build_msg(
            build_id, f"build should end SUCCESS after the retry, got {build.status}"
        )
        assert build.retry_count == 1, self._failed_build_msg(
            build_id, f"expected exactly one in-place retry, got {build.retry_count}"
        )

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

        expected = next(t for t in spec.target_expectations if t.target_name == "merge")
        # The SUCCESS run: all steps succeeded, the model is registered, and its merge
        # step reports reused_earlier_attempt (so it reused the output, not merged again).
        self._verify_target_and_steps(
            build_id, by_status[Status.SUCCESS], [Status.SUCCESS], expected
        )
        self._verify_target_failure_count(build_id, "merge", expected)
