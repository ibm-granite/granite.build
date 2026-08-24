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

"""In-place build retry in the local Bash environment.

A build retry reuses the *same* ``StoredBuild`` / build id instead of spawning a
new build. This test exercises the whole in-place loop end to end:

  1. The build's single target FAILS on its first attempt (the command emits its
     ``env_output`` artifact marker, drops a marker file, and ``exit 1``).
  2. The BuildRunner retries in place — same build id, ``retry_count`` bumped —
     and the command re-emits the same ``env_output``, finds the marker file, and
     ``exit 0``, so the target SUCCEEDS.

Verified across gb_builds / gb_targets / artifacts:
  * exactly ONE build record, whose uuid never changes, ending SUCCESS with
    ``retry_count == 1`` (no retry chain, no second build);
  * the target has a FAILED run and a SUCCESS run, the SUCCESS run linking back to
    the FAILED run via ``retry_of_target_id`` (the FAILED run links to nothing);
  * each run's steps mirror its outcome (FAILED run -> FAILED step, SUCCESS run ->
    SUCCESS step), via the shared ``_verify_target_and_steps`` harness;
  * the target has exactly the fixture's ``target_failure_count`` FAILED runs
    (one here) before the SUCCESS run;
  * the ``env_output`` marker is emitted on both attempts, but artifact counts are
    verified only against the SUCCESS run (which owns the single env_output
    artifact); the one registration is re-associated to the SUCCESS run: the build
    holds exactly one registration and its ``created_by_target_id`` points at the
    SUCCESS run (not the FAILED one), SUCCESS-status;
  * there are no "skipped" runs — every run is a real FAILED/SUCCESS record.
"""

from typing import Self, Tuple

import pytest
from libgbtest.buildrunner.buildtest import (
    AbstractBuildTest,
    BuildTestSpecification,
    ExpectedTarget,
    get_test_data_dir_for,
)
from libgbtest.buildrunner.utils import ExceptionRaisingThread
from libgbtest.constants import GBTEST_USER_NAME

from gbserver.buildrunner.buildrunner import BuildRunner
from gbserver.storage.artifact_registration import ArtifactRegistrationStatus
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.status import Status
from gbserver.utils.logger import get_logger

pytestmark = pytest.mark.standalone

logger = get_logger(__name__)


@pytest.mark.xdist_group(name="buildrunner_bash_retry")
class TestBuildRunnerRetryBash(AbstractBuildTest):
    """Verifies in-place build retry (fail-then-succeed) in the local Bash env."""

    def setup_method(self, method):
        # Run in-process via the local Bash environment — no cluster login.
        self.run_locally = True
        super().setup_method(method)

    def _get_spec(self) -> BuildTestSpecification:
        return BuildTestSpecification.from_yaml(
            get_test_data_dir_for(__file__) / "retry" / "buildtest.yaml"
        )

    def test_failed_target_reruns_in_place_and_links_success(self: Self):
        spec = self._get_spec()
        space = self._check_and_setup_space(spec)
        timeout_seconds = spec.timeout_minutes * 60

        stored_build = StoredBuild.create(
            name="test-retry",
            space_name=space.name,
            source_uri="",
            username=GBTEST_USER_NAME,
            build_yaml_path=spec.build_yaml,
            status=Status.PENDING,
        )
        build_id = stored_build.uuid
        self.storage.build_storage.add(stored_build)

        # Drive the build directly: BuildRunner's own loop retries in place, so the
        # first attempt fails, the build re-runs under the same id, and the second
        # attempt succeeds — all within this single start_and_wait().
        runner = BuildRunner(stored_build, space_uri=spec.space_uri, create_pr=False)
        runner_thread = ExceptionRaisingThread(
            name="Run in-place retry build", target=runner.start_and_wait, args=()
        )
        runner_thread.start()
        try:
            # The build fails its first attempt and retries in place under the same
            # id, so FAILED is a *transient* status here — wait through it for the
            # final SUCCESS rather than treating the first failure as terminal.
            self._wait_for_build_status(
                build_id,
                [Status.SUCCESS],
                timeout_seconds,
                transient_statuses=[Status.FAILED],
            )
        finally:
            runner_thread.join(timeout=60)

        self._assert_single_success_build(build_id)
        failed_run, success_run = self._assert_failed_then_success_runs(build_id)
        # Reuse the shared harness verification per run: it checks each run's step
        # count/status against the expectation. Artifact checks are gated to the
        # SUCCESS run — the winning run owns the single env_output artifact — while
        # the FAILED run's incidental output binding (which varies by environment)
        # is not asserted. The number of FAILED runs is verified against the
        # fixture's target_failure_count, and which run the one registration is
        # *attributed* to (the SUCCESS run, not the FAILED one) is asserted below.
        expected = self._expected_target(spec, "flaky-target")
        self._verify_target_and_steps(build_id, success_run, [Status.SUCCESS], expected)
        self._verify_target_and_steps(build_id, failed_run, [Status.FAILED], expected)
        self._verify_target_failure_count(build_id, "flaky-target", expected)
        self._assert_output_artifact_attributed_to_success(
            build_id, failed_run, success_run
        )

    def _assert_output_artifact_attributed_to_success(
        self: Self,
        build_id: str,
        failed_run: StoredTargetRun,
        success_run: StoredTargetRun,
    ) -> None:
        """The re-emitted env_output is attributed to the SUCCESS run, not the FAILED one.

        The command emits the same ``env_output`` artifact on both attempts. After
        the in-place retry the build holds exactly one such registration, and its
        ``created_by_target_id`` points at the SUCCESS run (the run that produced the
        successful output) — exercising the artifact re-association on retry. It is
        SUCCESS-status and never left PENDING.

        Args:
            build_id: the build under test.
            failed_run: the target's FAILED run (the registration must NOT be
                attributed to it, even though it also bound the emitted marker).
            success_run: the target's SUCCESS run (the registration is attributed
                to it).
        """
        artifacts = self.storage.artifact_registry.get_by_where(
            {"created_by_build_id": build_id}
        )
        assert len(artifacts) == 1, self._failed_build_msg(
            build_id,
            f"Expected exactly one output artifact for the build, got "
            f"{[(a.uri, a.created_by_target_id) for a in artifacts]}",
        )
        artifact = artifacts[0]
        assert (
            artifact.created_by_target_id == success_run.uuid
        ), self._failed_build_msg(
            build_id,
            f"env_output should be attributed to the SUCCESS run "
            f"({success_run.uuid}), but created_by_target_id is "
            f"{artifact.created_by_target_id} (FAILED run is {failed_run.uuid})",
        )
        assert (
            artifact.status == ArtifactRegistrationStatus.SUCCESS
        ), self._failed_build_msg(
            build_id,
            f"env_output should be SUCCESS, got {artifact.status}",
        )

    def _assert_single_success_build(self: Self, build_id: str) -> None:
        """Exactly one build record, same id, SUCCESS, retry_count == 1."""
        builds = self.storage.build_storage.get_by_uuid(None) or []
        assert len(builds) == 1, self._failed_build_msg(
            build_id,
            f"In-place retry must reuse one build id, found {len(builds)} builds",
        )
        build = builds[0]
        assert isinstance(build, StoredBuild)
        assert build.uuid == build_id, self._failed_build_msg(
            build_id, "Build id must not change across an in-place retry"
        )
        assert build.status == Status.SUCCESS, self._failed_build_msg(
            build_id, f"Build should end SUCCESS after the retry, got {build.status}"
        )
        assert build.retry_count == 1, self._failed_build_msg(
            build_id, f"Expected retry_count == 1, got {build.retry_count}"
        )

    def _assert_failed_then_success_runs(
        self: Self, build_id: str
    ) -> Tuple[StoredTargetRun, StoredTargetRun]:
        """The target has one FAILED run and one SUCCESS run linked back to it.

        Returns:
            (failed_run, success_run) so step/artifact checks can reuse them.
        """
        runs = self.storage.target_storage.get_by_where({"build_id": build_id})
        assert all(isinstance(r, StoredTargetRun) for r in runs)
        # A target that failed once and then succeeded on the in-place retry has
        # exactly two runs — a FAILED one and a SUCCESS one — and nothing else.
        by_status = {r.status: r for r in runs}
        assert len(runs) == 2, self._failed_build_msg(
            build_id,
            f"Expected exactly two target runs (FAILED + SUCCESS), got "
            f"{[(r.name, r.status) for r in runs]}",
        )
        assert (
            Status.FAILED in by_status and Status.SUCCESS in by_status
        ), self._failed_build_msg(
            build_id,
            f"Expected one FAILED and one SUCCESS run, got {list(by_status)}",
        )
        failed_run = by_status[Status.FAILED]
        success_run = by_status[Status.SUCCESS]
        # Same target, re-run in place.
        assert success_run.name == failed_run.name
        # The SUCCESS run links back to the FAILED run it retried.
        assert (
            success_run.retry_of_target_id == failed_run.uuid
        ), self._failed_build_msg(
            build_id,
            f"SUCCESS run retry_of_target_id ({success_run.retry_of_target_id}) "
            f"should point to the FAILED run ({failed_run.uuid})",
        )
        # The original FAILED run is not itself a retry of anything.
        assert failed_run.retry_of_target_id == "", self._failed_build_msg(
            build_id, "The FAILED run must not carry a retry_of_target_id"
        )
        return failed_run, success_run

    def _expected_target(
        self: Self, spec: BuildTestSpecification, name: str
    ) -> ExpectedTarget:
        """Return the ExpectedTarget named ``name`` from the loaded spec.

        Args:
            spec: the parsed buildtest specification.
            name: the target name whose expectation to return.

        Returns:
            The matching ExpectedTarget.

        Raises:
            AssertionError: if no expectation for ``name`` exists in the spec.
        """
        for expected in spec.target_expectations:
            if expected.target_name == name:
                return expected
        raise AssertionError(f"no expectation for target {name!r} in spec")
