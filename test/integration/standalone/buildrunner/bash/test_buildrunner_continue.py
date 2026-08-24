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

"""Build continuation with target reuse, in the local Bash environment.

Mirrors the build-level retry test (``test_buildrunner_retry.py``) but exercises
``gb build continue`` semantics: a *finished* build is continued by a *fresh*
BuildRunner rather than by the original runner's in-process retry loop.

Flow:
  1. Run a build whose single target succeeds (``command: exit 0``) to SUCCESS.
  2. Mark that build FAILED (so it is a plausible continuation candidate;
     continuation accepts any finished build), leaving its target/steps SUCCESS
     so they can be reused.
  3. Create a continuation via ``create_continuation_build`` (the same helper the
     ``POST /builds/continue`` endpoint uses) — a fresh build linked to the prior
     via ``retry_of_build_id`` with ``retry_count`` reset to 0 — and run it in a
     *new* BuildRunner. Because the target already succeeded in the chain, the
     continuation SKIPS it (``skipped_for_prerun_target_id`` points back to the
     original target). The continuation build completes SUCCESS.

The continuation linkage (``retry_of_build_id`` / ``retry_build_id`` /
``retry_count``) and the target-skip are verified across gb_builds and
gb_targets.
"""

from typing import Self

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
from gbserver.storage.stored_build import StoredBuild, create_continuation_build
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.status import Status
from gbserver.utils.logger import get_logger

pytestmark = pytest.mark.standalone

logger = get_logger(__name__)


@pytest.mark.xdist_group(name="buildrunner_bash_continue")
class TestBuildRunnerContinueBash(AbstractBuildTest):
    """Verifies build continuation and target reuse in the local Bash environment."""

    def setup_method(self, method):
        # Run in-process via the local Bash environment — no cluster login.
        self.run_locally = True
        super().setup_method(method)

    def _get_spec(self) -> BuildTestSpecification:
        return BuildTestSpecification.from_yaml(
            get_test_data_dir_for(__file__) / "continue" / "buildtest.yaml"
        )

    def test_buildrunner_continue_skips_succeeded_target(self: Self):
        spec = self._get_spec()
        space = self._check_and_setup_space(spec)
        timeout_seconds = spec.timeout_minutes * 60

        # --- Phase 1: run the build to SUCCESS ---
        original_build = StoredBuild.create(
            name="test-continue",
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

        # --- Phase 2: mark the successful build FAILED (a finished build to
        # continue). Targets/steps/artifacts are left SUCCESS so they are reused
        # (skipped) by the continuation. ---
        original_stored = self.storage.build_storage.get_by_uuid(original_id)
        assert isinstance(original_stored, StoredBuild)
        original_stored.status = Status.FAILED
        self.storage.build_storage.update(original_stored)

        # --- Phase 3: create the continuation (fresh build, fresh runner) ---
        # create_continuation_build stores the continuation as SUBMITTED, exactly
        # as POST /builds/continue does. In production the BuildWatcher then flips
        # SUBMITTED -> PENDING before dispatching a runner; here we drive the runner
        # directly, so make that same transition first (the runner only advances a
        # build whose status is PENDING/RUNNING/RETRY_PENDING).
        continuation = create_continuation_build(
            self.storage.build_storage, original_stored
        )
        continuation_id = continuation.uuid
        assert continuation_id != original_id
        assert continuation.status == Status.SUBMITTED
        continuation.status = Status.PENDING
        self.storage.build_storage.update(continuation)

        runner2 = BuildRunner(continuation, space_uri=spec.space_uri, create_pr=False)
        runner_thread = ExceptionRaisingThread(
            name="Run continuation build", target=runner2.start_and_wait, args=()
        )
        runner_thread.start()
        try:
            self._wait_for_build_status(
                continuation_id, [Status.SUCCESS], timeout_seconds
            )
        finally:
            runner_thread.join(timeout=60)

        # --- gb_builds: verify continuation linkage ---
        original = self.storage.build_storage.get_by_uuid(original_id)
        assert isinstance(original, StoredBuild)
        assert original.retry_build_id == continuation_id, self._failed_build_msg(
            original_id, "Original build should point to the continuation"
        )
        assert original.retry_of_build_id is None, self._failed_build_msg(
            original_id, "Original build should not have a retry_of_build_id"
        )

        cont = self.storage.build_storage.get_by_uuid(continuation_id)
        assert isinstance(cont, StoredBuild)
        assert cont.retry_of_build_id == original_id, self._failed_build_msg(
            continuation_id, "Continuation should point back to the prior chain root"
        )
        # max_retries is counted fresh for a continuation.
        assert cont.retry_count == 0, self._failed_build_msg(
            continuation_id, f"Expected retry_count=0, got {cont.retry_count}"
        )

        # --- gb_targets: every original target was skipped in the continuation ---
        original_targets = self.storage.target_storage.get_by_where(
            {"build_id": original_id}
        )
        assert len(original_targets) > 0, self._failed_build_msg(
            original_id, "Expected targets in original build"
        )
        for original_target in original_targets:
            assert isinstance(original_target, StoredTargetRun)
            cont_targets = self.storage.target_storage.get_by_where(
                {"build_id": continuation_id, "name": original_target.name}
            )
            assert len(cont_targets) == 1, self._failed_build_msg(
                continuation_id,
                f"Expected exactly one continuation target named '{original_target.name}'",
            )
            cont_target = cont_targets[0]
            assert isinstance(cont_target, StoredTargetRun)
            assert (
                cont_target.skipped_for_prerun_target_id == original_target.uuid
            ), self._failed_build_msg(
                continuation_id,
                f"Continuation target '{original_target.name}' "
                f"skipped_for_prerun_target_id ({cont_target.skipped_for_prerun_target_id}) "
                f"does not point to the original target ({original_target.uuid})",
            )
