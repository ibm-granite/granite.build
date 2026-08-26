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
``gb build restart`` semantics: a *finished* build is continued by a *fresh*
BuildRunner rather than by the original runner's in-process retry loop.

Option A — continuation reuses the SAME build id. There is no retry chain and no
skip record: continuing re-opens the finished build in place and re-dispatches it
onto a new runner, which reuses the target that already succeeded (it is not
re-run) rather than recording a separate skipped run.

Flow:
  1. Run a build whose single target succeeds (``command: exit 0``) to SUCCESS.
  2. Mark that build FAILED (a plausible continuation candidate; continuation
     accepts any finished build), leaving its target/steps SUCCESS so they can be
     reused.
  3. Continue it via ``reopen_finished_build`` (the same helper the
     ``POST /builds/continue`` endpoint uses) — the SAME build flips to SUBMITTED
     with ``retry_count`` reset to 0 — and run it in a *new* BuildRunner. Because
     the target already succeeded in this build, the continuation REUSES it (no
     re-run, no new record) and completes SUCCESS.

The in-place re-open (same id, ``retry_count`` reset) and the single-record target
reuse are verified across gb_builds and gb_targets.
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
from gbserver.storage.stored_build import StoredBuild, reopen_finished_build
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.status import Status
from gbserver.utils.logger import get_logger

pytestmark = pytest.mark.standalone

logger = get_logger(__name__)


@pytest.mark.xdist_group(name="buildrunner_bash_restart")
class TestBuildRunnerRestartBash(AbstractBuildTest):
    """Verifies build restart (continuation) and target reuse in the local Bash environment."""

    def setup_method(self, method):
        # Run in-process via the local Bash environment — no cluster login.
        self.run_locally = True
        super().setup_method(method)

    def _get_spec(self) -> BuildTestSpecification:
        return BuildTestSpecification.from_yaml(
            get_test_data_dir_for(__file__) / "continue" / "buildtest.yaml"
        )

    def test_buildrunner_restart_reuses_succeeded_target(self: Self):
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

        # Record the target runs produced by the first attempt; continuation must
        # reuse (not duplicate) these.
        targets_after_first = self.storage.target_storage.get_by_where(
            {"build_id": original_id}
        )
        assert len(targets_after_first) > 0, self._failed_build_msg(
            original_id, "Expected targets in the first attempt"
        )
        original_target_ids = {t.uuid for t in targets_after_first}

        # --- Phase 2: mark the successful build FAILED (a finished build to
        # continue). Targets/steps/artifacts are left SUCCESS so they are reused
        # by the continuation. ---
        original_stored = self.storage.build_storage.get_by_uuid(original_id)
        assert isinstance(original_stored, StoredBuild)
        original_stored.status = Status.FAILED
        self.storage.build_storage.update(original_stored)

        # --- Phase 3: continue the SAME build (in place, fresh runner) ---
        # reopen_finished_build flips the build to SUBMITTED with retry_count reset,
        # exactly as POST /builds/continue does. In production the BuildWatcher then
        # flips SUBMITTED -> PENDING before dispatching a runner; here we drive the
        # runner directly, so make that same transition first (the runner only
        # advances a build whose status is PENDING/RUNNING).
        reopened = reopen_finished_build(self.storage.build_storage, original_stored)
        assert isinstance(reopened, StoredBuild)
        # Same build id — no new build created.
        assert reopened.uuid == original_id
        assert reopened.status == Status.SUBMITTED
        # max_retries is counted fresh for a continuation.
        assert reopened.retry_count == 0
        reopened.status = Status.PENDING
        self.storage.build_storage.update(reopened)

        runner2 = BuildRunner(reopened, space_uri=spec.space_uri, create_pr=False)
        runner_thread = ExceptionRaisingThread(
            name="Run continuation build", target=runner2.start_and_wait, args=()
        )
        runner_thread.start()
        try:
            self._wait_for_build_status(original_id, [Status.SUCCESS], timeout_seconds)
        finally:
            runner_thread.join(timeout=60)

        # --- gb_builds: same build, back to SUCCESS, budget still reset ---
        # Continuation re-opens the finished build in place, so storage must still
        # hold exactly ONE build record — the continuation must not fork a new one.
        all_builds = self.storage.build_storage.get_by_uuid(None) or []
        assert len(all_builds) == 1, self._failed_build_msg(
            original_id,
            f"Continuation must reuse one build id, found {len(all_builds)} builds",
        )
        cont = self.storage.build_storage.get_by_uuid(original_id)
        assert isinstance(cont, StoredBuild)
        assert cont.status == Status.SUCCESS, self._failed_build_msg(
            original_id, f"Expected continuation to reach SUCCESS, got {cont.status}"
        )
        assert cont.retry_count == 0, self._failed_build_msg(
            original_id, f"Expected retry_count=0, got {cont.retry_count}"
        )

        # --- gb_targets: each target that already succeeded is reused, not
        # duplicated — the same SUCCESS run persists and no new run is written. ---
        for original_target in targets_after_first:
            assert isinstance(original_target, StoredTargetRun)
            reused = self.storage.target_storage.get_by_where(
                {"build_id": original_id, "name": original_target.name}
            )
            assert len(reused) == 1, self._failed_build_msg(
                original_id,
                f"Expected exactly one target run named '{original_target.name}' "
                f"after continuation (reuse must not duplicate it), got {len(reused)}",
            )
            reused_target = reused[0]
            assert isinstance(reused_target, StoredTargetRun)
            # It is the very same record produced by the first attempt.
            assert reused_target.uuid in original_target_ids, self._failed_build_msg(
                original_id,
                f"Target '{original_target.name}' was re-created on continuation "
                f"instead of reusing the original SUCCESS run",
            )
            assert reused_target.status == Status.SUCCESS
