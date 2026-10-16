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

"""Regression tests for BuildRunner._apply_run_timestamps.

The LSF state-mapping change maps suspended/queued job states (PSUSP/SSUSP/
USUSP/UNKWN) to granite.build PENDING, so a run can now transition
RUNNING -> PENDING and can terminate straight from PENDING (a job killed while
suspended). This shared helper stamps timestamps for both step runs and target
runs; these tests pin the transition state machine:

- started_at is stamped on the FIRST entry into RUNNING regardless of the prior
  status (a resumed job re-entering RUNNING must not push started_at later; a run
  entering RUNNING from a non-PENDING state such as SUBMITTED must still stamp).
- finished_at is stamped on the FIRST entry into ANY terminal status, whatever
  the prior status -- including PENDING -> FAILED, the edge case that a
  "was RUNNING" guard silently dropped -- and never re-stamped.
- A RUNNING -> PENDING transition must NOT mark the run finished.
"""

from datetime import datetime, timedelta

import pytest

from gbserver.buildrunner.buildrunner import BuildRunner
from gbserver.storage.stored_step_run import StoredStepRun
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.status import Status

_T0 = datetime(2026, 1, 1, 0, 0, 0)
_T1 = _T0 + timedelta(seconds=30)
_T2 = _T0 + timedelta(seconds=60)


def _step(status: Status, started_at=None, finished_at=None) -> StoredStepRun:
    """Build a minimal StoredStepRun in the given pre-transition state."""
    return StoredStepRun(
        build_id="b1",
        target_id="t1",
        definition_uri="step://x",
        status=status,
        started_at=started_at,
        finished_at=finished_at,
    )


def _target(status: Status, started_at=None, finished_at=None) -> StoredTargetRun:
    """Build a minimal StoredTargetRun in the given pre-transition state."""
    return StoredTargetRun(
        build_id="b1",
        environment_uri="env://x",
        status=status,
        started_at=started_at,
        finished_at=finished_at,
    )


def test_first_running_stamps_started_at():
    """PENDING -> RUNNING stamps started_at."""
    step = _step(Status.PENDING)
    BuildRunner._apply_run_timestamps(step, Status.RUNNING, _T0)
    assert step.started_at == _T0
    assert step.finished_at is None


def test_submitted_to_running_stamps_started_at():
    """A non-linear first transition (SUBMITTED -> RUNNING) still stamps
    started_at. A target run can be created SUBMITTED with started_at=None, and a
    "was PENDING" precondition would silently drop it -- the mirror of the
    finished_at regression."""
    step = _step(Status.SUBMITTED)
    BuildRunner._apply_run_timestamps(step, Status.RUNNING, _T0)
    assert step.started_at == _T0


def test_target_run_submitted_to_running_stamps_started_at():
    """Same guarantee for target runs, which are the ones actually created in a
    non-PENDING (SUBMITTED) state with started_at=None."""
    target = _target(Status.SUBMITTED)
    BuildRunner._apply_run_timestamps(target, Status.RUNNING, _T0)
    assert target.started_at == _T0


def test_reentry_into_running_does_not_restamp_started_at():
    """A resumed job re-entering RUNNING keeps its original started_at.

    The stored status is PENDING (the suspend-state mapping) but started_at is
    already set, so it must be preserved rather than pushed to the later time.
    """
    step = _step(Status.PENDING, started_at=_T0)
    BuildRunner._apply_run_timestamps(step, Status.RUNNING, _T1)
    assert step.started_at == _T0


def test_running_to_pending_does_not_finish():
    """RUNNING -> PENDING (job queued/suspended) must NOT stamp finished_at."""
    step = _step(Status.RUNNING, started_at=_T0)
    BuildRunner._apply_run_timestamps(step, Status.PENDING, _T1)
    assert step.finished_at is None


def test_running_to_success_stamps_finished_at():
    """The ordinary RUNNING -> SUCCESS path still stamps finished_at."""
    step = _step(Status.RUNNING, started_at=_T0)
    BuildRunner._apply_run_timestamps(step, Status.SUCCESS, _T2)
    assert step.finished_at == _T2


@pytest.mark.parametrize(
    "terminal", [Status.FAILED, Status.SUCCESS, Status.CANCELLED, Status.INVALID]
)
def test_pending_to_terminal_stamps_finished_at(terminal):
    """A step terminating straight from PENDING (e.g. killed while suspended)
    is still stamped finished_at -- the regression the old guard dropped."""
    step = _step(Status.PENDING, started_at=_T0)
    BuildRunner._apply_run_timestamps(step, terminal, _T2)
    assert step.finished_at == _T2


def test_finished_at_not_restamped_on_redelivered_terminal():
    """A second terminal event does not overwrite the original finished_at."""
    step = _step(Status.FAILED, started_at=_T0, finished_at=_T1)
    BuildRunner._apply_run_timestamps(step, Status.FAILED, _T2)
    assert step.finished_at == _T1


def test_target_run_pending_to_terminal_stamps_finished_at():
    """The same guarantee holds for target runs: a target terminating from a
    PENDING-mapped state is stamped finished_at (symmetric with the step fix)."""
    target = _target(Status.PENDING, started_at=_T0)
    BuildRunner._apply_run_timestamps(target, Status.FAILED, _T2)
    assert target.finished_at == _T2


def test_target_run_running_to_pending_does_not_finish():
    """A target RUNNING -> PENDING transition must NOT stamp finished_at."""
    target = _target(Status.RUNNING, started_at=_T0)
    BuildRunner._apply_run_timestamps(target, Status.PENDING, _T1)
    assert target.finished_at is None
