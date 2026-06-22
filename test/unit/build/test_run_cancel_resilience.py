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

"""Tests for cancellation resilience in Run and TargetRun.

Exercises the uncancel/re-await loop in Run.run()'s finally block and
verifies TargetRun.cancel() fans out to all step runs correctly.
"""

import asyncio
from asyncio import TaskGroup
from pathlib import Path
from typing import Optional, Self
from unittest.mock import MagicMock

import pytest

from gbserver.build.run import Run
from gbserver.build.targetrun import TargetRun
from gbserver.types.buildevent import EntityRunMetadata
from gbserver.types.status import Status


class CleanupTrackingRun(Run):
    """A Run subclass that tracks whether _cleanup() completed."""

    def __init__(self):
        entity = MagicMock()
        entity.build_id = "test-build"
        super().__init__(entity=entity, base_dir=Path("/tmp/test-run"))
        self.cleanup_started = False
        self.cleanup_completed = False
        self.cleanup_started_event = asyncio.Event()

    async def _run(self: Self, tg: Optional[TaskGroup] = None, **kwargs) -> None:
        await asyncio.sleep(3600)

    async def _cleanup(self: Self, tg: Optional[TaskGroup] = None, **kwargs) -> None:
        self.cleanup_started = True
        self.cleanup_started_event.set()
        await asyncio.sleep(0.5)
        self.cleanup_completed = True

    def get_runmetadata(self: Self) -> EntityRunMetadata:
        return EntityRunMetadata(build_id="test-build")


class TestRunCancelResilience:
    """Tests that cleanup survives repeated and concurrent cancellations."""

    @pytest.mark.asyncio
    async def test_cleanup_completes_through_repeated_cancellation(self) -> None:
        """Cleanup must finish even if cancel() is called many times while the
        finally block is awaiting _cleanup() — exercises the uncancel/re-await loop."""
        run = CleanupTrackingRun()
        task = asyncio.create_task(run.run())
        await asyncio.sleep(0.05)
        task.cancel()
        await run.cleanup_started_event.wait()

        # Hammer cancel() repeatedly while _cleanup()'s await is in flight.
        for _ in range(5):
            task.cancel()
            await asyncio.sleep(0)  # yield so each cancel hits a fresh await point

        with pytest.raises(asyncio.CancelledError):
            await task

        assert run.cleanup_completed, "cleanup must finish despite repeated cancels"

    @pytest.mark.asyncio
    async def test_cancellation_propagates_after_cleanup(self) -> None:
        """uncancel() must not swallow the cancellation: the task should still
        raise CancelledError and the run should end CANCELLED, not completed."""
        run = CleanupTrackingRun()
        task = asyncio.create_task(run.run())
        await asyncio.sleep(0.05)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert run.cleanup_completed
        assert task.cancelled() or task.cancelling() == 0  # cancellation consumed cleanly
        assert run.status is Status.CANCELLED


class TestTargetRunCancelFanOut:
    """Tests that TargetRun.cancel() propagates to all step runs."""

    def test_targetrun_cancel_cancels_all_step_runs(self) -> None:
        """TargetRun.cancel() must cancel all step runs with active tasks
        and safely skip step runs that have not started (task is None)."""
        from gbserver.build.target import Target

        target = MagicMock(spec=Target)
        target.build_id = "test-build"
        target.name = "test-target"
        target.username = "test-user"
        target.dir = Path("/tmp/test-run")
        target.inputs_status = {}
        # Make isinstance check pass
        target.__class__ = Target

        from asyncio import Queue

        target_run = TargetRun(target=target, event_q=Queue())
        # Assign a mock task so cancel() doesn't bail out early
        target_run.task = MagicMock()
        target_run.task.done.return_value = False
        target_run.task.cancel.return_value = True

        step_a = MagicMock()
        step_a.task = MagicMock()
        step_a.task.done.return_value = False

        step_b = MagicMock()
        step_b.task = None  # not-yet-started step

        target_run.target_step_runs = {step_a, step_b}

        target_run.cancel()

        step_a.task.cancel.assert_called_once()
        # step_b has no task — must not raise
