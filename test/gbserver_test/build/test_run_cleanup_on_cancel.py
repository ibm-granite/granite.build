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

"""Tests that Run._cleanup() completes even when the run task is cancelled.

The bug (issue #1880): When a Run is cancelled, CancelledError is caught in
Run.run() and re-raised. If the task receives another cancellation (e.g., from
a TaskGroup or repeated cancel calls), the ``await`` inside ``_cleanup()`` in
the finally block is interrupted, leaving K8s resources (helm releases, pods)
behind.

The fix (Task 2) will wrap the ``_cleanup()`` call in ``asyncio.shield()`` so
that cleanup always runs to completion regardless of further cancellations.
"""

import asyncio
from asyncio import TaskGroup
from pathlib import Path
from typing import Optional, Self
from unittest.mock import MagicMock

import pytest

from gbserver.build.run import Run
from gbserver.types.buildevent import EntityRunMetadata


class CleanupTrackingRun(Run):
    """A Run subclass that tracks whether _cleanup() completed."""

    def __init__(self):
        # Create a mock entity with just the attributes Run.__init__ needs
        entity = MagicMock()
        entity.build_id = "test-build"
        super().__init__(entity=entity, base_dir=Path("/tmp/test-run"))
        self.cleanup_started = False
        self.cleanup_completed = False
        self.cleanup_started_event = asyncio.Event()

    async def _run(self: Self, tg: Optional[TaskGroup] = None, **kwargs) -> None:
        # Simulate a long-running operation that will be cancelled
        await asyncio.sleep(3600)

    async def _cleanup(self: Self, tg: Optional[TaskGroup] = None, **kwargs) -> None:
        self.cleanup_started = True
        self.cleanup_started_event.set()
        # Simulate async cleanup work (e.g., helm uninstall subprocess).
        # This await will be interrupted if the task is cancelled again
        # while the finally block is executing.
        await asyncio.sleep(0.5)
        self.cleanup_completed = True

    def get_runmetadata(self: Self) -> EntityRunMetadata:
        return EntityRunMetadata(build_id="test-build")


class TestRunCleanupOnCancel:
    """Tests that _cleanup() completes even when the run is cancelled."""

    @pytest.mark.asyncio
    async def test_cleanup_completes_after_cancellation(self: Self) -> None:
        """When a run is cancelled and a second cancellation arrives during
        the finally block, _cleanup() should still run to completion.

        This simulates the real-world scenario where a TaskGroup or external
        caller cancels the task, and the cancellation propagates into the
        finally block's await points.
        """
        run = CleanupTrackingRun()
        task = asyncio.create_task(run.run())

        # Let the run start
        await asyncio.sleep(0.05)

        # Cancel the task (first cancellation — caught by except CancelledError)
        task.cancel()

        # Wait until _cleanup() has started (deterministic synchronization)
        await run.cleanup_started_event.wait()

        # Cancel again while the finally block is executing _cleanup().
        # This simulates a TaskGroup propagating cancellation or a repeated
        # cancel call, which will interrupt the await inside _cleanup().
        task.cancel()

        # Wait for the task to finish (will raise CancelledError).
        # With the robust shield pattern, cleanup is fully awaited before
        # the CancelledError propagates — no sleep needed.
        with pytest.raises(asyncio.CancelledError):
            await task

        assert run.cleanup_started, "_cleanup() should have started"
        assert (
            run.cleanup_completed
        ), "_cleanup() should have completed (not interrupted by CancelledError)"
