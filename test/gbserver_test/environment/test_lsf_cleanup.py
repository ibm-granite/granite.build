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

from typing import Self
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gbserver.environment.lsf import Lsf


def _make_lsf(use_ssh: bool = True) -> Lsf:
    """Create a minimal Lsf instance with mocked constructor dependencies."""
    with patch.object(Lsf, "__init__", lambda self, **kw: None):
        lsf = Lsf.__new__(Lsf)
    # Set minimum required attributes
    lsf.use_ssh = use_ssh
    lsf.launched_jobs = {}
    lsf.existing_jobids = {}
    lsf._ssh_tunnel = AsyncMock() if use_ssh else None
    lsf._send_message = MagicMock()
    lsf._dispatch_event = MagicMock()
    return lsf


class TestCleanupBsub:
    """Tests for cleanup_bsub accurate reporting and retry behavior."""

    @pytest.mark.asyncio
    async def test_successful_bkill_reports_killed(self: Self) -> None:
        """When bkill succeeds (rc=0), should report 'Killed LSF job'."""
        lsf = _make_lsf(use_ssh=True)
        lsf.launched_jobs["launch-1"] = "12345"
        lsf._ssh_tunnel.run_remote = AsyncMock(
            return_value=(0, "Job <12345> is being terminated\n", "")
        )

        await lsf.cleanup_bsub(
            launch_id="launch-1",
            run_metadata={"build_id": "b1"},
        )

        lsf._send_message.assert_called_once()
        msg = lsf._send_message.call_args[1].get("msg") or lsf._send_message.call_args[0][0]
        assert "Killed LSF job 12345" in msg

    @pytest.mark.asyncio
    async def test_failed_bkill_reports_failure(self: Self) -> None:
        """When bkill fails (rc!=0), should report failure, not success."""
        lsf = _make_lsf(use_ssh=True)
        lsf.launched_jobs["launch-1"] = "12345"
        lsf._ssh_tunnel.run_remote = AsyncMock(return_value=(255, "", "Permission denied"))

        await lsf.cleanup_bsub(
            launch_id="launch-1",
            run_metadata={"build_id": "b1"},
        )

        lsf._send_message.assert_called_once()
        msg = lsf._send_message.call_args[1].get("msg") or lsf._send_message.call_args[0][0]
        assert "Failed to kill LSF job 12345" in msg
        assert "Killed LSF job" not in msg

    @pytest.mark.asyncio
    async def test_job_already_finished_no_message(self: Self) -> None:
        """When bkill says 'Job has already finished', no message sent."""
        lsf = _make_lsf(use_ssh=True)
        lsf.launched_jobs["launch-1"] = "12345"
        lsf._ssh_tunnel.run_remote = AsyncMock(
            return_value=(255, "", "Job <12345>: Job has already finished")
        )

        await lsf.cleanup_bsub(
            launch_id="launch-1",
            run_metadata={"build_id": "b1"},
        )

        lsf._send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_retries_on_timeout(self: Self) -> None:
        """Should retry bkill up to 3 times on TimeoutError, then succeed."""
        lsf = _make_lsf(use_ssh=True)
        lsf.launched_jobs["launch-1"] = "12345"
        lsf._ssh_tunnel.run_remote = AsyncMock(
            side_effect=[
                TimeoutError("ssh timed out"),
                (0, "Job <12345> is being terminated\n", ""),
            ]
        )

        await lsf.cleanup_bsub(
            launch_id="launch-1",
            run_metadata={"build_id": "b1"},
        )

        assert lsf._ssh_tunnel.run_remote.call_count == 2
        lsf._send_message.assert_called_once()
        msg = lsf._send_message.call_args[1].get("msg") or lsf._send_message.call_args[0][0]
        assert "Killed LSF job 12345" in msg

    @pytest.mark.asyncio
    async def test_timeout_exhausts_retries(self: Self) -> None:
        """After 3 timeout failures, should raise TimeoutError."""
        lsf = _make_lsf(use_ssh=True)
        lsf.launched_jobs["launch-1"] = "12345"
        lsf._ssh_tunnel.run_remote = AsyncMock(side_effect=TimeoutError("ssh timed out"))

        with pytest.raises(RuntimeError):
            await lsf.cleanup_bsub(
                launch_id="launch-1",
                run_metadata={"build_id": "b1"},
            )

        assert lsf._ssh_tunnel.run_remote.call_count == 3
