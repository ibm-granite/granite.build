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

import asyncio
from typing import Self
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gbserver.environment.lsf import Lsf
from gbserver.types.errors import WorkloadFailedException


def _make_lsf(use_ssh: bool = True) -> Lsf:
    """Create a minimal Lsf instance with mocked constructor dependencies."""
    with patch.object(Lsf, "__init__", lambda self, **kw: None):
        lsf = Lsf.__new__(Lsf)
    # Set minimum required attributes
    lsf.use_ssh = use_ssh
    lsf._launched_jobs = {}
    lsf._existing_jobids = {}
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
        lsf._launched_jobs["launch-1"] = "12345"
        lsf._ssh_tunnel.run_remote = AsyncMock(
            return_value=(0, "Job <12345> is being terminated\n", "")
        )

        await lsf.cleanup_bsub(
            launch_id="launch-1",
            run_metadata={"build_id": "b1"},
        )

        lsf._send_message.assert_called_once()
        msg = (
            lsf._send_message.call_args[1].get("msg")
            or lsf._send_message.call_args[0][0]
        )
        assert "Killed LSF job 12345" in msg

    @pytest.mark.asyncio
    async def test_failed_bkill_reports_failure(self: Self) -> None:
        """When bkill fails (rc!=0), should report failure, not success."""
        lsf = _make_lsf(use_ssh=True)
        lsf._launched_jobs["launch-1"] = "12345"
        lsf._ssh_tunnel.run_remote = AsyncMock(
            return_value=(255, "", "Permission denied")
        )

        await lsf.cleanup_bsub(
            launch_id="launch-1",
            run_metadata={"build_id": "b1"},
        )

        lsf._send_message.assert_called_once()
        msg = (
            lsf._send_message.call_args[1].get("msg")
            or lsf._send_message.call_args[0][0]
        )
        assert "Failed to kill LSF job 12345" in msg
        assert "Killed LSF job" not in msg

    @pytest.mark.asyncio
    async def test_job_already_finished_no_message(self: Self) -> None:
        """When bkill says 'Job has already finished', no message sent."""
        lsf = _make_lsf(use_ssh=True)
        lsf._launched_jobs["launch-1"] = "12345"
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
        lsf._launched_jobs["launch-1"] = "12345"
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
        msg = (
            lsf._send_message.call_args[1].get("msg")
            or lsf._send_message.call_args[0][0]
        )
        assert "Killed LSF job 12345" in msg

    @pytest.mark.asyncio
    async def test_timeout_exhausts_retries(self: Self) -> None:
        """After 3 timeout failures, should raise TimeoutError."""
        lsf = _make_lsf(use_ssh=True)
        lsf._launched_jobs["launch-1"] = "12345"
        lsf._ssh_tunnel.run_remote = AsyncMock(
            side_effect=TimeoutError("ssh timed out")
        )

        with pytest.raises(RuntimeError):
            await lsf.cleanup_bsub(
                launch_id="launch-1",
                run_metadata={"build_id": "b1"},
            )

        assert lsf._ssh_tunnel.run_remote.call_count == 3


class TestRetryPendingAfterMonitor:
    """Tests for _retry_pending_after_monitor, the race guard that stops
    monitor_bsub_monitor from declaring success while a RetryHandler-owned
    relaunch (triggered by an emitted error event) is still in flight.
    """

    @staticmethod
    def _monitor(emitted_error_event: bool) -> MagicMock:
        """Build a stand-in LSFBsubMonitor exposing only emitted_error_event."""
        monitor = MagicMock()
        monitor.emitted_error_event = emitted_error_event
        return monitor

    @pytest.mark.asyncio
    async def test_returns_false_when_no_error_emitted(self: Self) -> None:
        """A clean DONE (no error event) must not wait — returns False at once."""
        lsf = _make_lsf(use_ssh=True)
        retry_complete_event = asyncio.Event()
        handler_task = asyncio.create_task(asyncio.Event().wait())
        try:
            result = await lsf._retry_pending_after_monitor(
                lsf_bsub_monitor=self._monitor(emitted_error_event=False),
                retry_complete_event=retry_complete_event,
                handler_task=handler_task,
            )
        finally:
            handler_task.cancel()

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_handler_disabled(self: Self) -> None:
        """With retries disabled (handler_task is None) there is nothing to wait
        for, so the guard returns False even though an error was emitted."""
        lsf = _make_lsf(use_ssh=True)

        result = await lsf._retry_pending_after_monitor(
            lsf_bsub_monitor=self._monitor(emitted_error_event=True),
            retry_complete_event=asyncio.Event(),
            handler_task=None,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_true_when_relaunch_completes(self: Self) -> None:
        """When the RetryHandler relaunches (sets retry_complete_event) while we
        wait, the guard returns True so the caller monitors the new job."""
        lsf = _make_lsf(use_ssh=True)
        retry_complete_event = asyncio.Event()
        # Handler task that never finishes on its own; the relaunch signal wins.
        handler_task = asyncio.create_task(asyncio.Event().wait())

        async def _signal_relaunch() -> None:
            retry_complete_event.set()

        signal_task = asyncio.create_task(_signal_relaunch())
        try:
            result = await lsf._retry_pending_after_monitor(
                lsf_bsub_monitor=self._monitor(emitted_error_event=True),
                retry_complete_event=retry_complete_event,
                handler_task=handler_task,
            )
        finally:
            handler_task.cancel()
            await signal_task

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_handler_gives_up(self: Self) -> None:
        """When the handler task finishes without setting retry_complete_event
        (retries exhausted / gave up), the guard returns False so the caller
        stops looping and the handler's exception can propagate."""
        lsf = _make_lsf(use_ssh=True)
        retry_complete_event = asyncio.Event()

        async def _handler_gives_up() -> None:
            return None

        handler_task = asyncio.create_task(_handler_gives_up())

        result = await lsf._retry_pending_after_monitor(
            lsf_bsub_monitor=self._monitor(emitted_error_event=True),
            retry_complete_event=retry_complete_event,
            handler_task=handler_task,
        )

        assert result is False
        assert retry_complete_event.is_set() is False

    @pytest.mark.asyncio
    async def test_raises_when_adjudication_times_out(self: Self) -> None:
        """Backstop: if neither a relaunch nor the handler task resolves within
        the adjudication timeout, fail the step loudly instead of hanging.

        This guards the pathological case where the handler neither retries nor
        recognizes the emitted error as terminal.
        """
        lsf = _make_lsf(use_ssh=True)
        retry_complete_event = asyncio.Event()
        # Handler task that never finishes and never signals a relaunch.
        handler_task = asyncio.create_task(asyncio.Event().wait())

        with patch(
            "gbserver.environment.lsf.GBSERVER_LSF_RETRY_ADJUDICATION_TIMEOUT",
            0.05,
        ):
            try:
                with pytest.raises(WorkloadFailedException):
                    await lsf._retry_pending_after_monitor(
                        lsf_bsub_monitor=self._monitor(emitted_error_event=True),
                        retry_complete_event=retry_complete_event,
                        handler_task=handler_task,
                    )
            finally:
                handler_task.cancel()
