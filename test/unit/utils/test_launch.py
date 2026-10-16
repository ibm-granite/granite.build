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

"""Unit tests for command launching and connection-closed retry classification."""

from unittest.mock import AsyncMock, patch

import pytest

from gbserver.types.errors import ErrConnectionClosed
from gbserver.utils.launch import (
    launch_command_and_raise_errors,
    launch_command_and_retry_or_raise_errors,
)

# The two dropped-connection stderr shapes seen from a failing scp over the
# LSF login-node tunnel (real bluevela log11 lines).
SCP_CONNECTION_CLOSED = "scp: Connection closed\n"
REMOTE_CLOSED = "Connection to localhost closed by remote host.\n"


class TestErrConnectionClosedMatcher:
    """The narrow matcher recognizes dropped connections but not benign text."""

    @pytest.mark.parametrize(
        "stderr",
        [SCP_CONNECTION_CLOSED, REMOTE_CLOSED, SCP_CONNECTION_CLOSED.encode("utf-8")],
    )
    def test_matches_dropped_connection(self, stderr):
        assert ErrConnectionClosed.matches_error_str(stderr) is True

    @pytest.mark.parametrize(
        "stderr",
        ["some unrelated failure", "Warning: Permanently added host", ""],
    )
    def test_ignores_unrelated_stderr(self, stderr):
        assert ErrConnectionClosed.matches_error_str(stderr) is False


def _mock_process(returncode: int, stderr: str) -> AsyncMock:
    """Build a mock subprocess whose communicate() yields the given stderr/rc.

    Args:
        returncode: Exit code the mock process reports.
        stderr: stderr text the mock process emits (stdout is empty).

    Returns:
        AsyncMock: A stand-in for an asyncio subprocess.
    """
    proc = AsyncMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(b"", stderr.encode("utf-8")))
    return proc


@pytest.mark.asyncio
async def test_raise_errors_maps_connection_closed_to_retryable():
    """A dropped-connection scp failure raises the retryable ErrConnectionClosed
    rather than a bare, non-retryable ValueError."""
    proc = _mock_process(returncode=1, stderr=SCP_CONNECTION_CLOSED)
    with patch("asyncio.create_subprocess_exec", return_value=proc):
        with pytest.raises(ErrConnectionClosed):
            await launch_command_and_raise_errors(
                command_list=["scp", "-r", "src", "host:dest"],
                launch_id="test-launch",
            )


@pytest.mark.asyncio
async def test_retry_recovers_after_connection_closed():
    """launch_command_and_retry_or_raise_errors retries a dropped connection and
    succeeds once the transfer goes through."""
    fail = _mock_process(returncode=1, stderr=REMOTE_CLOSED)
    ok = _mock_process(returncode=0, stderr="")

    # Skip tenacity's exponential backoff so the test runs instantly.
    with (
        patch("asyncio.create_subprocess_exec", side_effect=[fail, ok]) as exec_mock,
        patch("asyncio.sleep", new=AsyncMock()),
    ):
        process, _stdout, _stderr = await launch_command_and_retry_or_raise_errors(
            command_list=["scp", "-r", "src", "host:dest"],
            launch_id="test-launch",
        )

    assert process.returncode == 0
    assert exec_mock.call_count == 2  # failed once, retried, then succeeded
