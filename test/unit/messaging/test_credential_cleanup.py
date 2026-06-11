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

"""
Unit tests for credential_cleanup background task.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from gbserver.messaging.credential_cleanup import (
    CLEANUP_INTERVAL_SECONDS,
    run_cleanup_once,
    start_cleanup_loop,
)


class TestRunCleanupOnce:
    """Tests for run_cleanup_once."""

    @pytest.mark.asyncio
    async def test_returns_count_from_admin(self):
        """run_cleanup_once returns the count from admin.cleanup_expired_users."""
        mock_admin = AsyncMock()
        mock_admin.cleanup_expired_users.return_value = 3

        with patch(
            "gbserver.messaging.credential_cleanup._get_admin", return_value=mock_admin
        ):
            result = await run_cleanup_once()

        assert result == 3
        mock_admin.cleanup_expired_users.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_zero_on_exception(self):
        """run_cleanup_once returns 0 when an exception occurs."""
        mock_admin = AsyncMock()
        mock_admin.cleanup_expired_users.side_effect = RuntimeError("connection lost")

        with patch(
            "gbserver.messaging.credential_cleanup._get_admin", return_value=mock_admin
        ):
            result = await run_cleanup_once()

        assert result == 0

    @pytest.mark.asyncio
    async def test_returns_zero_count_without_logging_info(self):
        """run_cleanup_once returns 0 without info log when no users deleted."""
        mock_admin = AsyncMock()
        mock_admin.cleanup_expired_users.return_value = 0

        with (
            patch(
                "gbserver.messaging.credential_cleanup._get_admin",
                return_value=mock_admin,
            ),
            patch("gbserver.messaging.credential_cleanup.logger") as mock_logger,
        ):
            result = await run_cleanup_once()

        assert result == 0
        mock_logger.info.assert_not_called()


class TestStartCleanupLoop:
    """Tests for start_cleanup_loop."""

    @pytest.mark.asyncio
    async def test_calls_run_cleanup_once_repeatedly(self):
        """start_cleanup_loop calls run_cleanup_once in a loop."""
        call_count = 0

        async def fake_cleanup() -> int:
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                raise asyncio.CancelledError()
            return 0

        with (
            patch(
                "gbserver.messaging.credential_cleanup.run_cleanup_once",
                side_effect=fake_cleanup,
            ),
            patch(
                "gbserver.messaging.credential_cleanup.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            with pytest.raises(asyncio.CancelledError):
                await start_cleanup_loop()

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_sleeps_between_iterations(self):
        """start_cleanup_loop sleeps CLEANUP_INTERVAL_SECONDS between iterations."""
        call_count = 0

        async def fake_cleanup() -> int:
            nonlocal call_count
            call_count += 1
            return 0

        mock_sleep = AsyncMock(side_effect=[None, asyncio.CancelledError()])

        with (
            patch(
                "gbserver.messaging.credential_cleanup.run_cleanup_once",
                side_effect=fake_cleanup,
            ),
            patch(
                "gbserver.messaging.credential_cleanup.asyncio.sleep",
                mock_sleep,
            ),
        ):
            with pytest.raises(asyncio.CancelledError):
                await start_cleanup_loop()

        # Sleep was called with the correct interval
        mock_sleep.assert_called_with(CLEANUP_INTERVAL_SECONDS)
        assert call_count == 2
