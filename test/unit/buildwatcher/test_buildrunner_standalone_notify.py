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
Unit tests for BuildRunner.__dispatch_standalone_notification.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventStatusPayload,
    BuildEventType,
    EntityRunMetadata,
)
from gbserver.types.status import Status


def _make_status_event(build_id: str = "build-123") -> BuildEvent:
    """Helper to create a sample status BuildEvent."""
    return BuildEvent(
        run_metadata=EntityRunMetadata(
            build_id=build_id,
            target_name="my-target",
            targetstep_uri="train-step",
        ),
        type=BuildEventType.STATUS_EVENT,
        payload=BuildEventStatusPayload(
            status=Status.RUNNING,
            msg="Build is running",
        ),
        timestamp=datetime(2026, 1, 15, 10, 30, 0),
        source="build-framework",
    )


class TestDispatchStandaloneNotification:
    """Tests for BuildRunner.__dispatch_standalone_notification."""

    @pytest.mark.asyncio
    async def test_dispatch_noop_when_not_standalone(self):
        """When GB_ENVIRONMENT is not STANDALONE, dispatch is a no-op."""
        from gbserver.buildrunner.buildrunner import BuildRunner

        runner = MagicMock(spec=BuildRunner)
        runner._standalone_dispatcher = None

        event = _make_status_event()

        with patch(
            "gbserver.buildrunner.buildrunner.GB_ENVIRONMENT",
            "PROD",
        ), patch(
            "gbserver.buildrunner.buildrunner.GBSERVER_EVENT_PUBLISHING_ENABLED",
            False,
        ):
            await BuildRunner._BuildRunner__dispatch_standalone_notification(
                runner, event
            )

        # Dispatcher should not have been created
        assert runner._standalone_dispatcher is None

    @pytest.mark.asyncio
    async def test_dispatch_noop_when_event_publishing_enabled(self):
        """When GBSERVER_EVENT_PUBLISHING_ENABLED is True, dispatch is a no-op."""
        from gbserver.buildrunner.buildrunner import BuildRunner

        runner = MagicMock(spec=BuildRunner)
        runner._standalone_dispatcher = None

        event = _make_status_event()

        with patch(
            "gbserver.buildrunner.buildrunner.GB_ENVIRONMENT",
            "STANDALONE",
        ), patch(
            "gbserver.buildrunner.buildrunner.GBSERVER_EVENT_PUBLISHING_ENABLED",
            True,
        ):
            await BuildRunner._BuildRunner__dispatch_standalone_notification(
                runner, event
            )

        # Dispatcher should not have been created
        assert runner._standalone_dispatcher is None

    @pytest.mark.asyncio
    async def test_dispatch_calls_standalone_dispatcher(self):
        """When STANDALONE + publishing disabled, dispatches via StandaloneDispatcher."""
        from gbserver.buildrunner.buildrunner import BuildRunner

        runner = MagicMock(spec=BuildRunner)
        runner._standalone_dispatcher = None

        mock_dispatcher = AsyncMock()
        mock_dispatcher.dispatch = AsyncMock()

        event = _make_status_event()

        with patch(
            "gbserver.buildrunner.buildrunner.GB_ENVIRONMENT",
            "STANDALONE",
        ), patch(
            "gbserver.buildrunner.buildrunner.GBSERVER_EVENT_PUBLISHING_ENABLED",
            False,
        ), patch(
            "gbserver.notifications.dispatcher.StandaloneDispatcher",
            return_value=mock_dispatcher,
        ):
            await BuildRunner._BuildRunner__dispatch_standalone_notification(
                runner, event
            )

        mock_dispatcher.dispatch.assert_called_once_with(event)
        assert runner._standalone_dispatcher is mock_dispatcher

    @pytest.mark.asyncio
    async def test_dispatch_reuses_existing_dispatcher(self):
        """On subsequent calls, the existing dispatcher is reused."""
        from gbserver.buildrunner.buildrunner import BuildRunner

        runner = MagicMock(spec=BuildRunner)
        mock_dispatcher = AsyncMock()
        mock_dispatcher.dispatch = AsyncMock()
        runner._standalone_dispatcher = mock_dispatcher

        event = _make_status_event()

        with patch(
            "gbserver.buildrunner.buildrunner.GB_ENVIRONMENT",
            "STANDALONE",
        ), patch(
            "gbserver.buildrunner.buildrunner.GBSERVER_EVENT_PUBLISHING_ENABLED",
            False,
        ):
            await BuildRunner._BuildRunner__dispatch_standalone_notification(
                runner, event
            )

        mock_dispatcher.dispatch.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_dispatch_swallows_errors(self):
        """Errors from the dispatcher are caught and do not propagate."""
        from gbserver.buildrunner.buildrunner import BuildRunner

        runner = MagicMock(spec=BuildRunner)
        mock_dispatcher = AsyncMock()
        mock_dispatcher.dispatch = AsyncMock(
            side_effect=RuntimeError("notification failed")
        )
        runner._standalone_dispatcher = mock_dispatcher

        event = _make_status_event()

        with patch(
            "gbserver.buildrunner.buildrunner.GB_ENVIRONMENT",
            "STANDALONE",
        ), patch(
            "gbserver.buildrunner.buildrunner.GBSERVER_EVENT_PUBLISHING_ENABLED",
            False,
        ):
            # Should NOT raise
            await BuildRunner._BuildRunner__dispatch_standalone_notification(
                runner, event
            )

        mock_dispatcher.dispatch.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_dispatch_swallows_init_error(self):
        """If StandaloneDispatcher() init fails, error is caught."""
        from gbserver.buildrunner.buildrunner import BuildRunner

        runner = MagicMock(spec=BuildRunner)
        runner._standalone_dispatcher = None

        event = _make_status_event()

        with patch(
            "gbserver.buildrunner.buildrunner.GB_ENVIRONMENT",
            "STANDALONE",
        ), patch(
            "gbserver.buildrunner.buildrunner.GBSERVER_EVENT_PUBLISHING_ENABLED",
            False,
        ), patch(
            "gbserver.notifications.dispatcher.StandaloneDispatcher",
            side_effect=RuntimeError("config file missing"),
        ):
            # Should NOT raise
            await BuildRunner._BuildRunner__dispatch_standalone_notification(
                runner, event
            )

        # Dispatcher should remain None since init failed
        assert runner._standalone_dispatcher is None
