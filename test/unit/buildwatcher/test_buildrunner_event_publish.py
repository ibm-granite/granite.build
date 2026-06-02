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
Unit tests for BuildRunner.__dispatch_to_event_bus integration.
"""

import asyncio
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


class TestDispatchToEventBus:
    """Tests for BuildRunner.__dispatch_to_event_bus."""

    @pytest.mark.asyncio
    async def test_dispatch_disabled_returns_immediately(self):
        """When GBSERVER_EVENT_PUBLISHING_ENABLED is False, dispatch is a no-op."""
        from gbserver.buildrunner.buildrunner import BuildRunner

        runner = MagicMock(spec=BuildRunner)
        runner._event_publisher = None

        event = _make_status_event()

        with patch(
            "gbserver.buildrunner.buildrunner.GBSERVER_EVENT_PUBLISHING_ENABLED",
            False,
        ):
            # Call the unbound method directly
            await BuildRunner._BuildRunner__dispatch_to_event_bus(runner, event)

        # Publisher should not have been created
        assert runner._event_publisher is None

    @pytest.mark.asyncio
    async def test_dispatch_enabled_initializes_publisher_on_first_call(self):
        """When enabled, first call creates and sets up the publisher."""
        from gbserver.buildrunner.buildrunner import BuildRunner

        runner = MagicMock(spec=BuildRunner)
        runner._event_publisher = None

        mock_publisher = AsyncMock()
        mock_publisher.setup = AsyncMock()
        mock_publisher.publish_event = AsyncMock()

        event = _make_status_event()

        with patch(
            "gbserver.buildrunner.buildrunner.GBSERVER_EVENT_PUBLISHING_ENABLED",
            True,
        ), patch(
            "gbserver.messaging.build_event_publisher.BuildEventPublisher.from_env",
            return_value=mock_publisher,
        ):
            await BuildRunner._BuildRunner__dispatch_to_event_bus(runner, event)

        mock_publisher.setup.assert_called_once()
        mock_publisher.publish_event.assert_called_once_with(event)
        assert runner._event_publisher is mock_publisher

    @pytest.mark.asyncio
    async def test_dispatch_reuses_existing_publisher(self):
        """On subsequent calls, the existing publisher is reused (no re-init)."""
        from gbserver.buildrunner.buildrunner import BuildRunner

        runner = MagicMock(spec=BuildRunner)
        mock_publisher = AsyncMock()
        mock_publisher.publish_event = AsyncMock()
        runner._event_publisher = mock_publisher

        event = _make_status_event()

        with patch(
            "gbserver.buildrunner.buildrunner.GBSERVER_EVENT_PUBLISHING_ENABLED",
            True,
        ):
            await BuildRunner._BuildRunner__dispatch_to_event_bus(runner, event)

        # setup should NOT be called again
        mock_publisher.setup.assert_not_called()
        mock_publisher.publish_event.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_dispatch_swallows_publish_error(self):
        """Publish errors are logged as warnings, not raised."""
        from gbserver.buildrunner.buildrunner import BuildRunner

        runner = MagicMock(spec=BuildRunner)
        mock_publisher = AsyncMock()
        mock_publisher.publish_event = AsyncMock(
            side_effect=ConnectionError("RabbitMQ unavailable")
        )
        runner._event_publisher = mock_publisher

        event = _make_status_event()

        with patch(
            "gbserver.buildrunner.buildrunner.GBSERVER_EVENT_PUBLISHING_ENABLED",
            True,
        ):
            # Should NOT raise
            await BuildRunner._BuildRunner__dispatch_to_event_bus(runner, event)

        mock_publisher.publish_event.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_dispatch_swallows_setup_error(self):
        """If publisher.setup() fails, error is caught and logged."""
        from gbserver.buildrunner.buildrunner import BuildRunner

        runner = MagicMock(spec=BuildRunner)
        runner._event_publisher = None

        mock_publisher = AsyncMock()
        mock_publisher.setup = AsyncMock(side_effect=RuntimeError("connection refused"))
        mock_publisher.publish_event = AsyncMock()

        event = _make_status_event()

        with patch(
            "gbserver.buildrunner.buildrunner.GBSERVER_EVENT_PUBLISHING_ENABLED",
            True,
        ), patch(
            "gbserver.messaging.build_event_publisher.BuildEventPublisher.from_env",
            return_value=mock_publisher,
        ):
            # Should NOT raise
            await BuildRunner._BuildRunner__dispatch_to_event_bus(runner, event)

        # Setup was attempted
        mock_publisher.setup.assert_called_once()
        # publish_event should NOT be called since setup failed
        mock_publisher.publish_event.assert_not_called()


class TestDispatchToEventBusWiring:
    """Tests that __dispatch_to_event_bus is wired into the event processing path."""

    @pytest.mark.asyncio
    async def test_process_event_schedules_event_bus_dispatch(self):
        """Verify that __process_event schedules __dispatch_to_event_bus."""
        from gbserver.buildrunner.buildrunner import BuildRunner

        runner = MagicMock(spec=BuildRunner)
        runner._event_publisher = None

        event = _make_status_event()

        # We patch asyncio.ensure_future at the module level to verify it's called
        with patch(
            "gbserver.buildrunner.buildrunner.asyncio.ensure_future"
        ) as mock_ensure_future, patch(
            "gbserver.buildrunner.buildrunner.GBSERVER_EVENT_PUBLISHING_ENABLED",
            True,
        ):
            # Set up minimal mocks for __process_event to work
            runner.stored_build = MagicMock()
            runner.stored_build.status.is_finished.return_value = False
            runner.event_storage = MagicMock()

            # Call __process_event - it should schedule event bus dispatch
            BuildRunner._BuildRunner__process_event(runner, event)

            # ensure_future should have been called (event bus + standalone dispatch)
            assert mock_ensure_future.call_count >= 1
