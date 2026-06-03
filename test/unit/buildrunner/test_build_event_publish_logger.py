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

"""Tests for BuildEventPublishLogger."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gbserver.buildrunner.build_event_publish_logger import BuildEventPublishLogger
from gbserver.storage.stored_build import StoredBuild
from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventStatusPayload,
    BuildEventType,
    BuildLogLevel,
    EntityRunMetadata,
)
from gbserver.types.status import Status


@pytest.fixture
def mock_stored_build():
    build = MagicMock(spec=StoredBuild)
    build.uuid = "build-abc-123"
    build.username = "testuser"
    return build


@pytest.fixture
def mock_publisher():
    publisher = AsyncMock()
    publisher.setup = AsyncMock()
    publisher.publish_event = AsyncMock()
    return publisher


class TestBuildEventPublishLogger:
    """Tests for BuildEventPublishLogger."""

    @pytest.mark.asyncio
    async def test_log_publishes_status_event_to_rabbitmq(
        self, mock_stored_build, mock_publisher
    ):
        """A status triggering_event is forwarded to the publisher."""
        pub_logger = BuildEventPublishLogger(
            stored_build=mock_stored_build, publisher=mock_publisher
        )

        status_event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="build-abc-123"),
            type=BuildEventType.STATUS_EVENT,
            payload=BuildEventStatusPayload(status=Status.RUNNING, msg="running"),
            timestamp=datetime(2026, 6, 3, 10, 0, 0),
            source="build-runner",
        )

        pub_logger._log(
            level=BuildLogLevel.INFO,
            markdown="Build started",
            triggering_event=status_event,
        )

        # Give the fire-and-forget task a chance to run
        await asyncio.sleep(0)

        mock_publisher.setup.assert_called_once()
        mock_publisher.publish_event.assert_called_once_with(status_event)

    @pytest.mark.asyncio
    async def test_log_skips_non_status_events(
        self, mock_stored_build, mock_publisher
    ):
        """Non-status triggering events are NOT published."""
        pub_logger = BuildEventPublishLogger(
            stored_build=mock_stored_build, publisher=mock_publisher
        )

        message_event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="build-abc-123"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=None,
            source="build-runner",
        )

        pub_logger._log(
            level=BuildLogLevel.INFO,
            markdown="Some message",
            triggering_event=message_event,
        )

        await asyncio.sleep(0)

        mock_publisher.publish_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_log_skips_when_no_triggering_event(
        self, mock_stored_build, mock_publisher
    ):
        """If triggering_event is None, nothing is published."""
        pub_logger = BuildEventPublishLogger(
            stored_build=mock_stored_build, publisher=mock_publisher
        )

        pub_logger._log(
            level=BuildLogLevel.WARNING,
            markdown="Some warning",
            triggering_event=None,
        )

        await asyncio.sleep(0)

        mock_publisher.publish_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_publish_failure_does_not_raise(
        self, mock_stored_build, mock_publisher
    ):
        """Publisher errors are swallowed (fire-and-forget)."""
        mock_publisher.publish_event = AsyncMock(
            side_effect=ConnectionError("RabbitMQ down")
        )
        pub_logger = BuildEventPublishLogger(
            stored_build=mock_stored_build, publisher=mock_publisher
        )

        status_event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="build-abc-123"),
            type=BuildEventType.STATUS_EVENT,
            payload=BuildEventStatusPayload(status=Status.FAILED, msg="oom"),
            source="build-runner",
        )

        # Should not raise
        pub_logger._log(
            level=BuildLogLevel.ERROR,
            markdown="Build failed",
            triggering_event=status_event,
        )

        await asyncio.sleep(0)

    @pytest.mark.asyncio
    async def test_setup_called_only_once(
        self, mock_stored_build, mock_publisher
    ):
        """setup() is called on first publish, not subsequent ones."""
        pub_logger = BuildEventPublishLogger(
            stored_build=mock_stored_build, publisher=mock_publisher
        )

        status_event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="build-abc-123"),
            type=BuildEventType.STATUS_EVENT,
            payload=BuildEventStatusPayload(status=Status.RUNNING, msg="running"),
            source="build-runner",
        )

        pub_logger._log(level=BuildLogLevel.INFO, markdown="first", triggering_event=status_event)
        await asyncio.sleep(0)

        pub_logger._log(level=BuildLogLevel.INFO, markdown="second", triggering_event=status_event)
        await asyncio.sleep(0)

        # setup called only once, publish called twice
        mock_publisher.setup.assert_called_once()
        assert mock_publisher.publish_event.call_count == 2


class TestGetMessageLoggerIntegration:
    """Test that get_message_logger includes BuildEventPublishLogger when enabled."""

    def test_includes_publish_logger_when_enabled(self, mock_stored_build):
        """When GBSERVER_EVENT_PUBLISHING_ENABLED=True and RabbitMQ configured, publish logger included."""
        with patch(
            "gbserver.buildrunner.buildlogger.GBSERVER_EVENT_PUBLISHING_ENABLED", True
        ), patch(
            "gbserver.messaging.build_event_publisher.BuildEventPublisher.is_configured",
            return_value=True,
        ), patch(
            "gbserver.buildrunner.buildlogger.singleton_storage"
        ) as mock_storage:
            mock_storage.get_admin_storage.return_value.event_storage = MagicMock()
            mock_stored_build.source_uri = None

            from gbserver.buildrunner.buildlogger import get_message_logger

            get_message_logger.cache_clear()
            result = get_message_logger(mock_stored_build, "build-runner")

            from gbserver.buildrunner.build_event_publish_logger import (
                BuildEventPublishLogger,
            )
            from gbserver.buildrunner.buildlogger import BuildMultiMessageLogger

            assert isinstance(result, BuildMultiMessageLogger)
            has_publish_logger = any(
                isinstance(l, BuildEventPublishLogger) for l in result._loggers
            )
            assert has_publish_logger

    def test_excludes_publish_logger_when_disabled(self, mock_stored_build):
        """When GBSERVER_EVENT_PUBLISHING_ENABLED=False, no publish logger."""
        with patch(
            "gbserver.buildrunner.buildlogger.GBSERVER_EVENT_PUBLISHING_ENABLED", False
        ), patch(
            "gbserver.buildrunner.buildlogger.singleton_storage"
        ) as mock_storage:
            mock_storage.get_admin_storage.return_value.event_storage = MagicMock()
            mock_stored_build.source_uri = None

            from gbserver.buildrunner.buildlogger import get_message_logger

            get_message_logger.cache_clear()
            result = get_message_logger(mock_stored_build, "build-runner")

            from gbserver.buildrunner.build_event_publish_logger import (
                BuildEventPublishLogger,
            )

            # Should be a single logger (BuildEventMessageLogger), not multi
            if hasattr(result, "_loggers"):
                assert not any(
                    isinstance(l, BuildEventPublishLogger) for l in result._loggers
                )

    def test_excludes_publish_logger_when_rabbitmq_not_configured(self, mock_stored_build):
        """When enabled but RABBITMQ_HOST unset, publish logger is skipped."""
        with patch(
            "gbserver.buildrunner.buildlogger.GBSERVER_EVENT_PUBLISHING_ENABLED", True
        ), patch(
            "gbserver.messaging.build_event_publisher.BuildEventPublisher.is_configured",
            return_value=False,
        ), patch(
            "gbserver.buildrunner.buildlogger.singleton_storage"
        ) as mock_storage:
            mock_storage.get_admin_storage.return_value.event_storage = MagicMock()
            mock_stored_build.source_uri = None

            from gbserver.buildrunner.buildlogger import get_message_logger

            get_message_logger.cache_clear()
            result = get_message_logger(mock_stored_build, "build-runner")

            from gbserver.buildrunner.build_event_publish_logger import (
                BuildEventPublishLogger,
            )

            if hasattr(result, "_loggers"):
                assert not any(
                    isinstance(l, BuildEventPublishLogger) for l in result._loggers
                )
