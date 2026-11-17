#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""
Tests for LsfTransientErrorRetryStrategy.
"""

from typing import Self

import pytest

from gbserver.resilience.strategies.lsf_transient_error import (
    LsfTransientErrorRetryStrategy,
)
from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventMessagePayload,
    BuildEventType,
    EntityRunMetadata,
)
from gbserver.types.errors import ERR_LSF_CANNOT_OPEN_JOB_FILE


class TestLsfTransientErrorRetryStrategy:
    """Tests for LSF transient error detection strategy."""

    def test_detects_cannot_open_job_file(self: Self) -> None:
        """Test detection of the canonical LSF transient error."""
        strategy = LsfTransientErrorRetryStrategy()

        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=BuildEventMessagePayload(msg=ERR_LSF_CANNOT_OPEN_JOB_FILE),
        )

        assert strategy.should_retry(event)

    def test_detects_error_in_longer_message(self: Self) -> None:
        """Test detection when the error string is embedded in a longer message."""
        strategy = LsfTransientErrorRetryStrategy()

        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=BuildEventMessagePayload(
                msg=f"bsub: error: {ERR_LSF_CANNOT_OPEN_JOB_FILE} for job 12345"
            ),
        )

        assert strategy.should_retry(event)

    def test_ignores_non_message_events(self: Self) -> None:
        """Test that non-MESSAGE_EVENT types are ignored."""
        strategy = LsfTransientErrorRetryStrategy()

        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.STATUS_EVENT,
            payload=BuildEventMessagePayload(msg=ERR_LSF_CANNOT_OPEN_JOB_FILE),
        )

        assert not strategy.should_retry(event)

    def test_ignores_unrelated_errors(self: Self) -> None:
        """Test that unrelated error messages do not trigger retry."""
        strategy = LsfTransientErrorRetryStrategy()

        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=BuildEventMessagePayload(msg="Job 12345 exited with status 1"),
        )

        assert not strategy.should_retry(event)

    def test_ignores_none_payload(self: Self) -> None:
        """Test that events with None payload are handled gracefully."""
        strategy = LsfTransientErrorRetryStrategy()

        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=None,  # type: ignore[arg-type]
        )

        assert not strategy.should_retry(event)

    def test_get_retry_delay_returns_float(self: Self) -> None:
        """Test that get_retry_delay returns a float."""
        strategy = LsfTransientErrorRetryStrategy()
        delay = strategy.get_retry_delay(retry_count=0)
        assert isinstance(delay, float)
        assert delay >= 0
