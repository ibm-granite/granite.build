"""Tests for single-record failure-traceback logging (gbserver.build.run).

The log pipeline ingests one record per LINE, so a multi-line trace is split into
N records that get reordered — why traces looked missing from the runner log.
``_log_failure_trace`` collapses the trace into one record.
"""

import logging

import pytest

from gbserver.build.run import (
    _TRACE_LOG_MAX_CHARS,
    _TRACE_MARKER,
    _log_failure_trace,
)

MULTI_LINE_TRACE = (
    "Traceback (most recent call last):\n"
    '  File "/app/src/gbserver/environment/skypilot.py", line 2051, in _provision\n'
    "    request_id = await asyncio.shield(launch_fut)\n"
    "ValueError: Failed to get partitions for cluster bluevela\n"
)


def test_trace_is_emitted_as_exactly_one_record(caplog):
    """The whole trace occupies one record — the actual fix."""
    with caplog.at_level(logging.ERROR, logger="gbserver.build.run"):
        _log_failure_trace(MULTI_LINE_TRACE, "step-123")

    records = [r for r in caplog.records if _TRACE_MARKER in r.getMessage()]
    assert len(records) == 1
    message = records[0].getMessage()
    assert "\n" not in message, "trace must be collapsed to one physical line"
    assert "\r" not in message


def test_trace_preserves_frames_and_entity_id(caplog):
    """Collapsing must not lose content: every frame stays, in order."""
    with caplog.at_level(logging.ERROR, logger="gbserver.build.run"):
        _log_failure_trace(MULTI_LINE_TRACE, "step-123")

    message = next(
        r.getMessage() for r in caplog.records if _TRACE_MARKER in r.getMessage()
    )
    assert "step-123" in message
    assert "skypilot.py" in message
    assert "Failed to get partitions for cluster bluevela" in message
    # Escaped newlines keep frame boundaries legible and are trivially reversed.
    assert "\\n" in message
    assert message.index("Traceback") < message.index("ValueError")


def test_escaped_trace_round_trips():
    """The escaping must be reversible so a reader recovers the original trace."""
    escaped = (
        MULTI_LINE_TRACE.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "")
    )
    assert escaped.replace("\\n", "\n").replace("\\\\", "\\") == MULTI_LINE_TRACE


def test_backslashes_in_trace_are_not_ambiguous(caplog):
    """A literal backslash in the trace must not be confused with an escape."""
    trace = 'File "C:\\path\\to\\file.py"\nValueError: boom\n'
    with caplog.at_level(logging.ERROR, logger="gbserver.build.run"):
        _log_failure_trace(trace, "step-win")
    message = next(
        r.getMessage() for r in caplog.records if _TRACE_MARKER in r.getMessage()
    )
    assert "\n" not in message
    # Literal backslashes are doubled, so "\\n" (escaped newline) stays distinct
    # from a literal backslash followed by an 'n'.
    assert "C:\\\\path\\\\to\\\\file.py" in message


def test_long_trace_is_truncated_with_total_size(caplog):
    """A pathological trace is capped so it cannot flood the log pipeline."""
    huge = "x" * (_TRACE_LOG_MAX_CHARS + 5000)
    with caplog.at_level(logging.ERROR, logger="gbserver.build.run"):
        _log_failure_trace(huge, "step-huge")

    message = next(
        r.getMessage() for r in caplog.records if _TRACE_MARKER in r.getMessage()
    )
    assert "truncated" in message
    assert str(len(huge)) in message
    assert len(message) < _TRACE_LOG_MAX_CHARS + 500


@pytest.mark.parametrize("value", ["", None])
def test_empty_trace_does_not_raise(caplog, value):
    """A missing trace must never turn a build failure into a logging crash."""
    with caplog.at_level(logging.ERROR, logger="gbserver.build.run"):
        _log_failure_trace(value, "step-empty")
