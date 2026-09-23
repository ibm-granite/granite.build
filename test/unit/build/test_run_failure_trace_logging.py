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


@pytest.mark.parametrize(
    "filler,label",
    [
        ("\\", "backslash-heavy"),
        ("\n", "newline-heavy"),
    ],
)
def test_escape_expansion_cannot_exceed_the_cap(caplog, filler, label):
    """The cap must bound the ESCAPED record, not the raw input.

    Escaping doubles every backslash and newline, so capping the raw text first
    let a pathological trace emit a record up to 2x the limit — the flood the cap
    exists to prevent. Windows paths and repr()'d regexes produce exactly this.
    """
    huge = filler * (_TRACE_LOG_MAX_CHARS + 5000)
    with caplog.at_level(logging.ERROR, logger="gbserver.build.run"):
        _log_failure_trace(huge, f"step-{label}")

    message = next(
        r.getMessage() for r in caplog.records if _TRACE_MARKER in r.getMessage()
    )
    assert "truncated" in message
    assert str(len(huge)) in message, "reports the true pre-escape size"
    assert len(message) < _TRACE_LOG_MAX_CHARS + 500


def test_truncation_never_splits_an_escape_pair(caplog):
    """A cut must not leave a dangling backslash.

    Slicing escaped text can land mid-pair ("\\\\" -> "\\"), which would make the
    tail un-escape to something the original never contained.
    """
    # Place a backslash so its escaped pair straddles the cap boundary.
    raw = "a" * (_TRACE_LOG_MAX_CHARS - 1) + "\\" + "b" * 100
    with caplog.at_level(logging.ERROR, logger="gbserver.build.run"):
        _log_failure_trace(raw, "step-split")

    message = next(
        r.getMessage() for r in caplog.records if _TRACE_MARKER in r.getMessage()
    )
    body = message.split("... [truncated", 1)[0]
    trailing = len(body) - len(body.rstrip("\\"))
    assert trailing % 2 == 0, "odd trailing backslashes = split escape pair"
