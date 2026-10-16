"""Tests for autotune.callbacks.logging_service — buffer/flush/destination logic."""

import logging
import pickle
from unittest.mock import MagicMock

from autotune.callbacks.logging_service import (
    BufferedLogHandler,
    LogDestination,
    RecordType,
)


def _make_record(msg: str = "hello") -> logging.LogRecord:
    return logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=None,
        exc_info=None,
    )


class TestEnums:
    def test_log_destination_members(self):
        assert LogDestination.DATABASE.value == "database"
        assert LogDestination.HTTP.value == "http"

    def test_record_type_members(self):
        assert RecordType.RECORD_TRIAL.value == "record_trial"
        assert RecordType.UPDATE_STATUS.value == "update_status"
        assert RecordType.RECORD_RESULT.value == "insert_trial_result"


class TestInit:
    def test_no_destination_by_default(self):
        h = BufferedLogHandler()
        assert h.has_destination() is False
        assert h.get_destination_type() is None

    def test_database_destination(self):
        db = MagicMock()
        h = BufferedLogHandler(db=db)
        assert h.has_destination() is True
        assert h.get_destination_type() == LogDestination.DATABASE
        # create_logging_table called on init
        db.create_logging_table.assert_called_once()

    def test_endpoint_destination(self):
        h = BufferedLogHandler(endpoint_url="http://example.com")
        assert h.has_destination() is True
        assert h.get_destination_type() == LogDestination.HTTP

    def test_db_takes_precedence_over_endpoint(self):
        # Both set: db wins per _determine_destination order
        db = MagicMock()
        h = BufferedLogHandler(db=db, endpoint_url="http://example.com")
        assert h.get_destination_type() == LogDestination.DATABASE

    def test_buffer_size_minimum_one(self):
        # Constructor clamps buffer_size to >= 1
        h = BufferedLogHandler(buffer_size=0)
        assert h.buffer_size == 1


class TestEmit:
    def test_emit_without_job_id_does_not_buffer(self):
        h = BufferedLogHandler()
        h.emit(_make_record())
        assert h.get_buffer_size() == 0

    def test_emit_with_job_id_buffers(self):
        db = MagicMock()
        h = BufferedLogHandler(job_id="job-1", db=db, buffer_size=100)
        h.emit(_make_record())
        assert h.get_buffer_size() == 1

    def test_emit_buffer_flush_at_threshold(self):
        db = MagicMock()
        h = BufferedLogHandler(job_id="job-1", db=db, buffer_size=2)
        h.emit(_make_record("a"))
        assert h.get_buffer_size() == 1
        h.emit(_make_record("b"))
        # Threshold reached → flushed → buffer cleared
        assert h.get_buffer_size() == 0
        db.insert_logs.assert_called_once()

    def test_emit_auto_flush(self):
        db = MagicMock()
        h = BufferedLogHandler(job_id="job-1", db=db, buffer_size=100, auto_flush=True)
        h.emit(_make_record())
        # Auto-flush flushes immediately
        assert h.get_buffer_size() == 0


class TestFlush:
    def test_flush_empty_noop(self):
        db = MagicMock()
        h = BufferedLogHandler(job_id="job-1", db=db)
        h.flush()
        db.insert_logs.assert_not_called()

    def test_flush_without_destination_keeps_buffer(self):
        h = BufferedLogHandler(job_id="job-1")
        h.emit(_make_record())
        size_before = h.get_buffer_size()
        h.flush()
        # No destination → buffer is preserved
        assert h.get_buffer_size() == size_before

    def test_flush_clears_buffer_on_success(self):
        db = MagicMock()
        h = BufferedLogHandler(job_id="job-1", db=db, buffer_size=100)
        h.emit(_make_record())
        h.emit(_make_record())
        h.flush()
        assert h.get_buffer_size() == 0


class TestSetters:
    def test_set_database_switches_destination(self):
        h = BufferedLogHandler()
        db = MagicMock()
        h.set_database(db)
        assert h.get_destination_type() == LogDestination.DATABASE
        db.create_logging_table.assert_called_once()

    def test_set_endpoint_switches_destination(self):
        h = BufferedLogHandler()
        h.set_endpoint("http://example.com")
        assert h.get_destination_type() == LogDestination.HTTP

    def test_set_job_id_flushes(self):
        db = MagicMock()
        h = BufferedLogHandler(job_id="job-1", db=db, buffer_size=100)
        h.emit(_make_record())
        h.set_job_id("job-2")
        # set_job_id calls flush, which should clear the buffer
        assert h.get_buffer_size() == 0
        assert h.get_job_id() == "job-2"

    def test_set_trial_id_flushes(self):
        db = MagicMock()
        h = BufferedLogHandler(job_id="job-1", db=db, buffer_size=100)
        h.emit(_make_record())
        h.set_trial_id("trial-1")
        assert h.trial_id == "trial-1"


class TestPickling:
    """Pickle support is required for Ray to ship the handler to workers."""

    def test_getstate_excludes_lock(self):
        h = BufferedLogHandler(job_id="job-1")
        state = h.__getstate__()
        assert "lock" not in state
        assert "_flush_timer" not in state

    def test_pickle_roundtrip(self):
        h = BufferedLogHandler(job_id="job-1", endpoint_url="http://example.com")
        h.emit(_make_record())
        data = pickle.dumps(h)
        h2 = pickle.loads(data)
        assert h2.job_id == "job-1"
        assert h2.endpoint_url == "http://example.com"
        # lock & timer reinitialized
        assert h2.lock is not None


class TestSwitching:
    def test_switch_to_database(self):
        h = BufferedLogHandler(endpoint_url="http://example.com")
        assert h.get_destination_type() == LogDestination.HTTP
        db = MagicMock()
        h.switch_to_database(db)
        assert h.get_destination_type() == LogDestination.DATABASE

    def test_switch_to_endpoint(self):
        db = MagicMock()
        h = BufferedLogHandler(db=db)
        h.switch_to_endpoint("http://example.com")
        assert h.get_destination_type() == LogDestination.HTTP
