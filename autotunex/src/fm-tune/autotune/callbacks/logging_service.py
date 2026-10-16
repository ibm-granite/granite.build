# coding=utf-8
# Copyright 2023-present International Business Machines Corporation
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

import json
import logging
import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from enum import Enum
from logging import LogRecord
from typing import Dict, Optional, Protocol

import requests

# The trial that owns log records emitted by the *current* execution context.
#
# In the Tune driver process every trial's output converges on one root-logger
# handler: Ray runs with ``log_to_driver=True``, so each worker's stdout is
# forwarded into the driver, where ``main.py`` swaps ``sys.stdout`` for
# ``PrintLogger`` -> ``logging``. Those lines arrive on Ray's log-forwarding
# thread, interleaved with the Tune control thread that runs the trial
# callbacks. A single mutable ``handler.trial_id`` therefore labels each record
# with whichever trial most recently fired a callback, filing trial A's logs
# under trial B. A ContextVar is scoped to the thread/task that set it, so a
# callback can claim its own records without claiming everyone else's.
_current_trial_id: ContextVar[Optional[str]] = ContextVar("fmtune_current_trial_id", default=None)


@contextmanager
def trial_context(trial_id: Optional[str]):
    """Attribute log records emitted in this context to ``trial_id``.

    Scoped to the calling thread (and to the current asyncio task), so
    concurrent trials cannot overwrite each other's attribution.
    """
    token = _current_trial_id.set(trial_id)
    try:
        yield
    finally:
        _current_trial_id.reset(token)


def current_trial_id() -> Optional[str]:
    """The context-scoped trial id, or None outside any trial context."""
    return _current_trial_id.get()


class LogDestination(Enum):
    """Enumeration for log destinations."""

    DATABASE = "database"
    HTTP = "http"


class RecordType(Enum):
    """Enumeration for record type"""

    RECORD_TRIAL = "record_trial"
    UPDATE_STATUS = "update_status"
    RECORD_RESULT = "insert_trial_result"
    RECORD_METRICS = "record_metrics"


class DatabaseProtocol(Protocol):
    """Protocol defining the interface for database operations."""

    def create_logging_table(self) -> None:
        """Create the logging table if it doesn't exist."""
        ...

    def insert_logs(self, buffer: list) -> None:
        """Insert log entries into the database."""
        ...


class BufferedLogHandler(logging.Handler):
    def __init__(
        self,
        job_id: Optional[str] = None,
        trial_id: Optional[str] = None,
        buffer_size: int = 1024,
        auto_flush: bool = False,
        flush_interval: Optional[float] = None,
        db: Optional[DatabaseProtocol] = None,
        endpoint_url: Optional[str] = None,
        endpoint_headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        retry_attempts: int = 1,
    ):
        """
        Initialize the buffered log handler.

        Args:
            job_id: Unique identifier for the job
            trial_id: Optional identifier for specific trial runs
            buffer_size: Number of records to buffer before writing
            auto_flush: Whether to automatically flush on each emit
            flush_interval: Seconds between periodic background flushes.
                When set, a daemon timer flushes the buffer at this interval
                so logs reach the destination without waiting for buffer_size.
            db: Database service instance (can be set later)
            endpoint_url: HTTP endpoint URL for log submission
            endpoint_headers: Optional headers for HTTP requests
            timeout: HTTP request timeout in seconds
            retry_attempts: Number of retry attempts for HTTP requests
        """
        super().__init__()
        self.buffer_size = max(1, buffer_size)
        self.buffer = []
        self.db = db
        self.endpoint_url = endpoint_url
        self.endpoint_headers = endpoint_headers or {"Content-Type": "application/json"}
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.job_id = job_id
        self.trial_id = trial_id
        self.auto_flush = auto_flush
        self.flush_interval = flush_interval
        self.iteration = None
        self.epoch = None
        # Below 2 variable are added to resolve thread Rlock error while passing the handler into tuner callback
        self.lock = threading.RLock()
        self.data = {}
        self._flush_timer: Optional[threading.Timer] = None

        # Determine destination type
        self._destination = self._determine_destination()

        # Only create table if db is provided
        if self.db is not None:
            self.db.create_logging_table()

        # Start periodic flush timer if interval is set
        if self.flush_interval is not None and self.flush_interval > 0:
            self._start_flush_timer()

    def _determine_destination(self) -> Optional[LogDestination]:
        """Determine the current log destination based on available services."""
        if self.db is not None:
            return LogDestination.DATABASE
        elif self.endpoint_url is not None:
            return LogDestination.HTTP
        return None

    def _start_flush_timer(self):
        """Start (or restart) the periodic flush timer."""
        if self.flush_interval and self.flush_interval > 0:
            self._flush_timer = threading.Timer(self.flush_interval, self._periodic_flush)
            self._flush_timer.daemon = True
            self._flush_timer.start()

    def _periodic_flush(self):
        """Called by the timer — flush if buffer has entries, then reschedule."""
        try:
            self.flush()
        finally:
            self._start_flush_timer()

    def silent_log_operation(self, message, filename="silent.log"):
        # Your logging operation here
        log_path = os.getenv("LOG_PATH", "./logs")
        os.makedirs(f"{log_path}/logs", exist_ok=True)
        with open(f"{log_path}/logs/{filename}", "a") as f:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} - {message}\n")

    def set_database(self, db: DatabaseProtocol):
        """
        Set or update the database service instance.

        Args:
            db: Database service instance
        """
        self.db = db
        self._destination = LogDestination.DATABASE

        # Ensure table exists when database is set
        self.db.create_logging_table()

        # If there are buffered entries and job_id is set, flush them
        if self.buffer and self.job_id is not None:
            self.flush()

    def set_endpoint(
        self,
        endpoint_url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        retry_attempts: int = 3,
    ):
        """
        Set or update the HTTP endpoint for log submission.

        Args:
            endpoint_url: HTTP endpoint URL
            headers: Optional headers for HTTP requests
            timeout: HTTP request timeout in seconds
            retry_attempts: Number of retry attempts
        """
        self.endpoint_url = endpoint_url
        self.endpoint_headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self._destination = LogDestination.HTTP

        # If there are buffered entries and job_id is set, flush them
        if self.buffer and self.job_id is not None:
            self.flush()

    def _resolve_trial_id(self, record: LogRecord) -> Optional[str]:
        """Decide which trial owns ``record``.

        Precedence: an explicit ``trial_id`` on the record
        (``logger.info(..., extra={"trial_id": ...})``), then the
        context-scoped id set by :func:`trial_context`, then this handler's
        own default. The default is only meaningful in a trial worker
        process, which serves one trial at a time; in the driver it stays
        ``None`` so a job-level line is never filed under an unrelated trial.
        """
        explicit = getattr(record, "trial_id", None)
        if explicit is not None:
            return explicit
        scoped = _current_trial_id.get()
        if scoped is not None:
            return scoped
        return self.trial_id

    def emit(self, record: LogRecord):
        """
        Emit a record by adding it to the buffer and potentially flushing.
        If job_id is not set, it will only print to console.
        """
        try:
            # Format the log record
            log_entry = {
                "job_id": self.job_id,
                "trial_id": self._resolve_trial_id(record),
                "level": record.levelname,
                "filename": record.filename,
                "function": record.funcName,
                "line_number": record.lineno,
                "message": record.getMessage(),
                "iteration": self.iteration,
                "epoch": self.epoch,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "extra": getattr(record, "extra", None),
            }

            # Only buffer for logging if job_id is set
            if self.job_id is not None:
                with self.lock:
                    self.buffer.append(log_entry)
                    should_flush = self.auto_flush or len(self.buffer) >= self.buffer_size
                if should_flush:
                    self.flush()

        except Exception as e:
            self.handleError(record)
            self.silent_log_operation(f"Error in emit: {str(e)}")

    def _flush_to_database(self, batch: list):
        """Flush ``batch`` to database."""
        if self.db is not None:
            # Convert ISO string back to datetime for database
            db_buffer = []
            for entry in batch:
                db_entry = entry.copy()
                db_entry["timestamp"] = datetime.fromisoformat(entry["timestamp"])
                db_buffer.append(db_entry)

            self.db.insert_logs(buffer=db_buffer)

    def _flush_to_http(self, batch: list):
        """Flush ``batch`` to HTTP endpoint."""
        if self.endpoint_url is None:
            raise ValueError("Endpoint URL not set")
        self.silent_log_operation(f"logger uri:- {self.endpoint_url}/record_logs", "debug.log")
        for attempt in range(self.retry_attempts):
            url = f"{self.endpoint_url}/record_logs"
            try:
                response = requests.post(
                    url,
                    headers=self.endpoint_headers,
                    data=json.dumps(batch),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return  # Success, exit retry loop

            except requests.exceptions.RequestException as e:
                self.silent_log_operation(f"HTTP log submission attempt {attempt + 1} failed: {e}")
                if attempt == self.retry_attempts - 1:  # Last attempt
                    raise e

    def flush(self):
        """Write buffered records to the configured destination.

        The buffer is *detached* under the lock before the (blocking) send, so
        records emitted concurrently — Ray's log-forwarding thread is a steady
        source of them — land in the fresh buffer instead of being discarded by
        a post-send reset. The send itself runs outside the lock so a slow
        endpoint cannot stall every thread that logs.
        """
        with self.lock:
            if not self.buffer or self.job_id is None:
                return
            if self._destination is None:
                self.silent_log_operation(
                    f"Warning: {len(self.buffer)} log entries buffered but no destination available"
                )
                return
            batch, self.buffer = self.buffer, []

        try:
            if self._destination == LogDestination.DATABASE:
                self._flush_to_database(batch)
            else:
                self._flush_to_http(batch)

        except Exception as e:
            # Keep the batch for the next attempt (pre-existing retry
            # behaviour) and leave a durable copy on disk.
            with self.lock:
                self.buffer = batch + self.buffer
            log_path = os.getenv("LOG_PATH", "./logs")
            os.makedirs(f"{log_path}/logs", exist_ok=True)

            file_path = f"{log_path}/logs/{self.job_id}.json"
            with open(file_path, "w") as file:
                json.dump(batch, file, indent=4)
            self.silent_log_operation(f"Error in flush: {e}")
            # Optionally implement additional retry logic here

    def record_data(self, data, record_type: RecordType = RecordType.RECORD_TRIAL):
        """Insert trial data."""
        if self.endpoint_url is None:
            raise ValueError("Endpoint URL not set")
        self.silent_log_operation(
            f"record data uri: {f'{self.endpoint_url}/{record_type.value}'}",
            "debug.log",
        )
        for attempt in range(self.retry_attempts):
            try:
                url = f"{self.endpoint_url}/{record_type.value}"
                response = requests.post(
                    url,
                    headers=self.endpoint_headers,
                    data=json.dumps(data),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return  # Success, exit retry loop

            except requests.exceptions.RequestException as e:
                self.silent_log_operation(f"HTTP record data attempt {attempt + 1} failed: {e}")
                if attempt == self.retry_attempts - 1:  # Last attempt
                    self.silent_log_operation(f"Last attempt to record data failed: {e}")

    def close(self):
        """Ensure all records are written and close the handler."""
        if self._flush_timer is not None:
            self._flush_timer.cancel()
            self._flush_timer = None
        try:
            self.flush()
        finally:
            super().close()

    def set_trial_id(self, trial_id: str):
        """
        Update the trial ID for subsequent log entries.

        Args:
            trial_id: New trial identifier to use
        """
        self.flush()
        self.trial_id = trial_id

    def set_job_id(self, job_id: str):
        """
        Update the job ID for subsequent log entries.

        Args:
            job_id: New job identifier to use
        """
        self.flush()
        self.job_id = job_id

    def get_job_id(self) -> Optional[str]:
        """
        Get the current job ID.

        Returns:
            Current job identifier
        """
        return self.job_id

    def get_buffer_size(self) -> int:
        """
        Get the current number of buffered log entries.

        Returns:
            Number of entries in the buffer
        """
        return len(self.buffer)

    def has_destination(self) -> bool:
        """
        Check if a logging destination is available.

        Returns:
            True if either database or endpoint is set, False otherwise
        """
        return self._destination is not None

    def get_destination_type(self) -> Optional[LogDestination]:
        """
        Get the current destination type.

        Returns:
            Current destination type or None if no destination is set
        """
        return self._destination

    def switch_to_database(self, db: DatabaseProtocol):
        """
        Switch from HTTP endpoint to database logging.

        Args:
            db: Database service instance
        """
        self.flush()  # Flush any pending logs to current destination
        self.set_database(db)

    def switch_to_endpoint(self, endpoint_url: str, headers: Optional[Dict[str, str]] = None):
        """
        Switch from database to HTTP endpoint logging.

        Args:
            endpoint_url: HTTP endpoint URL
            headers: Optional headers for HTTP requests
        """
        self.flush()  # Flush any pending logs to current destination
        self.set_endpoint(endpoint_url, headers)

    # Below 2 functions are added to resolve thread Rlock error while passing the handler into tuner callback
    def __getstate__(self):
        # Remove unpicklable objects from state
        state = self.__dict__.copy()
        if "lock" in state:
            del state["lock"]
        if "_flush_timer" in state:
            del state["_flush_timer"]
        return state

    def __setstate__(self, state):
        # Restore state and reinitialize unpicklable objects
        self.__dict__.update(state)
        self.lock = threading.RLock()
        self._flush_timer = None
        if self.flush_interval and self.flush_interval > 0:
            self._start_flush_timer()
