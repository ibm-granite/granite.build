# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""Centralized logging configuration for the api-bridge.

Call :func:`setup_logging` once, as early as possible at process start (before
the FastAPI app is created). It configures the *root* logger so that every
``logging.getLogger(__name__)` in the codebase inherits a single stdout handler
with a consistent format.

Two env vars drive behaviour:

- ``LOG_LEVEL``  — root level (DEBUG/INFO/WARNING/ERROR). Default: ``INFO``.
- ``LOG_FORMAT`` — ``text`` (human-readable, default) or ``json`` (one JSON
  object per line, for log aggregators like Loki/ELK/Datadog).

Logs go to stdout, which supervisord + ``PYTHONUNBUFFERED=1`` forward straight
to container logs.
"""

import datetime
import json
import logging
import os
import sys

_TEXT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"

# Extra LogRecord attributes the middleware attaches; surfaced as top-level
# keys in JSON output and ignored everywhere else.
_EXTRA_KEYS = (
    "request_id",
    "method",
    "path",
    "status_code",
    "duration_ms",
    "caller",
)

# Standard LogRecord attributes — used to detect *ad-hoc* extras passed via
# ``logger.info(..., extra={...})`` so JSON output can include them too.
_RESERVED = set(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime"}


class _RenameUvicornErrorFilter(logging.Filter):
    """Relabel uvicorn's general logger from ``uvicorn.error`` to ``uvicorn``.

    uvicorn routes all of its lifecycle/server messages (startup, shutdown,
    "Uvicorn running on…") through a logger literally named ``uvicorn.error``,
    regardless of severity — so INFO lines show up tagged ``.error``, which
    reads like a problem. This filter rewrites the record name for display;
    actual errors still carry their ERROR level.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == "uvicorn.error":
            record.name = "uvicorn"
        return True


class JsonFormatter(logging.Formatter):
    """Minimal stdlib-only JSON line formatter."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.datetime.fromtimestamp(record.created, tz=datetime.UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Known middleware extras (only when set).
        for key in _EXTRA_KEYS:
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value

        # Any other ad-hoc extras passed through ``extra={...}``.
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key not in payload:
                payload[key] = value

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(payload, default=str)


def setup_logging() -> None:
    """Configure the root logger. Idempotent — safe to call more than once."""
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = os.getenv("LOG_FORMAT", "text").lower()

    handler = logging.StreamHandler(sys.stdout)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(_TEXT_FORMAT))

    root = logging.getLogger()
    # Replace any pre-existing handlers (e.g. a stray basicConfig from an
    # imported module, or a previous call under uvicorn --reload) so we don't
    # emit every line twice.
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)

    # Route uvicorn's own loggers through our handler at the same level so the
    # access log and our application logs share one consistent format.
    rename_filter = _RenameUvicornErrorFilter()
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
        uvicorn_logger.setLevel(level)
        # Idempotent: clear any prior copy before re-adding (matters under reload).
        for existing_filter in [
            f for f in uvicorn_logger.filters if isinstance(f, _RenameUvicornErrorFilter)
        ]:
            uvicorn_logger.removeFilter(existing_filter)
        uvicorn_logger.addFilter(rename_filter)

    logging.getLogger("api-bridge").info(
        "Logging configured (level=%s, format=%s)", level_name, fmt
    )
