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

"""Utility functions for better errors."""

import asyncio
from typing import Optional

from gbserver.types.constants import FETCH_CLOUD_LOGS_MAX_RETRIES
from gbserver.types.errors import LogMonitoringFailedException, WorkloadFailedException
from gbserver.utils.cloud_logquery import get_log_manager
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


def get_readable_error_message(e: Exception, err_stack: str) -> str:
    """Get a readable error message to post to the pull request."""
    logger.debug("get_readable_error_message start")
    readable_error = unwrap_errors(e)
    # For a failure raised inside a remote API server, err_stack is just
    # "<Type>: <message>" (the re-raised object has no __traceback__), so append
    # the server-side traceback that does name the failing call/path.
    server_stack = remote_stacktrace(e)
    if server_stack is not None and server_stack not in err_stack:
        err_stack = (
            f"{err_stack.rstrip()}\n\n"
            f"--- Traceback from the remote API server ---\n{server_stack}"
        )
    body = f"""
The run failed due to exception(s):
{readable_error}

<details>

<summary>See more details</summary>

### Full Stack Trace

```
{err_stack}
```

</details>
"""
    logger.debug("get_readable_error_message end")
    return body


def format_oserror(e: OSError) -> str:
    """Render an OSError as ``[Errno N] strerror: 'filename'``.

    Includes errno and filename/filename2 when set so the failing path is
    visible. When none are set (e.g. a bare TimeoutError/ConnectionError, both
    OSError subclasses) falls back to ``str(e)`` to avoid noise.
    """
    if e.errno is None and not e.filename:
        return str(e)
    parts = []
    if e.errno is not None:
        parts.append(f"[Errno {e.errno}]")
    parts.append(e.strerror or str(e))
    msg = " ".join(parts)
    if e.filename:
        msg += f": {e.filename!r}"
        if e.filename2:
            msg += f" -> {e.filename2!r}"
    return msg


def remote_stacktrace(e: BaseException) -> Optional[str]:
    """Return a traceback recorded on a remote server for ``e``, if any.

    SkyPilot's client re-raises the *API server's* exception object on our side
    (``sky/client/sdk.py:_raise_exception_object_on_client``). The object is
    unpickled fresh, so ``__traceback__`` and ``__cause__`` are empty and
    ``logger.exception``/``exc_info=True`` render a single bare line with no
    frames. The server-side traceback survives as a ``stacktrace`` string
    attribute (set by ``sky/server/requests/requests.py:set_exception_stacktrace``),
    which is the only place the failing path appears for errors raised remotely —
    an ``OSError`` built as ``OSError(errno, strerror)`` carries no ``filename``
    for :func:`format_oserror` to report.

    Searches the whole exception chain (``__cause__``/``__context__`` and
    ``(Base)ExceptionGroup`` members), not just ``e``: by the time a failure
    reaches a reporting layer it is typically wrapped — the production trace read
    ``RunFailed -> ValueError: failed during loading artifacts`` over the
    original OSError — and looking only at the outermost exception would skip the
    server traceback exactly where it is needed. Mirrors the traversal in
    :func:`unwrap_errors`.

    Returns None when no exception in the chain carries a non-empty ``stacktrace``
    string, so locally-raised exceptions (which have a real traceback) are
    unaffected.
    """
    # Cycles are possible via __context__; bound the walk by identity.
    seen: set[int] = set()

    def _walk(exc: Optional[BaseException]) -> Optional[str]:
        if exc is None or id(exc) in seen:
            return None
        seen.add(id(exc))
        stacktrace = getattr(exc, "stacktrace", None)
        if isinstance(stacktrace, str) and stacktrace.strip():
            return stacktrace
        if isinstance(exc, BaseExceptionGroup):
            for member in exc.exceptions:
                found = _walk(member)
                if found is not None:
                    return found
        # __cause__ first (explicit `raise ... from`), then the implicit context.
        return _walk(exc.__cause__) or _walk(exc.__context__)

    return _walk(e)


def format_failure_reason(e: BaseException) -> str:
    """One-line failure reason (no traceback): the same leaf as
    :func:`unwrap_errors`, collapsed to a single line — for a log line or stored
    ``failure_reason``. ``fetch_logs=False`` keeps it a bare reason (no cloud-log
    fetch), so calling it per layer is cheap."""
    return " ".join(unwrap_errors(e, fetch_logs=False).split())


def unwrap_errors(e: BaseException, fetch_logs: bool = True) -> str:
    """Unwrap nested Exception(Group)s to create a readable message.

    ``fetch_logs`` (default True) inlines the build's step logs for
    Workload/LogMonitoring failures; pass False for a bare one-line reason."""
    assert isinstance(
        e, BaseException
    ), f"unwrap_errors called with non-exception type: {type(e)} {e}"
    if isinstance(e, BaseExceptionGroup):
        # Filter out CancelledError — these are sibling tasks cancelled by the
        # TaskGroup when a real failure occurred, not the failure itself.
        real_exceptions = [
            exc for exc in e.exceptions if not isinstance(exc, asyncio.CancelledError)
        ]
        if real_exceptions:
            return "\n".join(unwrap_errors(exc, fetch_logs) for exc in real_exceptions)
        return str(e)
    if e.__cause__ is not None:
        return unwrap_errors(e.__cause__, fetch_logs)
    if isinstance(e, KeyError):
        return "key error: " + str(e)
    if isinstance(e, ValueError):
        return "value error: " + str(e)
    if isinstance(e, OSError):
        return format_oserror(e)
    if isinstance(e, LogMonitoringFailedException):
        build_id = e.build_id
        if not fetch_logs or FETCH_CLOUD_LOGS_MAX_RETRIES <= 0:
            return "log monitoring failed (fetching build logs is disabled): " + str(e)
        log_manager = None
        try:
            log_manager = get_log_manager()
        except Exception as log_ex:
            logger.error("failed to get the log_manager, error: %s", log_ex)
        if log_manager is not None and build_id != "":
            try:
                logs_str = log_manager.get_build_logs(build_id=build_id)
                return (
                    "log monitoring failed: fetched the step logs:\n\n```\n"
                    + logs_str
                    + "\n```\n\n"
                )
            except Exception as logfetche:
                logger.error(
                    "failed to fetch the logs for the build %s : %s",
                    build_id,
                    logfetche,
                )
        return "log monitoring failed (also failed to fetch build logs): " + str(e)
    if isinstance(e, WorkloadFailedException):
        build_id = e.build_id
        if not fetch_logs or FETCH_CLOUD_LOGS_MAX_RETRIES <= 0:
            return "workload failed: " + str(e)
        log_manager = None
        try:
            log_manager = get_log_manager()
        except Exception as log_ex:
            logger.error("failed to get the log_manager, error: %s", log_ex)
        if log_manager is not None and build_id != "":
            try:
                logs_str = log_manager.get_build_logs(build_id=build_id)
                return (
                    "workload failed: fetched the step logs:\n\n```\n"
                    + logs_str
                    + "\n```\n\n"
                )
            except Exception as logfetche:
                logger.error(
                    "failed to fetch the logs for the build %s : %s",
                    build_id,
                    logfetche,
                )
        return "workload failed: " + str(e)
    return str(e)
