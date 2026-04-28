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

from gbserver.types.constants import FETCH_CLOUD_LOGS_MAX_RETRIES
from gbserver.types.errors import LogMonitoringFailedException, WorkloadFailedException
from gbserver.utils.cloud_logquery import get_log_manager
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


def get_readable_error_message(e: Exception, err_stack: str) -> str:
    """Get a readable error message to post to the pull request."""
    logger.debug("get_readable_error_message start")
    readable_error = unwrap_errors(e)
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


def unwrap_errors(e: BaseException) -> str:
    """Unwrap nested Exception(Group)s to create a readable message."""
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
            return "\n".join(unwrap_errors(exc) for exc in real_exceptions)
        return str(e)
    if e.__cause__ is not None:
        return unwrap_errors(e.__cause__)
    if isinstance(e, KeyError):
        return "key error: " + str(e)
    if isinstance(e, ValueError):
        return "value error: " + str(e)
    if isinstance(e, LogMonitoringFailedException):
        build_id = e.build_id
        if FETCH_CLOUD_LOGS_MAX_RETRIES <= 0:
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
        if FETCH_CLOUD_LOGS_MAX_RETRIES <= 0:
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
