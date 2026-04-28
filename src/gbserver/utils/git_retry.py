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
Retry logic for Git and GitHub API operations with transient error detection.
"""

import logging
import random
import shutil
import time
from pathlib import Path
from typing import Optional, Tuple

import requests
from git.exc import GitCommandError, GitError
from tenacity import (
    RetryCallState,
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from gbserver.types.constants import (
    GIT_CLONE_MAX_RETRIES,
    GIT_CLONE_RETRY_MAX_WAIT,
    GIT_CLONE_RETRY_MIN_WAIT,
    GITHUB_API_MAX_RETRIES,
    GITHUB_API_RETRY_BASE_DELAY,
    GITHUB_API_RETRY_MAX_DELAY,
)
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

# Error patterns that indicate transient network errors that should be retried
RETRYABLE_ERROR_PATTERNS = [
    # Network connection errors
    "connection reset by peer",
    "connection timed out",
    "connection refused",
    "could not resolve host",
    "temporary failure in name resolution",
    "network is unreachable",
    "no route to host",
    # Server errors
    "http 502",
    "http 503",
    "http 504",
    "bad gateway",
    "service unavailable",
    "gateway timeout",
    # SSH errors
    "ssh connection error",
    "connection closed by remote host",
    # Git protocol errors
    "the remote end hung up unexpectedly",
    "rpc failed",
    "early eof",
    "index-pack failed",
    "unable to access",
    "failed to connect",
]

# Error patterns that indicate permanent errors that should NOT be retried
NON_RETRYABLE_ERROR_PATTERNS = [
    # Authentication failures
    "authentication failed",
    "permission denied",
    "invalid credentials",
    "http 401",
    "http 403",
    "fatal: authentication failed",
    # Repository not found
    "repository not found",
    "http 404",
    "could not read from remote repository",
    "fatal: repository",
    # Invalid configuration
    "invalid username or password",
    "host key verification failed",
    "bad owner or permissions",
    # Disk space
    "no space left on device",
    "disk quota exceeded",
]


def is_transient_git_error(exception: BaseException) -> bool:
    """
    Determine if a git operation error is transient and should be retried.

    Args:
        exception: The exception raised during git operation

    Returns:
        True if the error is transient and should be retried, False otherwise
    """
    if not isinstance(exception, (GitCommandError, GitError)):
        return False

    error_msg = str(exception).lower()

    # Check for non-retryable errors first (fail fast on permanent errors)
    for pattern in NON_RETRYABLE_ERROR_PATTERNS:
        if pattern.lower() in error_msg:
            logger.debug(f"Non-retryable git error detected: {pattern}")
            return False

    # Check for retryable errors
    for pattern in RETRYABLE_ERROR_PATTERNS:
        if pattern.lower() in error_msg:
            logger.debug(f"Retryable git error detected: {pattern}")
            return True

    # Default to retry for GitCommandError (network errors often don't have clear messages)
    # But not for other GitError types which might be programming errors
    if isinstance(exception, GitCommandError):
        logger.debug("Unknown GitCommandError, defaulting to retry")
        return True

    return False


def cleanup_failed_clone(retry_state):
    """
    Cleanup callback that removes partial clone directories before retry attempts.

    When git clone fails partway through (e.g., network timeout), it may leave behind
    a partial directory with some downloaded objects. This callback runs before each
    retry attempt to clean up the target directory, preventing "destination path
    already exists and is not an empty directory" errors.

    Args:
        retry_state: The RetryCallState from tenacity containing attempt information
    """
    # This callback only runs before retry attempts (not before the first attempt)
    # If we're here, the previous attempt failed with a transient error

    # Extract the path parameter - it's typically the 2nd positional arg (after self)
    # or might be in kwargs as 'path', 'dest', or 'to_path'
    path = None
    if len(retry_state.args) >= 2:
        path = retry_state.args[1]
    else:
        # Check common kwarg names used in different clone methods
        for key in ["path", "dest", "to_path"]:
            if key in retry_state.kwargs:
                path = retry_state.kwargs[key]
                break

    if path:
        path_obj = Path(path)
        if path_obj.exists():
            logger.warning("Cleaning up partial clone directory before retry: %s", path)
            shutil.rmtree(path_obj, ignore_errors=True)


git_clone_retry = retry(
    retry=retry_if_exception(is_transient_git_error),
    wait=wait_random_exponential(
        multiplier=1,
        min=GIT_CLONE_RETRY_MIN_WAIT,
        max=GIT_CLONE_RETRY_MAX_WAIT,
    ),
    stop=stop_after_attempt(GIT_CLONE_MAX_RETRIES),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    before=cleanup_failed_clone,
    reraise=True,
)
"""
Decorator for adding retry logic to git operations.

This decorator will retry git operations that fail with transient network errors,
using exponential backoff with jitter. Permanent errors (authentication failures,
repository not found, etc.) will fail immediately without retrying.

The decorator automatically cleans up partial clone directories after transient failures
to prevent "destination path already exists and is not an empty directory" errors on retry.

Usage:
    @git_clone_retry
    def clone_repo(...):
        return Repo.clone_from(...)

Environment Variables:
    GBSERVER_GIT_CLONE_MAX_RETRIES: Maximum number of retry attempts (default: 5)
    GBSERVER_GIT_CLONE_RETRY_MIN_WAIT: Minimum wait time in seconds (default: 1)
    GBSERVER_GIT_CLONE_RETRY_MAX_WAIT: Maximum wait time in seconds (default: 30)
"""


# -----------------------------------------------------------------------------
# GitHub API Retry Logic
# -----------------------------------------------------------------------------


def _calculate_github_backoff_delay(
    retry_count: int,
    base_delay: float = GITHUB_API_RETRY_BASE_DELAY,
    max_delay: float = GITHUB_API_RETRY_MAX_DELAY,
) -> float:
    """Calculate exponential backoff delay with jitter for GitHub API requests."""
    delay = min(base_delay * (2 ** (retry_count - 1)), max_delay)
    # Add jitter (0.5 to 1.5 times the delay)
    jitter = 0.5 + random.random()
    return delay * jitter


def _is_github_rate_limit_error(
    response: requests.Response,
) -> Tuple[bool, Optional[float]]:
    """
    Check if the response indicates a GitHub rate limit error.
    Returns (is_rate_limited, retry_after_seconds).
    """
    if response.status_code == 403:
        # Check for rate limit headers (lowercase as per GitHub docs)
        # https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api
        remaining = response.headers.get("x-ratelimit-remaining")
        if remaining is not None and int(remaining) == 0:
            reset_time = response.headers.get("x-ratelimit-reset")
            if reset_time:
                retry_after = float(max(0, int(reset_time) - int(time.time())))
                return True, retry_after
            return True, None
        # Also check for "rate limit" in error message
        try:
            error_msg = response.json().get("message", "").lower()
            if "rate limit" in error_msg or "abuse" in error_msg:
                return True, None
        except Exception:
            pass
    # Check Retry-After header (can appear on various status codes)
    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            return True, float(retry_after)
        except ValueError:
            return True, None
    return False, None


def _should_retry_github_status_code(status_code: int) -> bool:
    """Determine if we should retry based on GitHub API status code."""
    # Retry on server errors (5xx), rate limits (403), and too many requests (429)
    return status_code >= 500 or status_code == 403 or status_code == 429


def is_retryable_github_error(exception: BaseException) -> bool:
    """Check if exception should trigger a retry for GitHub API requests."""
    if isinstance(exception, requests.HTTPError) and exception.response is not None:
        return _should_retry_github_status_code(exception.response.status_code)
    return False


def _wait_for_github_rate_limit(retry_state: RetryCallState) -> float:
    """
    Custom wait strategy that respects GitHub rate limit headers.
    Falls back to exponential backoff with jitter if no rate limit info available.
    """
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exception, requests.HTTPError) and exception.response is not None:
        is_rate_limit, retry_after = _is_github_rate_limit_error(exception.response)
        if is_rate_limit and retry_after is not None:
            return retry_after
    # Fall back to exponential backoff with jitter
    return _calculate_github_backoff_delay(retry_state.attempt_number)


github_api_retry = retry(
    retry=retry_if_exception(is_retryable_github_error),
    wait=_wait_for_github_rate_limit,
    stop=stop_after_attempt(GITHUB_API_MAX_RETRIES),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
"""
Decorator for adding retry logic to GitHub API requests.

This decorator will retry GitHub API requests that fail with transient errors
(5xx server errors, 403 rate limits, 429 too many requests), using exponential
backoff with jitter. It respects GitHub's rate limit headers (x-ratelimit-reset,
retry-after) when determining wait times.

Permanent errors (401 unauthorized, 404 not found, 400 bad request, etc.) will
fail immediately without retrying.

Usage:
    @github_api_retry
    def fetch_data(self):
        response = requests.get(url, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

Environment Variables:
    GBSERVER_GITHUB_API_MAX_RETRIES: Maximum number of retry attempts (default: 10)
    GBSERVER_GITHUB_API_RETRY_BASE_DELAY: Base delay in seconds for exponential backoff (default: 1.0)
    GBSERVER_GITHUB_API_RETRY_MAX_DELAY: Maximum delay in seconds (default: 60.0)
"""
