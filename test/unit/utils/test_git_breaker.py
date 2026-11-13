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

"""Optional git/GitHub interactions give up per build run instead of stalling it."""

import pytest
import requests

import gbserver.utils.git_retry as git_retry
from gbserver.utils.git_retry import (
    GitAccessBreaker,
    end_build_git_run,
    get_build_git_breaker,
    github_api_retry,
)


def _fail():
    raise RuntimeError("github down")


def test_opens_after_consecutive_failures_then_skips():
    breaker = GitAccessBreaker("b", max_failures=2, deadline=5)
    calls = []

    def fn():
        calls.append(1)
        _fail()

    assert breaker.call("x", fn, default="d") == "d"
    assert not breaker.is_open
    assert breaker.call("x", fn) is None
    assert breaker.is_open
    assert breaker.call("x", fn) is None
    assert len(calls) == 2  # the third call was skipped


def test_success_resets_the_count():
    breaker = GitAccessBreaker("b", max_failures=2, deadline=5)
    breaker.call("x", _fail)
    assert breaker.call("x", lambda: "ok") == "ok"
    breaker.call("x", _fail)
    assert not breaker.is_open


def _http_error(status: int) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(response=response)


def test_deadline_cuts_github_retries_short(monkeypatch):
    """A retryable 503 would retry GITHUB_API_MAX_RETRIES times with backoff;
    inside a breaker call it stops at the deadline."""
    monkeypatch.setattr(git_retry, "_calculate_github_backoff_delay", lambda *_: 0)
    clock = iter(range(1000))  # each read of the clock advances one "second"
    monkeypatch.setattr(git_retry.time, "monotonic", lambda: float(next(clock)))
    attempts = []

    @github_api_retry
    def flaky():
        attempts.append(1)
        raise _http_error(503)

    GitAccessBreaker("b", max_failures=1, deadline=3).call("x", flaky)
    assert 1 <= len(attempts) < git_retry.GITHUB_API_MAX_RETRIES


def test_no_deadline_outside_a_breaker(monkeypatch):
    """Other callers keep the full retry count."""
    monkeypatch.setattr(git_retry, "_calculate_github_backoff_delay", lambda *_: 0)
    attempts = []

    @github_api_retry
    def flaky():
        attempts.append(1)
        raise _http_error(503)

    with pytest.raises(requests.HTTPError):
        flaky()
    assert len(attempts) == git_retry.GITHUB_API_MAX_RETRIES


def test_registry_is_per_run():
    breaker = get_build_git_breaker("build-1")
    assert get_build_git_breaker("build-1") is breaker
    end_build_git_run("build-1")
    assert get_build_git_breaker("build-1") is not breaker
    end_build_git_run("build-1")


def test_registry_is_bounded(monkeypatch):
    monkeypatch.setattr(git_retry, "_MAX_BREAKERS", 3)
    for i in range(10):
        get_build_git_breaker(f"bounded-{i}")
    assert len(git_retry._breakers) <= 3
    for i in range(10):
        end_build_git_run(f"bounded-{i}")
