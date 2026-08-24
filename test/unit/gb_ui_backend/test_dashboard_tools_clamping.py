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

"""Regression tests for the code-review finding that search_build_errors and
compare_builds passed model-supplied limit/days_back straight into a SQL
query / archive-scan with no clamp — unlike search_builds and
search_build_yaml, which do clamp. An arbitrarily large value could
materialize a huge result set (search_build_errors) or scan an
unnecessarily wide archive window (compare_builds) from a read-only,
auto-approved tool with no confirmation gate.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gb_ui_backend.services.chat_agents import dashboard_tools
from gb_ui_backend.services.chat_agents.dashboard_tools import (
    _MAX_DAYS_BACK,
    _MAX_SEARCH_ERRORS_LIMIT,
    DashboardToolError,
    compare_builds,
    search_build_errors,
)


class _FakeResult:
    def scalars(self):
        return self

    def all(self):
        return []


class _FakeSession:
    def __init__(self):
        self.executed_stmt = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def execute(self, stmt):
        self.executed_stmt = stmt
        return _FakeResult()


def _compiled_sql(stmt) -> str:
    return str(stmt.compile(compile_kwargs={"literal_binds": True}))


@pytest.mark.asyncio
class TestSearchBuildErrorsClamping:
    async def test_excessive_limit_is_clamped(self, monkeypatch):
        session = _FakeSession()
        monkeypatch.setattr(
            dashboard_tools, "get_config", lambda: SimpleNamespace(db_enabled=True)
        )
        monkeypatch.setattr(
            dashboard_tools, "_get_session_factory", lambda: (lambda: session)
        )

        await search_build_errors("boom", limit=10_000_000)

        assert f"LIMIT {_MAX_SEARCH_ERRORS_LIMIT}" in _compiled_sql(
            session.executed_stmt
        )

    async def test_excessive_days_back_is_clamped(self, monkeypatch):
        session = _FakeSession()
        monkeypatch.setattr(
            dashboard_tools, "get_config", lambda: SimpleNamespace(db_enabled=True)
        )
        monkeypatch.setattr(
            dashboard_tools, "_get_session_factory", lambda: (lambda: session)
        )

        await search_build_errors("boom", days_back=100_000)

        sql = _compiled_sql(session.executed_stmt)
        earliest_allowed = datetime.now(timezone.utc) - timedelta(
            days=_MAX_DAYS_BACK, hours=1
        )
        # The bound >= threshold must not predate _MAX_DAYS_BACK ago — a
        # 100,000-day request must not reach back further than that.
        threshold_str = sql.split(">= '")[1].split("'")[0]
        threshold = datetime.fromisoformat(threshold_str)
        assert threshold > earliest_allowed

    async def test_reasonable_values_pass_through_unclamped(self, monkeypatch):
        session = _FakeSession()
        monkeypatch.setattr(
            dashboard_tools, "get_config", lambda: SimpleNamespace(db_enabled=True)
        )
        monkeypatch.setattr(
            dashboard_tools, "_get_session_factory", lambda: (lambda: session)
        )

        await search_build_errors("boom", days_back=7, limit=50)

        assert "LIMIT 50" in _compiled_sql(session.executed_stmt)


@pytest.mark.asyncio
class TestCompareBuildsClamping:
    async def test_excessive_days_back_is_clamped_before_the_archive_scan(
        self, monkeypatch
    ):
        source = SimpleNamespace()
        source.get_build = AsyncMock(
            side_effect=lambda bid: {
                "uuid": bid,
                "name": bid,
                "space_name": "s",
                "username": "u",
                "status": "COMPLETED",
            }
        )
        source.list_builds_for_dp_scan = AsyncMock(return_value=([], None))
        monkeypatch.setattr(dashboard_tools, "get_gbserver_source", lambda: source)

        await compare_builds(["b1", "b2"], days_back=999_999)

        _args, kwargs = source.list_builds_for_dp_scan.call_args
        assert kwargs["days_back"] <= _MAX_DAYS_BACK

    async def test_reasonable_days_back_passes_through_unclamped(self, monkeypatch):
        source = SimpleNamespace()
        source.get_build = AsyncMock(
            side_effect=lambda bid: {
                "uuid": bid,
                "name": bid,
                "space_name": "s",
                "username": "u",
                "status": "COMPLETED",
            }
        )
        source.list_builds_for_dp_scan = AsyncMock(return_value=([], None))
        monkeypatch.setattr(dashboard_tools, "get_gbserver_source", lambda: source)

        await compare_builds(["b1", "b2"], days_back=14)

        _args, kwargs = source.list_builds_for_dp_scan.call_args
        assert kwargs["days_back"] == 14


@pytest.mark.asyncio
class TestCompareBuildsConcurrentFetch:
    """Code-review observation: compare_builds fetched each build ID via a
    sequential for-loop of `await source.get_build(bid)` calls — N
    independent network/DB round trips paying N times the latency instead
    of roughly one."""

    async def test_builds_are_fetched_concurrently_not_sequentially(self, monkeypatch):
        build_ids = ["b1", "b2", "b3"]
        started = asyncio.Event()
        call_count = 0

        async def _get_build(bid):
            nonlocal call_count
            call_count += 1
            if call_count == len(build_ids):
                started.set()
            else:
                # Would hang (and time out) if calls ran sequentially — the
                # last call can only ever start once all the earlier ones
                # already have, which can't happen if each one is awaited
                # to completion before the next begins.
                await asyncio.wait_for(started.wait(), timeout=1.0)
            return {
                "uuid": bid,
                "name": bid,
                "space_name": "s",
                "username": "u",
                "status": "COMPLETED",
            }

        source = SimpleNamespace()
        source.get_build = _get_build
        source.list_builds_for_dp_scan = AsyncMock(return_value=([], None))
        monkeypatch.setattr(dashboard_tools, "get_gbserver_source", lambda: source)

        await compare_builds(build_ids, days_back=14)

        assert call_count == 3

    async def test_error_names_the_first_missing_build_in_list_order(self, monkeypatch):
        """Even if a later build's fetch resolves first, the error must
        still name the first missing build in the caller's own list order
        — matching the old sequential-raise behavior exactly."""

        async def _get_build(bid):
            if bid == "b2":
                return None  # resolves immediately
            await asyncio.sleep(0.02)  # b1 resolves slower
            return None

        source = SimpleNamespace()
        source.get_build = _get_build
        source.list_builds_for_dp_scan = AsyncMock(return_value=([], None))
        monkeypatch.setattr(dashboard_tools, "get_gbserver_source", lambda: source)

        with pytest.raises(DashboardToolError, match="Build b1 not found"):
            await compare_builds(["b1", "b2"], days_back=14)
