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

"""Regression tests for search_build_yaml's ReDoS mitigation.

Code-review finding: the old `asyncio.wait_for(asyncio.to_thread(re.search,
...), timeout=2.0)` mitigation didn't work — stdlib `re` never releases the
GIL during a match, so a catastrophically backtracking pattern would block
the whole single-threaded server for the full backtrack duration (easily
minutes), not just this one call; wait_for's timeout can't fire until the
blocking call returns on its own. Switching to the third-party `regex`
package's own `timeout=` kwarg gives a real, hard bound enforced inside its
matching loop, not dependent on GIL release.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gb_ui_backend.services.chat_agents import dashboard_tools
from gb_ui_backend.services.chat_agents.dashboard_tools import (
    DashboardToolError,
    search_build_yaml,
)


def _fake_source(builds):
    source = SimpleNamespace()
    source.list_builds_for_dp_scan = AsyncMock(return_value=(builds, None))
    return source


@pytest.mark.asyncio
class TestSearchBuildYaml:
    async def test_matching_pattern_is_found(self, monkeypatch):
        builds = [
            {
                "uuid": "b1",
                "name": "build-1",
                "status": "COMPLETED",
                "username": "u",
                "yaml_content": "target: train_lora",
            },
            {
                "uuid": "b2",
                "name": "build-2",
                "status": "COMPLETED",
                "username": "u",
                "yaml_content": "target: eval",
            },
        ]
        monkeypatch.setattr(
            dashboard_tools, "get_gbserver_source", lambda: _fake_source(builds)
        )

        result = await search_build_yaml("train_lora")

        assert result["scanned"] == 2
        assert [m["uuid"] for m in result["matches"]] == ["b1"]
        assert "timed_out_for" not in result

    async def test_invalid_pattern_raises_dashboard_tool_error(self, monkeypatch):
        monkeypatch.setattr(
            dashboard_tools, "get_gbserver_source", lambda: _fake_source([])
        )

        with pytest.raises(DashboardToolError, match="Invalid regex pattern"):
            await search_build_yaml("(unclosed[")

    async def test_no_gbserver_source_raises_dashboard_tool_error(self, monkeypatch):
        monkeypatch.setattr(dashboard_tools, "get_gbserver_source", lambda: None)

        with pytest.raises(DashboardToolError, match="isn't connected"):
            await search_build_yaml("anything")

    async def test_a_pattern_that_times_out_is_recorded_and_does_not_hang(
        self, monkeypatch
    ):
        """Exercises the actual code path this fix touches: a match that
        exceeds the timeout raises regex's own TimeoutError (real
        catastrophic-backtracking timing is flaky to reproduce
        deterministically in a test — the `regex` package's backtracking
        optimizer is in fact resistant to several classic pathological
        patterns, which is a feature, not something to fight in a test) —
        the important thing this covers is that the exception is caught,
        the build lands in timed_out_for rather than matches, and the call
        returns promptly rather than propagating or hanging."""
        builds = [
            {
                "uuid": "b1",
                "name": "build-1",
                "status": "COMPLETED",
                "username": "u",
                "yaml_content": "irrelevant",
            },
            {
                "uuid": "b2",
                "name": "build-2",
                "status": "COMPLETED",
                "username": "u",
                "yaml_content": "target: fine",
            },
        ]
        monkeypatch.setattr(
            dashboard_tools, "get_gbserver_source", lambda: _fake_source(builds)
        )

        real_compile = dashboard_tools.regex.compile

        def _compile_with_timeout_for_b1(pattern, flags=0):
            compiled = real_compile(pattern, flags)

            def _search(text, timeout=None):
                if text == "irrelevant":
                    raise TimeoutError("simulated catastrophic-backtracking timeout")
                return compiled.search(text)

            return SimpleNamespace(search=_search)

        monkeypatch.setattr(
            dashboard_tools.regex, "compile", _compile_with_timeout_for_b1
        )

        start = time.monotonic()
        result = await asyncio.wait_for(search_build_yaml("target"), timeout=5.0)
        elapsed = time.monotonic() - start

        assert elapsed < 1.0
        assert result["timed_out_for"] == ["b1"]
        assert [m["uuid"] for m in result["matches"]] == ["b2"]

    async def test_multiple_builds_are_each_scanned_independently(self, monkeypatch):
        builds = [
            {
                "uuid": f"b{i}",
                "name": f"build-{i}",
                "status": "COMPLETED",
                "username": "u",
                "yaml_content": f"target: t{i}",
            }
            for i in range(3)
        ]
        monkeypatch.setattr(
            dashboard_tools, "get_gbserver_source", lambda: _fake_source(builds)
        )

        result = await search_build_yaml("t1")

        assert [m["uuid"] for m in result["matches"]] == ["b1"]
        assert result["scanned"] == 3
