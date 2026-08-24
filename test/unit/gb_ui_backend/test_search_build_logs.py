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

"""Regression tests for a code-review finding in search_build_logs: `tail`
was clamped for the cloud-logs API page_size, but the final in-memory slice
used the raw, unclamped value — `lines[-tail:]`. Since `-0 == 0` in Python,
`tail=0` (a natural way to ask "just tell me if anything matched") returned
the WHOLE log instead of nothing, and a negative `tail` sliced from the
head instead of the tail.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gb_ui_backend.services.chat_agents import dashboard_tools
from gb_ui_backend.services.chat_agents.dashboard_tools import search_build_logs


def _configure(monkeypatch, lines: list[str]):
    monkeypatch.setattr(
        dashboard_tools,
        "get_config",
        lambda: SimpleNamespace(cloud_logs_url="http://fake", cloud_logs_api_key="key"),
    )
    client = SimpleNamespace()
    client.query_logs = AsyncMock(return_value={"fake": "response"})
    client.parse_logs = lambda response: list(lines)
    monkeypatch.setattr(dashboard_tools, "get_cloud_logs_client", lambda *a: client)
    return client


@pytest.mark.asyncio
class TestSearchBuildLogsTailSlicing:
    async def test_tail_zero_returns_no_lines_not_the_whole_log(self, monkeypatch):
        lines = [f"line-{i}" for i in range(50)]
        _configure(monkeypatch, lines)

        result = await search_build_logs("b1", tail=0)

        assert result == []

    async def test_negative_tail_returns_no_lines_not_a_head_slice(self, monkeypatch):
        lines = [f"line-{i}" for i in range(50)]
        _configure(monkeypatch, lines)

        result = await search_build_logs("b1", tail=-5)

        assert result == []

    async def test_positive_tail_returns_the_last_n_lines(self, monkeypatch):
        lines = [f"line-{i}" for i in range(50)]
        _configure(monkeypatch, lines)

        result = await search_build_logs("b1", tail=3)

        assert result == ["line-47", "line-48", "line-49"]

    async def test_tail_is_clamped_before_being_used_as_the_api_page_size(
        self, monkeypatch
    ):
        lines = [f"line-{i}" for i in range(10)]
        client = _configure(monkeypatch, lines)

        await search_build_logs("b1", tail=1_000_000)

        _args, kwargs = client.query_logs.call_args
        assert kwargs["page_size"] == 2000
