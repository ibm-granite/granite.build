#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""Regression test for a code-review finding: tool_registry.py's _wrap()
used a hand-rolled `hasattr(result, "__await__")` check to decide whether to
await a tool function's result, instead of the standard library's
`inspect.isawaitable`. Covers both sync and async tool functions to lock in
that the swap didn't change behavior.
"""

from __future__ import annotations

import pytest

from gb_ui_backend.services.chat_agents.tool_registry import _wrap


@pytest.mark.asyncio
class TestWrapSupportsSyncAndAsyncTools:
    async def test_sync_tool_function_result_is_returned_directly(self):
        tool = _wrap("sync_tool", "desc", {}, lambda: "sync result")

        assert await tool.handler({}) == "sync result"

    async def test_async_tool_function_result_is_awaited(self):
        async def async_fn():
            return "async result"

        tool = _wrap("async_tool", "desc", {}, async_fn)

        assert await tool.handler({}) == "async result"
