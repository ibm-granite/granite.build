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

"""Regression tests for chat's gbmcp-less (degraded) mode.

`ToolLoopBackend` used to require gbmcp: __init__ raised RuntimeError without
`mcp`, and _resolve_gbmcp_bin() ran unconditionally at session start. Both ship
with the top-level granite.build distribution but NOT with
granite-build-analytics (src/gb_ui_backend's own pyproject.toml), so a consumer
installing only that distribution got a chat feature that advertised itself as
enabled via /chat/status — which only checks LLM config — and then failed on the
very first message. Chat now detects availability once and falls back to the
dashboard tools, which are pure Python and need no subprocess.

Why this file exists at all: **this repo's own CI always has gbmcp on PATH**, so
no other test in the suite can ever exercise that fallback. Every assertion here
covers an install shape our CI never reaches, which is exactly the shape an
external consumer runs. Without it, a future refactor re-breaks that consumer
silently and we hear about it from the consumer rather than from a red build.

Deliberately NOT in test_session_lifecycle.py, whose `backend` fixture patches
stdio_client/ClientSession/build_gbmcp_tools — that fixture builds the
gbmcp-*enabled* shape. The fixture here touches none of those names, so these
tests also pass in a venv where `mcp` isn't importable at all (where patching
`tool_loop_backend.stdio_client` would itself raise AttributeError).

Four things are covered, matching the four moving parts of the fallback:

  1. _gbmcp_available()'s own branches, including that it short-circuits before
     probing for the binary when `mcp` is missing.
  2. The mcp_session=None session-owner path: what it serves, that a real turn
     runs on it, that it closes without leaking its owner task, and both of
     confirm_action()'s degraded outcomes — the reachable one (pop returns
     not-found first) and the Optional-deref guard behind it, which needs an
     injected confirmation to reach at all.
  3. The conditional system prompt, both ways, across the cloud-logs matrix.
  4. Import safety with `mcp` genuinely absent (subprocess), plus a sweep
     proving no gbmcp tool name reaches the model through *either* channel —
     the system prompt or a tool description.
"""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from typing import Any, AsyncIterator

import pytest

from gb_ui_backend.config import Config
from gb_ui_backend.services.chat_agents import tool_loop_backend
from gb_ui_backend.services.chat_agents.gbmcp_policy import ALL_GBMCP_TOOLS
from gb_ui_backend.services.chat_agents.tool_loop_backend import (
    NAVIGATION_TOOL_NAME,
    ToolLoopBackend,
)
from gb_ui_backend.services.chat_agents.tool_registry import build_dashboard_tools

# The gbmcp tools the enabled system prompt names explicitly. Kept separate from
# ALL_GBMCP_TOOLS so the prompt assertions below test both directions: these must
# be absent when degraded and present when enabled. (ALL_GBMCP_TOOLS is the right
# set for the leak sweep, but the prompt never mentions all 18.)
_PROMPT_NAMED_GBMCP_TOOLS = (
    "secret_list",
    "secret_get",
    "secret_create",
    "secret_update",
    "build_start",
    "gbserver_stop",
    "build_status",
    "build_describe",
    "build_log",
)


def _names_in(text: str, names) -> set[str]:
    """Which of `names` appear in `text` as whole words.

    Word-boundary matching rather than a plain substring test, because
    `search_build_logs` — a *dashboard* tool, present even when degraded —
    contains `build_log`. A naive `"build_log" not in text` assertion passes only
    while cloud logs are unconfigured and silently starts failing the moment the
    matrix below covers cloud_logs_available=True. `_` is a \\w character, so
    \\bbuild_log\\b correctly does not match inside search_build_logs.
    """
    return {n for n in names if re.search(rf"\b{re.escape(n)}\b", text)}


class _StubProvider:
    """Minimal provider: echoes the user message back through history so a full
    stream_turn can be driven with no LLM credentials and no model round-trip."""

    PROVIDER_NAME = "stub"

    def __init__(self) -> None:
        self.model = "stub-model"
        self.tools_seen: list[list[Any]] = []

    async def run_turn(
        self,
        history: list[Any],
        tools: list[Any],
        user_message: str,
        event_queue: Any,
        interrupt_event: Any,
    ) -> AsyncIterator[dict]:
        # Recorded so a test can assert on exactly the tool list the model was
        # offered, which is the thing that actually matters here.
        self.tools_seen.append(list(tools))
        history.append({"role": "user", "content": user_message})
        reply = f"echo: {user_message}"
        history.append({"role": "assistant", "content": reply})
        yield {"type": "text_delta", "text": reply}


@pytest.fixture
def degraded_backend(monkeypatch):
    """A backend that believes gbmcp is unavailable.

    Patches only _gbmcp_available (read as a module global in __init__) and
    _build_provider. Notably absent: stdio_client/ClientSession/build_gbmcp_tools
    — see this module's docstring.
    """
    stub_provider = _StubProvider()
    monkeypatch.setattr(tool_loop_backend, "_gbmcp_available", lambda: False)
    monkeypatch.setattr(
        tool_loop_backend, "_build_provider", lambda config, prompt: stub_provider
    )
    backend = ToolLoopBackend(Config(_env_file=None))
    backend._stub_provider = stub_provider  # type: ignore[attr-defined]
    return backend


class TestGbmcpAvailabilityDetection:
    """_gbmcp_available() needs both the `mcp` client library and the gbmcp
    console script; either one missing means degraded."""

    def test_missing_mcp_short_circuits_before_probing_for_the_binary(
        self, monkeypatch
    ):
        """Order matters: without `mcp` there is nothing to talk to gbmcp over,
        so the binary probe (a filesystem stat plus a PATH scan) must not run."""

        def _must_not_be_called() -> str:
            raise AssertionError(
                "_resolve_gbmcp_bin() must not be probed when mcp is unavailable"
            )

        monkeypatch.setattr(tool_loop_backend, "_MCP_AVAILABLE", False)
        monkeypatch.setattr(
            tool_loop_backend, "_resolve_gbmcp_bin", _must_not_be_called
        )

        assert tool_loop_backend._gbmcp_available() is False

    def test_unresolvable_gbmcp_script_means_unavailable(self, monkeypatch):
        """`mcp` present but no gbmcp script — the granite-build-analytics
        install shape once its [chat] extra pulls in `mcp`."""

        def _raise() -> str:
            raise RuntimeError("gbmcp console script not found")

        monkeypatch.setattr(tool_loop_backend, "_MCP_AVAILABLE", True)
        monkeypatch.setattr(tool_loop_backend, "_resolve_gbmcp_bin", _raise)

        assert tool_loop_backend._gbmcp_available() is False

    def test_both_present_means_available(self, monkeypatch):
        monkeypatch.setattr(tool_loop_backend, "_MCP_AVAILABLE", True)
        monkeypatch.setattr(
            tool_loop_backend, "_resolve_gbmcp_bin", lambda: "/venv/bin/gbmcp"
        )

        assert tool_loop_backend._gbmcp_available() is True


@pytest.mark.asyncio
class TestDegradedSessionServesReadOnlyToolsOnly:
    async def test_backend_constructs_without_gbmcp(self, degraded_backend):
        """The original bug: this raised RuntimeError in __init__."""
        assert degraded_backend._gbmcp_enabled is False
        assert degraded_backend.describe()["provider"] == "stub"

    async def test_session_has_no_mcp_session_and_only_read_only_tools(
        self, degraded_backend
    ):
        session = await degraded_backend._get_or_create_session("s1")

        assert session.mcp_session is None

        names = {t.name for t in session.tools}
        # suggest_navigation + the dashboard tools, and nothing else. Asserted as
        # equality rather than a few `in` checks so a gbmcp tool leaking into this
        # path fails here even if it isn't one of the names spelled out below.
        expected = {NAVIGATION_TOOL_NAME} | {
            t.name
            for t in build_dashboard_tools(
                degraded_backend._config, gbmcp_available=False
            )
        }
        assert names == expected
        assert NAVIGATION_TOOL_NAME in names
        assert not names & set(ALL_GBMCP_TOOLS)

        # Nothing can populate this without the confirmable-gbmcp handlers, which
        # is the premise confirm_action()'s None guard relies on.
        assert session.pending_confirmations == {}

    async def test_confirm_action_reports_not_found_instead_of_raising(
        self, degraded_backend
    ):
        """The normal degraded path through confirm_action(): pending_confirmations
        can never be populated (only build_confirmable_gbmcp_tools' handlers write
        to it, and those aren't built here), so the pop returns first and the
        Optional mcp_session is never reached.

        Note this asserts the *observable* safety, not the None-guard itself — the
        guard is unreachable from here by construction, and deleting it leaves this
        test passing. See
        test_confirm_action_on_an_injected_confirmation_reports_gbmcp_is_absent for
        coverage of the guard.
        """
        await degraded_backend._get_or_create_session("s1")

        for approved in (True, False):
            assert await degraded_backend.confirm_action(
                "s1", "bogus-confirmation-id", approved=approved
            ) == {"found": False}

    async def test_confirm_action_on_an_injected_confirmation_reports_gbmcp_is_absent(
        self, degraded_backend
    ):
        """Exercises the mcp_session None-guard directly, by being the "future
        caller that populates pending_confirmations some other way" the guard exists
        for. Injecting the entry is the only way to reach it — which is the point:
        the guard is defense-in-depth, so without this the suite would pass with it
        deleted.

        Both with and without the guard the failure is caught by confirm_action's
        except and reported rather than raised, so the assertion is on *which*
        failure: a deliberate "gbmcp is not available" message, not an
        AttributeError on None that happened to land in the generic handler.
        """
        session = await degraded_backend._get_or_create_session("s1")
        session.pending_confirmations["injected-id"] = {
            "action": "build_start",
            "args": {},
        }

        result = await degraded_backend.confirm_action(
            "s1", "injected-id", approved=True
        )

        assert result["found"] is True
        assert result["is_error"] is True
        assert "gbmcp is not available" in result["result"]
        # The tell-tale of an unguarded None deref reaching the generic handler.
        assert "NoneType" not in result["result"]

    async def test_unknown_session_is_reported_not_found(self, degraded_backend):
        assert await degraded_backend.confirm_action("nope", "any-id", True) == {
            "found": False
        }

    async def test_a_full_turn_completes(self, degraded_backend):
        """Constructing the session isn't enough — the original failure was on the
        first *message*, so drive one all the way through."""
        events = [
            event
            async for event in degraded_backend.stream_turn("s1", "hello", None, None)
        ]

        assert any(e.get("type") == "text_delta" for e in events)
        assert degraded_backend._sessions["s1"].history == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "echo: hello"},
        ]

        # The tool list the model was actually offered on a real turn.
        offered = {t.name for t in degraded_backend._stub_provider.tools_seen[0]}
        assert not offered & set(ALL_GBMCP_TOOLS)

    async def test_close_session_leaves_no_owner_task_behind(self, degraded_backend):
        session = await degraded_backend._get_or_create_session("s1")
        assert not session.owner_task.done()

        await degraded_backend.close_session("s1")

        assert "s1" not in degraded_backend._sessions
        assert session.owner_task.done()
        # _close_session_owner swallows and logs, so a failing owner task would
        # otherwise close "successfully" and hide the breakage.
        assert session.owner_task.exception() is None


class TestSystemPromptIsConditional:
    """Every tool the enabled prompt names is a gbmcp tool. Describing them with
    no such tool available has the model announce capabilities it cannot perform
    and then confabulate results, so the degraded prompt must not name them."""

    @pytest.mark.parametrize("cloud_logs_available", [False, True])
    def test_degraded_prompt_names_no_gbmcp_tools(self, cloud_logs_available):
        prompt = tool_loop_backend._build_system_prompt(
            cloud_logs_available=cloud_logs_available, gbmcp_available=False
        )

        leaked = _names_in(prompt, ALL_GBMCP_TOOLS)
        assert leaked == set(), f"degraded prompt names gbmcp tools: {sorted(leaked)}"

    @pytest.mark.parametrize("cloud_logs_available", [False, True])
    def test_degraded_prompt_states_it_is_read_only(self, cloud_logs_available):
        prompt = tool_loop_backend._build_system_prompt(
            cloud_logs_available=cloud_logs_available, gbmcp_available=False
        )

        assert "read-only" in prompt.lower()
        # Cancellation still has somewhere useful to send the user, so the prompt
        # must keep pointing there instead of only refusing.
        assert NAVIGATION_TOOL_NAME in prompt
        assert "build_detail" in prompt

    @pytest.mark.parametrize("cloud_logs_available", [False, True])
    def test_enabled_prompt_still_names_them(self, cloud_logs_available):
        """The other direction — without this, deleting the whole block would
        also pass the degraded assertions above."""
        prompt = tool_loop_backend._build_system_prompt(
            cloud_logs_available=cloud_logs_available, gbmcp_available=True
        )

        named = _names_in(prompt, _PROMPT_NAMED_GBMCP_TOOLS)
        assert named == set(_PROMPT_NAMED_GBMCP_TOOLS)

    def test_search_build_logs_tracks_cloud_logs_not_gbmcp(self):
        """search_build_logs is a dashboard tool gated on cloud-logs config, not on
        gbmcp — so it survives in the degraded prompt. Only its "prefer this over
        build_log" comparison drops, since build_log is the gbmcp-only half."""
        degraded = tool_loop_backend._build_system_prompt(
            cloud_logs_available=True, gbmcp_available=False
        )
        enabled = tool_loop_backend._build_system_prompt(
            cloud_logs_available=True, gbmcp_available=True
        )

        assert "search_build_logs" in degraded
        assert "search_build_logs" in enabled
        assert _names_in(degraded, ["build_log"]) == set()
        assert _names_in(enabled, ["build_log"]) == {"build_log"}

        for prompt in (
            tool_loop_backend._build_system_prompt(
                cloud_logs_available=False, gbmcp_available=False
            ),
            tool_loop_backend._build_system_prompt(
                cloud_logs_available=False, gbmcp_available=True
            ),
        ):
            assert "search_build_logs" not in prompt


class TestNoGbmcpToolNamesReachTheModelAtAll:
    """The prompt is only one of the two channels that describe tools to the
    model — every ToolSpec's own description is sent too. Both must be clean, or
    the model is told to reach for a tool that isn't in its tool list."""

    def test_degraded_tool_descriptions_name_no_gbmcp_tools(self):
        for cloud_logs_available in (False, True):
            config = Config(
                _env_file=None,
                **(
                    {
                        "cloud_logs_url": "https://logs.example.com",
                        "cloud_logs_api_key": "x",
                    }
                    if cloud_logs_available
                    else {}
                ),
            )
            for tool in build_dashboard_tools(config, gbmcp_available=False):
                leaked = _names_in(tool.description, ALL_GBMCP_TOOLS)
                assert leaked == set(), (
                    f"{tool.name}'s description names gbmcp tools "
                    f"{sorted(leaked)} in a deployment that has none"
                )

    def test_enabled_tool_descriptions_still_cross_reference_gbmcp(self):
        """The cross-references are useful when the tools exist — the degraded
        variant drops them rather than deleting them outright."""
        config = Config(
            _env_file=None,
            cloud_logs_url="https://logs.example.com",
            cloud_logs_api_key="x",
        )
        descriptions = {
            t.name: t.description
            for t in build_dashboard_tools(config, gbmcp_available=True)
        }

        assert _names_in(descriptions["search_builds"], ALL_GBMCP_TOOLS) == {
            "build_status",
            "build_describe",
        }
        assert _names_in(descriptions["search_build_logs"], ALL_GBMCP_TOOLS) == {
            "build_log"
        }

    def test_tool_names_and_schemas_do_not_vary_by_mode(self):
        """Only description prose is conditional. Names and JSON schemas must be
        identical, so nothing downstream (the ToolSpec sharing asserted in
        test_chat_page_context.py, or a provider's schema handling) shifts."""
        config = Config(_env_file=None)
        enabled = build_dashboard_tools(config, gbmcp_available=True)
        degraded = build_dashboard_tools(config, gbmcp_available=False)

        assert [t.name for t in enabled] == [t.name for t in degraded]
        assert [t.parameters for t in enabled] == [t.parameters for t in degraded]

    def test_default_is_the_gbmcp_enabled_wording(self):
        """The kwarg defaults to True so the one production call site is the only
        thing that decides, and existing callers/tests keep today's behavior."""
        config = Config(_env_file=None)
        assert [t.description for t in build_dashboard_tools(config)] == [
            t.description for t in build_dashboard_tools(config, gbmcp_available=True)
        ]


# Runs in a subprocess: `mcp` is importable in this repo's venv, and blocking it
# in-process would need importlib.reload, which creates a second module object
# while every other module keeps referring to the first. Precedent for shelling
# out: test/unit/standalone/test_gbcli_entry_points.py.
_MCP_ABSENT_SCRIPT = textwrap.dedent('''
    import sys


    class _BlockMcp:
        """Raises on any attempt to import mcp, mimicking a venv without it."""

        def find_spec(self, name, path=None, target=None):
            if name == "mcp" or name.startswith("mcp."):
                raise ImportError(f"{name} is blocked by this test")
            return None


    sys.meta_path.insert(0, _BlockMcp())

    from gb_ui_backend.config import Config
    from gb_ui_backend.services.chat_agents import tool_loop_backend, tool_registry

    assert tool_loop_backend._MCP_AVAILABLE is False, "_MCP_AVAILABLE should be False"
    assert tool_loop_backend._gbmcp_available() is False, "should report unavailable"
    assert "mcp" not in sys.modules, "something imported mcp at runtime"

    # tool_registry keeps ClientSession under TYPE_CHECKING; if that import moved
    # back to module scope, importing it above would already have raised.
    assert callable(tool_registry.build_gbmcp_tools)
    assert callable(tool_registry.build_confirmable_gbmcp_tools)

    tools = tool_registry.build_dashboard_tools(
        Config(_env_file=None), gbmcp_available=False
    )
    assert len(tools) == 9, f"expected 9 dashboard tools, got {len(tools)}"

    print("MCP_ABSENT_OK")
    ''')


class TestImportsSurviveMcpBeingAbsent:
    def test_modules_import_and_report_unavailable_without_mcp(self):
        """The only test here that exercises the real install shape rather than
        simulating it with monkeypatch."""
        result = subprocess.run(
            [sys.executable, "-c", _MCP_ABSENT_SCRIPT],
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0, (
            f"importing chat modules without mcp failed\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert "MCP_ABSENT_OK" in result.stdout
