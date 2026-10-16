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

"""Regression test for the gbmcp tool policy: every tool falls into exactly
one of ALLOWED_GBMCP_TOOLS (executes directly), CONFIRMABLE_GBMCP_TOOLS
(available, but only via propose-then-confirm — see tool_registry.py's
build_confirmable_gbmcp_tools() and ToolLoopBackend.confirm_action()), or
DISALLOWED_GBMCP_TOOLS (never described to the model at all). This is a
deliberate, explicit product decision (not a default to be widened/narrowed
casually) — this test exists so a future edit to gbmcp_policy.py that
accidentally moves a tool between buckets, or drops one, fails loudly
instead of silently changing what an LLM-driven chat surface can do.

build_start/build_cancel/gbserver_stop were briefly allowed to execute
directly during development, then walked back after a security review (see
gbmcp_policy.py's module docstring). build_cancel is handled through
suggest_navigation instead (a real page — the build detail page — already
has the actual Cancel button). build_start/gbserver_start/gbserver_stop have
no equivalent page, so instead of being declined outright, they're
confirmable: the model can call them, but only ever proposes the action —
real execution happens later, outside the model loop, only if the user
clicks Approve. gbserver_start is gated for the same reason as
gbserver_stop even though it's a no-op in the common standalone deployment
(chat mounted into the very gbserver process it would start) — see
gbmcp_policy.py's module docstring.
"""

from gb_ui_backend.services.chat_agents.gbmcp_policy import (
    ALL_GBMCP_TOOLS,
    ALLOWED_GBMCP_TOOLS,
    CONFIRMABLE_GBMCP_TOOLS,
    DISALLOWED_GBMCP_TOOLS,
    KNOWN_MUTATING_GBMCP_TOOLS,
)

_DISALLOWED = {"secret_delete", "build_cancel"}
_CONFIRMABLE = {"build_start", "gbserver_start", "gbserver_stop"}


class TestGbmcpPolicy:
    def test_disallowed_tools_are_exactly_these_two(self):
        assert set(DISALLOWED_GBMCP_TOOLS) == _DISALLOWED

    def test_confirmable_tools_are_exactly_these_three(self):
        assert set(CONFIRMABLE_GBMCP_TOOLS) == _CONFIRMABLE

    def test_allowed_tools_are_everything_else(self):
        assert (
            set(ALLOWED_GBMCP_TOOLS)
            == set(ALL_GBMCP_TOOLS) - _DISALLOWED - _CONFIRMABLE
        )

    def test_the_three_buckets_partition_all_tools_with_no_overlap(self):
        allowed, confirmable, disallowed = (
            set(ALLOWED_GBMCP_TOOLS),
            set(CONFIRMABLE_GBMCP_TOOLS),
            set(DISALLOWED_GBMCP_TOOLS),
        )
        assert allowed & confirmable == set()
        assert allowed & disallowed == set()
        assert confirmable & disallowed == set()
        assert allowed | confirmable | disallowed == set(ALL_GBMCP_TOOLS)

    def test_secret_crud_other_than_delete_is_still_directly_allowed(self):
        """These never execute a real mutation — get/create/update only ever
        return a CLI command for the user to run themselves, and list is
        read-only — so they were never part of the injection-impact concern
        that got build_start/build_cancel/gbserver_stop gated/moved."""
        for tool in ("secret_list", "secret_get", "secret_create", "secret_update"):
            assert (
                tool in ALLOWED_GBMCP_TOOLS
            ), f"{tool} should still be directly allowed"

    def test_all_18_known_gbmcp_tools_are_accounted_for(self):
        assert len(ALL_GBMCP_TOOLS) == 18
        assert len(set(ALL_GBMCP_TOOLS)) == 18  # no duplicates

    def test_no_known_mutating_tool_is_auto_approved(self):
        """Guards the code-review concern that a future gbmcp addition could
        silently land in ALLOWED_GBMCP_TOOLS without a page or confirm gate —
        see gbmcp_policy.py's KNOWN_MUTATING_GBMCP_TOOLS docstring."""
        assert set(KNOWN_MUTATING_GBMCP_TOOLS) & set(ALLOWED_GBMCP_TOOLS) == set()
