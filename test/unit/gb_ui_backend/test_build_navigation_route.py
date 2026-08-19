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

"""Regression test for a code-review finding: build_navigation_route used
`reason` verbatim as the ui_action's `label`. ChatWidget.tsx only renders
the navigation card when `label` is truthy, so an empty/whitespace `reason`
from the model silently dropped the card while the tool still reported
success back to the model — which then believed it had shown the user an
offer that never actually appeared.
"""

from __future__ import annotations

from gb_ui_backend.services.chat_agents.ui_actions import build_navigation_route


class TestBuildNavigationRouteLabelFallback:
    def test_empty_reason_falls_back_to_a_non_empty_label(self):
        result = build_navigation_route("dashboard", "")

        assert result["label"]

    def test_whitespace_only_reason_falls_back_to_a_non_empty_label(self):
        result = build_navigation_route("dashboard", "   ")

        assert result["label"].strip() == result["label"]
        assert result["label"]

    def test_real_reason_is_used_verbatim(self):
        result = build_navigation_route("dashboard", "you asked about recent builds")

        assert result["label"] == "you asked about recent builds"
