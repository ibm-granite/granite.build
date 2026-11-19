#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

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
