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

"""Unit tests for `gb build status` skipped-target filtering.

Targets reused from an earlier attempt (retry or continuation) carry a
`skipped_for_prerun_target_id`. They run no steps and produce no artifacts, so
`execution_status_plain_output` hides them by default and only shows them when
`--show-skipped-targets` is passed. Both the targets overview (summary) and the
per-target sections are affected.
"""

import pytest

from gbcli.commands.command_build import execution_status_plain_output

pytestmark = pytest.mark.standalone


_DETAILS = {
    "build_id": "b-123",
    "name": "demo",
    "description": "",
    "status": "success",
    "started_at": "2026-08-20T10:00:00Z",
    "updated_at": "2026-08-20T10:05:00Z",
    "source_pr": "",
}


def _target(status="success", skipped_id=""):
    return {
        "status": status,
        "build_id": "b-123",
        "skipped_for_prerun_target_id": skipped_id,
        "input_artifacts": [],
        "output_artifacts": [],
        "steps": [],
    }


def _targets():
    # One target that actually ran, one reused/skipped from a prior attempt.
    return {
        "ran-target (u1)": _target(status="success"),
        "skipped-target (u2)": _target(status="success", skipped_id="prior-u2"),
    }


def test_skipped_targets_hidden_by_default():
    out = execution_status_plain_output(
        _DETAILS, _targets(), history=[], show_events=False
    )
    assert "ran-target (u1)" in out
    # Hidden from both the overview and the per-target sections. Match on the
    # unique target key (the flag name in the hint also contains the substring
    # "skipped-target", so assert on the target identity instead).
    assert "skipped-target (u2)" not in out
    assert "SKIPPED" not in out
    # A hint tells the user how to reveal them.
    assert "1 skipped target hidden" in out
    assert "--show-skipped-targets" in out


def test_skipped_targets_shown_with_flag():
    out = execution_status_plain_output(
        _DETAILS,
        _targets(),
        history=[],
        show_events=False,
        show_skipped_targets=True,
    )
    assert "ran-target (u1)" in out
    assert "skipped-target (u2)" in out
    assert "SKIPPED" in out
    # No "hidden" hint when nothing is hidden.
    assert "hidden" not in out


def test_no_hint_when_no_skipped_targets():
    targets = {"ran-target (u1)": _target(status="success")}
    out = execution_status_plain_output(
        _DETAILS, targets, history=[], show_events=False
    )
    assert "ran-target" in out
    assert "hidden" not in out


def test_hiding_skipped_targets_preserves_original_numbering():
    """Hiding a skipped target must not renumber the survivors: a "Target #N"
    always reflects the build's real target order, so a visible target keeps its
    original position even when earlier targets are hidden."""
    # Order: #1 is skipped, #2 actually ran. Hiding #1 must leave the runner as #2.
    targets = {
        "skipped-first (u1)": _target(status="success", skipped_id="prior-u1"),
        "ran-second (u2)": _target(status="success"),
    }
    out = execution_status_plain_output(
        _DETAILS, targets, history=[], show_events=False
    )
    assert "Target #2 ran-second (u2)" in out
    # It must NOT be renumbered to #1.
    assert "Target #1 ran-second (u2)" not in out
