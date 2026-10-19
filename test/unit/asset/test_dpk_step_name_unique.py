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

"""Exactly one step named ``dpk`` ships in this repo.

The step's URI is ``space://steps/data/dpk``. A bare ``space://steps/dpk`` is still
looked up by name: the resolver's env-class-match tier searches
``steps/**/dpk/step.yaml`` under every base_uri and, when several match, silently
picks one (fewest ``environment_configs`` keys first, then path order). The BlueVela
fixtures rely on that lookup on purpose, because in the gb-test assets repo the step
lives at ``steps/skypilot/dpk``, where ``skypilot`` is an env-partition dir.

A second ``dpk`` anywhere the resolver looks would make that lookup ambiguous without
any error. This includes a stale ``steps/dpk`` left behind by a move, or a ``dpk`` in
another category. The old in-repo URI ``space://steps/dpk`` is no longer supported
(see CHANGELOG.md), so nothing here pins its behaviour.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# Every tree in this repo that a space's base_uris can reach: the published assets
# (configurations/assets, chained by configurations/spaces/*) and the builtin steps.
SEARCH_ROOTS = [
    REPO_ROOT / "configurations" / "assets",
    REPO_ROOT / "src" / "gbserver" / "builtins",
]


def test_exactly_one_step_is_named_dpk():
    """Glob the same ``steps/**/dpk/step.yaml`` pattern the resolver uses."""
    found = sorted(
        p.parent.relative_to(REPO_ROOT).as_posix()
        for root in SEARCH_ROOTS
        for p in root.rglob("steps/**/dpk/step.yaml")
    )
    assert found == ["configurations/assets/environments/skypilot/steps/data/dpk"], (
        "a bare space://steps/dpk lookup silently picks one of several steps named "
        f"dpk; found {found}"
    )
