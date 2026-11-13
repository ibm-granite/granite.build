#!/usr/bin/env python3

# Copyright Granite.Build Authors
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

The step's URI is ``space://steps/data/dpk``. ``data`` is a category folder, so it is
part of the name: a bare ``space://steps/dpk`` does not reach it. A bare name only
matches a ``dpk`` directly under ``steps/`` or under an env-class folder there (e.g.
``steps/skypilot/dpk``); see "Step names" in docs/environments/step-resolution.md.

A second ``dpk`` would make the name ambiguous. A stale ``steps/dpk`` left behind by
a move, or one under an env-class folder, would be what a bare ``space://steps/dpk``
resolves to; a ``dpk`` in another category would give two steps the same leaf name.
So this test checks the broader ``steps/**/dpk/step.yaml`` pattern, not just the
names the resolver matches. The old in-repo URI ``space://steps/dpk`` is no longer
supported (see CHANGELOG.md).
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
    """Glob ``steps/**/dpk/step.yaml``, which covers every place a ``dpk`` could sit."""
    found = sorted(
        p.parent.relative_to(REPO_ROOT).as_posix()
        for root in SEARCH_ROOTS
        for p in root.rglob("steps/**/dpk/step.yaml")
    )
    assert found == ["configurations/assets/environments/skypilot/steps/data/dpk"], (
        "more than one step is named dpk, so its name is ambiguous; " f"found {found}"
    )
