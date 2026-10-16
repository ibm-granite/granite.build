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

"""Every recipe's ``step_uri`` names a step that exists.

Renaming or moving a step leaves no error behind in a recipe that still uses the
old name, and the recipe's own tests do not catch it either: they check the
rendered config, not that the URI resolves. Three recipes kept
``space://steps/distill-sft-baseline`` for weeks after the step became
``distill-sft``. The reference sat behind ``INCLUDE_SFT``, off by default, so it
would only have failed on the first run that turned SFT on.

This scans the raw build.yaml text rather than a rendered build, so URIs inside
template branches that are off by default are checked too.
"""

import pathlib
import re

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[3]
_ASSETS = _REPO / "configurations" / "assets"
_BUILTINS = _REPO / "src" / "gbserver" / "builtins" / "steps"
_STEP_URI = re.compile(r"step_uri:\s*[\"']?space://steps/([\w./-]+?)[\"']?\s*$", re.M)


def _recipe_step_uris():
    for build in sorted((_REPO / "recipes").glob("**/build.yaml")):
        for path in sorted(set(_STEP_URI.findall(build.read_text(encoding="utf-8")))):
            yield pytest.param(build, path, id=f"{build.parent.name}:{path}")


def _step_exists(path: str) -> bool:
    """A published step at any environment level, or a gbserver builtin."""
    return any(_ASSETS.glob(f"**/steps/{path}/step.yaml")) or any(
        _BUILTINS.glob(f"*/{path}/step.yaml")
    )


def test_recipes_are_scanned():
    """The regex still matches the recipes' format, or the check below is empty."""
    assert len(list(_recipe_step_uris())) > 50


@pytest.mark.parametrize("build, path", _recipe_step_uris())
def test_step_uri_names_an_existing_step(build, path):
    assert _step_exists(path), (
        f"{build.relative_to(_REPO)} uses space://steps/{path}, but no "
        f"steps/{path}/step.yaml is published under configurations/assets/ or "
        f"src/gbserver/builtins/steps/"
    )
