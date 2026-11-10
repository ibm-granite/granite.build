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

"""The eval recipes mirror the step tree: recipes/evals/<family>/<benchmark>/skypilot.

The 41 benchmark recipes live under ``recipes/evals/<family>/<benchmark>/skypilot/``,
one per benchmark, grouped by the step family that runs it — mirroring
``steps/eval/<family>/skypilot/`` so the step <-> recipe mapping is structural.
A recipe that referenced another family's step would still run, but the grouping
is what reviewers navigate by, so this suite pins it to disk.
"""

import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
RECIPES_DIR = REPO_ROOT / "recipes" / "evals"

# The granite-evals image families, mirroring steps/eval/ and the contract
# tests' _FAMILIES (steps/eval also holds eval-example, a pre-existing
# exemplar step outside this contract).
FAMILIES = (
    "bfcl",
    "bird",
    "ifbench",
    "judged",
    "lmeval",
    "nemoskills",
    "swebench",
    "tau",
    "tbench",
)

_STEP_REF = re.compile(r"space://steps/eval/([a-z0-9-]+)")


def _recipe_yamls() -> list[pathlib.Path]:
    return sorted(RECIPES_DIR.glob("*/*/skypilot/build.yaml"))


def _rel(path: pathlib.Path) -> str:
    return path.relative_to(RECIPES_DIR).as_posix()


def test_every_recipe_sits_at_family_benchmark_skypilot():
    """The glob must see every build.yaml: none may hide at another depth."""
    everywhere = {
        p.relative_to(RECIPES_DIR).as_posix() for p in RECIPES_DIR.rglob("build.yaml")
    }
    assert everywhere == {_rel(p) for p in _recipe_yamls()}


def test_family_directories_are_the_step_families():
    assert {p.name for p in RECIPES_DIR.iterdir() if p.is_dir()} == set(FAMILIES)


@pytest.mark.parametrize("family", FAMILIES)
def test_every_family_has_a_recipe(family):
    assert any(p.parents[2].name == family for p in _recipe_yamls())


@pytest.mark.parametrize("family", FAMILIES)
def test_every_family_backs_onto_a_real_step(family):
    """A recipe family dir with no step (or unpublished step) is a typo."""
    assert (
        REPO_ROOT / "steps" / "eval" / family / "skypilot" / "step-template.yaml"
    ).is_file()
    assert (
        REPO_ROOT
        / "configurations"
        / "assets"
        / "environments"
        / "skypilot"
        / "steps"
        / "eval"
        / family
        / "step.yaml"
    ).is_file()


@pytest.mark.parametrize("build_yaml", _recipe_yamls(), ids=_rel)
def test_recipe_references_its_own_familys_step(build_yaml):
    """The <family> path segment and the step_uri must agree."""
    family = build_yaml.parents[2].name
    referenced = set(_STEP_REF.findall(build_yaml.read_text(encoding="utf-8")))
    assert referenced == {family}, (
        f"{_rel(build_yaml)} sits under family {family!r} but references "
        f"{sorted(referenced)}"
    )
