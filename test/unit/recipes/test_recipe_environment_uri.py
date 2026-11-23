# SPDX-License-Identifier: Apache-2.0

"""A recipe that declares ENVIRONMENT_URI routes every target through it.

The parameter exists so moving a recipe to another environment is a one-line change in
its parameters.yaml, or in the site file a recipe copies from its site.example.yaml (in
the recipe's directory or any directory above it, up to recipes/). A target that still
spells the URI literally
would stay behind on the old environment, silently, after that change.
"""

import pathlib
import re

import pytest

_RECIPES = pathlib.Path(__file__).resolve().parents[3] / "recipes"


def _declares(path):
    """True if path is a file with a top-level ENVIRONMENT_URI key."""
    return path.is_file() and bool(
        re.search(r"^ENVIRONMENT_URI:", path.read_text(encoding="utf-8"), re.M)
    )


def _site_example_declares(recipe):
    """True if a site.example.yaml in recipe or a directory above it, up to
    recipes/, declares ENVIRONMENT_URI."""
    for d in (recipe, *recipe.parents):
        if _declares(d / "site.example.yaml"):
            return True
        if d == _RECIPES:
            return False
    return False


_DECLARING = sorted(
    p.parent
    for p in _RECIPES.glob("**/build.yaml")
    if _declares(p.parent / "parameters.yaml") or _site_example_declares(p.parent)
)


def test_at_least_one_recipe_declares_it():
    assert _DECLARING


@pytest.mark.parametrize(
    "recipe", _DECLARING, ids=lambda p: str(p.relative_to(_RECIPES))
)
def test_every_target_uses_the_parameter(recipe):
    uris = re.findall(
        r"^\s*environment_uri:\s*(.+?)\s*$",
        (recipe / "build.yaml").read_text(encoding="utf-8"),
        re.M,
    )
    assert uris
    assert set(uris) == {'"$${ENVIRONMENT_URI}"'}
