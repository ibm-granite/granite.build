"""A recipe that declares ENVIRONMENT_URI routes every target through it.

The parameter exists so moving a recipe to another environment is a one-line change in
its parameters.yaml. A target that still spells the URI literally would stay behind on
the old environment, silently, after that change.
"""

import pathlib
import re

import pytest

_RECIPES = pathlib.Path(__file__).resolve().parents[3] / "recipes"
_DECLARING = sorted(
    p.parent
    for p in _RECIPES.glob("*/*/*/parameters.yaml")
    if re.search(r"^ENVIRONMENT_URI:", p.read_text(encoding="utf-8"), re.M)
)


def test_at_least_one_recipe_declares_it():
    assert _DECLARING


@pytest.mark.parametrize("recipe", _DECLARING, ids=lambda p: p.name)
def test_every_target_uses_the_parameter(recipe):
    uris = re.findall(
        r"^\s*environment_uri:\s*(.+?)\s*$",
        (recipe / "build.yaml").read_text(encoding="utf-8"),
        re.M,
    )
    assert uris
    assert set(uris) == {'"$${ENVIRONMENT_URI}"'}
