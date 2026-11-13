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
    for p in _RECIPES.glob("**/parameters.yaml")
    if re.search(r"^ENVIRONMENT_URI:", p.read_text(encoding="utf-8"), re.M)
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
