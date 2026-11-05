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

"""The granite-switch AWS recipes are templates anyone can run.

They and the environment templates they use must name no private data, code or
cluster, and no resource in a particular AWS account: a user brings their own
data and puts their own ids in an uncommitted copy (recipes/granite-switch/
skypilot/aws/README.md).
"""

import pathlib
import re

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[3]
_FILES = sorted(
    p
    for d in (
        _REPO / "recipes" / "granite-switch" / "skypilot" / "aws",
        _REPO / "configurations" / "assets" / "environments" / "skypilot" / "aws",
    )
    for p in d.rglob("*")
    if p.is_file()
    and p.suffix in (".yaml", ".md", ".sh")
    and (p.parent.name.startswith("gb-shared") or "recipes" in p.parts)
)
# Private data, code and clusters; AWS resource ids other than the placeholder.
_FORBIDDEN = re.compile(
    r"ibm-research|touma|intrinsics_training|bluevela|/proj\b|\.ibm\.com"
    r"|\b(fs|sg|vpc|subnet|fsap)-(?!REPLACE-ME\b)[0-9a-f]{8,17}\b|\b\d{12}\b",
    re.IGNORECASE,
)


def test_the_scan_covers_the_recipes_and_environments():
    names = {p.relative_to(_REPO).as_posix() for p in _FILES}
    assert any(n.endswith("aws/answerability-lora/build.yaml") for n in names)
    assert any(n.endswith("aws/answerability-sr/parameters.yaml") for n in names)
    assert any(n.endswith("aws/gb-shared/environment.yaml") for n in names)
    assert any(n.endswith("aws/gb-shared-usw2/environment.yaml") for n in names)


@pytest.mark.parametrize("path", _FILES, ids=lambda p: p.relative_to(_REPO).as_posix())
def test_names_nothing_private(path):
    hits = [
        f"{i}: {line.strip()}"
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if _FORBIDDEN.search(line)
    ]
    assert not hits
