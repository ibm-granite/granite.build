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

"""The granite-switch recipes carry two blocks of inline code in two copies
each: the answerability-lora prepare script (slurm-bluevela and aws) and the
answerability-sr training-config generator (lsf-bluevela and aws).

`gb build start` uploads only build.yaml, so a recipe cannot ship a shared
script beside it yet (ibm-granite/granite.build#628); until it can
(ibm-granite/granite.build#650), these tests keep each pair identical, so
the same parameters prepare the same dataset and write the same SR config on
every cluster. Change both copies together.
"""

import difflib
import pathlib
import textwrap

import pytest

_SKYPILOT = (
    pathlib.Path(__file__).resolve().parents[3] / "recipes/granite-switch/skypilot"
)


def _block(path, start, end):
    """The lines of path from the first line containing start through the
    next line containing end, dedented together (Python stays valid)."""
    lines = path.read_text().splitlines()
    first = next(i for i, line in enumerate(lines) if start in line)
    last = next(i for i in range(first + 1, len(lines)) if end in lines[i])
    return textwrap.dedent("\n".join(lines[first : last + 1])).splitlines()


def _assert_same(a, b, start, end):
    """Fail with a unified diff when the block differs between the two files."""
    left, right = _block(a, start, end), _block(b, start, end)
    diff = "\n".join(
        difflib.unified_diff(
            left,
            right,
            str(a),
            str(b),
            lineterm="",
        )
    )
    assert not diff, f"the two copies differ; change both together:\n{diff}"


@pytest.mark.parametrize(
    "name,a,b,start,end",
    [
        (
            "prepare script",
            "slurm-bluevela/answerability-lora/build.yaml",
            "aws/answerability-lora/build.yaml",
            "<<'PY'",
            "PY",
        ),
        (
            "SR config generator",
            "lsf-bluevela/answerability-sr/build.yaml",
            "aws/answerability-sr/build.yaml",
            "printf '%s\\n'",
            '> "$CFG";',
        ),
    ],
)
def test_the_copies_are_identical(name, a, b, start, end):
    assert name
    _assert_same(_SKYPILOT / a, _SKYPILOT / b, start, end)


def test_a_changed_line_in_one_copy_fails(tmp_path):
    """The check is not vacuous: one edited line in one copy is reported."""
    src = _SKYPILOT / "aws/answerability-lora/build.yaml"
    edited = tmp_path / "build.yaml"
    edited.write_text(
        src.read_text().replace("MAX_DROPPED = 0.01", "MAX_DROPPED = 0.02", 1)
    )
    with pytest.raises(AssertionError, match=r"\+ *MAX_DROPPED = 0.02"):
        _assert_same(src, edited, "<<'PY'", "PY")
