#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""The granite-switch recipes carry one block of inline code in two copies:
the answerability-lora prepare script (slurm-bluevela and aws).

`gb build start` uploads only build.yaml, so a recipe cannot ship a shared
script beside it yet (ibm-granite/granite.build#628); until it can
(ibm-granite/granite.build#650), these tests keep the two copies identical,
so the same parameters prepare the same dataset on every cluster. Change both
copies together.
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
