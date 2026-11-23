#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""The granite-switch recipes carry one block of inline code in two copies:
the answerability-lora prepare script (slurm and aws).

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
_SLURM = _SKYPILOT / "slurm/answerability-lora/build.yaml"
_AWS = _SKYPILOT / "aws/answerability-lora/build.yaml"
# The prepare script is the heredoc from the `<<'PY'` line to its `PY` line.
_START, _END = "<<'PY'", "PY"


def _block(path, start, end):
    """The lines of path from the first line containing start through the
    next line that is just end, dedented together (Python stays valid)."""
    lines = path.read_text().splitlines()
    first = next(i for i, line in enumerate(lines) if start in line)
    last = next(i for i in range(first + 1, len(lines)) if lines[i].strip() == end)
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


def test_the_prepare_script_copies_are_identical():
    _assert_same(_SLURM, _AWS, _START, _END)


def test_a_changed_line_in_one_copy_fails(tmp_path):
    """The check is not vacuous: one edited line in one copy is reported."""
    edited = tmp_path / "build.yaml"
    edited.write_text(
        _AWS.read_text().replace("MAX_DROPPED = 0.01", "MAX_DROPPED = 0.02", 1)
    )
    with pytest.raises(AssertionError, match=r"\+ *MAX_DROPPED = 0.02"):
        _assert_same(_AWS, edited, _START, _END)


def test_the_block_ends_at_the_terminator_not_at_a_line_naming_it(tmp_path):
    """A line inside the heredoc that contains PY (PYTHONHASHSEED, COPY) does
    not cut the compared block short."""
    path = tmp_path / "build.yaml"
    path.write_text("  run <<'PY'\n  PYTHONHASHSEED = 0\n  x = 1\n  PY\n  after\n")
    assert _block(path, _START, _END) == [
        "run <<'PY'",
        "PYTHONHASHSEED = 0",
        "x = 1",
        "PY",
    ]
