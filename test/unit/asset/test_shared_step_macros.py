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

"""steps/train/shared/q.j2, the shell-quoting macro that openinstruct-sft,
hf-peft-sft and fms-hf-tuning gb-include. The steps' own render tests don't run in
CI, so its behaviour is pinned here."""

import pathlib

import pytest

from gbserver.utils.template import fill_template

_Q = (
    pathlib.Path(__file__).resolve().parents[3] / "steps" / "train" / "shared" / "q.j2"
).read_text()


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, ""),
        ("", ""),
        ("a b", "a b"),
        ("it's", "it'\"'\"'s"),
        (2, "2"),
        (False, "False"),
    ],
)
def test_q_quotes_for_a_single_quoted_shell_word(value, expected):
    """null renders as empty (unset), not as the word None; a ' is closed,
    escaped and reopened."""
    assert fill_template(_Q + "{{- q(v) }}", {"v": value}, strict=True) == expected
