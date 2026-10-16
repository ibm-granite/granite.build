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

"""Target numbering in the plain build-status output under in-place retry.

A target that FAILED then re-ran to SUCCESS contributes two "name (uuid)" keys on
the one build id; both runs must share a single logical target number, and later
targets must not shift as runs accumulate.
"""

import pytest

from gbcli.commands.command_build import _number_logical_targets

pytestmark = pytest.mark.standalone


def _targets(*items):
    """Build the ordered "name (uuid)" -> value mapping process_target_runs emits.

    Args:
        items: (name, uuid) pairs, in the order the runs appear.
    """
    return {f"{name} ({uuid})": {"name": name} for name, uuid in items}


def test_failed_then_success_run_share_one_number():
    # targetA succeeded; targetB failed then retried to SUCCESS (two runs).
    targets = _targets(
        ("targetA", "tA1"),
        ("targetB", "tB1"),
        ("targetB", "tB2"),
    )

    numbers = _number_logical_targets(targets)

    assert numbers["targetA (tA1)"] == 1
    # Both of targetB's runs are #2; a later target would not be pushed to #3.
    assert numbers["targetB (tB1)"] == 2
    assert numbers["targetB (tB2)"] == 2


def test_numbers_do_not_shift_when_a_retry_is_inserted():
    without_retry = _number_logical_targets(
        _targets(("targetA", "tA1"), ("targetB", "tB1"))
    )
    with_retry = _number_logical_targets(
        _targets(("targetA", "tA1"), ("targetB", "tB1"), ("targetB", "tB2"))
    )

    # targetB stays #2 whether or not it has retried; nothing is renumbered.
    assert without_retry["targetB (tB1)"] == 2
    assert with_retry["targetB (tB1)"] == 2
    assert with_retry["targetB (tB2)"] == 2


def test_distinct_targets_number_in_first_appearance_order():
    numbers = _number_logical_targets(
        _targets(("targetA", "tA1"), ("targetB", "tB1"), ("targetC", "tC1"))
    )

    assert [numbers[k] for k in numbers] == [1, 2, 3]


def test_empty_mapping_is_empty():
    assert _number_logical_targets({}) == {}
