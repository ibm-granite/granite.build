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

"""Unit tests for build-status target processing across a retry chain.

``process_target_runs`` / ``process_target_runs_to_json`` flatten the merged
target runs from every chain member. A target reused from a previous attempt is
"skipped": it carries ``skipped_for_prerun_target_id`` and has no ``started_at``,
which must not break the start-time sort. All collaborators are pure — no
infrastructure required.
"""

import pytest

from gbcli.services.service_build import (
    process_target_runs,
    process_target_runs_to_json,
)

pytestmark = pytest.mark.standalone


def _target_run(name, build_id, uuid, status, started_at, chain_index, skipped_for=""):
    return {
        "target": {
            "name": name,
            "build_id": build_id,
            "uuid": uuid,
            "status": status,
            "started_at": started_at,
            "skipped_for_prerun_target_id": skipped_for,
        },
        # Stamped during the chain merge in build_status; the attempt's position
        # in the chain (root == 0) is the primary sort key.
        "_chain_index": chain_index,
        "input_artifacts": [],
        "output_artifacts": [],
        "steps": [],
    }


def _merged_runs():
    # A retry chain: root (chain_index 0) ran targetA (ok) + targetB (failed);
    # the retry (chain_index 1) skipped targetA (no started_at) and re-ran
    # targetB. Returned out of order to exercise the sort.
    return [
        _target_run("targetB", "retry", "tB2", "success", "2020-01-01T00:02:00Z", 1),
        _target_run("targetA", "retry", "tA2", "success", None, 1, skipped_for="tA1"),
        _target_run("targetA", "root", "tA1", "success", "2020-01-01T00:00:00Z", 0),
        _target_run("targetB", "root", "tB1", "failed", "2020-01-01T00:01:00Z", 0),
    ]


def test_plain_sorts_oldest_to_newest_by_attempt_then_start():
    targets = process_target_runs(_merged_runs())

    # Oldest -> newest: the root attempt's targets first (by start time), then
    # the retry attempt's. The skipped target (no started_at) sorts ahead of the
    # re-run within its own attempt rather than jumping to the front overall.
    assert list(targets) == [
        "targetA (tA1)",
        "targetB (tB1)",
        "targetA (tA2)",
        "targetB (tB2)",
    ]

    skipped = targets["targetA (tA2)"]
    assert skipped["skipped_for_prerun_target_id"] == "tA1"
    assert skipped["build_id"] == "retry"
    ran = targets["targetA (tA1)"]
    assert ran["skipped_for_prerun_target_id"] == ""
    assert ran["build_id"] == "root"


def test_json_carries_skip_marker_and_build_id():
    targets = process_target_runs_to_json(_merged_runs())

    by_id = {t["target_id"]: t for t in targets}
    assert by_id["tA2"]["skipped_for_prerun_target_id"] == "tA1"
    assert by_id["tA2"]["build_id"] == "retry"
    assert by_id["tA1"]["skipped_for_prerun_target_id"] == ""
    assert by_id["tB1"]["build_id"] == "root"
