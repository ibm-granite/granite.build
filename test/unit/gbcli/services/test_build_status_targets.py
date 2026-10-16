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

"""Unit tests for build-status target processing under in-place retry.

``process_target_runs`` / ``process_target_runs_to_json`` flatten the target runs
of a single build. With in-place retry every run lives on the one build id, so
they are ordered purely by start time — a target's FAILED run therefore lays out
ahead of the SUCCESS run that retried it, and the SUCCESS run carries
``retry_of_target_id`` pointing back at the FAILED run. All collaborators are
pure — no infrastructure required.
"""

import pytest

from gbcli.services.service_build import (
    process_target_runs,
    process_target_runs_to_json,
)

pytestmark = pytest.mark.standalone

_BUILD_ID = "build-1"


def _target_run(name, uuid, status, started_at, retry_of=""):
    return {
        "target": {
            "name": name,
            "build_id": _BUILD_ID,
            "uuid": uuid,
            "status": status,
            "started_at": started_at,
            "retry_of_target_id": retry_of,
        },
        "input_artifacts": [],
        "output_artifacts": [],
        "steps": [],
    }


def _runs():
    # One build: targetA succeeded on the first attempt; targetB failed then was
    # re-run and succeeded, so its SUCCESS run links back to the FAILED run via
    # retry_of_target_id. Returned out of order to exercise the start-time sort.
    return [
        _target_run(
            "targetB", "tB2", "success", "2020-01-01T00:02:00Z", retry_of="tB1"
        ),
        _target_run("targetB", "tB1", "failed", "2020-01-01T00:01:00Z"),
        _target_run("targetA", "tA1", "success", "2020-01-01T00:00:00Z"),
    ]


def test_plain_sorts_oldest_to_newest_by_start():
    targets = process_target_runs(_runs())

    # Oldest -> newest by start time; targetB's failed run precedes its retry.
    assert list(targets) == [
        "targetA (tA1)",
        "targetB (tB1)",
        "targetB (tB2)",
    ]

    retried = targets["targetB (tB2)"]
    assert retried["retry_of_target_id"] == "tB1"
    assert retried["build_id"] == _BUILD_ID
    # Each run carries its plain logical-target name (used to number targets so a
    # FAILED run and its SUCCESS retry share one number rather than two).
    assert retried["name"] == "targetB"
    failed = targets["targetB (tB1)"]
    assert failed["retry_of_target_id"] == ""
    assert failed["build_id"] == _BUILD_ID
    assert failed["name"] == "targetB"


def test_json_carries_retry_link_and_build_id():
    targets = process_target_runs_to_json(_runs())

    by_id = {t["target_id"]: t for t in targets}
    assert by_id["tB2"]["retry_of_target_id"] == "tB1"
    assert by_id["tB2"]["build_id"] == _BUILD_ID
    assert by_id["tB1"]["retry_of_target_id"] == ""
    assert by_id["tA1"]["build_id"] == _BUILD_ID
