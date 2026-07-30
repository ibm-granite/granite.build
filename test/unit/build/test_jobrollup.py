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

"""Unit tests for the retry-chain job roll-up.

All collaborators are pure: tests build StoredBuild / StoredTargetRun objects by
hand and pass them in root-first, so no storage or FastAPI is involved.
"""

import pytest

from gbserver.build.jobrollup import resolve_spec_targets
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.status import Status

pytestmark = pytest.mark.standalone


def _build(status=Status.SUCCESS, retry_count=0, retry_of=None, targets=None):
    return StoredBuild(
        name="test",
        space_name="testspace",
        source_uri="",
        username="tester",
        status=status,
        retry_count=retry_count,
        retry_of_build_id=retry_of,
        targets=targets,
    )


def _run(build_id, name, status=Status.SUCCESS, started_at=None, skipped_for=""):
    return StoredTargetRun(
        name=name,
        build_id=build_id,
        environment_uri="space://environments/bash",
        status=status,
        started_at=started_at,
        skipped_for_prerun_target_id=skipped_for,
    )


def test_spec_targets_prefers_the_root_target_list():
    # root.targets is authoritative: it is the only way to know targetC exists in
    # the spec, since a target whose dependency failed has no run at all.
    root = _build(targets=["targetB", "targetA", "targetC"])
    chain = [(root, [_run(root.uuid, "targetA"), _run(root.uuid, "targetB")])]

    assert resolve_spec_targets(root, chain) == ["targetB", "targetA", "targetC"]


def test_spec_targets_fall_back_to_observed_names_sorted():
    root = _build(targets=None)
    chain = [(root, [_run(root.uuid, "targetB"), _run(root.uuid, "targetA")])]

    assert resolve_spec_targets(root, chain) == ["targetA", "targetB"]


def test_spec_targets_dedupes_the_root_list():
    root = _build(targets=["targetA", "targetA", "targetB"])

    assert resolve_spec_targets(root, [(root, [])]) == ["targetA", "targetB"]
