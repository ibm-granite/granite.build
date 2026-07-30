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
hand and pass them in root-first, so no database access and no HTTP layer is
involved.
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

    assert resolve_spec_targets(chain) == ["targetB", "targetA", "targetC"]


def test_spec_targets_fall_back_to_observed_names_sorted():
    root = _build(targets=None)
    chain = [(root, [_run(root.uuid, "targetB"), _run(root.uuid, "targetA")])]

    assert resolve_spec_targets(chain) == ["targetA", "targetB"]


def test_spec_targets_dedupes_the_root_list():
    root = _build(targets=["targetA", "targetA", "targetB"])

    assert resolve_spec_targets([(root, [])]) == ["targetA", "targetB"]


def test_spec_targets_treat_an_empty_root_list_like_none():
    # [] and None both mean "not specified". Pinned so a future change to
    # `if root.targets is not None` cannot silently zero out the denominator.
    root = _build(targets=[])

    assert resolve_spec_targets([(root, [_run(root.uuid, "targetA")])]) == ["targetA"]


def test_spec_targets_fallback_unions_names_across_members():
    root = _build(targets=None)
    retry = _build(targets=None, retry_count=1, retry_of=root.uuid)
    chain = [
        (root, [_run(root.uuid, "targetB")]),
        (retry, [_run(retry.uuid, "targetA"), _run(retry.uuid, "targetB")]),
    ]

    assert resolve_spec_targets(chain) == ["targetA", "targetB"]
