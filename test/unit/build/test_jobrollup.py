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

from gbserver.build.jobrollup import pick_winning_runs, resolve_spec_targets
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


def _chain_root_failed_retry_succeeded():
    """The issue's scenario: root ran targetA (ok) + targetB (failed); the retry
    skipped targetA for reuse and re-ran targetB successfully."""
    root = _build(Status.FAILED, retry_count=0, targets=["targetA", "targetB"])
    retry = _build(
        Status.SUCCESS,
        retry_count=1,
        retry_of=root.uuid,
        targets=["targetA", "targetB"],
    )
    root_a = _run(root.uuid, "targetA", Status.SUCCESS, "2020-01-01T00:00:00Z")
    root_b = _run(root.uuid, "targetB", Status.FAILED, "2020-01-01T00:01:00Z")
    # A reused target is SUCCESS with no start time and a pointer back to the run
    # that really executed. It produces no artifacts of its own.
    retry_a = _run(retry.uuid, "targetA", Status.SUCCESS, None, skipped_for=root_a.uuid)
    retry_b = _run(retry.uuid, "targetB", Status.SUCCESS, "2020-01-01T00:02:00Z")
    return [(root, [root_a, root_b]), (retry, [retry_a, retry_b])], root_a, retry_b


def test_winner_for_a_reused_target_is_the_run_that_executed():
    chain, root_a, _retry_b = _chain_root_failed_retry_succeeded()

    winners = pick_winning_runs(chain)

    # Reuse only ever matches a run inside the same chain, so the run that
    # executed targetA is in the group alongside the retry's skip marker and is
    # selected directly. No pointer had to be followed, so none is recorded.
    assert winners["targetA"].run.uuid == root_a.uuid
    assert winners["targetA"].reused_from_target_run_id == ""


def test_winner_carries_the_build_that_owns_the_run():
    # create_jobstats_for_target rejects a run whose build_id does not match the
    # build passed alongside it, so the pairing must hold for every winner —
    # including one reached through a pointer into an earlier attempt.
    chain, _root_a, _retry_b = _chain_root_failed_retry_succeeded()

    winners = pick_winning_runs(chain)

    assert winners
    for name, winner in winners.items():
        assert winner.run.build_id == winner.build.uuid, name


def test_winner_for_a_rerun_target_is_the_successful_attempt():
    chain, _root_a, retry_b = _chain_root_failed_retry_succeeded()

    winners = pick_winning_runs(chain)

    assert winners["targetB"].run.uuid == retry_b.uuid
    assert winners["targetB"].reused_from_target_run_id == ""


def test_targets_that_never_succeeded_have_no_winner():
    root = _build(Status.FAILED, targets=["targetA"])
    chain = [
        (root, [_run(root.uuid, "targetA", Status.FAILED, "2020-01-01T00:00:00Z")])
    ]

    assert not pick_winning_runs(chain)


def test_earliest_executing_success_wins_over_a_later_one():
    root = _build(Status.FAILED, targets=["targetA"])
    retry = _build(Status.SUCCESS, retry_count=1, retry_of=root.uuid)
    first = _run(root.uuid, "targetA", Status.SUCCESS, "2020-01-01T00:00:00Z")
    second = _run(retry.uuid, "targetA", Status.SUCCESS, "2020-01-01T00:05:00Z")
    chain = [(root, [first]), (retry, [second])]

    assert pick_winning_runs(chain)["targetA"].run.uuid == first.uuid


def test_winner_is_dereferenced_for_a_cross_name_reuse():
    # Legitimate, not anomalous: the definition hash excludes the target name, so
    # targetB was skipped for an identically-configured targetA run. targetB's
    # group therefore holds nothing but the skip marker and has no executed
    # success of its own, so the pointer must be followed to the run that produced
    # the artifacts. The pointer is recorded because provenance is indirect: the
    # winning run is not targetB's own run.
    root = _build(Status.SUCCESS, targets=["targetA", "targetB"])
    retry = _build(Status.SUCCESS, retry_count=1, retry_of=root.uuid)
    produced = _run(root.uuid, "targetA", Status.SUCCESS, "2020-01-01T00:00:00Z")
    marker = _run(
        retry.uuid, "targetB", Status.SUCCESS, None, skipped_for=produced.uuid
    )
    chain = [(root, [produced]), (retry, [marker])]

    winner = pick_winning_runs(chain)["targetB"]

    assert winner.run.uuid == produced.uuid
    assert winner.build.uuid == root.uuid
    assert winner.reused_from_target_run_id == produced.uuid


def test_unresolvable_skip_pointer_falls_back_without_losing_the_pointer():
    # The pointer normally targets a run in the same chain. If it cannot be
    # resolved we must still report something rather than raise, and must not
    # silently drop the pointer.
    root = _build(Status.SUCCESS, targets=["targetA"])
    orphan = _run(
        root.uuid, "targetA", Status.SUCCESS, None, skipped_for="missing-uuid"
    )
    chain = [(root, [orphan])]

    winner = pick_winning_runs(chain)["targetA"]

    assert winner.run.uuid == orphan.uuid
    assert winner.reused_from_target_run_id == "missing-uuid"


def test_winner_run_is_successful_even_when_the_pointer_is_not():
    # Task 3's counts partition depends on a winner always meaning "this target
    # succeeded", so a pointer into a failed run must not drag a FAILED run in.
    root = _build(Status.FAILED, targets=["targetA"])
    failed = _run(root.uuid, "targetB", Status.FAILED, "2020-01-01T00:00:00Z")
    marker = _run(root.uuid, "targetA", Status.SUCCESS, None, skipped_for=failed.uuid)
    chain = [(root, [failed, marker])]

    winner = pick_winning_runs(chain)["targetA"]

    assert winner.run.status == Status.SUCCESS
    assert winner.reused_from_target_run_id == failed.uuid


def test_duplicate_runs_in_one_member_pick_the_earliest():
    # The grouping sorts within a member by start time, so the earlier run wins
    # even when storage hands them back in the other order.
    root = _build(Status.SUCCESS, targets=["targetA"])
    later = _run(root.uuid, "targetA", Status.SUCCESS, "2020-01-01T00:05:00Z")
    earlier = _run(root.uuid, "targetA", Status.SUCCESS, "2020-01-01T00:00:00Z")
    chain = [(root, [later, earlier])]

    assert pick_winning_runs(chain)["targetA"].run.uuid == earlier.uuid


def test_empty_chain_has_no_winners():
    assert not pick_winning_runs([])
