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

from gbserver.build.jobrollup import pick_winning_runs, resolve_spec_targets, roll_up
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


def test_chain_that_finished_every_target_is_a_successful_job():
    # The bug in issue #222: the root stays FAILED, but the job did complete.
    chain, _root_a, _retry_b = _chain_root_failed_retry_succeeded()

    summary = roll_up(chain)

    assert summary.status == Status.SUCCESS
    assert summary.job_id == chain[0][0].uuid
    assert summary.attempts == 2
    assert summary.build_ids == [chain[0][0].uuid, chain[1][0].uuid]
    # attempt_builds carries the same members, root first, each with its own
    # honest per-attempt status — the root stayed FAILED, the retry SUCCEEDED.
    assert [(ab.build_id, ab.status) for ab in summary.attempt_builds] == [
        (chain[0][0].uuid, Status.FAILED),
        (chain[1][0].uuid, Status.SUCCESS),
    ]
    # pylint cannot see through pydantic's Field(default_factory=...) and infers
    # summary.counts as FieldInfo, tripping a false no-member; disable per line.
    assert summary.counts.model_dump() == {  # pylint: disable=no-member
        "total": 2,
        "succeeded": 2,
        "failed": 0,
        "running": 0,
        "not_run": 0,
    }
    # Each build keeps its own honest per-attempt status; nothing was mutated.
    assert chain[0][0].status == Status.FAILED


def test_partial_completion_is_failed_with_a_nonzero_success_count():
    # Exhausted chain: targetA succeeded, targetB never did, targetC never ran
    # because it depended on targetB. "Partial" is FAILED with succeeded > 0
    # rather than a new Status member.
    root = _build(Status.FAILED, targets=["targetA", "targetB", "targetC"])
    chain = [
        (
            root,
            [
                _run(root.uuid, "targetA", Status.SUCCESS, "2020-01-01T00:00:00Z"),
                _run(root.uuid, "targetB", Status.FAILED, "2020-01-01T00:01:00Z"),
            ],
        )
    ]

    summary = roll_up(chain)

    assert summary.status == Status.FAILED
    assert summary.counts.model_dump() == {  # pylint: disable=no-member
        "total": 3,
        "succeeded": 1,
        "failed": 1,
        "running": 0,
        "not_run": 1,
    }
    outcomes = {o.name: o for o in summary.targets}
    assert outcomes["targetC"].status is None
    assert outcomes["targetC"].target_run_id == ""
    assert outcomes["targetA"].status == Status.SUCCESS


def test_outcome_order_follows_the_spec_target_list():
    root = _build(Status.FAILED, targets=["targetC", "targetA", "targetB"])
    chain = [
        (root, [_run(root.uuid, "targetA", Status.FAILED, "2020-01-01T00:00:00Z")])
    ]

    assert [o.name for o in roll_up(chain).targets] == ["targetC", "targetA", "targetB"]


def test_outcome_name_is_the_spec_name_not_the_winning_runs_name():
    # Cross-name reuse is legitimate: the definition hash excludes the target
    # name, so targetB can be skipped for an identically-configured targetA run.
    # The outcome must still be reported under the spec name it satisfies.
    root = _build(Status.SUCCESS, targets=["targetA", "targetB"])
    retry = _build(Status.SUCCESS, retry_count=1, retry_of=root.uuid)
    produced = _run(root.uuid, "targetA", Status.SUCCESS, "2020-01-01T00:00:00Z")
    marker = _run(
        retry.uuid, "targetB", Status.SUCCESS, None, skipped_for=produced.uuid
    )
    chain = [(root, [produced]), (retry, [marker])]

    outcomes = {o.name: o for o in roll_up(chain).targets}

    assert set(outcomes) == {"targetA", "targetB"}
    assert outcomes["targetB"].target_run_id == produced.uuid
    assert outcomes["targetB"].reused_from_target_run_id == produced.uuid


def test_an_in_flight_member_makes_the_job_running():
    root = _build(Status.FAILED, targets=["targetA"])
    retry = _build(Status.RETRY_PENDING, retry_count=1, retry_of=root.uuid)
    chain = [
        (root, [_run(root.uuid, "targetA", Status.FAILED, "2020-01-01T00:00:00Z")]),
        (retry, []),
    ]

    assert roll_up(chain).status == Status.RUNNING


def test_a_running_target_is_counted_separately_from_failed():
    # An in-progress job must never report having failed a target it has not
    # failed, which is why `running` is its own bucket.
    root = _build(Status.RUNNING, targets=["targetA", "targetB"])
    chain = [
        (
            root,
            [
                _run(root.uuid, "targetA", Status.SUCCESS, "2020-01-01T00:00:00Z"),
                _run(root.uuid, "targetB", Status.RUNNING, "2020-01-01T00:01:00Z"),
            ],
        )
    ]

    summary = roll_up(chain)

    assert summary.status == Status.RUNNING
    assert summary.counts.running == 1  # pylint: disable=no-member
    assert summary.counts.failed == 0  # pylint: disable=no-member


def test_cancel_requested_on_any_member_wins():
    # Cancelling any member cancels the whole chain, so the job reports it even
    # though that member is otherwise finished.
    root = _build(Status.CANCEL_REQUESTED, targets=["targetA"])
    chain = [
        (root, [_run(root.uuid, "targetA", Status.SUCCESS, "2020-01-01T00:00:00Z")])
    ]

    assert roll_up(chain).status == Status.CANCEL_REQUESTED


def test_cancelled_chain_is_not_reported_as_failed():
    root = _build(Status.CANCELLED, targets=["targetA", "targetB"])
    chain = [
        (root, [_run(root.uuid, "targetA", Status.SUCCESS, "2020-01-01T00:00:00Z")])
    ]

    assert roll_up(chain).status == Status.CANCELLED


def test_invalid_root_with_nothing_succeeded_is_invalid():
    root = _build(Status.INVALID, targets=["targetA"])

    assert roll_up([(root, [])]).status == Status.INVALID


def test_job_with_no_spec_targets_defers_to_the_latest_attempt():
    # Nothing to aggregate, so reporting FAILED would be a lie.
    root = _build(Status.SUCCESS, targets=None)

    assert roll_up([(root, [])]).status == Status.SUCCESS


def test_empty_chain_returns_an_empty_summary_rather_than_raising():
    # The roll-up is additive to endpoints that already work, so it must never
    # turn a working request into a 500.
    summary = roll_up([])

    assert summary.job_id == ""
    assert summary.targets == []


def test_attempt_records_the_owning_members_retry_count():
    chain, _root_a, _retry_b = _chain_root_failed_retry_succeeded()

    outcomes = {o.name: o for o in roll_up(chain).targets}

    # targetA succeeded in the root (attempt 0); targetB in the retry (attempt 1).
    assert outcomes["targetA"].attempt == 0
    assert outcomes["targetB"].attempt == 1


def test_counts_always_partition_the_spec_targets():
    chain, _a, _b = _chain_root_failed_retry_succeeded()
    counts = roll_up(chain).counts.model_dump()  # pylint: disable=no-member

    assert counts["total"] == (
        counts["succeeded"] + counts["failed"] + counts["running"] + counts["not_run"]
    )


def test_non_authoritative_targets_do_not_promote_a_failed_build_to_success():
    # targets=None (build-everything default): counts.total is only what ran, so
    # "all counted succeeded" must NOT win unless the newest attempt succeeded.
    root = _build(Status.FAILED, targets=None)
    chain = [
        (root, [_run(root.uuid, "targetA", Status.SUCCESS, "2020-01-01T00:00:00Z")])
    ]

    assert roll_up(chain).status == Status.FAILED


def test_non_authoritative_cancelled_build_with_a_succeeded_run_is_cancelled():
    root = _build(Status.CANCELLED, targets=None)
    chain = [
        (root, [_run(root.uuid, "targetA", Status.SUCCESS, "2020-01-01T00:00:00Z")])
    ]

    assert roll_up(chain).status == Status.CANCELLED


def test_non_authoritative_invalid_build_with_a_succeeded_run_is_failed():
    # A build that produced a successful run but ended INVALID is reported FAILED,
    # not INVALID: the INVALID row requires nothing to have succeeded.
    root = _build(Status.INVALID, targets=None)
    chain = [
        (root, [_run(root.uuid, "targetA", Status.SUCCESS, "2020-01-01T00:00:00Z")])
    ]

    assert roll_up(chain).status == Status.FAILED


def test_genuine_completion_still_succeeds_without_an_authoritative_target_list():
    # The #222 fix must survive the guard: root FAILED, retry re-ran the failed
    # target to SUCCESS, targets=None. The newest attempt is SUCCESS, so the job
    # is SUCCESS even though the denominator is only what ran.
    root = _build(Status.FAILED, targets=None)
    retry = _build(Status.SUCCESS, retry_count=1, retry_of=root.uuid, targets=None)
    root_a = _run(root.uuid, "targetA", Status.SUCCESS, "2020-01-01T00:00:00Z")
    root_b = _run(root.uuid, "targetB", Status.FAILED, "2020-01-01T00:01:00Z")
    retry_b = _run(retry.uuid, "targetB", Status.SUCCESS, "2020-01-01T00:02:00Z")
    chain = [(root, [root_a, root_b]), (retry, [retry_b])]

    assert roll_up(chain).status == Status.SUCCESS


def test_a_completed_but_cancelled_job_is_success_when_targets_are_authoritative():
    # Pins branch order: succeeded==total (row 4) beats CANCELLED (row 5). A
    # cancel that landed after every spec target already succeeded is too late to
    # remove anything from the result. Reachable: cancel on the root, the
    # in-flight retry finishes the remaining target.
    root = _build(Status.CANCELLED, retry_count=0, targets=["targetA", "targetB"])
    retry = _build(
        Status.SUCCESS,
        retry_count=1,
        retry_of=root.uuid,
        targets=["targetA", "targetB"],
    )
    root_a = _run(root.uuid, "targetA", Status.SUCCESS, "2020-01-01T00:00:00Z")
    retry_b = _run(retry.uuid, "targetB", Status.SUCCESS, "2020-01-01T00:02:00Z")
    chain = [(root, [root_a]), (retry, [retry_b])]

    assert roll_up(chain).status == Status.SUCCESS


def test_never_succeeded_target_reports_its_latest_attempt():
    # Pins _outcome_for's entries[-1]: a target that failed in the root and again
    # in the retry must report the retry's run (attempt 1), not the root's, so a
    # user opens the most recent logs.
    root = _build(Status.FAILED, retry_count=0, targets=["targetA"])
    retry = _build(
        Status.FAILED, retry_count=1, retry_of=root.uuid, targets=["targetA"]
    )
    root_a = _run(root.uuid, "targetA", Status.FAILED, "2020-01-01T00:00:00Z")
    retry_a = _run(retry.uuid, "targetA", Status.FAILED, "2020-01-01T00:02:00Z")
    chain = [(root, [root_a]), (retry, [retry_a])]

    outcome = roll_up(chain).targets[0]

    assert outcome.attempt == 1
    assert outcome.target_run_id == retry_a.uuid


def test_invalid_conjunct_only_wins_when_nothing_succeeded():
    # Pins the `and counts.succeeded == 0` conjunct on the INVALID row: a chain
    # with a success plus an INVALID member is FAILED (partial), not INVALID.
    root = _build(Status.FAILED, retry_count=0, targets=["targetA", "targetB"])
    retry = _build(
        Status.INVALID,
        retry_count=1,
        retry_of=root.uuid,
        targets=["targetA", "targetB"],
    )
    root_a = _run(root.uuid, "targetA", Status.SUCCESS, "2020-01-01T00:00:00Z")
    root_b = _run(root.uuid, "targetB", Status.FAILED, "2020-01-01T00:01:00Z")
    chain = [(root, [root_a, root_b]), (retry, [])]

    assert roll_up(chain).status == Status.FAILED
