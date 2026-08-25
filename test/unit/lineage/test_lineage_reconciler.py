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

"""Unit tests for the admin-DB lineage reconciliation (the central mechanism).

These tests use an in-memory stub admin storage and a stub lineage store, so they
run in CI without a cluster, PostgreSQL, or wandb credentials.

They verify the two nested selections — builds from the checkpoint forward
(``select_builds_from_checkpoint``) and one build's successful targets
(``select_recordable_targets``) — that ``reconcile_build`` records the unrecorded
ones through the single leaf and reports per-build confirmation, and that a failed
dedup query fails CLOSED: nothing recorded, the failure surfaced, and the build
never reported confirmed.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from gbserver.lineage.lineage_reconciler import (
    UTC_MIN,
    as_aware,
    expected_run_count,
    get_most_recent_build,
    is_permanent_sink_failure,
    reconcile_build,
    record_selected_targets,
    record_target_lineage,
    select_builds_from_checkpoint,
    select_recordable_targets,
)
from gbserver.lineage.lineage_seeding import (
    BACKFILL_BUILD_ID,
    SEED_ALL,
    SEED_FROM_LATEST,
    LineageSeedError,
    _build_checkpoint,
)
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.status import Status

# Aware UTC, matching what a real finished_at/created_time carries. A naive value
# here would be read as *local* (see as_aware) and shift every expectation by the
# test machine's UTC offset.
_BASE = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

_SCAN_PAGE_SIZE = 200


def _target(
    build_id: str,
    uuid: str,
    status: Status = Status.SUCCESS,
    finished_at: datetime = None,
    output_artifacts: dict[str, list[str]] = None,
    retry_of_target_id: str = "",
    name: str = "",
) -> StoredTargetRun:
    # Default the target name to the uuid so distinct runs get distinct names;
    # select_recordable_targets deduplicates SUCCESS runs *per target name*, so a
    # shared/empty name would wrongly collapse unrelated runs. Pass name
    # explicitly to model two runs of the *same* target.
    return StoredTargetRun(
        uuid=uuid,
        build_id=build_id,
        environment_uri="env://test",
        status=status,
        finished_at=finished_at,
        output_artifacts=output_artifacts or {},
        retry_of_target_id=retry_of_target_id,
        name=name or uuid,
    )


def _build(
    uuid: str, created_time: datetime = _BASE, status: Status = Status.SUCCESS
) -> StoredBuild:
    """A StoredBuild with a pinned uuid/created_time/status.

    ``created_time`` is non-nullable on the model (it has a default_factory), so
    there is deliberately no "no creation time" case here: such a build cannot be
    constructed. The reconciler's None-guards are defensive against a row arriving
    by some other path, not a state a test can reach.
    """
    build = StoredBuild(
        name=f"build-{uuid}",
        space_name="sp",
        source_uri="https://x",
        username="u",
    )
    build.uuid = uuid
    build.created_time = created_time
    build.status = status
    return build


def _admin_storage_with(targets: list[StoredTargetRun]) -> MagicMock:
    """Stub admin storage whose target_storage pages one build's SUCCESS targets.

    Asserts the reconciler filters server-side by SUCCESS *and* by build id — the
    selection is build-scoped now, so a query without a build_id would be reading
    unrelated history — and that it honors the newest-``finished_at``-first
    pagination contract.
    """
    storage = MagicMock()
    successful = sorted(
        (t for t in targets if t.status == Status.SUCCESS),
        key=lambda t: (t.finished_at is not None, t.finished_at or _BASE),
        reverse=True,
    )

    def _get_by_where(where, query_control=None):
        assert where.get("status") == Status.SUCCESS.name
        assert "build_id" in where, "target selection must be scoped to one build"
        matching = [t for t in successful if t.build_id == where["build_id"]]
        assert query_control is not None
        pagination = query_control.pagination
        assert pagination is not None and pagination.size == _SCAN_PAGE_SIZE
        assert query_control.sort_orders
        assert query_control.sort_orders[0].column == "finished_at"
        assert query_control.sort_orders[0].ascending is False
        start = pagination.index * pagination.size
        return matching[start : start + pagination.size]

    storage.target_storage.get_by_where.side_effect = _get_by_where
    return storage


def _admin_storage_returning(pages: list[list[StoredTargetRun]]) -> MagicMock:
    """Stub admin storage that returns exactly the given pages, order preserved.

    Unlike ``_admin_storage_with`` this does not re-sort, so a test can hand the
    scan NULL-``finished_at`` rows interleaved among finished ones to prove the
    walk is not truncated by an out-of-contract ordering.
    """
    storage = MagicMock()

    def _get_by_where(where, query_control=None):
        idx = query_control.pagination.index
        return pages[idx] if idx < len(pages) else []

    storage.target_storage.get_by_where.side_effect = _get_by_where
    return storage


def _admin_storage_with_builds(builds: list[StoredBuild]) -> MagicMock:
    """Stub admin storage whose build_storage pages builds newest-created-first."""
    storage = MagicMock()
    ordered = sorted(
        builds,
        key=lambda b: (b.created_time is not None, b.created_time or _BASE),
        reverse=True,
    )

    def _get_by_where(where=None, query_control=None):
        assert query_control is not None
        assert query_control.sort_orders
        assert query_control.sort_orders[0].column == "created_time"
        assert query_control.sort_orders[0].ascending is False
        pagination = query_control.pagination
        start = pagination.index * pagination.size
        return ordered[start : start + pagination.size]

    storage.build_storage.get_by_where.side_effect = _get_by_where
    storage.build_storage.get_by_uuid.side_effect = lambda uuid: next(
        (b for b in builds if b.uuid == uuid), None
    )
    return storage


class _StubStore:
    """Lineage store stub: records into a set, tracks calls, dedupes per-sink."""

    def __init__(
        self,
        already_recorded: set[str] = None,
        fail: set[str] = None,
        query_error: Exception = None,
    ):
        self._recorded = set(already_recorded or set())
        self._fail = set(fail or set())
        self.query_error = query_error
        self.recorded_calls: list[tuple[str, str]] = []
        # Captures the expected_counts the reconciler passed on the last call, so
        # tests can assert it derived per-target run counts from the targets.
        self.last_expected_counts: dict[str, int] | None = None

    def add_jobstats_for_build_target(self, storage, build_id, target_id):
        if target_id in self._fail:
            raise RuntimeError("boom")
        self.recorded_calls.append((build_id, target_id))
        self._recorded.add(target_id)

    def filter_unrecorded(
        self,
        target_ids: set[str],
        expected_counts: dict[str, int] = None,
        on_query_error=None,
    ) -> set[str]:
        self.last_expected_counts = expected_counts
        if self.query_error is not None:
            # Mirrors the real store: fail CLOSED and report through the callback.
            if on_query_error is not None:
                on_query_error(self.query_error)
            return set()
        return set(target_ids) - self._recorded


class TestSelectBuildsFromCheckpoint:
    def test_includes_the_anchor_and_newer_builds(self):
        """The cutoff is inclusive, so the anchor build is re-selected.

        A pass that crashed partway through the anchor left targets unrecorded, and
        re-selecting it every scan is what recovers them.
        """
        storage = _admin_storage_with_builds(
            [
                _build("old", _BASE - timedelta(hours=1)),
                _build("anchor", _BASE),
                _build("newer", _BASE + timedelta(hours=1)),
            ]
        )

        found = select_builds_from_checkpoint(storage, _BASE)

        assert [b.uuid for b in found] == ["anchor", "newer"]

    def test_returns_oldest_created_first(self):
        """Processing order is oldest-first, which the contiguous advance needs.

        The query itself pages newest-first (to stop at the cutoff); reversing it
        here is what lets the caller stop at the first unfinished build.
        """
        storage = _admin_storage_with_builds(
            [
                _build("c", _BASE + timedelta(minutes=2)),
                _build("a", _BASE),
                _build("b", _BASE + timedelta(minutes=1)),
            ]
        )

        found = select_builds_from_checkpoint(storage, _BASE)

        assert [b.uuid for b in found] == ["a", "b", "c"]

    def test_excludes_requested_ids(self):
        storage = _admin_storage_with_builds(
            [_build("a", _BASE), _build("b", _BASE + timedelta(minutes=1))]
        )

        found = select_builds_from_checkpoint(storage, _BASE, exclude_ids={"a"})

        assert [b.uuid for b in found] == ["b"]

    def test_utc_min_cutoff_reaches_all_history(self):
        """The backfill anchor selects everything, however old."""
        storage = _admin_storage_with_builds(
            [_build("ancient", _BASE - timedelta(days=365)), _build("a", _BASE)]
        )

        found = select_builds_from_checkpoint(storage, UTC_MIN)

        assert [b.uuid for b in found] == ["ancient", "a"]

    def test_pages_past_a_full_first_page(self):
        """The walk keeps paging while rows stay at or above the cutoff."""
        builds = [
            _build(f"b{i}", _BASE + timedelta(seconds=i))
            for i in range(_SCAN_PAGE_SIZE + 5)
        ]
        storage = _admin_storage_with_builds(builds)

        found = select_builds_from_checkpoint(storage, _BASE)

        assert len(found) == _SCAN_PAGE_SIZE + 5

    def test_mixed_offsets_compare_as_instants(self):
        """Two aware timestamps compare as instants regardless of their offsets."""
        offset = timezone(timedelta(hours=-5))
        # Same instant as _BASE, written with a different offset.
        same_instant = datetime(2025, 12, 31, 19, 0, 0, tzinfo=offset)
        storage = _admin_storage_with_builds([_build("a", same_instant)])

        found = select_builds_from_checkpoint(storage, _BASE)

        assert [b.uuid for b in found] == ["a"]


def _admin_storage_with_misordered_builds(builds: list[StoredBuild]) -> MagicMock:
    """Stub whose build pages come back in a WRONG order, on purpose.

    Reproduces what SQLite actually does to this column: it stores
    DateTime(timezone=True) as TEXT and orders it as TEXT, and the column holds two
    spellings ("2026-08-21 21:13:24.581351" from SQLAlchemy and
    "2026-08-21T15:18:38.948Z" from older writers). ' ' sorts before 'T', so a
    genuinely newer build can be returned *below* a much older one.

    Returns the list exactly as given, so a test can hand over an order no correct
    implementation may rely on.
    """
    storage = MagicMock()

    def _get_by_where(where=None, query_control=None):
        pagination = query_control.pagination
        start = pagination.index * pagination.size
        return builds[start : start + pagination.size]

    storage.build_storage.get_by_where.side_effect = _get_by_where
    return storage


class TestBuildWalkDoesNotTrustTheOrdering:
    """The walk must not stop early on the backend's ordering.

    SQLite's text collation on this column interleaves rows whose real instants are
    months apart, so an early return on "this row is behind the cutoff" can end the
    walk before reaching genuinely newer builds -- reporting nothing to record while
    real lineage sits unread. Silent data loss, not a slow scan.
    """

    def test_newer_build_after_an_older_one_is_still_found(self):
        # Order deliberately wrong: the old row comes first, the new one after it.
        old_build = _build("old", _BASE - timedelta(days=30))
        new_build = _build("new", _BASE + timedelta(hours=1))
        storage = _admin_storage_with_misordered_builds([old_build, new_build])

        found = select_builds_from_checkpoint(storage, _BASE)

        assert [b.uuid for b in found] == ["new"], (
            "the walk stopped at the out-of-order old row and never reached the "
            "newer build"
        )

    def test_result_is_sorted_by_real_instant_not_by_arrival(self):
        """The caller's contiguous advance needs true chronological order."""
        b_mid = _build("mid", _BASE + timedelta(minutes=2))
        b_first = _build("first", _BASE)
        b_last = _build("last", _BASE + timedelta(minutes=5))
        storage = _admin_storage_with_misordered_builds([b_mid, b_first, b_last])

        found = select_builds_from_checkpoint(storage, _BASE)

        assert [b.uuid for b in found] == ["first", "mid", "last"]

    def test_most_recent_build_is_the_true_maximum(self):
        """Seeding must anchor at the real newest build, not the first row seen.

        Anchoring at an arbitrary older build would silently re-drive history from
        there.
        """
        storage = _admin_storage_with_misordered_builds(
            [
                _build("looks-first", _BASE),
                _build("actually-newest", _BASE + timedelta(days=1)),
                _build("older", _BASE - timedelta(days=1)),
            ]
        )

        assert get_most_recent_build(storage).uuid == "actually-newest"


class TestGetMostRecentBuild:
    def test_returns_the_newest_build(self):
        storage = _admin_storage_with_builds(
            [_build("a", _BASE), _build("b", _BASE + timedelta(minutes=1))]
        )

        assert get_most_recent_build(storage).uuid == "b"

    def test_returns_none_on_an_empty_db(self):
        assert get_most_recent_build(_admin_storage_with_builds([])) is None


class TestSelectRecordableTargets:
    def test_selects_only_successful_targets(self):
        storage = _admin_storage_with(
            [
                _target("b1", "t1", finished_at=_BASE),
                _target("b1", "t2", status=Status.FAILED, finished_at=_BASE),
            ]
        )

        found = select_recordable_targets(storage, build_id="b1")

        assert [t.uuid for t in found] == ["t1"]

    def test_scoped_to_one_build(self):
        """Only the named build's targets, so no unrelated history is recorded."""
        storage = _admin_storage_with(
            [
                _target("b1", "t1", finished_at=_BASE),
                _target("b2", "t2", finished_at=_BASE),
            ]
        )

        found = select_recordable_targets(storage, build_id="b1")

        assert [t.uuid for t in found] == ["t1"]

    def test_null_finished_at_is_skipped_not_a_stop(self):
        """A NULL row is not yet complete, but must not truncate the page walk."""
        storage = _admin_storage_returning(
            [
                [
                    _target("b1", "t-null", finished_at=None),
                    _target("b1", "t1", finished_at=_BASE),
                ]
            ]
        )

        found = select_recordable_targets(storage, build_id="b1")

        assert [t.uuid for t in found] == ["t1"]

    def test_pages_past_a_full_first_page(self):
        page = [
            _target("b1", f"t{i}", finished_at=_BASE + timedelta(seconds=i))
            for i in range(_SCAN_PAGE_SIZE)
        ]
        storage = _admin_storage_returning(
            [page, [_target("b1", "tail", finished_at=_BASE)]]
        )

        found = select_recordable_targets(storage, build_id="b1")

        assert len(found) == _SCAN_PAGE_SIZE + 1

    def test_dedupes_repeated_success_runs_to_the_latest(self):
        """In-place retry reuses one build id, so a target can have >1 SUCCESS run
        (a prior success with unregistered artifacts is re-run; a reuse-disabled
        build re-runs every target). Only the latest is recordable — otherwise the
        target is written to the sink twice.
        """
        storage = _admin_storage_with(
            [
                _target("b1", "t-old", name="targetA", finished_at=_BASE),
                _target(
                    "b1",
                    "t-new",
                    name="targetA",
                    finished_at=_BASE + timedelta(minutes=5),
                ),
                _target("b1", "t-other", name="targetB", finished_at=_BASE),
            ]
        )

        found = select_recordable_targets(storage, build_id="b1")

        # The superseded run is dropped; the other target is untouched.
        assert sorted(t.uuid for t in found) == ["t-new", "t-other"]


class TestReconcileBuild:
    def test_records_each_unrecorded_target(self):
        storage = _admin_storage_with(
            [
                _target("b1", "t1", finished_at=_BASE),
                _target("b1", "t2", finished_at=_BASE + timedelta(minutes=1)),
            ]
        )
        store = _StubStore()

        result = reconcile_build(store, storage, build_id="b1")

        assert set(store.recorded_calls) == {("b1", "t1"), ("b1", "t2")}
        assert result.newly_recorded == 2
        assert result.all_confirmed is True

    def test_already_recorded_targets_are_skipped(self):
        """Dedup is the only thing preventing duplicates with random run ids."""
        storage = _admin_storage_with([_target("b1", "t1", finished_at=_BASE)])
        store = _StubStore(already_recorded={"t1"})

        result = reconcile_build(store, storage, build_id="b1")

        assert store.recorded_calls == []
        assert result.newly_recorded == 0
        assert result.all_confirmed is True

    def test_build_with_no_targets_is_trivially_confirmed(self):
        """Otherwise the checkpoint would pin behind a build with no lineage."""
        storage = _admin_storage_with([])
        store = _StubStore()

        result = reconcile_build(store, storage, build_id="b1")

        assert result.all_confirmed is True
        assert result.newly_recorded == 0

    def test_failure_does_not_abort_the_build_but_blocks_confirmation(self):
        """A failing target must not stop its siblings, nor be reported confirmed.

        Reporting confirmed would let the checkpoint advance past a build whose
        lineage is still missing, and nothing sweeps behind the mark.
        """
        storage = _admin_storage_with(
            [
                _target("b1", "t1", finished_at=_BASE),
                _target("b1", "t2", finished_at=_BASE + timedelta(minutes=1)),
            ]
        )
        store = _StubStore(fail={"t1"})

        result = reconcile_build(store, storage, build_id="b1")

        assert ("b1", "t2") in store.recorded_calls
        assert result.all_confirmed is False

    def test_on_error_callback_receives_the_failure(self):
        storage = _admin_storage_with([_target("b1", "t1", finished_at=_BASE)])
        store = _StubStore(fail={"t1"})
        seen: list[tuple[str, str, Exception]] = []

        reconcile_build(
            store,
            storage,
            build_id="b1",
            on_error=lambda b, t, e: seen.append((b, t, e)),
        )

        assert len(seen) == 1
        assert seen[0][0] == "b1" and seen[0][1] == "t1"

    def test_on_success_callback_fires_per_recorded_target(self):
        storage = _admin_storage_with([_target("b1", "t1", finished_at=_BASE)])
        store = _StubStore()
        seen: list[tuple[str, str]] = []

        reconcile_build(
            store, storage, build_id="b1", on_success=lambda b, t: seen.append((b, t))
        )

        assert seen == [("b1", "t1")]

    def test_skipped_targets_are_reported_as_a_known_gap(self):
        """A dropped target confirms the build *with* a gap the caller can log.

        Holding the mark instead would wedge every newer build behind lineage that
        will never land — what the durable drop set exists to prevent.
        """
        storage = _admin_storage_with([_target("b1", "t1", finished_at=_BASE)])
        store = _StubStore()

        result = reconcile_build(store, storage, build_id="b1", skip={"t1"})

        assert store.recorded_calls == []
        assert result.dropped == {"t1"}
        assert result.all_confirmed is True

    def test_passes_expected_run_counts_derived_from_outputs(self):
        """The dedup count must match how many runs the sink will emit."""
        storage = _admin_storage_with(
            [
                _target(
                    "b1",
                    "t1",
                    finished_at=_BASE,
                    output_artifacts={"a": ["o1", "o2"]},
                ),
                _target("b1", "t2", finished_at=_BASE),
            ]
        )
        store = _StubStore()

        reconcile_build(store, storage, build_id="b1")

        assert store.last_expected_counts == {"t1": 2, "t2": 1}

    def test_retried_target_counts_from_its_own_outputs(self):
        """A re-run (retry) target is a real SUCCESS run counted by its own outputs.

        In-place retry keeps both the FAILED and the SUCCESS run in one build;
        there is no skip/swap concept, so a retried target's expected count comes
        directly from its own ``output_artifacts`` like any other run. 't2' below
        retried a prior failed run (``retry_of_target_id``) and produced one output,
        so it must contribute a count of 1 — no different from a first-attempt run.
        """
        storage = _admin_storage_with(
            [
                _target(
                    "b1",
                    "t1",
                    finished_at=_BASE,
                    output_artifacts={"a": ["o1", "o2"]},
                ),
                _target(
                    "b1",
                    "t2",
                    finished_at=_BASE,
                    output_artifacts={"a": ["o3"]},
                    retry_of_target_id="t2-failed",
                ),
            ]
        )
        store = _StubStore()

        reconcile_build(store, storage, build_id="b1")

        assert store.last_expected_counts == {"t1": 2, "t2": 1}

    def test_retried_target_is_recorded(self):
        """A retried SUCCESS target records its lineage like any other run."""
        storage = _admin_storage_with(
            [
                _target(
                    "b1",
                    "t2",
                    finished_at=_BASE,
                    output_artifacts={"a": ["o3"]},
                    retry_of_target_id="t2-failed",
                ),
            ]
        )
        store = _StubStore()

        result = reconcile_build(store, storage, build_id="b1")

        assert store.recorded_calls == [("b1", "t2")]
        assert result.newly_recorded == 1
        assert result.all_confirmed

    # ---- fail-closed dedup -----------------------------------------------

    def test_dedup_failure_records_nothing(self):
        """Writing on an unanswered query would duplicate runs, not resume them."""
        storage = _admin_storage_with([_target("b1", "t1", finished_at=_BASE)])
        store = _StubStore(query_error=RuntimeError("timeout"))

        result = reconcile_build(store, storage, build_id="b1")

        assert store.recorded_calls == []
        assert result.newly_recorded == 0

    def test_dedup_failure_is_reported_and_not_confirmed(self):
        """An empty candidate set must never be read as "all recorded".

        ``all_confirmed`` staying False is what stops the caller advancing its
        checkpoint over work that was never done.
        """
        storage = _admin_storage_with([_target("b1", "t1", finished_at=_BASE)])
        failure = RuntimeError("timeout")
        store = _StubStore(query_error=failure)

        result = reconcile_build(store, storage, build_id="b1")

        assert result.dedup_query_failed is True
        assert result.query_failure is failure
        assert result.all_confirmed is False


class TestIsPermanentSinkFailure:
    def test_auth_failure_is_permanent(self):
        assert is_permanent_sink_failure(RuntimeError("permission denied")) is True

    def test_missing_project_is_permanent(self):
        assert (
            is_permanent_sink_failure(RuntimeError("could not find project x")) is True
        )

    def test_network_failure_is_transient(self):
        assert is_permanent_sink_failure(RuntimeError("connection timed out")) is False

    def test_unknown_message_is_transient(self):
        """The default is transient — the safe direction.

        Retrying a permanent failure only costs queries; treating a transient one as
        permanent would switch recording off over a network blip.
        """
        assert is_permanent_sink_failure(RuntimeError("something odd")) is False

    def test_same_type_classifies_differently_by_message(self):
        """The exception *type* cannot separate these.

        wandb raises the same error class for a permanent refusal and for an
        ordinary network blip, which is why this matches on the message.
        """

        class CommError(Exception):
            pass

        assert is_permanent_sink_failure(CommError("unauthorized")) is True
        assert is_permanent_sink_failure(CommError("connection reset")) is False

    def test_wrapped_cause_is_inspected(self):
        """The real cause usually arrives wrapped by the sink's own handling."""
        try:
            try:
                raise RuntimeError("invalid api key")
            except RuntimeError as inner:
                raise ValueError("recording failed") from inner
        except ValueError as outer:
            assert is_permanent_sink_failure(outer) is True


class TestRecordTargetLineage:
    def test_leaf_calls_store_with_ids(self):
        store = MagicMock()
        storage = MagicMock()

        record_target_lineage(store, storage, build_id="b1", target_id="t1")

        store.add_jobstats_for_build_target.assert_called_once_with(
            storage, build_id="b1", target_id="t1"
        )


class TestExpectedRunCount:
    def test_counts_all_output_artifacts_across_lists(self):
        t = _target("b1", "t1", output_artifacts={"a": ["o1"], "b": ["o2", "o3"]})
        assert expected_run_count(t) == 3

    def test_no_outputs_expects_one_run(self):
        assert expected_run_count(_target("b1", "t1")) == 1


class TestRecordSelectedTargets:
    def test_selected_push_uses_the_same_leaf(self):
        """The D-seam: an explicit selection records via the single leaf."""
        store = MagicMock()
        storage = MagicMock()

        record_selected_targets(store, storage, [("b1", "t1"), ("b2", "t2")])

        assert store.add_jobstats_for_build_target.call_count == 2
        store.add_jobstats_for_build_target.assert_any_call(
            storage, build_id="b1", target_id="t1"
        )
        store.add_jobstats_for_build_target.assert_any_call(
            storage, build_id="b2", target_id="t2"
        )


class TestSeedAnchors:
    """Where ``lineage-watch --base-build-id`` places the checkpoint."""

    def test_all_anchors_at_utc_min(self):
        checkpoint = _build_checkpoint(_admin_storage_with_builds([]), SEED_ALL)

        assert checkpoint["build_id"] == BACKFILL_BUILD_ID
        assert checkpoint["created_time"] == UTC_MIN.isoformat()

    def test_from_latest_anchors_at_the_newest_build(self):
        """The anchored build is recorded whole, so no oldest-target step is needed."""
        storage = _admin_storage_with_builds(
            [_build("a", _BASE), _build("b", _BASE + timedelta(minutes=1))]
        )

        checkpoint = _build_checkpoint(storage, SEED_FROM_LATEST)

        assert checkpoint["build_id"] == "b"
        assert checkpoint["created_time"] == (_BASE + timedelta(minutes=1)).isoformat()

    def test_build_id_anchors_at_that_build(self):
        storage = _admin_storage_with_builds(
            [_build("a", _BASE), _build("b", _BASE + timedelta(minutes=1))]
        )

        checkpoint = _build_checkpoint(storage, "a")

        assert checkpoint["build_id"] == "a"
        assert checkpoint["created_time"] == _BASE.isoformat()

    def test_from_latest_on_an_empty_db_raises(self):
        """Better to fail loudly than to seed an anchor that means "everything"."""
        with pytest.raises(LineageSeedError):
            _build_checkpoint(_admin_storage_with_builds([]), SEED_FROM_LATEST)

    def test_unknown_build_id_raises(self):
        with pytest.raises(LineageSeedError):
            _build_checkpoint(_admin_storage_with_builds([_build("a", _BASE)]), "gone")
