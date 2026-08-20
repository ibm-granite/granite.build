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

These tests use an in-memory stub admin storage and a stub lineage store, so
they run in CI without a cluster, PostgreSQL, or wandb credentials. They verify
that reconciliation selects successful targets from the admin DB by a
``finished_at`` time watermark and records each through the single leaf, that the
watermark is required (a full-history backfill must be asked for explicitly, with
``datetime.min``, rather than being what an omitted argument means), that
steady-state scans read only newly-finished targets, and that the per-sink
``filter_unrecorded`` check decides what each sink actually records.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from gbserver.lineage.lineage_reconciler import (
    UTC_MIN,
    _expected_run_count,
    as_aware,
    get_most_recent_successful_target,
    get_oldest_successful_target,
    reconcile_once,
    record_selected_targets,
    record_target_lineage,
    select_recordable_targets,
)
from gbserver.lineage.lineage_seeding import (
    BACKFILL_BUILD_ID,
    SEED_ALL,
    SEED_FROM_LATEST,
    LineageSeedError,
    _build_checkpoint,
)
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.status import Status

# Aware UTC, matching what a real finished_at carries: utils.get_time() stamps
# them with datetime.now().astimezone(). A naive value here would be read as
# *local* (see as_aware) and shift every expectation by the test machine's
# UTC offset.
_BASE = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def _target(
    build_id: str,
    uuid: str,
    status: Status = Status.SUCCESS,
    finished_at: datetime = None,
    output_artifacts: dict[str, list[str]] = None,
    skipped_for_prerun_target_id: str = "",
) -> StoredTargetRun:
    return StoredTargetRun(
        uuid=uuid,
        build_id=build_id,
        environment_uri="env://test",
        status=status,
        finished_at=finished_at,
        output_artifacts=output_artifacts or {},
        skipped_for_prerun_target_id=skipped_for_prerun_target_id,
    )


_SCAN_PAGE_SIZE = 200


def _admin_storage_with(targets: list[StoredTargetRun]) -> MagicMock:
    """Stub admin storage whose target_storage pages SUCCESS targets.

    Asserts the reconciler queries by SUCCESS status (rather than scanning all
    targets) and honors the newest-``finished_at``-first pagination contract, so
    both the selection filter and the bounded-scan behavior are pinned. Returns
    targets sorted by ``finished_at`` descending (None sorts last) to mimic the
    server-side ordering.
    """
    storage = MagicMock()
    successful = sorted(
        (t for t in targets if t.status == Status.SUCCESS),
        key=lambda t: (t.finished_at is not None, t.finished_at or _BASE),
        reverse=True,
    )

    def _get_by_where(where, query_control=None):
        assert where == {"status": Status.SUCCESS.name}
        assert query_control is not None
        pagination = query_control.pagination
        assert pagination is not None and pagination.size == _SCAN_PAGE_SIZE
        assert query_control.sort_orders
        assert query_control.sort_orders[0].column == "finished_at"
        assert query_control.sort_orders[0].ascending is False
        start = pagination.index * pagination.size
        return successful[start : start + pagination.size]

    storage.target_storage.get_by_where.side_effect = _get_by_where
    return storage


def _admin_storage_returning(pages: list[list[StoredTargetRun]]) -> MagicMock:
    """Stub admin storage that returns exactly the given pages, order preserved.

    Unlike ``_admin_storage_with`` this does not re-sort, so a test can hand the
    reconciler NULL-``finished_at`` rows interleaved among finished ones to prove
    the scan is not truncated by an out-of-contract ordering.
    """
    storage = MagicMock()

    def _get_by_where(where, query_control=None):
        idx = query_control.pagination.index
        return pages[idx] if idx < len(pages) else []

    storage.target_storage.get_by_where.side_effect = _get_by_where
    return storage


class _StubStore:
    """Lineage store stub: records into a set, tracks calls, dedupes per-sink."""

    def __init__(self, already_recorded: set[str] = None, fail: set[str] = None):
        self._recorded = set(already_recorded or set())
        self._fail = set(fail or set())
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
        self, target_ids: set[str], expected_counts: dict[str, int] = None
    ) -> set[str]:
        self.last_expected_counts = expected_counts
        return set(target_ids) - self._recorded


class TestSelectRecordableTargets:
    def test_selects_only_successful_targets(self):
        storage = _admin_storage_with(
            [
                _target("b1", "t1", Status.SUCCESS, _BASE),
                _target("b1", "t2", Status.SUCCESS, _BASE + timedelta(seconds=1)),
            ]
        )
        selected = select_recordable_targets(storage, finished_after=_BASE)
        assert {t.uuid for t in selected} == {"t1", "t2"}

    def test_watermark_selects_only_newly_finished(self):
        t1 = _target("b1", "t1", finished_at=_BASE)
        t2 = _target("b1", "t2", finished_at=_BASE + timedelta(seconds=10))
        t3 = _target("b1", "t3", finished_at=_BASE + timedelta(seconds=20))
        storage = _admin_storage_with([t1, t2, t3])

        selected = select_recordable_targets(
            storage, finished_after=_BASE + timedelta(seconds=5)
        )
        # t1 finished before the watermark; only t2 and t3 are selected.
        assert {t.uuid for t in selected} == {"t2", "t3"}

    def test_watermark_boundary_is_inclusive(self):
        t1 = _target("b1", "t1", finished_at=_BASE)
        storage = _admin_storage_with([t1])
        selected = select_recordable_targets(storage, finished_after=_BASE)
        assert {t.uuid for t in selected} == {"t1"}

    def test_interleaved_null_finished_at_does_not_truncate_scan(self):
        # A NULL-finished_at row appearing before a still-recordable finished one
        # (out-of-contract ordering) must be skipped, not treated as a watermark
        # crossing that ends the walk.
        t_new = _target("b1", "t_new", finished_at=_BASE + timedelta(seconds=20))
        t_null = _target("b1", "t_null", finished_at=None)
        t_recordable = _target(
            "b1", "t_recordable", finished_at=_BASE + timedelta(seconds=10)
        )
        storage = _admin_storage_returning([[t_new, t_null, t_recordable]])

        selected = select_recordable_targets(
            storage, finished_after=_BASE + timedelta(seconds=5)
        )
        # t_null is skipped; t_recordable is still newer than the watermark and
        # must not be dropped by an early return on the NULL row.
        assert {t.uuid for t in selected} == {"t_new", "t_recordable"}

    def test_tz_aware_finished_at_does_not_break_watermark_compare(self):
        # Mixed awareness must not abort the scan: comparing a naive datetime
        # against an aware one raises TypeError. Both sides are normalized to
        # aware UTC instants, so the walk works whichever awareness the backend
        # yields. Here the aware value is one hour after the watermark, so it is
        # newer and must be selected.
        aware = _BASE + timedelta(hours=1)
        t_aware = _target("b1", "t_aware", finished_at=aware)
        older = _target("b1", "t_older", finished_at=_BASE - timedelta(seconds=10))
        storage = _admin_storage_returning([[t_aware, older]])

        selected = select_recordable_targets(storage, finished_after=_BASE)
        assert {t.uuid for t in selected} == {"t_aware"}

    def test_overflowing_aware_finished_at_does_not_abort_scan(self):
        # The --base-build-id all anchor is datetime.min itself, and a backend may
        # hand it back aware with a positive UTC offset. Shifting that to UTC overflows
        # datetime.min, which would raise OverflowError out of the whole scan
        # (through reconcile_once and start()) rather than merely comparing wrong.
        # It must be clamped to the bound instead, leaving the walk intact.
        overflowing = datetime.min.replace(tzinfo=timezone(timedelta(hours=5)))
        t_min = _target("b1", "t_min", finished_at=overflowing)
        t_new = _target("b1", "t_new", finished_at=_BASE + timedelta(seconds=10))
        storage = _admin_storage_returning([[t_new, t_min]])

        selected = select_recordable_targets(storage, finished_after=_BASE)
        # t_min clamps to datetime.min, which is older than the watermark, so it
        # ends the walk — but t_new, read before it, is still selected.
        assert {t.uuid for t in selected} == {"t_new"}

    def test_overflowing_aware_watermark_does_not_abort_scan(self):
        # Same overflow on the cutoff side: a datetime.min watermark read back
        # aware must clamp to datetime.min (selecting everything, per
        # --base-build-id all) rather than raising out of the scan.
        cutoff = datetime.min.replace(tzinfo=timezone(timedelta(hours=5)))
        t1 = _target("b1", "t1", finished_at=_BASE)
        storage = _admin_storage_with([t1])

        selected = select_recordable_targets(storage, finished_after=cutoff)
        assert {t.uuid for t in selected} == {"t1"}


class TestGetMostRecentSuccessfulTarget:
    """Seeding anchor lookup — must survive a pre-``finished_at`` NULL backlog."""

    def test_pages_past_a_full_first_page_of_nulls(self):
        # finished_at stamping was added after rows already existed, so real
        # deployments hold SUCCESS targets with finished_at NULL — and PostgreSQL
        # sorts NULLs FIRST under DESC (the sort is a bare desc(), no NULLS LAST).
        # Reading only page 0 would return None here, making `--base-build-id`
        # raise LineageSeedError on exactly the deployments that have history to
        # anchor against — a crashloop, since --base-build-id is meant to stay in
        # the pod spec.
        nulls = [_target("b1", f"t_null_{i}") for i in range(_SCAN_PAGE_SIZE)]
        anchor = _target("b2", "t_anchor", finished_at=_BASE)
        storage = _admin_storage_returning([nulls, [anchor]])

        found = get_most_recent_successful_target(storage)
        assert found is not None and found.uuid == "t_anchor"

    def test_stops_at_a_short_page_of_nulls(self):
        # A short page is the last one: no non-NULL row exists, so this must
        # return None rather than paging forever.
        storage = _admin_storage_returning([[_target("b1", "t_null")]])
        assert get_most_recent_successful_target(storage) is None
        assert storage.target_storage.get_by_where.call_count == 1


class TestSeedAnchors:
    """Where each --base-build-id spec places the checkpoint."""

    @staticmethod
    def _storage(targets):
        storage = MagicMock()

        def _get_by_where(where, query_control=None):
            rows = [
                t
                for t in targets
                if "build_id" not in where or t.build_id == where["build_id"]
            ]
            rows.sort(key=lambda t: t.finished_at, reverse=True)
            return rows if query_control.pagination.index == 0 else []

        storage.target_storage.get_by_where.side_effect = _get_by_where
        return storage

    def test_from_latest_anchors_at_the_newest_build_s_oldest_target(self):
        # Two steps: the most recent completion names the build, then the anchor is
        # that build's *oldest* target. Anchoring at the newest completion would
        # start recording mid-build and skip the rest of its already-finished
        # targets, while still naming their build in the checkpoint.
        new_last = _target("b_new", "n3", finished_at=_BASE + timedelta(minutes=20))
        new_mid = _target("b_new", "n2", finished_at=_BASE + timedelta(minutes=10))
        new_first = _target("b_new", "n1", finished_at=_BASE)
        old = _target("b_old", "o1", finished_at=_BASE - timedelta(days=1))
        storage = self._storage([new_last, new_mid, new_first, old])

        checkpoint = _build_checkpoint(storage, SEED_FROM_LATEST)

        assert checkpoint["build_id"] == "b_new"
        # The newest build's first completion, not its last.
        assert checkpoint["finished_at"] == as_aware(_BASE).isoformat()

    def test_build_id_anchors_at_that_build_s_oldest_target(self):
        first = _target("b1", "t1", finished_at=_BASE)
        last = _target("b1", "t2", finished_at=_BASE + timedelta(minutes=30))
        storage = self._storage([last, first])

        checkpoint = _build_checkpoint(storage, "b1")

        assert checkpoint["build_id"] == "b1"
        assert checkpoint["finished_at"] == as_aware(_BASE).isoformat()

    def test_all_anchors_at_utc_min(self):
        checkpoint = _build_checkpoint(self._storage([]), SEED_ALL)
        assert checkpoint["build_id"] == BACKFILL_BUILD_ID
        assert checkpoint["finished_at"] == UTC_MIN.isoformat()

    def test_from_latest_on_an_empty_db_raises(self):
        # Nothing to anchor against; the CLI turns this into a ClickException
        # rather than seeding something arbitrary.
        with pytest.raises(LineageSeedError):
            _build_checkpoint(self._storage([]), SEED_FROM_LATEST)


class TestGetOldestSuccessfulTarget:
    """Anchoring a checkpoint at a build must not skip that build's own targets."""

    def test_returns_the_oldest_finished_target(self):
        # The anchor for a build id: its *oldest* completion. The watermark is
        # inclusive but forward-only, so anchoring at the newest would exclude
        # every earlier target of the same build.
        t_old = _target("b1", "t_old", finished_at=_BASE)
        t_mid = _target("b1", "t_mid", finished_at=_BASE + timedelta(minutes=10))
        t_new = _target("b1", "t_new", finished_at=_BASE + timedelta(minutes=20))
        storage = _admin_storage_with([t_old, t_mid, t_new])

        found = get_oldest_successful_target(storage)
        assert found is not None and found.uuid == "t_old"

    def test_scoped_to_one_build(self):
        # Only the requested build's targets may anchor it. The build_id must reach
        # the query as a server-side filter (it is a real column), not be applied
        # after the fact — otherwise another build's older target would drag the
        # anchor down and re-drive unrelated lineage.
        mine = _target("b1", "t_mine", finished_at=_BASE)
        storage = _admin_storage_returning([[mine]])

        found = get_oldest_successful_target(storage, build_id="b1")
        assert found is not None and found.uuid == "t_mine"
        where = storage.target_storage.get_by_where.call_args.args[0]
        assert where == {"status": Status.SUCCESS.name, "build_id": "b1"}

    def test_pages_to_the_end_to_find_the_oldest(self):
        # Ordering is newest-first, so the oldest row is on the LAST page: a
        # single-page read would return the wrong anchor.
        first_page = [
            _target("b1", f"t_{i}", finished_at=_BASE + timedelta(minutes=i + 1))
            for i in range(_SCAN_PAGE_SIZE)
        ]
        oldest = _target("b1", "t_oldest", finished_at=_BASE)
        storage = _admin_storage_returning([first_page, [oldest]])

        found = get_oldest_successful_target(storage)
        assert found is not None and found.uuid == "t_oldest"

    def test_skips_null_finished_at(self):
        # A NULL finished_at is not a completion; it must never become the anchor
        # (and the NULL backlog PostgreSQL sorts first must not be mistaken for
        # the oldest target).
        t_null = _target("b1", "t_null", finished_at=None)
        t_real = _target("b1", "t_real", finished_at=_BASE)
        storage = _admin_storage_returning([[t_null, t_real]])

        found = get_oldest_successful_target(storage)
        assert found is not None and found.uuid == "t_real"

    def test_returns_none_when_no_finished_target_exists(self):
        storage = _admin_storage_returning([[_target("b1", "t_null")]])
        assert get_oldest_successful_target(storage) is None

    def test_mixed_awareness_does_not_raise(self):
        # finished_at may arrive aware or naive depending on the backend; comparing
        # the two directly raises TypeError, which would abort seeding entirely.
        aware = _target("b1", "t_aware", finished_at=_BASE + timedelta(hours=1))
        naive = _target(
            "b1",
            "t_naive",
            finished_at=(_BASE + timedelta(hours=2)).replace(tzinfo=None),
        )
        storage = _admin_storage_returning([[naive, aware]])

        found = get_oldest_successful_target(storage)
        assert found is not None and found.uuid == "t_aware"


class TestRecordTargetLineage:
    def test_leaf_calls_store_with_ids(self):
        store = MagicMock()
        storage = MagicMock()
        record_target_lineage(store, storage, build_id="b1", target_id="t1")
        store.add_jobstats_for_build_target.assert_called_once_with(
            storage, build_id="b1", target_id="t1"
        )


class TestReconcileOnce:
    def test_records_each_successful_target(self):
        store = _StubStore()
        storage = _admin_storage_with(
            [
                _target("b1", "t1", finished_at=_BASE),
                _target("b1", "t2", finished_at=_BASE + timedelta(seconds=1)),
            ]
        )

        recorded = reconcile_once(store, storage, finished_after=_BASE)

        assert recorded == 2
        # Recorded oldest-finished-first.
        assert [c[1] for c in store.recorded_calls] == ["t1", "t2"]

    def test_already_recorded_targets_are_skipped(self):
        store = _StubStore(already_recorded={"t1"})
        storage = _admin_storage_with(
            [
                _target("b1", "t1", finished_at=_BASE),
                _target("b1", "t2", finished_at=_BASE + timedelta(seconds=1)),
            ]
        )

        reconcile_once(store, storage, finished_after=_BASE)

        # Only the not-yet-recorded target is recorded this pass.
        assert store.recorded_calls == [("b1", "t2")]

    def test_backfill_pages_through_all_targets(self):
        """``datetime.min`` is the deliberate full-history backfill.

        There is no longer an implicit "no watermark" mode — a caller that wants
        every successful target ever must ask for it — and the walk must page
        past the first page to deliver it.
        """
        store = _StubStore()
        targets = [
            _target("b1", f"t{i}", finished_at=_BASE + timedelta(seconds=i))
            for i in range(_SCAN_PAGE_SIZE + 5)
        ]
        storage = _admin_storage_with(targets)

        reconcile_once(store, storage, finished_after=datetime.min)
        assert {c[1] for c in store.recorded_calls} == {t.uuid for t in targets}

    def test_steady_state_stops_at_watermark(self):
        store = _StubStore()
        targets = [
            _target("b1", f"t{i}", finished_at=_BASE + timedelta(seconds=i))
            for i in range(_SCAN_PAGE_SIZE + 5)
        ]
        storage = _admin_storage_with(targets)

        # Watermark just below the newest few: only those newer are recorded,
        # and the walk stops before reading the whole table.
        watermark = _BASE + timedelta(seconds=_SCAN_PAGE_SIZE + 2)
        reconcile_once(store, storage, finished_after=watermark)

        recorded = {c[1] for c in store.recorded_calls}
        assert recorded == {
            f"t{i}"
            for i in range(_SCAN_PAGE_SIZE + 5)
            if _BASE + timedelta(seconds=i) >= watermark
        }
        # get_by_where was called only once (single partial page), not paged
        # through the whole table.
        assert storage.target_storage.get_by_where.call_count == 1

    def test_failure_does_not_abort_scan_and_target_retried_next_scan(self):
        store = _StubStore(fail={"t1"})
        storage = _admin_storage_with(
            [
                _target("b1", "t1", finished_at=_BASE),
                _target("b1", "t2", finished_at=_BASE + timedelta(seconds=1)),
            ]
        )

        reconcile_once(store, storage, finished_after=_BASE)
        # Scan continued past the failure and recorded t2.
        assert ("b1", "t2") in store.recorded_calls
        assert ("b1", "t1") not in store.recorded_calls

        # Next scan: t1 no longer fails and is retried (still unrecorded).
        store._fail = set()
        reconcile_once(store, storage, finished_after=_BASE)
        assert ("b1", "t1") in store.recorded_calls

    def test_on_error_callback_invoked_on_failure(self):
        store = _StubStore(fail={"t1"})
        storage = _admin_storage_with([_target("b1", "t1", finished_at=_BASE)])
        errors = []

        reconcile_once(
            store,
            storage,
            finished_after=_BASE,
            on_error=lambda b, t, e: errors.append((b, t, str(e))),
        )

        assert ("b1", "t1") not in store.recorded_calls
        assert errors == [("b1", "t1", "boom")]

    def test_skip_set_excludes_targets_and_does_not_advance_watermark(self):
        store = _StubStore()
        t1 = _target("b1", "t1", finished_at=_BASE)
        t2 = _target("b1", "t2", finished_at=_BASE + timedelta(seconds=5))
        storage = _admin_storage_with([t1, t2])

        advances = []
        reconcile_once(
            store,
            storage,
            finished_after=_BASE,
            skip={"t2"},
            on_checkpoint_advance=lambda build_id, finished_at: advances.append(
                (build_id, finished_at)
            ),
        )

        # t2 is skipped (dropped), t1 still recorded.
        assert store.recorded_calls == [("b1", "t1")]
        # The checkpoint only advances over the actually-recorded target; a
        # skipped target must not carry it past t1.
        assert advances == [("b1", _BASE)]

    def test_on_checkpoint_advance_fires_per_target_oldest_first(self):
        store = _StubStore()
        t1 = _target("b1", "t1", finished_at=_BASE)
        t2 = _target("b2", "t2", finished_at=_BASE + timedelta(seconds=5))
        storage = _admin_storage_with([t1, t2])
        advances = []

        reconcile_once(
            store,
            storage,
            finished_after=_BASE,
            on_checkpoint_advance=lambda build_id, finished_at: advances.append(
                (build_id, finished_at)
            ),
        )

        assert advances == [
            ("b1", _BASE),
            ("b2", _BASE + timedelta(seconds=5)),
        ]

    def test_on_checkpoint_advance_stops_at_failed_target(self):
        """The checkpoint must not advance past a target that failed to record.

        t2 fails while t3 (newer) still records — a failure does not abort the
        scan. But the checkpoint may only cover the *contiguous* oldest-first run
        of recorded targets: advancing to t3 would durably move the watermark
        past t2's unrecorded lineage, so the next scan would not re-surface t2 and
        a restart (retry state is in-memory only) would drop it permanently.
        """
        store = _StubStore(fail={"t2"})
        t1 = _target("b1", "t1", finished_at=_BASE)
        t2 = _target("b1", "t2", finished_at=_BASE + timedelta(seconds=5))
        t3 = _target("b1", "t3", finished_at=_BASE + timedelta(seconds=10))
        storage = _admin_storage_with([t1, t2, t3])
        advances = []

        reconcile_once(
            store,
            storage,
            finished_after=_BASE,
            on_checkpoint_advance=lambda build_id, finished_at: advances.append(
                (build_id, finished_at)
            ),
        )

        # t3 is still recorded (the scan does not abort)...
        assert ("b1", "t3") in store.recorded_calls
        # ...but the checkpoint stops at t1, the last target before the failure.
        assert advances == [("b1", _BASE)]

    def test_passes_expected_run_counts_derived_from_outputs(self):
        store = _StubStore()
        # t1: two output-artifact names, the second holding two artifacts -> 3
        # runs. t2: no outputs -> the single "no-output" run.
        t1 = _target(
            "b1",
            "t1",
            finished_at=_BASE,
            output_artifacts={"a": ["o1"], "b": ["o2", "o3"]},
        )
        t2 = _target("b1", "t2", finished_at=_BASE + timedelta(seconds=1))
        storage = _admin_storage_with([t1, t2])

        reconcile_once(store, storage, finished_after=_BASE)

        assert store.last_expected_counts == {"t1": 3, "t2": 1}

    def test_skipped_for_prerun_target_omitted_from_expected_counts(self):
        store = _StubStore()
        # A skipped-for-prerun target records the *original* target's outputs, so
        # its own output_artifacts give the wrong count; it must be omitted and
        # fall back to the presence check.
        t1 = _target(
            "b1",
            "t1",
            finished_at=_BASE,
            output_artifacts={"a": ["o1"]},
            skipped_for_prerun_target_id="orig",
        )
        storage = _admin_storage_with([t1])

        reconcile_once(store, storage, finished_after=_BASE)

        assert store.last_expected_counts == {}


class TestExpectedRunCount:
    def test_counts_all_output_artifacts_across_lists(self):
        t = _target("b1", "t1", output_artifacts={"a": ["o1"], "b": ["o2", "o3"]})
        assert _expected_run_count(t) == 3

    def test_no_outputs_expects_one_run(self):
        assert _expected_run_count(_target("b1", "t1")) == 1


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


class TestStopAtAnchor:
    """The ``stop_at`` anchor: walk back to a named row, not to a time cutoff.

    The steady-state scan used to bound itself with ``finished_after`` alone
    (``watermark - _WATERMARK_OVERLAP``), which silently dropped lineage:
    ``finished_at`` is stamped when the *event is created*
    (``BuildEvent.timestamp``, ``default_factory=get_time``), not when the row is
    written, so a row can become visible carrying a timestamp already behind an
    advanced watermark. With concurrent builds of differing durations, targets
    interleave in ``finished_at`` order, so this is routine rather than an edge.

    ``stop_at`` replaces the fixed window with a row identity — ``(build_id,
    finished_at)`` — so the walk retreats to a known point regardless of how far
    back in time that is. The query filters only on ``status``, never
    ``build_id``, so sweeping that range picks up the straggling targets of
    *every* build, which is what closes the gap.
    """

    def test_includes_the_anchor_target(self):
        # Inclusive, mirroring the ``>=`` semantics of the finished_after cutoff:
        # the anchor row itself is returned (re-reading it is a harmless
        # idempotent no-op) rather than being treated as already-handled.
        #
        # This also covers the within-build case: rows above the anchor are kept
        # whatever build they belong to, so the anchor build's own later targets
        # are never behind the stop point. That the *anchor* is the build's oldest
        # target (rather than the checkpoint's own row) is a `_reconcile`
        # decision, pinned by
        # test_staggered_targets_within_checkpoint_build_are_not_lost.
        old = _target("b1", "t_old", finished_at=_BASE)
        anchor = _target("b2", "t_anchor", finished_at=_BASE + timedelta(seconds=10))
        newer = _target("b3", "t_newer", finished_at=_BASE + timedelta(seconds=20))
        storage = _admin_storage_with([old, anchor, newer])

        selected = select_recordable_targets(
            storage,
            finished_after=UTC_MIN,
            stop_at=("b2", _BASE + timedelta(seconds=10)),
        )

        assert {t.uuid for t in selected} == {"t_anchor", "t_newer"}

    def test_covers_concurrent_build_gap(self):
        """The reviewer's case: a quick build that finished behind the watermark.

        Build C is a long build whose targets finished at T and T+15; the T+15
        one recorded last, so C is the checkpoint's build. Build B started later
        but finished *earlier* (T+5), so it sits between C's two targets. Under
        the old one-minute window B was excluded from the query outright and,
        because the watermark only moves forward, was never re-surfaced — its
        lineage was lost permanently.

        The anchor is C's *oldest* target (T), which is what ``_reconcile``
        derives via ``get_oldest_successful_target``. Anchoring on C's *newest*
        row would stop the walk immediately and still lose B — which is precisely
        why the oldest is the right anchor.
        """
        c_early = _target("build-c", "t_c_early", finished_at=_BASE)
        b_target = _target("build-b", "t_b", finished_at=_BASE + timedelta(seconds=5))
        c_late = _target(
            "build-c", "t_c_late", finished_at=_BASE + timedelta(seconds=15)
        )
        storage = _admin_storage_with([c_early, b_target, c_late])

        selected = select_recordable_targets(
            storage,
            finished_after=UTC_MIN,
            stop_at=("build-c", _BASE),
        )

        # The straggler the fixed window dropped is back in scope.
        assert {t.uuid for t in selected} == {"t_c_early", "t_b", "t_c_late"}

    def test_falls_back_to_full_scan_when_anchor_row_not_found(self):
        """A missing anchor degrades toward scanning *more*, never less.

        The anchor row can legitimately be absent — pruned, or the
        ``--base-build-id all`` ``datetime.min`` sentinel that matches no real
        row. Returning everything read is the safe direction: re-reads are
        idempotent no-ops, whereas truncating would drop lineage silently.
        """
        t1 = _target("b1", "t1", finished_at=_BASE)
        t2 = _target("b1", "t2", finished_at=_BASE + timedelta(seconds=10))
        storage = _admin_storage_with([t1, t2])

        selected = select_recordable_targets(
            storage,
            finished_after=UTC_MIN,
            stop_at=("no-such-build", _BASE + timedelta(seconds=99)),
        )

        assert {t.uuid for t in selected} == {"t1", "t2"}

    def test_disambiguates_same_instant_different_build(self):
        """Matching is on ``(build_id, finished_at)``, not the timestamp alone.

        Concurrent builds can complete targets in the same instant. Stopping on
        the timestamp alone would end the walk at whichever row the backend
        happened to return first, dropping the other build's target.
        """
        same_instant = _BASE + timedelta(seconds=10)
        # The other build's row comes FIRST, so a match on the timestamp alone
        # would stop the walk here and never reach the real anchor — dropping it.
        other = _target("build-other", "t_other", finished_at=same_instant)
        anchor = _target("build-anchor", "t_anchor", finished_at=same_instant)
        older = _target("build-older", "t_older", finished_at=_BASE)
        storage = _admin_storage_returning([[other, anchor, older]])

        selected = select_recordable_targets(
            storage,
            finished_after=UTC_MIN,
            stop_at=("build-anchor", same_instant),
        )

        # Both same-instant rows are kept (the walk passed the other build's and
        # stopped at the anchor); the older row is never reached.
        assert {t.uuid for t in selected} == {"t_other", "t_anchor"}
