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
``finished_at`` time watermark and records each through the single leaf, that a
full catch-up (no watermark) recovers targets that appeared while the recorder
was down (no restart blind spot), that steady-state scans read only newly-
finished targets, and that the per-sink ``filter_unrecorded`` check decides what
each sink actually records.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from gbserver.lineage.lineage_reconciler import (
    ReconcileResult,
    _expected_run_count,
    reconcile_once,
    record_selected_targets,
    record_target_lineage,
    select_recordable_targets,
)
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.status import Status

_BASE = datetime(2026, 1, 1, 0, 0, 0)


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
        selected = select_recordable_targets(storage)
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
        # A storage backend may hand finished_at back timezone-aware even though
        # it was written naive. Comparing a naive watermark against an aware
        # finished_at (or vice versa) would raise TypeError and abort the scan;
        # both sides are normalized to naive UTC so the walk still works. Here
        # _BASE is treated as UTC, so the aware value one hour later in UTC (=
        # _BASE + 1h) is newer than the watermark and must be selected.
        aware = (_BASE + timedelta(hours=1)).replace(tzinfo=timezone.utc)
        t_aware = _target("b1", "t_aware", finished_at=aware)
        older = _target("b1", "t_older", finished_at=_BASE - timedelta(seconds=10))
        storage = _admin_storage_returning([[t_aware, older]])

        selected = select_recordable_targets(storage, finished_after=_BASE)
        assert {t.uuid for t in selected} == {"t_aware"}


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

        result = reconcile_once(store, storage)

        assert {c[1] for c in store.recorded_calls} == {"t1", "t2"}
        assert result == ReconcileResult(_BASE + timedelta(seconds=1), "b1")
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

        reconcile_once(store, storage)

        # Only the not-yet-recorded target is recorded this pass.
        assert store.recorded_calls == [("b1", "t2")]

    def test_full_catch_up_recovers_targets_seen_while_down(self):
        """A scan with no watermark records everything in the DB.

        This is the restart-blind-spot fix: a target that succeeded while the
        recorder was down is present in the admin DB and picked up on the next
        scan, with no event replay required.
        """
        store = _StubStore()
        storage = _admin_storage_with(
            [
                _target("b1", "t1", finished_at=_BASE),
                _target("b2", "t2", finished_at=_BASE + timedelta(seconds=1)),
            ]
        )

        reconcile_once(store, storage, finished_after=None)

        assert {c[1] for c in store.recorded_calls} == {"t1", "t2"}

    def test_full_catch_up_pages_through_all_targets(self):
        store = _StubStore()
        targets = [
            _target("b1", f"t{i}", finished_at=_BASE + timedelta(seconds=i))
            for i in range(_SCAN_PAGE_SIZE + 5)
        ]
        storage = _admin_storage_with(targets)

        reconcile_once(store, storage, finished_after=None)
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

        reconcile_once(store, storage)
        # Scan continued past the failure and recorded t2.
        assert ("b1", "t2") in store.recorded_calls
        assert ("b1", "t1") not in store.recorded_calls

        # Next scan: t1 no longer fails and is retried (still unrecorded).
        store._fail = set()
        reconcile_once(store, storage)
        assert ("b1", "t1") in store.recorded_calls

    def test_on_error_callback_invoked_on_failure(self):
        store = _StubStore(fail={"t1"})
        storage = _admin_storage_with([_target("b1", "t1", finished_at=_BASE)])
        errors = []

        reconcile_once(
            store,
            storage,
            on_error=lambda b, t, e: errors.append((b, t, str(e))),
        )

        assert ("b1", "t1") not in store.recorded_calls
        assert errors == [("b1", "t1", "boom")]

    def test_skip_set_excludes_targets_and_does_not_advance_watermark(self):
        store = _StubStore()
        t1 = _target("b1", "t1", finished_at=_BASE)
        t2 = _target("b1", "t2", finished_at=_BASE + timedelta(seconds=5))
        storage = _admin_storage_with([t1, t2])

        result = reconcile_once(store, storage, skip={"t2"})

        # t2 is skipped (dropped), t1 still recorded.
        assert store.recorded_calls == [("b1", "t1")]
        # Watermark only reflects the actually-recorded target; a skipped
        # target must not advance a persisted checkpoint past it.
        assert result == ReconcileResult(_BASE, "b1")

    def test_on_checkpoint_advance_fires_per_target_oldest_first(self):
        store = _StubStore()
        t1 = _target("b1", "t1", finished_at=_BASE)
        t2 = _target("b2", "t2", finished_at=_BASE + timedelta(seconds=5))
        storage = _admin_storage_with([t1, t2])
        advances = []

        reconcile_once(
            store,
            storage,
            on_checkpoint_advance=lambda build_id, finished_at: advances.append(
                (build_id, finished_at)
            ),
        )

        assert advances == [
            ("b1", _BASE),
            ("b2", _BASE + timedelta(seconds=5)),
        ]

    def test_on_checkpoint_advance_stops_at_last_successful_target(self):
        store = _StubStore(fail={"t2"})
        t1 = _target("b1", "t1", finished_at=_BASE)
        t2 = _target("b1", "t2", finished_at=_BASE + timedelta(seconds=5))
        t3 = _target("b1", "t3", finished_at=_BASE + timedelta(seconds=10))
        storage = _admin_storage_with([t1, t2, t3])
        advances = []

        result = reconcile_once(
            store,
            storage,
            on_checkpoint_advance=lambda build_id, finished_at: advances.append(
                (build_id, finished_at)
            ),
        )

        # t2 fails; t1 and t3 still record (failure does not abort the scan),
        # but the checkpoint callback only fires for actually-recorded targets.
        assert advances == [
            ("b1", _BASE),
            ("b1", _BASE + timedelta(seconds=10)),
        ]
        assert result == ReconcileResult(_BASE + timedelta(seconds=10), "b1")

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

        reconcile_once(store, storage)

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

        reconcile_once(store, storage)

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
