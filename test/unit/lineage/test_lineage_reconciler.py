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
that reconciliation selects successful targets from the admin DB and records
each through the single leaf, that a full rescan recovers targets that appeared
while the recorder was down (no restart blind spot), and that the leaf is the
one path all recording (scan + selective push) flows through.
"""

from unittest.mock import MagicMock

import pytest

from gbserver.lineage.lineage_reconciler import (
    reconcile_once,
    record_selected_targets,
    record_target_lineage,
    select_recordable_targets,
)
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.status import Status


def _target(
    build_id: str, uuid: str, status: Status = Status.SUCCESS
) -> StoredTargetRun:
    return StoredTargetRun(
        uuid=uuid,
        build_id=build_id,
        environment_uri="env://test",
        status=status,
    )


def _admin_storage_with(targets: list[StoredTargetRun]) -> MagicMock:
    """Stub admin storage whose target_storage returns the given SUCCESS targets.

    Asserts the reconciler queries by SUCCESS status rather than scanning all
    targets, so the selection contract is pinned.
    """
    storage = MagicMock()

    def _get_by_where(where):
        assert where == {"status": Status.SUCCESS.name}
        return [t for t in targets if t.status == Status.SUCCESS]

    storage.target_storage.get_by_where.side_effect = _get_by_where
    return storage


class TestSelectRecordableTargets:
    def test_selects_only_successful_targets(self):
        storage = _admin_storage_with(
            [
                _target("b1", "t1", Status.SUCCESS),
                _target("b1", "t2", Status.SUCCESS),
            ]
        )
        selected = select_recordable_targets(storage)
        assert {t.uuid for t in selected} == {"t1", "t2"}


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
        store = MagicMock()
        storage = _admin_storage_with([_target("b1", "t1"), _target("b1", "t2")])

        recorded = reconcile_once(store, storage)

        assert recorded == {"t1", "t2"}
        assert store.add_jobstats_for_build_target.call_count == 2

    def test_already_recorded_targets_are_skipped(self):
        store = MagicMock()
        storage = _admin_storage_with([_target("b1", "t1"), _target("b1", "t2")])

        recorded = reconcile_once(store, storage, already_recorded={"t1"})

        # Only the not-yet-recorded target is recorded this pass.
        store.add_jobstats_for_build_target.assert_called_once_with(
            storage, build_id="b1", target_id="t2"
        )
        assert recorded == {"t1", "t2"}

    def test_full_rescan_recovers_targets_seen_while_down(self):
        """A fresh scan (empty already_recorded) records everything in the DB.

        This is the restart-blind-spot fix: a target that succeeded while the
        recorder was down is present in the admin DB and picked up on the next
        scan, with no event replay required.
        """
        store = MagicMock()
        # Two targets succeeded "while the watcher was down".
        storage = _admin_storage_with([_target("b1", "t1"), _target("b2", "t2")])

        recorded = reconcile_once(store, storage, already_recorded=set())

        assert recorded == {"t1", "t2"}
        assert store.add_jobstats_for_build_target.call_count == 2

    def test_failure_does_not_abort_scan_and_target_retried_next_scan(self):
        store = MagicMock()
        storage = _admin_storage_with([_target("b1", "t1"), _target("b1", "t2")])
        # t1 fails, t2 succeeds on the first scan.
        store.add_jobstats_for_build_target.side_effect = [RuntimeError("boom"), None]

        recorded = reconcile_once(store, storage)

        # Scan continued past the failure and recorded t2.
        assert recorded == {"t2"}
        assert store.add_jobstats_for_build_target.call_count == 2

        # Next scan: t1 (still not recorded) is retried and now succeeds.
        store.add_jobstats_for_build_target.side_effect = None
        recorded = reconcile_once(store, storage, already_recorded=recorded)
        assert recorded == {"t1", "t2"}

    def test_on_error_callback_invoked_on_failure(self):
        store = MagicMock()
        storage = _admin_storage_with([_target("b1", "t1")])
        store.add_jobstats_for_build_target.side_effect = RuntimeError("boom")
        errors = []

        recorded = reconcile_once(
            store,
            storage,
            on_error=lambda b, t, e: errors.append((b, t, str(e))),
        )

        assert recorded == set()  # not recorded
        assert errors == [("b1", "t1", "boom")]


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
