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

"""Unit tests for the LineageWatcher async lineage-recording agent.

The watcher drives admin-DB reconciliation (see ``lineage_reconciler``) on an
interval; these tests stub the admin storage and the lineage store, then drive
``_reconcile`` directly (bypassing the background thread). They run in CI without
a cluster, PostgreSQL, or wandb credentials.

They cover that the watcher records successful targets, skips already-recorded
ones across scans, retries a transiently-failing target and drops a persistently
failing one after ``_MAX_RECORD_ATTEMPTS`` so it cannot wedge later scans.
"""

from unittest.mock import MagicMock, patch

import pytest

from gbserver.lineage.lineage_watcher import LineageWatcher
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.status import Status


def _target(build_id: str, uuid: str) -> StoredTargetRun:
    return StoredTargetRun(
        uuid=uuid,
        build_id=build_id,
        environment_uri="env://test",
        status=Status.SUCCESS,
    )


@pytest.mark.live("storage", "lineage")
class TestLineageWatcher:
    """Reconciliation and retry behaviour of LineageWatcher._reconcile."""

    @pytest.fixture(autouse=True)
    def _stub_storage(self):
        """Stub admin storage whose target_storage returns configurable targets."""
        self._targets: list[StoredTargetRun] = []
        admin_storage = MagicMock()
        admin_storage.target_storage.get_by_where.side_effect = lambda where: list(
            self._targets
        )
        self.storage = admin_storage
        with patch(
            "gbserver.lineage.lineage_watcher.get_admin_storage",
            return_value=admin_storage,
        ):
            yield

    def _make_watcher(self) -> tuple[LineageWatcher, MagicMock]:
        watcher = LineageWatcher()
        store = MagicMock()
        watcher._store = store
        return watcher, store

    def test_successful_target_records_lineage(self):
        self._targets = [_target("build-1", "target-1")]
        watcher, store = self._make_watcher()

        watcher._reconcile()

        store.add_jobstats_for_build_target.assert_called_once_with(
            self.storage, build_id="build-1", target_id="target-1"
        )
        assert watcher._recorded == {"target-1"}

    def test_already_recorded_target_not_reprocessed(self):
        self._targets = [_target("build-2", "target-2")]
        watcher, store = self._make_watcher()

        watcher._reconcile()
        assert store.add_jobstats_for_build_target.call_count == 1

        # A second scan over the same DB must not re-record.
        watcher._reconcile()
        assert store.add_jobstats_for_build_target.call_count == 1

    def test_failure_does_not_abort_batch(self):
        self._targets = [_target("build-a", "target-a"), _target("build-b", "target-b")]
        watcher, store = self._make_watcher()
        store.add_jobstats_for_build_target.side_effect = [RuntimeError("boom"), None]

        watcher._reconcile()

        assert store.add_jobstats_for_build_target.call_count == 2

    def test_transient_failure_is_retried_on_next_scan(self):
        self._targets = [_target("build-r", "target-r")]
        watcher, store = self._make_watcher()
        store.add_jobstats_for_build_target.side_effect = [RuntimeError("boom"), None]

        # First scan: fails, target queued for retry, not marked recorded.
        watcher._reconcile()
        assert watcher._failed_attempts == {"target-r": 1}
        assert "target-r" not in watcher._recorded

        # Second scan: retried (target still in the DB, not recorded) and clears.
        watcher._reconcile()
        assert store.add_jobstats_for_build_target.call_count == 2
        assert watcher._failed_attempts == {}
        assert watcher._recorded == {"target-r"}

    def test_persistent_failure_is_dropped_after_max_attempts(self):
        self._targets = [_target("build-p", "target-p")]
        watcher, store = self._make_watcher()
        store.add_jobstats_for_build_target.side_effect = RuntimeError("boom")

        # Scan enough times to exhaust the retry budget.
        for _ in range(LineageWatcher._MAX_RECORD_ATTEMPTS + 2):
            watcher._reconcile()

        assert (
            store.add_jobstats_for_build_target.call_count
            == LineageWatcher._MAX_RECORD_ATTEMPTS
        )
        assert watcher._failed_attempts == {}
        # Dropped target is marked recorded so it stops wedging every scan.
        assert "target-p" in watcher._recorded

    def test_full_rescan_records_targets_seen_while_down(self):
        """Fresh watcher (empty _recorded) records everything already in the DB.

        This is the restart-blind-spot fix: targets that succeeded while the
        watcher was down are recovered on the first scan after restart.
        """
        self._targets = [_target("b1", "t1"), _target("b2", "t2")]
        watcher, store = self._make_watcher()
        assert watcher._recorded == set()

        watcher._reconcile()

        assert watcher._recorded == {"t1", "t2"}
        assert store.add_jobstats_for_build_target.call_count == 2


@pytest.mark.live("storage", "lineage")
class TestLineageWatcherSeed:
    """start() seeds _recorded from the store to avoid re-emitting history."""

    def _start_without_thread(self, watcher: LineageWatcher) -> None:
        """Run start()'s seeding without launching the background thread."""
        with patch("threading.Thread"):
            watcher.start()

    def test_start_seeds_recorded_from_store(self):
        """A restart seeds the skip set from the store's already-recorded ids."""
        store = MagicMock()
        store.list_recorded_target_ids.return_value = {"t1", "t2"}
        watcher = LineageWatcher()
        with patch(
            "gbserver.lineage.lineage_watcher.get_lineage_store", return_value=store
        ):
            self._start_without_thread(watcher)

        store.list_recorded_target_ids.assert_called_once_with()
        assert watcher._recorded == {"t1", "t2"}

    def test_seeded_targets_are_not_re_emitted(self):
        """A target seeded as recorded is skipped, not re-driven through the leaf.

        This is the efficiency win: after a restart, history already in the store
        is not re-emitted. The seed is combined with a stubbed admin storage so we
        can drive a scan and confirm the seeded target never reaches the leaf.
        """
        store = MagicMock()
        store.list_recorded_target_ids.return_value = {"t1"}
        admin_storage = MagicMock()
        admin_storage.target_storage.get_by_where.side_effect = lambda where: [
            _target("b1", "t1"),
            _target("b2", "t2"),
        ]
        watcher = LineageWatcher()
        with patch(
            "gbserver.lineage.lineage_watcher.get_lineage_store", return_value=store
        ):
            self._start_without_thread(watcher)
        with patch(
            "gbserver.lineage.lineage_watcher.get_admin_storage",
            return_value=admin_storage,
        ):
            watcher._reconcile()

        # Only the un-seeded target t2 is recorded; the seeded t1 is skipped.
        store.add_jobstats_for_build_target.assert_called_once_with(
            admin_storage, build_id="b2", target_id="t2"
        )
        assert watcher._recorded == {"t1", "t2"}

    def test_start_tolerates_empty_seed(self):
        """An empty seed (e.g. store read failed) leaves a normal full-rescan state."""
        store = MagicMock()
        store.list_recorded_target_ids.return_value = set()
        watcher = LineageWatcher()
        with patch(
            "gbserver.lineage.lineage_watcher.get_lineage_store", return_value=store
        ):
            self._start_without_thread(watcher)

        assert watcher._recorded == set()
