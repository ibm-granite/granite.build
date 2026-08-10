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

They cover that the watcher records successful targets, advances its
``finished_at`` watermark so steady-state scans read only newly-finished
targets, does not re-record what a sink already has (per-sink
``filter_unrecorded``), retries a transiently-failing target, and drops a
persistently failing one after ``_MAX_RECORD_ATTEMPTS`` so it cannot wedge later
scans.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from gbserver.lineage.lineage_watcher import LineageWatcher
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.status import Status

_BASE = datetime(2026, 1, 1, 0, 0, 0)


def _target(build_id: str, uuid: str, finished_at: datetime = None) -> StoredTargetRun:
    return StoredTargetRun(
        uuid=uuid,
        build_id=build_id,
        environment_uri="env://test",
        status=Status.SUCCESS,
        finished_at=finished_at if finished_at is not None else _BASE,
    )


class _StubStore:
    """Lineage store stub: records into a set, dedupes per-sink, can be told to
    fail specific targets."""

    def __init__(self, fail: set = None):
        self._recorded: set = set()
        self._fail: set = set(fail or set())
        self.calls: list = []

    def add_jobstats_for_build_target(self, storage, build_id, target_id):
        if target_id in self._fail:
            raise RuntimeError("boom")
        self.calls.append((build_id, target_id))
        self._recorded.add(target_id)

    def filter_unrecorded(self, target_ids: set) -> set:
        return set(target_ids) - self._recorded


@pytest.mark.live("storage", "lineage")
class TestLineageWatcher:
    """Reconciliation and retry behaviour of LineageWatcher._reconcile."""

    @pytest.fixture(autouse=True)
    def _stub_storage(self):
        """Stub admin storage whose target_storage returns configurable targets,
        ordered newest-``finished_at``-first and honoring pagination."""
        self._targets: list[StoredTargetRun] = []
        admin_storage = MagicMock()

        def _get_by_where(where, query_control=None):
            ordered = sorted(
                self._targets,
                key=lambda t: (t.finished_at is not None, t.finished_at or _BASE),
                reverse=True,
            )
            if query_control is not None and query_control.pagination is not None:
                p = query_control.pagination
                start = p.index * p.size
                return ordered[start : start + p.size]
            return ordered

        admin_storage.target_storage.get_by_where.side_effect = _get_by_where
        self.storage = admin_storage
        with patch(
            "gbserver.lineage.lineage_watcher.get_admin_storage",
            return_value=admin_storage,
        ):
            yield

    def _make_watcher(self, fail: set = None) -> tuple[LineageWatcher, _StubStore]:
        watcher = LineageWatcher()
        store = _StubStore(fail=fail)
        watcher._store = store
        return watcher, store

    def test_successful_target_records_lineage(self):
        self._targets = [_target("build-1", "target-1", _BASE)]
        watcher, store = self._make_watcher()

        watcher._reconcile()

        assert store.calls == [("build-1", "target-1")]
        assert watcher._last_seen == _BASE

    def test_already_recorded_target_not_reprocessed(self):
        self._targets = [_target("build-2", "target-2", _BASE)]
        watcher, store = self._make_watcher()

        watcher._reconcile()
        assert len(store.calls) == 1

        # A second scan over the same DB must not re-record (filter_unrecorded).
        watcher._reconcile()
        assert len(store.calls) == 1

    def test_watermark_advances_and_steady_state_reads_only_new(self):
        self._targets = [_target("b1", "t1", _BASE)]
        watcher, store = self._make_watcher()

        watcher._reconcile()
        assert watcher._last_seen == _BASE

        # A newer target appears; the next scan picks it up and advances.
        new_at = _BASE + timedelta(seconds=30)
        self._targets.append(_target("b2", "t2", new_at))
        watcher._reconcile()

        assert ("b2", "t2") in store.calls
        assert watcher._last_seen == new_at

    def test_failure_does_not_abort_batch(self):
        self._targets = [
            _target("build-a", "target-a", _BASE),
            _target("build-b", "target-b", _BASE + timedelta(seconds=1)),
        ]
        watcher, store = self._make_watcher(fail={"target-a"})

        watcher._reconcile()

        # target-b still recorded despite target-a failing.
        assert ("build-b", "target-b") in store.calls

    def test_transient_failure_is_retried_on_next_scan(self):
        self._targets = [_target("build-r", "target-r", _BASE)]
        watcher, store = self._make_watcher(fail={"target-r"})

        # First scan: fails, target queued for retry, not recorded.
        watcher._reconcile()
        assert watcher._failed_attempts == {"target-r": 1}
        assert "target-r" not in store._recorded

        # Second scan: no longer failing, retried and clears (overlap guard
        # re-surfaces it since the watermark did not pass it).
        store._fail = set()
        watcher._reconcile()
        assert ("build-r", "target-r") in store.calls
        # Recovery clears the retry counter (via on_success): the target drops
        # out of the unrecorded set afterward, so on_error is never called for
        # it again and a lingering entry would leak for the process lifetime.
        assert watcher._failed_attempts == {}

    def test_persistent_failure_is_dropped_after_max_attempts(self):
        self._targets = [_target("build-p", "target-p", _BASE)]
        watcher, store = self._make_watcher(fail={"target-p"})

        for _ in range(LineageWatcher._MAX_RECORD_ATTEMPTS + 2):
            watcher._reconcile()

        assert len(store.calls) == 0
        assert watcher._failed_attempts == {}
        # Dropped target is in the skip set so it stops wedging every scan.
        assert "target-p" in watcher._dropped

    def test_first_scan_is_full_catch_up(self):
        """Fresh watcher (no watermark) records everything already in the DB.

        This is the restart-blind-spot fix: targets that succeeded while the
        watcher was down are recovered on the first scan after restart.
        """
        self._targets = [
            _target("b1", "t1", _BASE),
            _target("b2", "t2", _BASE + timedelta(seconds=1)),
        ]
        watcher, store = self._make_watcher()
        assert watcher._last_seen is None

        watcher._reconcile()

        assert {c[1] for c in store.calls} == {"t1", "t2"}
        assert watcher._last_seen == _BASE + timedelta(seconds=1)

    def test_stop_resets_watermark_for_full_catch_up(self):
        self._targets = [_target("b1", "t1", _BASE)]
        watcher, _ = self._make_watcher()
        watcher._reconcile()
        assert watcher._last_seen == _BASE

        watcher.stop()
        assert watcher._last_seen is None
