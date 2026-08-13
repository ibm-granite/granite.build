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
``start()``/``_reconcile`` directly (bypassing the background thread). They run
in CI without a cluster, PostgreSQL, or wandb credentials.

They cover that the watcher records successful targets, persists its
``finished_at``/``build_id`` checkpoint to ``gb_status`` immediately after each
successfully-recorded target so steady-state scans (and restarts) read only
newly-finished targets, does not re-record what a sink already has (per-sink
``filter_unrecorded``), retries a transiently-failing target, drops a
persistently failing one after ``_MAX_RECORD_ATTEMPTS`` so it cannot wedge later
scans, and that ``start()`` loads/seeds/verifies the checkpoint correctly.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from gbserver.lineage.lineage_reconciler import LINEAGE_WATCHER_CHECKPOINT_KEY
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

    def filter_unrecorded(self, target_ids: set, expected_counts=None) -> set:
        return set(target_ids) - self._recorded


class _StubStatusStorage:
    """In-memory stand-in for ``status_storage`` (the ``gb_status`` key-value store)."""

    def __init__(self):
        self._values: dict = {}

    def get_value(self, key):
        return self._values.get(key)

    def set_value(self, key, value):
        self._values[key] = value


@pytest.mark.live("storage", "lineage")
class TestLineageWatcher:
    """Reconciliation and retry behaviour of LineageWatcher._reconcile."""

    @pytest.fixture(autouse=True)
    def _stub_storage(self):
        """Stub admin storage whose target_storage returns configurable targets,
        ordered newest-``finished_at``-first and honoring pagination and a
        ``build_id`` filter (used by checkpoint verification on start())."""
        self._targets: list[StoredTargetRun] = []
        admin_storage = MagicMock()

        def _get_by_where(where, query_control=None):
            matching = self._targets
            if where and "build_id" in where:
                matching = [t for t in matching if t.build_id == where["build_id"]]
            ordered = sorted(
                matching,
                key=lambda t: (t.finished_at is not None, t.finished_at or _BASE),
                reverse=True,
            )
            if query_control is not None and query_control.pagination is not None:
                p = query_control.pagination
                start = p.index * p.size
                return ordered[start : start + p.size]
            return ordered

        admin_storage.target_storage.get_by_where.side_effect = _get_by_where
        admin_storage.status_storage = _StubStatusStorage()
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

    def test_first_scan_after_seed_and_verify_records_only_the_next_target(self):
        """A fresh watcher seeds its checkpoint from the newest target, records
        it via the start()-time verification (not the scan itself), and the
        first ``_reconcile()`` only picks up anything newer than that.
        """
        self._targets = [
            _target("b1", "t1", _BASE),
            _target("b2", "t2", _BASE + timedelta(seconds=1)),
        ]
        watcher, store = self._make_watcher()
        watcher._load_or_seed_checkpoint(self.storage)
        # t2 (the newest) was seeded and verified/recorded at start()-time.
        assert watcher._last_seen == _BASE + timedelta(seconds=1)
        assert ("b2", "t2") in store.calls

        store.calls.clear()
        # A genuinely newer target appears. The scan re-reads t1 too (it falls
        # within the watermark-overlap window and was never actually recorded
        # in the stub store — only the checkpointed t2 was) but that re-read is
        # a harmless idempotent no-op in a real store; what matters is the scan
        # is bounded to the overlap window, not a full-DB rescan.
        self._targets.append(_target("b3", "t3", _BASE + timedelta(minutes=1)))
        watcher._reconcile()
        assert {c[1] for c in store.calls} == {"t1", "t3"}

    def test_stop_does_not_reset_watermark(self):
        self._targets = [_target("b1", "t1", _BASE)]
        watcher, _ = self._make_watcher()
        watcher._reconcile()
        assert watcher._last_seen == _BASE

        watcher.stop()
        # The checkpoint is persisted; a restart reloads it rather than needing
        # in-memory state to survive, so stop() leaves it as-is.
        assert watcher._last_seen == _BASE


@pytest.mark.live("storage", "lineage")
class TestLineageWatcherCheckpoint:
    """``start()``'s load/seed/verify flow, driven directly via
    ``_load_or_seed_checkpoint`` (bypassing the background thread)."""

    @pytest.fixture(autouse=True)
    def _stub_storage(self):
        self._targets: list[StoredTargetRun] = []
        admin_storage = MagicMock()

        def _get_by_where(where, query_control=None):
            matching = self._targets
            if where and "build_id" in where:
                matching = [t for t in matching if t.build_id == where["build_id"]]
            ordered = sorted(
                matching,
                key=lambda t: (t.finished_at is not None, t.finished_at or _BASE),
                reverse=True,
            )
            if query_control is not None and query_control.pagination is not None:
                p = query_control.pagination
                start = p.index * p.size
                return ordered[start : start + p.size]
            return ordered

        admin_storage.target_storage.get_by_where.side_effect = _get_by_where
        admin_storage.status_storage = _StubStatusStorage()
        self.storage = admin_storage
        yield

    def _make_watcher(self, fail: set = None) -> tuple[LineageWatcher, _StubStore]:
        watcher = LineageWatcher()
        store = _StubStore(fail=fail)
        watcher._store = store
        return watcher, store

    def test_seed_when_checkpoint_missing_with_existing_target_records_it(self):
        self._targets = [_target("b1", "t1", _BASE)]
        watcher, store = self._make_watcher()

        watcher._load_or_seed_checkpoint(self.storage)

        assert watcher._last_seen == _BASE
        assert ("b1", "t1") in store.calls
        assert self.storage.status_storage.get_value(
            LINEAGE_WATCHER_CHECKPOINT_KEY
        ) == {"build_id": "b1", "finished_at": _BASE.isoformat()}

    def test_seed_when_checkpoint_missing_and_no_targets_yet_writes_nothing(self):
        self._targets = []
        watcher, store = self._make_watcher()

        watcher._load_or_seed_checkpoint(self.storage)

        assert watcher._last_seen is None
        assert store.calls == []
        assert (
            self.storage.status_storage.get_value(LINEAGE_WATCHER_CHECKPOINT_KEY)
            is None
        )

    def test_load_checkpoint_already_recorded_is_a_no_op(self):
        self._targets = [_target("b1", "t1", _BASE)]
        watcher, store = self._make_watcher()
        self.storage.status_storage.set_value(
            LINEAGE_WATCHER_CHECKPOINT_KEY,
            {"build_id": "b1", "finished_at": _BASE.isoformat()},
        )
        store.calls.append(("b1", "t1"))  # Pre-record it in the stub store.
        store._recorded.add("t1")

        watcher._load_or_seed_checkpoint(self.storage)

        # Already recorded: filter_unrecorded excludes it, no duplicate call.
        assert store.calls == [("b1", "t1")]
        assert watcher._last_seen == _BASE

    def test_seed_not_persisted_when_its_target_fails_to_record(self):
        """A freshly-seeded checkpoint must not be written if recording failed.

        Persisting it would durably advance the watermark past a target whose
        lineage was never recorded (the failure is logged and swallowed), leaving
        nothing to retry it. Leaving it unwritten means the next start() re-seeds
        and retries.
        """
        self._targets = [_target("b1", "t1", _BASE)]
        watcher, store = self._make_watcher(fail={"t1"})

        watcher._load_or_seed_checkpoint(self.storage)

        assert store.calls == []
        assert (
            self.storage.status_storage.get_value(LINEAGE_WATCHER_CHECKPOINT_KEY) is None
        )
        assert watcher._last_seen is None

        # A later start() re-seeds and retries now that recording works.
        store._fail = set()
        watcher._load_or_seed_checkpoint(self.storage)
        assert ("b1", "t1") in store.calls
        assert watcher._last_seen == _BASE

    def test_verification_is_scoped_to_the_checkpoint_build(self):
        """Start-time verification only records the checkpoint's own build."""
        self._targets = [
            _target("b1", "t1", _BASE),
            _target("b2", "t2", _BASE + timedelta(seconds=5)),
        ]
        watcher, store = self._make_watcher()
        self.storage.status_storage.set_value(
            LINEAGE_WATCHER_CHECKPOINT_KEY,
            {"build_id": "b1", "finished_at": _BASE.isoformat()},
        )

        watcher._load_or_seed_checkpoint(self.storage)

        # b2's target is newer but belongs to another build: the steady-state
        # scan picks it up, not checkpoint verification.
        assert store.calls == [("b1", "t1")]

    def test_verification_handles_prerun_skipped_target(self):
        """A prerun-skipped target in the checkpoint's build records cleanly.

        It has no expected-run count of its own (it records the *original*
        target's outputs), so it must fall back to the presence check rather than
        being passed to filter_unrecorded with a missing count.
        """
        skipped = _target("b1", "t1", _BASE)
        skipped.skipped_for_prerun_target_id = "orig-target"
        self._targets = [skipped]
        watcher, store = self._make_watcher()

        watcher._load_or_seed_checkpoint(self.storage)

        assert ("b1", "t1") in store.calls
        assert watcher._last_seen == _BASE

    def test_load_checkpoint_not_actually_recorded_gets_recorded(self):
        self._targets = [_target("b1", "t1", _BASE)]
        watcher, store = self._make_watcher()
        # Checkpoint says t1 is done, but the store never actually recorded it
        # (e.g. a crash between recording and persisting the checkpoint).
        self.storage.status_storage.set_value(
            LINEAGE_WATCHER_CHECKPOINT_KEY,
            {"build_id": "b1", "finished_at": _BASE.isoformat()},
        )

        watcher._load_or_seed_checkpoint(self.storage)

        assert ("b1", "t1") in store.calls
        assert watcher._last_seen == _BASE
