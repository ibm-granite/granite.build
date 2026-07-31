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

These tests build an isolated SQLite event storage (unique table per test, no
external infra) and a stub lineage store, then drive the watcher's
event-filtering and watermark logic directly, bypassing the background thread.
They run in CI without a cluster, PostgreSQL, or wandb credentials.

They also guard the regression where the watcher filtered on a non-existent
``BuildEventType.TARGET_SUCCESS`` and therefore never recorded lineage: a
successful target completion is a STATUS_EVENT whose run-metadata type is
"Target" and whose payload status is SUCCESS.
"""

import itertools
import os
from unittest.mock import MagicMock, patch

import pytest

# Unique-per-process table name suffix so SQLite's on-disk tables never carry
# state across test runs or between tests, without needing to drop tables
# (which can contend on the shared DB file lock).
_TABLE_COUNTER = itertools.count()

from gbserver.lineage.lineage_watcher import LineageWatcher
from gbserver.storage.sqlite.sqlite_storage import SqliteEventStorage
from gbserver.storage.stored_event import StoredEvent
from gbserver.types.buildevent import (
    ArtifactPushedEventPayload,
    BuildEvent,
    BuildEventStatusPayload,
    BuildEventType,
    EntityRunMetadata,
)
from gbserver.types.status import Status


def _target_status_event(
    build_id: str,
    targetrun_id: str,
    status: Status,
    run_type: str = "Target",
) -> StoredEvent:
    """Build a STATUS_EVENT for a target run in the given status."""
    run_metadata = EntityRunMetadata(
        build_id=build_id,
        type=run_type,
        targetrun_id=targetrun_id,
    )
    build_event = BuildEvent(
        run_metadata=run_metadata,
        type=BuildEventType.STATUS_EVENT,
        payload=BuildEventStatusPayload(status=status, msg="msg"),
        source="test",
    )
    return StoredEvent(build_event=build_event)


def _artifact_event(build_id: str) -> StoredEvent:
    """Build a non-status event that must never trigger lineage recording."""
    run_metadata = EntityRunMetadata(build_id=build_id, type="Target")
    build_event = BuildEvent(
        run_metadata=run_metadata,
        type=BuildEventType.ARTIFACT_PUSHED_EVENT,
        payload=ArtifactPushedEventPayload(data={}, uri="https://localhost/x"),
        source="test",
    )
    return StoredEvent(build_event=build_event)


@pytest.mark.live("storage", "lineage")
class TestLineageWatcher:
    """Filtering and watermark behaviour of LineageWatcher._process_new_events."""

    @pytest.fixture(autouse=True)
    def _isolated_storage(self, request):
        """Provide an isolated SQLite event storage in a unique table.

        A unique table name per test keeps events from leaking between tests
        without depending on the global storage singleton or PostgreSQL.
        """
        table_name = f"gb_events_test_{os.getpid()}_{next(_TABLE_COUNTER)}"
        event_storage = SqliteEventStorage(table_name=table_name)

        admin_storage = MagicMock()
        admin_storage.event_storage = event_storage
        self.storage = admin_storage

        with patch(
            "gbserver.lineage.lineage_watcher.get_admin_storage",
            return_value=admin_storage,
        ):
            yield

    def _make_watcher(self) -> tuple[LineageWatcher, MagicMock]:
        """Create a watcher wired to a stub lineage store."""
        watcher = LineageWatcher()
        store = MagicMock()
        watcher._store = store
        return watcher, store

    def test_target_success_records_lineage(self):
        """A successful target status event records lineage with its ids."""
        self.storage.event_storage.add(
            _target_status_event("build-1", "target-1", Status.SUCCESS)
        )

        watcher, store = self._make_watcher()
        watcher._watermark = 0
        watcher._process_new_events()

        store.add_jobstats_for_build_target.assert_called_once_with(
            self.storage, build_id="build-1", target_id="target-1"
        )

    @pytest.mark.parametrize(
        "event_factory",
        [
            lambda: _target_status_event("build-1", "t-1", Status.FAILED),
            lambda: _target_status_event(
                "build-1", "t-1", Status.SUCCESS, run_type="TargetStep"
            ),
            lambda: _target_status_event(
                "build-1", "t-1", Status.SUCCESS, run_type="Build"
            ),
            lambda: _artifact_event("build-1"),
        ],
        ids=["failed_target", "target_step", "build", "artifact_pushed"],
    )
    def test_non_target_success_events_ignored(self, event_factory):
        """Events that are not a successful target completion record no lineage."""
        self.storage.event_storage.add(event_factory())

        watcher, store = self._make_watcher()
        watcher._watermark = 0
        watcher._process_new_events()

        store.add_jobstats_for_build_target.assert_not_called()

    def test_events_at_or_below_watermark_are_skipped(self):
        """Events already covered by the watermark are not reprocessed."""
        self.storage.event_storage.add(
            _target_status_event("build-old", "target-old", Status.SUCCESS)
        )
        seeded_watermark = self.storage.event_storage.get_max_index()

        watcher, store = self._make_watcher()
        watcher._watermark = seeded_watermark
        watcher._process_new_events()

        store.add_jobstats_for_build_target.assert_not_called()

    def test_watermark_advances_and_avoids_reprocessing(self):
        """After processing, the watermark advances past consumed events."""
        self.storage.event_storage.add(
            _target_status_event("build-2", "target-2", Status.SUCCESS)
        )

        watcher, store = self._make_watcher()
        watcher._watermark = 0
        watcher._process_new_events()
        assert store.add_jobstats_for_build_target.call_count == 1

        # A second pass with the advanced watermark must not reprocess.
        watcher._process_new_events()
        assert store.add_jobstats_for_build_target.call_count == 1

    def test_lineage_failure_does_not_abort_batch(self):
        """A failure recording one event does not prevent recording the next."""
        self.storage.event_storage.add(
            _target_status_event("build-a", "target-a", Status.SUCCESS)
        )
        self.storage.event_storage.add(
            _target_status_event("build-b", "target-b", Status.SUCCESS)
        )

        watcher, store = self._make_watcher()
        watcher._watermark = 0
        store.add_jobstats_for_build_target.side_effect = [RuntimeError("boom"), None]

        watcher._process_new_events()

        assert store.add_jobstats_for_build_target.call_count == 2

    def test_transient_failure_is_retried_on_next_poll(self):
        """A target that fails once is retried on a later poll and succeeds."""
        self.storage.event_storage.add(
            _target_status_event("build-r", "target-r", Status.SUCCESS)
        )

        watcher, store = self._make_watcher()
        watcher._watermark = 0
        store.add_jobstats_for_build_target.side_effect = [RuntimeError("boom"), None]

        # First poll: fails, watermark advances past the event, target queued.
        watcher._process_new_events()
        assert watcher._pending_retries == {("build-r", "target-r"): 1}

        # Second poll: no new events, but the queued target is retried and clears.
        watcher._process_new_events()
        assert store.add_jobstats_for_build_target.call_count == 2
        assert watcher._pending_retries == {}

    def test_persistent_failure_is_dropped_after_max_attempts(self):
        """A target that keeps failing is dropped and stops being retried."""
        self.storage.event_storage.add(
            _target_status_event("build-p", "target-p", Status.SUCCESS)
        )

        watcher, store = self._make_watcher()
        watcher._watermark = 0
        store.add_jobstats_for_build_target.side_effect = RuntimeError("boom")

        # Poll enough times to exhaust the retry budget.
        for _ in range(LineageWatcher._MAX_RECORD_ATTEMPTS + 2):
            watcher._process_new_events()

        assert (
            store.add_jobstats_for_build_target.call_count
            == LineageWatcher._MAX_RECORD_ATTEMPTS
        )
        assert watcher._pending_retries == {}

    def test_get_max_index_empty(self):
        """get_max_index returns 0 when no events exist."""
        assert self.storage.event_storage.get_max_index() == 0

    def test_get_events_after_index_ordering(self):
        """get_events_after_index returns ascending events strictly after index."""
        self.storage.event_storage.add(
            _target_status_event("build-1", "target-1", Status.SUCCESS)
        )
        self.storage.event_storage.add(
            _target_status_event("build-2", "target-2", Status.SUCCESS)
        )

        max_index = self.storage.event_storage.get_max_index()
        assert max_index >= 2

        # Nothing is newer than the max index.
        assert self.storage.event_storage.get_events_after_index(max_index) == []

        # Everything is newer than index 0, returned in ascending order.
        events = self.storage.event_storage.get_events_after_index(0)
        assert len(events) >= 2
        indices = [index for index, _event in events]
        assert indices == sorted(indices)
        assert all(isinstance(event, StoredEvent) for _index, event in events)

    def test_get_events_after_index_pages_beyond_page_size(self):
        """A backlog larger than the internal page_size (100) is fully returned.

        Exercises the multi-page ``while`` loop and the descending-order
        stop-at-watermark break in get_events_after_index, not just the single
        steady-state page.
        """
        total = 250  # > page_size (100), forcing multiple pages
        for i in range(total):
            self.storage.event_storage.add(
                _target_status_event(f"build-{i}", f"target-{i}", Status.SUCCESS)
            )

        # From index 0, every event is newer: all pages must be walked.
        events = self.storage.event_storage.get_events_after_index(0)
        assert len(events) == total
        indices = [index for index, _event in events]
        assert indices == sorted(indices)  # ascending for the caller
        assert len(set(indices)) == total  # no duplicates across page boundaries

        # A watermark mid-backlog returns only the strictly-newer tail, and the
        # descending scan stops once it crosses the watermark.
        cutoff = indices[total // 2]
        newer = self.storage.event_storage.get_events_after_index(cutoff)
        assert [idx for idx, _ in newer] == [i for i in indices if i > cutoff]
