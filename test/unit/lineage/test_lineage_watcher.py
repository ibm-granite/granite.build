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

They cover the build-scoped selection (the checkpoint's build plus everything
created at or after it), the *contiguous* checkpoint advance that never steps over
a build still running, the fail-closed dedup contract (an unanswered query aborts
the whole pass and never advances the mark) including how a permanent sink failure
switches recording off, the retry/drop budget for an individual target, checkpoint
migration from the older target-shaped value, and that a *missing* checkpoint
records nothing at all rather than being seeded implicitly.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from gbserver.lineage.lineage_reconciler import (
    LINEAGE_WATCHER_CHECKPOINT_KEY,
    LINEAGE_WATCHER_CHECKPOINT_VERSION,
    LINEAGE_WATCHER_DROPPED_KEY,
)
from gbserver.lineage.lineage_seeding import BACKFILL_BUILD_ID
from gbserver.lineage.lineage_watcher import LineageWatcher
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.status import Status

# Aware UTC, matching what a real created_time/finished_at carries. A naive value
# here would be interpreted as *local* (see as_aware), so the expected cutoffs
# would shift by the test machine's UTC offset and the suite would only pass in UTC.
_BASE = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def _target(build_id: str, uuid: str, finished_at: datetime = None) -> StoredTargetRun:
    return StoredTargetRun(
        uuid=uuid,
        build_id=build_id,
        environment_uri="env://test",
        status=Status.SUCCESS,
        finished_at=finished_at if finished_at is not None else _BASE,
    )


def _build(uuid: str, created_time: datetime, status: Status) -> StoredBuild:
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


class _StubStore:
    """Lineage store stub: records into a set, dedupes per-sink, can be told to
    fail specific targets or to fail the dedup query itself."""

    def __init__(self, fail: set = None, query_error: Exception = None):
        self._recorded: set = set()
        self._fail: set = set(fail or set())
        self.query_error = query_error
        self.calls: list = []

    def add_jobstats_for_build_target(self, storage, build_id, target_id):
        if target_id in self._fail:
            raise RuntimeError("boom")
        self.calls.append((build_id, target_id))
        self._recorded.add(target_id)

    def filter_unrecorded(
        self, target_ids: set, expected_counts=None, on_query_error=None
    ) -> set:
        if self.query_error is not None:
            # Mirrors the real store: fail CLOSED (record nothing) and report the
            # failure through the callback, so an empty set is never mistaken for
            # "everything already recorded".
            if on_query_error is not None:
                on_query_error(self.query_error)
            return set()
        return set(target_ids) - self._recorded


class _StubKeyValuePairStorage:
    """In-memory stand-in for ``kv_pair_storage`` (the ``gb_kv_pairs`` store)."""

    def __init__(self):
        self._values: dict = {}

    def get_value(self, key):
        return self._values.get(key)

    def set_value(self, key, value):
        self._values[key] = value


@pytest.mark.live("storage", "lineage")
class TestLineageWatcher:
    """Selection, checkpoint advance and retry behaviour of ``_reconcile``."""

    @pytest.fixture(autouse=True)
    def _stub_storage(self):
        """Stub admin storage over configurable builds and targets.

        ``target_storage`` orders newest-``finished_at``-first and honors the
        ``build_id`` filter and pagination; ``build_storage`` orders by the
        ``created_time`` sort the build walk asks for, so the "stop at the cutoff"
        logic is exercised rather than bypassed.
        """
        self._targets: list[StoredTargetRun] = []
        self._builds: list[StoredBuild] = []
        admin_storage = MagicMock()

        def _targets_by_where(where, query_control=None):
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

        def _builds_by_where(where=None, query_control=None):
            ordered = sorted(
                self._builds,
                key=lambda b: (b.created_time is not None, b.created_time or _BASE),
                reverse=True,
            )
            if query_control is not None and query_control.pagination is not None:
                p = query_control.pagination
                start = p.index * p.size
                return ordered[start : start + p.size]
            return ordered

        admin_storage.target_storage.get_by_where.side_effect = _targets_by_where
        admin_storage.build_storage.get_by_where.side_effect = _builds_by_where
        admin_storage.build_storage.get_by_uuid.side_effect = lambda uuid: next(
            (b for b in self._builds if b.uuid == uuid), None
        )
        admin_storage.kv_pair_storage = _StubKeyValuePairStorage()
        self.storage = admin_storage
        with patch(
            "gbserver.lineage.lineage_watcher.get_admin_storage",
            return_value=admin_storage,
        ):
            yield

    def _seed(self, build_id: str, created_time: datetime) -> None:
        """Write a v2 checkpoint, the way ``lineage-watch --base-build-id`` does."""
        self.storage.kv_pair_storage.set_value(
            LINEAGE_WATCHER_CHECKPOINT_KEY,
            {
                "build_id": build_id,
                "created_time": created_time.isoformat(),
                "version": LINEAGE_WATCHER_CHECKPOINT_VERSION,
            },
        )

    def _checkpoint_build(self) -> str | None:
        """The build id the durable checkpoint names, or None if unset.

        Read back from storage rather than from the watcher: the checkpoint is the
        only place the mark lives, and it is what survives a restart.
        """
        value = self.storage.kv_pair_storage.get_value(LINEAGE_WATCHER_CHECKPOINT_KEY)
        return None if value is None else value.get("build_id")

    def _make_watcher(
        self, fail: set = None, query_error: Exception = None
    ) -> tuple[LineageWatcher, _StubStore]:
        watcher = LineageWatcher()
        store = _StubStore(fail=fail, query_error=query_error)
        watcher._store = store
        return watcher, store

    def _three_builds(self, middle_status: Status) -> tuple[str, str, str]:
        """A(finished) -> B(``middle_status``) -> C(finished), one target each.

        Returns their ids oldest-first. A is the seeded anchor.
        """
        a = _build("A", _BASE, Status.SUCCESS)
        b = _build("B", _BASE + timedelta(minutes=1), middle_status)
        c = _build("C", _BASE + timedelta(minutes=2), Status.SUCCESS)
        self._builds = [a, b, c]
        self._targets = [
            _target("A", "t-a"),
            _target("B", "t-b"),
            _target("C", "t-c"),
        ]
        self._seed("A", _BASE)
        return a.uuid, b.uuid, c.uuid

    # ---- selection -------------------------------------------------------

    def test_unseeded_watcher_records_nothing(self):
        """No checkpoint means record nothing — never an implicit full backfill.

        An unseeded deployment must not decide for the operator where recording
        begins; the alternative (defaulting to "everything") would drive the
        platform's whole history into the sink on first boot.
        """
        watcher, store = self._make_watcher()
        self._builds = [_build("A", _BASE, Status.SUCCESS)]
        self._targets = [_target("A", "t-a")]

        watcher._reconcile()

        assert store.calls == []
        assert self._checkpoint_build() is None

    def test_anchor_build_is_included_in_the_range(self):
        """The checkpoint's own build is re-selected, not skipped.

        The cutoff is ``>=`` so a pass that crashed partway through the anchor
        build can still finish it; excluding the anchor would strand those targets
        with nothing to bring them back (this is what replaces the old start-up
        verification sweep).
        """
        watcher, store = self._make_watcher()
        self._builds = [_build("A", _BASE, Status.SUCCESS)]
        self._targets = [_target("A", "t-a")]
        self._seed("A", _BASE)

        watcher._reconcile()

        assert ("A", "t-a") in store.calls

    def test_build_older_than_the_anchor_is_not_selected(self):
        """History behind the anchor stays out of range.

        The anchor is the operator's "start here" decision; walking behind it would
        re-drive arbitrarily much old history into the sink.
        """
        watcher, store = self._make_watcher()
        self._builds = [
            _build("OLD", _BASE - timedelta(days=1), Status.SUCCESS),
            _build("A", _BASE, Status.SUCCESS),
        ]
        self._targets = [_target("OLD", "t-old"), _target("A", "t-a")]
        self._seed("A", _BASE)

        watcher._reconcile()

        recorded = {target_id for _build_id, target_id in store.calls}
        assert recorded == {"t-a"}

    def test_new_build_is_picked_up_on_a_later_scan(self):
        """The build list is rebuilt every scan, so new builds need no registration."""
        watcher, store = self._make_watcher()
        self._builds = [_build("A", _BASE, Status.SUCCESS)]
        self._targets = [_target("A", "t-a")]
        self._seed("A", _BASE)
        watcher._reconcile()

        self._builds.append(_build("B", _BASE + timedelta(minutes=1), Status.SUCCESS))
        self._targets.append(_target("B", "t-b"))
        watcher._reconcile()

        assert ("B", "t-b") in store.calls

    def test_already_recorded_target_is_not_re_recorded(self):
        """Dedup is what prevents duplicates now that run ids are random."""
        watcher, store = self._make_watcher()
        self._builds = [_build("A", _BASE, Status.SUCCESS)]
        self._targets = [_target("A", "t-a")]
        self._seed("A", _BASE)

        watcher._reconcile()
        watcher._reconcile()

        assert store.calls == [("A", "t-a")]

    def test_finished_confirmed_build_is_not_re_read(self):
        """A finished, confirmed build is skipped without re-reading its targets.

        This is the mitigation for a pinned cutoff: without it, every scan would
        re-read the targets of every build above the mark for as long as one build
        stays stuck.
        """
        watcher, _store = self._make_watcher()
        self._builds = [
            _build("A", _BASE, Status.SUCCESS),
            _build("B", _BASE + timedelta(minutes=1), Status.RUNNING),
        ]
        self._targets = [_target("A", "t-a"), _target("B", "t-b")]
        self._seed("A", _BASE)
        watcher._reconcile()

        self.storage.target_storage.get_by_where.reset_mock()
        watcher._reconcile()

        read_builds = {
            call.args[0].get("build_id")
            for call in self.storage.target_storage.get_by_where.call_args_list
        }
        assert "A" not in read_builds, "a confirmed finished build was re-read"
        assert "B" in read_builds, "the unfinished build must still be re-read"

    # ---- contiguous checkpoint advance ------------------------------------

    def test_checkpoint_stays_at_a_running_build(self):
        """A running build blocks the mark, but later builds still record.

        Recording and advancing are deliberately separate: C's lineage is written
        immediately, while the mark waits so B cannot fall out of range while it can
        still produce targets.
        """
        watcher, store = self._make_watcher()
        _a, _b, _c = self._three_builds(middle_status=Status.RUNNING)

        watcher._reconcile()

        recorded = {target_id for _build_id, target_id in store.calls}
        assert recorded == {"t-a", "t-b", "t-c"}
        assert self._checkpoint_build() == "A"

    def test_checkpoint_never_skips_a_non_finished_build(self):
        """The advance stops at the first unfinished build, never jumping past it.

        Jumping to C would move the cutoff beyond B, and since nothing sweeps behind
        the anchor, B's remaining lineage would be lost for good.
        """
        watcher, _store = self._make_watcher()
        self._builds = [
            _build("A", _BASE, Status.SUCCESS),
            _build("B", _BASE + timedelta(minutes=1), Status.RUNNING),
            _build("C", _BASE + timedelta(minutes=2), Status.SUCCESS),
        ]
        self._targets = [_target("B", "t-b"), _target("C", "t-c")]
        self._seed("A", _BASE)

        watcher._reconcile()

        assert self._checkpoint_build() == "A"

    def test_checkpoint_advances_over_contiguous_finished_builds(self):
        """Once the blocking build finishes, one pass walks the whole run."""
        watcher, _store = self._make_watcher()
        self._three_builds(middle_status=Status.RUNNING)
        watcher._reconcile()
        assert self._checkpoint_build() == "A"

        self._builds[1].status = Status.SUCCESS
        watcher._reconcile()

        assert self._checkpoint_build() == "C"

    def test_unconfirmed_finished_build_blocks_the_advance(self):
        """A finished build whose target failed to record does not move the mark.

        Advancing past it would put it behind the cutoff with its lineage still
        missing and no later scan able to reach it.
        """
        watcher, _store = self._make_watcher(fail={"t-b"})
        self._three_builds(middle_status=Status.SUCCESS)

        watcher._reconcile()

        assert self._checkpoint_build() == "A"

    def test_build_with_no_targets_still_advances_the_checkpoint(self):
        """A build that produced no recordable target is trivially complete.

        Treating "nothing to record" as unconfirmed would pin the mark forever
        behind a build that will never have lineage.
        """
        watcher, _store = self._make_watcher()
        self._builds = [
            _build("A", _BASE, Status.SUCCESS),
            _build("B", _BASE + timedelta(minutes=1), Status.SUCCESS),
        ]
        self._targets = [_target("A", "t-a")]
        self._seed("A", _BASE)

        watcher._reconcile()

        assert self._checkpoint_build() == "B"

    # ---- fail-closed dedup ------------------------------------------------

    def test_dedup_failure_aborts_the_pass_and_records_nothing(self):
        """An unanswered dedup query must not be read as "nothing recorded".

        With random run ids, writing on an unanswered query duplicates runs rather
        than resuming them, so the pass stops and the mark stays put.
        """
        watcher, store = self._make_watcher(query_error=RuntimeError("timeout"))
        self._three_builds(middle_status=Status.SUCCESS)

        watcher._reconcile()

        assert store.calls == []
        assert self._checkpoint_build() == "A"

    def test_dedup_failure_does_not_process_later_builds(self):
        """The abort is per-pass, not per-build.

        A sink that cannot answer for one build will not answer for the next, so
        continuing would just accumulate duplicate-risk writes.
        """
        watcher, store = self._make_watcher()
        self._three_builds(middle_status=Status.SUCCESS)
        store.query_error = RuntimeError("timeout")

        watcher._reconcile()

        assert store.calls == []

    def test_pass_recovers_after_a_transient_dedup_failure(self):
        """Nothing is lost by aborting: the next scan re-selects everything."""
        watcher, store = self._make_watcher(query_error=RuntimeError("timeout"))
        self._three_builds(middle_status=Status.SUCCESS)
        watcher._reconcile()

        store.query_error = None
        watcher._reconcile()

        recorded = {target_id for _build_id, target_id in store.calls}
        assert recorded == {"t-a", "t-b", "t-c"}
        assert self._checkpoint_build() == "C"

    def test_permanent_dedup_failure_disables_recording(self):
        """A failure no retry can clear switches recording off instead of looping.

        Retrying forever would leave the watcher aborting every pass in silence with
        the mark frozen — indistinguishable from a healthy idle watcher.
        """
        watcher, store = self._make_watcher(
            query_error=RuntimeError("permission denied for project")
        )
        self._three_builds(middle_status=Status.SUCCESS)

        watcher._reconcile()

        assert watcher._recording_disabled is True
        assert store.calls == []
        assert self._checkpoint_build() == "A"

    def test_disabled_recording_stops_touching_the_sink(self):
        """Once disabled, later scans do not query or write, and stay alive.

        The process deliberately keeps running (rather than exiting) so the
        CRITICAL log is the signal; it must not silently resume either.
        """
        watcher, store = self._make_watcher(query_error=RuntimeError("invalid api key"))
        self._three_builds(middle_status=Status.SUCCESS)
        watcher._reconcile()

        store.query_error = None
        watcher._reconcile()

        assert store.calls == [], "a disabled watcher must not resume on its own"

    def test_transient_failure_is_not_treated_as_permanent(self):
        """An unrecognized failure counts as transient — the safe direction.

        Misclassifying a network blip as permanent would switch off recording for a
        condition that would have cleared on the next scan.
        """
        watcher, _store = self._make_watcher(
            query_error=RuntimeError("connection reset by peer")
        )
        self._three_builds(middle_status=Status.SUCCESS)

        watcher._reconcile()

        assert watcher._recording_disabled is False

    # ---- per-target retry and drop ---------------------------------------

    def test_failure_does_not_abort_the_build(self):
        """One failing target must not stop its build's other targets."""
        watcher, store = self._make_watcher(fail={"t-1"})
        self._builds = [_build("A", _BASE, Status.SUCCESS)]
        self._targets = [_target("A", "t-1"), _target("A", "t-2")]
        self._seed("A", _BASE)

        watcher._reconcile()

        assert ("A", "t-2") in store.calls

    def test_transient_failure_is_retried_on_the_next_scan(self):
        watcher, store = self._make_watcher(fail={"t-1"})
        self._builds = [_build("A", _BASE, Status.SUCCESS)]
        self._targets = [_target("A", "t-1")]
        self._seed("A", _BASE)
        watcher._reconcile()
        assert store.calls == []

        store._fail.clear()
        watcher._reconcile()

        assert store.calls == [("A", "t-1")]

    def test_persistent_failure_is_dropped_after_max_attempts(self):
        """A target that always fails is given up on, durably.

        Otherwise it pins the checkpoint forever: the mark refuses to pass a build
        with unrecorded lineage, so an un-droppable target wedges everything newer.
        """
        watcher, _store = self._make_watcher(fail={"t-1"})
        self._builds = [_build("A", _BASE, Status.SUCCESS)]
        self._targets = [_target("A", "t-1")]
        self._seed("A", _BASE)

        for _ in range(LineageWatcher._MAX_RECORD_ATTEMPTS):
            watcher._reconcile()

        assert "t-1" in watcher._dropped
        persisted = self.storage.kv_pair_storage.get_value(LINEAGE_WATCHER_DROPPED_KEY)
        assert persisted == {"target_ids": ["t-1"]}

    def test_transient_failure_gets_its_full_retry_budget(self):
        """Every failure is retryable: no rejection is treated as permanent here.

        The one that used to be (a run id the sink had seen and deleted) cannot
        occur now that ids are fresh random uuids.
        """
        watcher, _store = self._make_watcher(fail={"t-1"})
        self._builds = [_build("A", _BASE, Status.SUCCESS)]
        self._targets = [_target("A", "t-1")]
        self._seed("A", _BASE)

        watcher._reconcile()

        assert "t-1" not in watcher._dropped
        assert watcher._failed_attempts["t-1"] == 1

    def test_dropped_target_does_not_pin_the_checkpoint(self):
        """Once dropped, a target stops blocking the advance.

        The build is confirmed *with a known gap* — logged at ERROR — rather than
        holding the mark for lineage that will never land.
        """
        watcher, _store = self._make_watcher(fail={"t-a"})
        self._builds = [_build("A", _BASE, Status.SUCCESS)]
        self._targets = [_target("A", "t-a")]
        self._seed("A", _BASE)

        for _ in range(LineageWatcher._MAX_RECORD_ATTEMPTS + 1):
            watcher._reconcile()

        assert "t-a" in watcher._dropped
        assert self._checkpoint_build() == "A"

    def test_dropped_target_survives_a_restart(self):
        """The drop decision is durable, so a restart does not resurrect it."""
        watcher, _store = self._make_watcher(fail={"t-1"})
        self._builds = [_build("A", _BASE, Status.SUCCESS)]
        self._targets = [_target("A", "t-1")]
        self._seed("A", _BASE)
        for _ in range(LineageWatcher._MAX_RECORD_ATTEMPTS):
            watcher._reconcile()

        fresh = LineageWatcher()
        fresh._store = _StubStore()
        fresh._load_dropped(self.storage)

        assert "t-1" in fresh._dropped

    # ---- checkpoint value handling ---------------------------------------

    def test_backfill_anchor_records_everything(self):
        """The backfill sentinel names no build and reaches all history."""
        watcher, store = self._make_watcher()
        self._builds = [_build("A", _BASE - timedelta(days=30), Status.SUCCESS)]
        self._targets = [_target("A", "t-a")]
        self.storage.kv_pair_storage.set_value(
            LINEAGE_WATCHER_CHECKPOINT_KEY,
            {
                "build_id": BACKFILL_BUILD_ID,
                "created_time": datetime.min.replace(tzinfo=timezone.utc).isoformat(),
                "version": LINEAGE_WATCHER_CHECKPOINT_VERSION,
            },
        )

        watcher._reconcile()

        assert ("A", "t-a") in store.calls

    def test_legacy_target_shaped_checkpoint_is_migrated(self):
        """A v1 value keeps its place and is rewritten build-shaped.

        Only its ``build_id`` is reused: the v1 timestamp measured a *target*'s
        finish, so reusing it would put the cutoff at the wrong instant.
        """
        watcher, store = self._make_watcher()
        self._builds = [
            _build("A", _BASE, Status.SUCCESS),
            _build("B", _BASE + timedelta(minutes=1), Status.SUCCESS),
        ]
        self._targets = [_target("A", "t-a"), _target("B", "t-b")]
        self.storage.kv_pair_storage.set_value(
            LINEAGE_WATCHER_CHECKPOINT_KEY,
            {"build_id": "A", "finished_at": _BASE.isoformat()},
        )

        watcher._reconcile()

        value = self.storage.kv_pair_storage.get_value(LINEAGE_WATCHER_CHECKPOINT_KEY)
        assert value["version"] == LINEAGE_WATCHER_CHECKPOINT_VERSION
        assert "created_time" in value
        recorded = {target_id for _build_id, target_id in store.calls}
        assert recorded == {"t-a", "t-b"}

    def test_legacy_checkpoint_for_a_missing_build_records_nothing(self):
        """An unresolvable v1 anchor records nothing rather than everything.

        Falling back to "no cutoff" would silently turn a broken checkpoint into a
        full historical backfill.
        """
        watcher, store = self._make_watcher()
        self._builds = [_build("A", _BASE, Status.SUCCESS)]
        self._targets = [_target("A", "t-a")]
        self.storage.kv_pair_storage.set_value(
            LINEAGE_WATCHER_CHECKPOINT_KEY,
            {"build_id": "GONE", "finished_at": _BASE.isoformat()},
        )

        watcher._reconcile()

        assert store.calls == []

    def test_malformed_checkpoint_records_nothing_instead_of_raising(self):
        watcher, store = self._make_watcher()
        self._builds = [_build("A", _BASE, Status.SUCCESS)]
        self._targets = [_target("A", "t-a")]
        self.storage.kv_pair_storage.set_value(
            LINEAGE_WATCHER_CHECKPOINT_KEY, {"created_time": _BASE.isoformat()}
        )

        watcher._reconcile()

        assert store.calls == []

    def test_unparseable_checkpoint_records_nothing_instead_of_raising(self):
        watcher, store = self._make_watcher()
        self._builds = [_build("A", _BASE, Status.SUCCESS)]
        self._targets = [_target("A", "t-a")]
        self.storage.kv_pair_storage.set_value(
            LINEAGE_WATCHER_CHECKPOINT_KEY,
            {"build_id": "A", "created_time": "not-a-timestamp", "version": 2},
        )

        watcher._reconcile()

        assert store.calls == []

    def test_checkpoint_keeps_the_build_timestamp_form(self):
        """The stored timestamp matches the build row rather than being re-zoned.

        Rewriting offsets is what previously made the same row appear hours apart
        depending on which table it was read from.
        """
        watcher, _store = self._make_watcher()
        offset = timezone(timedelta(hours=-5))
        created = datetime(2026, 1, 1, 6, 0, 0, tzinfo=offset)
        self._builds = [
            _build("A", _BASE, Status.SUCCESS),
            _build("B", created, Status.SUCCESS),
        ]
        self._targets = [_target("A", "t-a"), _target("B", "t-b")]
        self._seed("A", _BASE)

        watcher._reconcile()

        value = self.storage.kv_pair_storage.get_value(LINEAGE_WATCHER_CHECKPOINT_KEY)
        assert value["build_id"] == "B"
        assert value["created_time"] == created.isoformat()
