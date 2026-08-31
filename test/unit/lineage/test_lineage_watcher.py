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


def _target(
    build_id: str,
    uuid: str,
    finished_at: datetime = None,
    name: str = "",
) -> StoredTargetRun:
    # Default ``name`` to ``uuid`` so each distinct uuid is a distinct *logical*
    # target: select_recordable_targets dedupes SUCCESS runs by name (in-place
    # retry can leave more than one SUCCESS run for one logical target), so
    # targets sharing a name collapse to one. These tests use distinct uuids to
    # mean distinct targets; pass ``name`` explicitly to model two runs of one.
    return StoredTargetRun(
        uuid=uuid,
        build_id=build_id,
        name=name or uuid,
        environment_uri="env://test",
        status=Status.SUCCESS,
        finished_at=finished_at if finished_at is not None else _BASE,
        # One output artifact so the target is recordable: these tests exercise
        # watcher/checkpoint mechanics, and select_recordable_targets drops
        # targets with no input and no output artifacts. The uuid need not
        # resolve in the registry -- nothing here reads the artifact itself.
        output_artifacts={"out0": [f"artifact-{uuid}"]},
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

    def test_running_base_holds_the_mark_but_later_builds_still_record(self):
        """The mark stops *on* a running build, and recording runs past it anyway.

        A is complete, so the mark steps onto B; B is running, so it stays there.
        Recording and advancing are deliberately separate: C's lineage is written
        on this same pass, while the mark waits on B so B cannot fall out of range
        while it can still produce targets.
        """
        watcher, store = self._make_watcher()
        _a, _b, _c = self._three_builds(middle_status=Status.RUNNING)

        watcher._reconcile()

        recorded = {target_id for _build_id, target_id in store.calls}
        assert recorded == {"t-a", "t-b", "t-c"}
        assert self._checkpoint_build() == "B", "A is complete, so the mark leaves it"

        # B still runs, so a further scan must not step over it onto C.
        watcher._reconcile()
        assert self._checkpoint_build() == "B"

    def test_targets_appearing_after_an_empty_scan_are_still_recorded(self):
        """A build scanned while it had no targets must be re-read once it has them.

        The real sequence from production: the watcher selects a build that is still
        RUNNING and has no gb_targets rows yet. That pass has nothing to record, so
        it reports all_confirmed -- a finished build with genuinely no lineage must
        not pin the checkpoint. But the confirmation rests on an empty set; the sink
        was never asked about anything.

        Caching that as complete is the bug: the skip gate is "cached complete AND
        finished", and when the build then succeeds and its target appears, both
        halves hold and the build is skipped without ever re-reading its targets.
        The lineage was recorded only after a service restart, which is the one
        thing that clears the in-memory set.
        """
        watcher, store = self._make_watcher()
        build = _build("B1", _BASE, Status.RUNNING)
        self._builds = [build]
        self._targets = []
        self._seed("B1", _BASE)

        watcher._reconcile()
        assert store.calls == [], "nothing to record while the build has no targets"
        assert "B1" not in watcher._complete_builds, (
            "an empty pass must not be cached as complete: the sink was never asked "
            "about any target, so there is nothing to skip re-reading later"
        )

        # By the time the build reads SUCCESS its target rows are already persisted
        # (buildrunner finalizes children before the parent), so a later scan sees
        # both -- provided the build was not cached as complete by the empty pass.
        build.status = Status.SUCCESS
        self._targets = [_target("B1", "t-b1")]

        watcher._reconcile()
        recorded = {target_id for _build_id, target_id in store.calls}
        assert recorded == {"t-b1"}, (
            "the target that appeared after the empty scan must be recorded without "
            "needing a restart to clear the completed-build cache"
        )

    def test_checkpoint_never_steps_off_a_non_finished_build(self):
        """The mark never leaves a running build, never jumping past it to C.

        This is the safety invariant of the walk. Moving the cutoff beyond B while
        B can still produce targets would lose them for good: nothing sweeps behind
        the anchor, so a target B emits after the mark passed it is unreachable.
        """
        watcher, _store = self._make_watcher()
        self._builds = [
            _build("A", _BASE, Status.SUCCESS),
            _build("B", _BASE + timedelta(minutes=1), Status.RUNNING),
            _build("C", _BASE + timedelta(minutes=2), Status.SUCCESS),
        ]
        self._targets = [_target("B", "t-b"), _target("C", "t-c")]
        self._seed("B", _BASE + timedelta(minutes=1))

        # Many scans, so a slow drift past B would show up rather than hide.
        for _ in range(5):
            watcher._reconcile()

        assert self._checkpoint_build() == "B"

    def test_checkpoint_advances_one_build_per_pass(self):
        """The mark steps base -> next -> next, one build per scan.

        Not a jump to the far end of an already-complete run: stepping keeps the
        durable mark close to the work, so a process that dies mid-catch-up resumes
        one build back instead of redoing the whole run.
        """
        watcher, _store = self._make_watcher()
        self._three_builds(middle_status=Status.RUNNING)
        # A is complete, so the mark steps onto B -- and stops there, because B is
        # running, even though C is finished and confirmed.
        watcher._reconcile()
        assert self._checkpoint_build() == "B"
        watcher._reconcile()
        assert self._checkpoint_build() == "B", "a running base must hold the mark"

        # B finishes and its lineage is confirmed, so the mark may leave it.
        self._builds[1].status = Status.SUCCESS
        watcher._reconcile()
        assert self._checkpoint_build() == "C"

        # Nothing left to move to; the mark stays put rather than drifting.
        watcher._reconcile()
        assert self._checkpoint_build() == "C"

    def test_confirmations_are_pruned_to_the_builds_still_in_range(self):
        """The confirmed-build set tracks the live window, not every build ever seen.

        Selection is ">= the anchor", so once the mark has moved past a build that
        build is never read again and its confirmation is dead weight. Without the
        prune the set grows by one uuid per build ever confirmed and is cleared only
        by a restart -- a slow leak in a long-lived daemon.
        """
        watcher, _store = self._make_watcher()
        self._three_builds(middle_status=Status.SUCCESS)

        watcher._reconcile()
        assert self._checkpoint_build() == "B"
        # A was confirmed this pass and is still selected (it is the anchor), so it
        # is kept -- the prune must not drop the build the mark just left until the
        # cutoff actually excludes it.
        assert watcher._complete_builds == {"A", "B", "C"}

        watcher._reconcile()
        assert self._checkpoint_build() == "C"
        # Now the cutoff is B, so A can never be selected again and is dropped.
        assert watcher._complete_builds == {"B", "C"}

        watcher._reconcile()
        assert watcher._complete_builds == {"C"}, "only the live window survives"

    def test_loop_does_not_wait_the_interval_while_it_is_advancing(self):
        """A scan that moved the mark is followed immediately by the next one.

        This is the catch-up path: with one advance per scan, sleeping between
        steps would stretch a backlog of N builds over N intervals.
        """
        watcher, _store = self._make_watcher()
        watcher.monitoring_interval = 3600.0
        waits: list[float] = []
        # Three advances, then no more progress -- which must end the catch-up and
        # fall back to the interval.
        results = iter([True, True, True, False])

        def _reconcile():
            try:
                return next(results)
            except StopIteration:
                watcher.stop_event.set()
                return False

        def _wait(timeout=None):
            waits.append(timeout)
            # Let the loop exit once it reaches the real interval wait.
            if timeout == watcher.monitoring_interval:
                watcher.stop_event.set()
            return watcher.stop_event.is_set()

        with (
            patch.object(watcher, "_reconcile", side_effect=_reconcile),
            patch.object(watcher.stop_event, "wait", side_effect=_wait),
        ):
            watcher._run()

        assert waits[:3] == [0, 0, 0], "advancing scans must not wait the interval"
        assert (
            waits[3] == watcher.monitoring_interval
        ), "the first scan with nothing to do falls back to the interval"

    def test_loop_waits_the_interval_after_a_crashed_scan(self):
        """A scan that raised is not progress: back off instead of hot-looping."""
        watcher, _store = self._make_watcher()
        watcher.monitoring_interval = 3600.0
        waits: list[float] = []

        def _wait(timeout=None):
            waits.append(timeout)
            watcher.stop_event.set()
            return True

        with (
            patch.object(watcher, "_reconcile", side_effect=RuntimeError("boom")),
            patch.object(watcher.stop_event, "wait", side_effect=_wait),
        ):
            watcher._run()

        assert waits == [watcher.monitoring_interval]

    def test_reconcile_reports_whether_the_mark_moved(self):
        """The return value is what lets the loop catch up without sleeping.

        One build per scan means a backlog of N builds needs N scans; if those
        scans were spaced by the full monitoring interval, catching up would take
        N intervals. ``_reconcile`` reporting "the mark moved" is what lets the
        loop run the next scan immediately (see ``_run``).
        """
        watcher, _store = self._make_watcher()
        self._three_builds(middle_status=Status.SUCCESS)

        # Two advances available (A -> B -> C), then nothing left to move to.
        assert watcher._reconcile() is True
        assert watcher._reconcile() is True
        assert watcher._reconcile() is False, "no advance left is not progress"

    def test_a_blocked_advance_is_not_reported_as_progress(self):
        """A running build holding the mark must not spin the loop.

        The interval is the only thing pacing the watcher while it waits for a
        build to finish, so "could not advance" has to read as no progress.
        """
        watcher, _store = self._make_watcher()
        self._three_builds(middle_status=Status.RUNNING)

        # The first scan does progress: A is complete, so the mark steps onto B.
        assert watcher._reconcile() is True
        # B is running, so there is nowhere to go. That must read as no progress,
        # or the loop would spin instead of waiting for B to finish.
        assert watcher._reconcile() is False

    def test_advance_ignores_a_build_sorting_ahead_of_the_anchor(self):
        """A same-instant build ordering before the anchor is not the destination.

        ``created_time`` is stamped in Python, so two builds created in the same
        instant have no defined order between them and one can sort ahead of the
        anchor. Advancing onto it would move the mark *backwards* in the list and
        put the anchor's successor behind the new cutoff.
        """
        watcher, _store = self._make_watcher()
        # X shares A's timestamp and is returned before it; B is genuinely newer.
        self._builds = [
            _build("X", _BASE, Status.SUCCESS),
            _build("A", _BASE, Status.SUCCESS),
            _build("B", _BASE + timedelta(minutes=1), Status.SUCCESS),
        ]
        self._targets = [_target("X", "t-x"), _target("A", "t-a"), _target("B", "t-b")]
        self._seed("A", _BASE)

        watcher._reconcile()

        assert self._checkpoint_build() == "B", (
            "the mark must step to the anchor's successor, not to a build "
            "sorting ahead of it"
        )

    def test_unconfirmed_finished_build_blocks_the_advance(self):
        """A finished build whose target failed to record does not move the mark.

        The gate is on the build being left: B is finished but its lineage never
        reached the sink, so the mark stays on B. Stepping off would put it behind
        the cutoff with its lineage still missing and no later scan able to reach
        it -- the second half of the gate, alongside "still running".
        """
        watcher, _store = self._make_watcher(fail={"t-b"})
        self._three_builds(middle_status=Status.SUCCESS)

        # A is complete, so the mark reaches B; B's target keeps failing, so it
        # stays there rather than drifting onto C. Kept under
        # ``_MAX_RECORD_ATTEMPTS`` scans: once the target is *dropped* the build
        # counts as confirmed on purpose (a dropped target must not pin the mark
        # forever), which is a different behaviour, covered elsewhere.
        assert watcher._MAX_RECORD_ATTEMPTS > 2, "scan count below relies on this"
        watcher._reconcile()
        assert self._checkpoint_build() == "B"
        watcher._reconcile()
        assert self._checkpoint_build() == "B"

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

    def test_finished_no_target_anchor_does_not_wedge_the_checkpoint(self):
        """The anchor itself has no targets: the mark must still step off it.

        The previous test seeds an anchor that *has* a target, so the advance is
        gated on a build that gets cached either way -- the no-target build is
        only the destination and never the anchor. Here the empty build is the
        anchor, which is the case that wedges: it is finished, so it will never
        gain a target, but _advance_checkpoint requires the anchor in
        _complete_builds. Refusing to cache it pins the mark on it forever and
        blocks every newer build's lineage behind it.
        """
        watcher, store = self._make_watcher()
        self._builds = [
            _build("A", _BASE, Status.SUCCESS),
            _build("B", _BASE + timedelta(minutes=1), Status.SUCCESS),
        ]
        self._targets = [_target("B", "t-b")]
        self._seed("A", _BASE)

        # Two scans: the first advances off the empty anchor A, the second must
        # then advance off B. A single scan passes even when wedged, because the
        # first pass is the one that reads A while it is still the anchor.
        watcher._reconcile()
        assert self._checkpoint_build() == "B", "the mark wedged on empty anchor A"

        watcher._reconcile()

        assert {target_id for _build_id, target_id in store.calls} == {"t-b"}

    def test_failed_anchor_with_no_targets_does_not_wedge_the_checkpoint(self):
        """A FAILED build is the common shape of a finished build with no lineage.

        ``select_builds_from_checkpoint`` has no status filter, so a FAILED build
        does become an anchor -- and it records nothing by definition. If that
        wedged the mark, one failed build would stop lineage for the platform.
        """
        watcher, _store = self._make_watcher()
        self._builds = [
            _build("A", _BASE, Status.FAILED),
            _build("B", _BASE + timedelta(minutes=1), Status.SUCCESS),
        ]
        self._targets = [_target("B", "t-b")]
        self._seed("A", _BASE)

        watcher._reconcile()

        assert self._checkpoint_build() == "B"

    def test_running_build_with_no_targets_is_not_cached_as_complete(self):
        """The original bug: an empty pass on a RUNNING build must not be cached.

        Caching it arms the skip gate ("cached complete AND finished") on a build
        whose targets are written moments later, so they are never re-read and the
        lineage appears only after a restart. Finished-and-empty is cached;
        running-and-empty is not, and the build state is the whole distinction.
        """
        watcher, store = self._make_watcher()
        self._builds = [_build("A", _BASE, Status.RUNNING)]
        self._targets = []
        self._seed("A", _BASE)

        watcher._reconcile()
        assert "A" not in watcher._complete_builds

        # A finishes and its targets land: the next scan must re-read them.
        self._builds = [_build("A", _BASE, Status.SUCCESS)]
        self._targets = [_target("A", "t-a")]

        watcher._reconcile()

        assert {target_id for _build_id, target_id in store.calls} == {"t-a"}

    def test_running_build_with_only_dropped_targets_is_not_cached_as_complete(self):
        """The sibling of the bug above, on the all-targets-dropped path.

        A RUNNING build whose only targets so far are all in the durable drop set
        also confirms without a sink answer: every target is filtered out of
        `candidates`, so filter_unrecorded is never called. That pass used to report
        all_confirmed without flagging itself unqueried, so it was cached complete
        and the skip gate ("cached complete AND finished") then swallowed the
        non-dropped target that arrived before the build finished -- the same
        restart-only recovery as the empty case.
        """
        watcher, store = self._make_watcher()
        self.storage.kv_pair_storage.set_value(
            LINEAGE_WATCHER_DROPPED_KEY, {"target_ids": ["t-dropped"]}
        )
        self._builds = [_build("A", _BASE, Status.RUNNING)]
        self._targets = [_target("A", "t-dropped")]
        self._seed("A", _BASE)

        watcher._reconcile()
        assert store.calls == [], "the only target is permanently dropped"
        assert "A" not in watcher._complete_builds, (
            "an all-dropped pass on a RUNNING build asked the sink nothing, so it "
            "must not arm the skip gate against targets still to come"
        )

        # A gains a target that is *not* dropped, then finishes: the next scan must
        # re-read its targets rather than skip the build.
        self._builds = [_build("A", _BASE, Status.SUCCESS)]
        self._targets = [_target("A", "t-dropped"), _target("A", "t-live")]

        watcher._reconcile()

        assert {target_id for _build_id, target_id in store.calls} == {"t-live"}

    def test_finished_build_with_only_dropped_targets_is_cached_as_complete(self):
        """The other half: a *finished* all-dropped build must still be cached.

        _advance_checkpoint requires the anchor in _complete_builds, so refusing to
        cache this build pins the mark on it forever and blocks every newer build's
        lineage -- exactly what the durable drop set exists to prevent. Flagging the
        pass unqueried must not cost that.
        """
        watcher, _store = self._make_watcher()
        self.storage.kv_pair_storage.set_value(
            LINEAGE_WATCHER_DROPPED_KEY, {"target_ids": ["t-dropped"]}
        )
        self._builds = [
            _build("A", _BASE, Status.SUCCESS),
            _build("B", _BASE + timedelta(minutes=1), Status.SUCCESS),
        ]
        self._targets = [_target("A", "t-dropped"), _target("B", "t-b")]
        self._seed("A", _BASE)

        watcher._reconcile()

        assert "A" in watcher._complete_builds
        assert self._checkpoint_build() == "B", "the mark must not wedge on A"

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
        # All three recorded on the recovery pass, but the mark still advances one
        # build at a time, so it lands on B rather than C.
        assert self._checkpoint_build() == "B"

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

    def test_clearing_the_row_retries_the_target_without_a_restart(self):
        """The point of re-reading every scan: `lineage-init --clear` needs no restart.

        Previously _dropped was loaded once in start(), so a cleared row was invisible
        to a live watcher -- and worse, the next drop persisted the whole stale set,
        undoing the operator's clear.
        """
        watcher, store = self._make_watcher(fail={"t-1"})
        self._builds = [_build("A", _BASE, Status.SUCCESS)]
        self._targets = [_target("A", "t-1")]
        self._seed("A", _BASE)
        for _ in range(LineageWatcher._MAX_RECORD_ATTEMPTS):
            watcher._reconcile()
        assert "t-1" in watcher._dropped

        # What the CLI writes, on the same live instance.
        self.storage.kv_pair_storage.set_value(
            LINEAGE_WATCHER_DROPPED_KEY, {"target_ids": []}
        )
        store._fail = set()
        watcher._reconcile()

        assert "t-1" not in watcher._dropped, "the clear must take effect on this scan"
        assert "t-1" in store._recorded, "the target must actually be retried"

    def test_the_watcher_does_not_undo_a_clear_on_its_next_drop(self):
        """A later drop must persist only what is still dropped, not a stale set."""
        watcher, store = self._make_watcher(fail={"t-1", "t-2"})
        self._builds = [_build("A", _BASE, Status.SUCCESS)]
        self._targets = [_target("A", "t-1"), _target("A", "t-2")]
        self._seed("A", _BASE)
        for _ in range(LineageWatcher._MAX_RECORD_ATTEMPTS):
            watcher._reconcile()
        assert watcher._dropped == {"t-1", "t-2"}

        self.storage.kv_pair_storage.set_value(
            LINEAGE_WATCHER_DROPPED_KEY, {"target_ids": []}
        )
        # t-2 keeps failing and gets re-dropped; t-1 must not come back with it.
        store._fail = {"t-2"}
        for _ in range(LineageWatcher._MAX_RECORD_ATTEMPTS + 1):
            watcher._reconcile()

        persisted = self.storage.kv_pair_storage.get_value(LINEAGE_WATCHER_DROPPED_KEY)
        assert "t-1" not in persisted["target_ids"], "the clear was undone"

    def test_an_unpersisted_drop_survives_the_next_scans_reload(self):
        """A drop whose persist failed must not be resurrected by the reload.

        _persist_dropped is non-raising, so the decision can exist only in memory;
        treating the row as authoritative without adding those back would re-record a
        hopeless target on every scan.
        """
        watcher, _store = self._make_watcher()
        watcher._dropped = {"t-9"}
        failing = MagicMock()
        failing.kv_pair_storage.set_value.side_effect = RuntimeError("kv down")
        watcher._persist_dropped(failing)
        assert watcher._dropped_unpersisted == {"t-9"}

        # The durable row never got it, but the reload must not drop it.
        watcher._load_dropped(self.storage)

        assert "t-9" in watcher._dropped

    def test_a_successful_persist_hands_ids_over_to_the_row(self):
        """After a good persist the ids live in the row, so a clear can reach them."""
        watcher, _store = self._make_watcher()
        watcher._dropped = {"t-9"}
        watcher._dropped_unpersisted = {"t-9"}
        watcher._persist_dropped(self.storage)

        assert watcher._dropped_unpersisted == set()

        self.storage.kv_pair_storage.set_value(
            LINEAGE_WATCHER_DROPPED_KEY, {"target_ids": []}
        )
        watcher._load_dropped(self.storage)

        assert watcher._dropped == set(), "a persisted id must follow the row"

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

    @pytest.mark.parametrize(
        "bad_value",
        [
            "oops",
            ["t-1"],
            7,
            {"target_ids": "t-1"},
            {"target_ids": 3},
            # These pass a container-only check but make _persist_dropped's sorted()
            # raise on the next drop, which unwinds into _run and fails every scan.
            {"target_ids": ["t-1", 7]},
            {"target_ids": [1, 2]},
            {"target_ids": [None]},
        ],
    )
    def test_an_unusable_drop_set_starts_empty_instead_of_raising(self, bad_value):
        """A corrupt row must not abort start(): a dead watcher records nothing.

        Empty is the only readable fallback, but it is the wedge the drop set
        exists to prevent, so it is logged at ERROR rather than swallowed.
        """
        self.storage.kv_pair_storage.set_value(LINEAGE_WATCHER_DROPPED_KEY, bad_value)

        fresh = LineageWatcher()
        fresh._store = _StubStore()
        fresh._load_dropped(self.storage)

        assert fresh._dropped == set()

    def test_an_unusable_drop_set_is_left_on_disk_for_inspection(self):
        """Loading never rewrites the bad row -- the operator still needs to see it."""
        self.storage.kv_pair_storage.set_value(LINEAGE_WATCHER_DROPPED_KEY, "oops")

        fresh = LineageWatcher()
        fresh._store = _StubStore()
        fresh._load_dropped(self.storage)

        assert (
            self.storage.kv_pair_storage.get_value(LINEAGE_WATCHER_DROPPED_KEY)
            == "oops"
        )

    def test_persisting_after_a_corrupt_load_does_not_raise(self):
        """The load fallback must leave a persistable set, or the watcher wedges.

        A mixed list loaded as-is makes ``_persist_dropped``'s sorted() raise on the
        next drop; ``on_error`` runs outside ``reconcile_build``'s try/except, so that
        unwinds into ``_run`` and fails every scan from then on -- worse than the
        empty-set fallback the guard chooses.
        """
        self.storage.kv_pair_storage.set_value(
            LINEAGE_WATCHER_DROPPED_KEY, {"target_ids": ["t-1", 7]}
        )

        fresh = LineageWatcher()
        fresh._store = _StubStore()
        fresh._load_dropped(self.storage)
        fresh._dropped.add("t-2")
        fresh._persist_dropped(self.storage)

        assert self.storage.kv_pair_storage.get_value(LINEAGE_WATCHER_DROPPED_KEY) == {
            "target_ids": ["t-2"]
        }

    def test_a_failing_persist_never_escapes_to_abort_the_scan(self):
        """Durability is best-effort; the scan loop is not.

        ``_on_record_error`` persists from inside ``reconcile_build``, so a raise here
        would kill the iteration and every one after it.
        """
        storage = MagicMock()
        storage.kv_pair_storage.set_value.side_effect = RuntimeError("kv down")

        fresh = LineageWatcher()
        fresh._store = _StubStore()
        fresh._dropped = {"t-1"}
        fresh._persist_dropped(storage)

        assert fresh._dropped == {"t-1"}

    def test_a_failing_drop_set_read_starts_empty_instead_of_raising(self):
        """A storage error is treated the same as a corrupt shape."""
        storage = MagicMock()
        storage.kv_pair_storage.get_value.side_effect = RuntimeError("kv down")

        fresh = LineageWatcher()
        fresh._store = _StubStore()
        fresh._load_dropped(storage)

        assert fresh._dropped == set()

    @pytest.mark.parametrize("bad_value", ["oops", {"target_ids": ["t-9", 7]}])
    def test_an_unusable_row_keeps_the_current_set(self, bad_value):
        """A bad row must not retry what this process knows is hopeless.

        _load_dropped now runs every scan, so "clear on failure" would re-record a
        dropped target on every pass for as long as the row stays corrupt. Keeping
        the in-memory set is the conservative direction.
        """
        self.storage.kv_pair_storage.set_value(LINEAGE_WATCHER_DROPPED_KEY, bad_value)

        fresh = LineageWatcher()
        fresh._store = _StubStore()
        fresh._dropped = {"t-known"}
        fresh._load_dropped(self.storage)

        assert fresh._dropped == {"t-known"}

    def test_starting_resets_a_set_left_by_a_previous_run(self):
        """start() re-reads from scratch, so a stale instance does not leak state."""
        self.storage.kv_pair_storage.set_value(
            LINEAGE_WATCHER_DROPPED_KEY, {"target_ids": ["t-row"]}
        )

        fresh = LineageWatcher()
        fresh._dropped = {"t-stale"}
        fresh._dropped_unpersisted = {"t-stale"}
        with (
            patch(
                "gbserver.lineage.lineage_watcher.get_lineage_store",
                return_value=_StubStore(),
            ),
            patch(
                "gbserver.lineage.lineage_watcher.get_admin_storage",
                return_value=self.storage,
            ),
            patch.object(LineageWatcher, "_run"),
        ):
            fresh.start()
            fresh.stop_event.set()

        assert fresh._dropped == {"t-row"}, "the stale set must not survive start()"

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

    def test_backfill_anchor_is_replaced_by_a_real_build(self):
        """The sentinel must be stepped off, not kept forever.

        It resolves to a UTC_MIN cutoff, so a checkpoint left on it re-selects the
        platform's whole history on every single scan. Advancing onto the first
        complete build is what retires it. The sentinel names no build, so it is
        never in the selected list -- the advance must not depend on finding it
        there.
        """
        watcher, _store = self._make_watcher()
        self._builds = [
            _build("A", _BASE, Status.SUCCESS),
            _build("B", _BASE + timedelta(minutes=1), Status.SUCCESS),
        ]
        self._targets = [_target("A", "t-a"), _target("B", "t-b")]
        self.storage.kv_pair_storage.set_value(
            LINEAGE_WATCHER_CHECKPOINT_KEY,
            {
                "build_id": BACKFILL_BUILD_ID,
                "created_time": datetime.min.replace(tzinfo=timezone.utc).isoformat(),
                "version": LINEAGE_WATCHER_CHECKPOINT_VERSION,
            },
        )

        watcher._reconcile()

        assert self._checkpoint_build() == "A", (
            "the backfill sentinel was not retired; every later scan would "
            "re-select all history"
        )

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
