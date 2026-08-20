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
``finished_at``/``build_id`` checkpoint to ``gb_kv_pairs`` immediately after each
successfully-recorded target so steady-state scans (and restarts) read only
newly-finished targets, does not re-record what a sink already has (per-sink
``filter_unrecorded``), retries a transiently-failing target, drops a
persistently failing one after ``_MAX_RECORD_ATTEMPTS`` so it cannot wedge later
scans, that ``start()`` loads and verifies the checkpoint correctly, and that a
*missing* checkpoint records nothing at all rather than being seeded implicitly.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from gbserver.lineage.lineage_reconciler import (
    LINEAGE_WATCHER_CHECKPOINT_KEY,
    LINEAGE_WATCHER_DROPPED_KEY,
)
from gbserver.lineage.lineage_watcher import LineageWatcher
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.status import Status

# Aware UTC, matching what a real finished_at carries: utils.get_time() stamps
# them with datetime.now().astimezone(). A naive value here would be interpreted
# as *local* (see as_aware), so the expected watermarks would shift by the
# test machine's UTC offset and the suite would only pass in UTC.
_BASE = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


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


class _StubKeyValuePairStorage:
    """In-memory stand-in for ``kv_pair_storage`` (the ``gb_kv_pairs`` key-value store)."""

    def __init__(self):
        self._values: dict = {}

    def get_value(self, key):
        return self._values.get(key)

    def set_value(self, key, value):
        self._values[key] = value


def _seed(storage, build_id: str, finished_at: datetime) -> None:
    """Write the lineage checkpoint, the way ``lineage-watch --base-build-id`` does."""
    storage.kv_pair_storage.set_value(
        LINEAGE_WATCHER_CHECKPOINT_KEY,
        {"build_id": build_id, "finished_at": finished_at.isoformat()},
    )


def _watermark(storage) -> datetime | None:
    """Read the checkpoint's watermark, or None if the key is unset.

    The watcher keeps no in-memory copy — the checkpoint is the only place the
    watermark lives — so assertions about "where the watcher got to" read it back
    from the durable store, which is also what survives a restart.
    """
    checkpoint = storage.kv_pair_storage.get_value(LINEAGE_WATCHER_CHECKPOINT_KEY)
    return (
        None
        if checkpoint is None
        else datetime.fromisoformat(checkpoint["finished_at"])
    )


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
        admin_storage.kv_pair_storage = _StubKeyValuePairStorage()
        self.storage = admin_storage
        with patch(
            "gbserver.lineage.lineage_watcher.get_admin_storage",
            return_value=admin_storage,
        ):
            yield

    def _make_watcher(
        self, fail: set = None, since: datetime = None
    ) -> tuple[LineageWatcher, _StubStore]:
        """Build a watcher against an already-seeded checkpoint.

        These tests exercise reconciliation/retry/drop behaviour, which only runs
        once the checkpoint exists — an unseeded watcher records nothing by design
        (see ``LineageWatcher._verify_checkpoint``). ``since`` defaults to a
        watermark old enough that every ``_BASE``-era target in the fixture falls
        within it, so a test that cares about recording rather than about where
        the watermark starts need not set it.
        """
        watcher = LineageWatcher()
        store = _StubStore(fail=fail)
        watcher._store = store
        _seed(self.storage, "seed-build", since or _BASE - timedelta(days=1))
        return watcher, store

    def test_unseeded_watcher_records_nothing(self):
        """With no checkpoint, a scan is a no-op: the key is never created
        implicitly and nothing is recorded, however many targets are waiting.

        This is the requirement that keeps a fresh deployment from silently
        backfilling the whole admin DB.
        """
        self._targets = [_target("build-1", "target-1", _BASE)]
        watcher, store = self._make_watcher()
        # Drop the seeded key: this is the unseeded state start() leaves behind.
        self.storage.kv_pair_storage._values.clear()

        watcher._reconcile()

        assert store.calls == []
        assert _watermark(self.storage) is None

    def test_backfill_anchor_checkpoint_still_records(self):
        """A ``datetime.min`` checkpoint (the ``--all`` backfill anchor) records.

        The anchor matches no real row, so ``select_recordable_targets`` walks the
        whole table and returns everything — which is exactly what a full-history
        backfill asks for. Previously this path did arithmetic on the watermark
        (``datetime.min - _WATERMARK_OVERLAP``), which raised ``OverflowError``
        before any recording and failed every scan forever; there is no such
        subtraction left to overflow.
        """
        self._targets = [_target("build-1", "target-1", _BASE)]
        watcher, store = self._make_watcher(since=datetime.min)

        # Must not raise (the loop catches, so assert on the effect too).
        watcher._reconcile()

        assert store.calls == [("build-1", "target-1")]
        assert _watermark(self.storage) == _BASE

    def test_concurrent_build_gap_is_recorded(self):
        """The reviewer's case, end to end: a quick build behind the watermark.

        Build C is long-running with targets at T and T+15; the T+15 one recorded
        last, so C is the checkpoint's build. Build B started later and finished
        *earlier* (T+5), landing between C's two targets. Under the old fixed
        one-minute window B fell outside the query and, the watermark being
        monotonic, was never re-surfaced — its lineage was lost silently. The
        anchor is C's oldest target, so the walk reaches back past B.
        """
        self._targets = [
            _target("build-c", "t_c_early", _BASE),
            _target("build-b", "t_b", _BASE + timedelta(seconds=5)),
            _target("build-c", "t_c_late", _BASE + timedelta(seconds=15)),
        ]
        watcher, store = self._make_watcher()
        _seed(self.storage, "build-c", _BASE + timedelta(seconds=15))

        watcher._reconcile()

        assert "t_b" in {c[1] for c in store.calls}

    def test_staggered_targets_within_checkpoint_build_are_not_lost(self):
        """The checkpoint's build keeps its own earlier targets in scope.

        The checkpoint names whichever target recorded last (here C's T+15 one),
        but a build's targets finish at staggered times. Anchoring on that row
        would leave C's own T+5 target behind the stop point; anchoring on the
        build's *oldest* target includes it.
        """
        self._targets = [
            _target("build-c", "t_early", _BASE + timedelta(seconds=5)),
            _target("build-c", "t_late", _BASE + timedelta(seconds=15)),
        ]
        watcher, store = self._make_watcher()
        _seed(self.storage, "build-c", _BASE + timedelta(seconds=15))

        watcher._reconcile()

        assert {c[1] for c in store.calls} == {"t_early", "t_late"}

    def test_target_behind_the_watermark_does_not_lower_it(self):
        """Recording a target behind the watermark must not move it backward.

        Regression: ``_on_checkpoint_advance`` wrote unconditionally, so a target
        that legitimately finished behind the current watermark (its row became
        visible late — see the anchor rationale in ``_reconcile``) dragged the
        durable watermark below its prior high mark. With no newer target in the
        same pass to push it back up it stayed lowered, and every later scan
        re-read that range. The target is still recorded; only the watermark write
        is suppressed.

        The checkpoint names ``build-1`` so the anchor resolves to that build's
        oldest target, which is what brings the behind-the-watermark row into
        scope in the first place.
        """
        behind = _BASE - timedelta(seconds=30)
        self._targets = [_target("build-1", "target-1", behind)]
        watcher, store = self._make_watcher()
        _seed(self.storage, "build-1", _BASE)

        watcher._reconcile()

        assert store.calls == [("build-1", "target-1")]
        assert _watermark(self.storage) == _BASE

    def test_watermark_still_advances_for_a_newer_target(self):
        """The monotonic guard must not block genuine forward progress."""
        ahead = _BASE + timedelta(seconds=30)
        self._targets = [_target("build-1", "target-1", ahead)]
        watcher, store = self._make_watcher(since=_BASE)

        watcher._reconcile()

        assert store.calls == [("build-1", "target-1")]
        assert _watermark(self.storage) == ahead

    def test_timezone_aware_checkpoint_watermark_records(self):
        """A checkpoint seeded with a timezone-aware ``finished_at`` still records.

        Regression: the checkpoint is written straight from a stored target's
        ``finished_at``, which a storage backend or DB driver may hand back
        timezone-aware (this is why ``as_aware`` exists at all). An aware
        watermark made ``_reconcile``'s ``watermark - datetime.min`` raise
        ``TypeError: can't subtract offset-naive and offset-aware datetimes``
        before any recording, so every scan failed and nothing was ever
        recorded. The watermark is normalized to naive UTC on read instead.
        """
        self._targets = [_target("build-1", "target-1", _BASE)]
        watcher, store = self._make_watcher()
        # UTC+0 keeps the instant identical to the naive seed, so this isolates
        # awareness itself rather than also shifting the watermark.
        aware = (_BASE - timedelta(days=1)).replace(tzinfo=timezone.utc)
        _seed(self.storage, "seed-build", aware)

        watcher._reconcile()

        assert store.calls == [("build-1", "target-1")]

    def test_timezone_aware_target_finished_at_advances_the_checkpoint(self):
        """An aware ``finished_at`` must still advance the checkpoint.

        Regression: the monotonic guard in ``_on_checkpoint_advance`` compared the
        target's ``finished_at`` — passed through raw by ``reconcile_once`` — against
        the normalized watermark from ``_checkpoint_watermark``. In production
        ``finished_at`` is aware (stamped from ``BuildEvent.timestamp``, i.e.
        ``get_time()``), so the comparison raised ``TypeError: can't compare
        offset-naive and offset-aware datetimes``.

        The raise lands inside ``reconcile_once``'s per-target ``try``, so it was
        misattributed to *recording* — which had already succeeded — blocking the
        checkpoint and, after ``_MAX_RECORD_ATTEMPTS`` scans, dropping the target
        durably. Hence both assertions: recording alone passed throughout, so only
        the watermark catches this.

        The sibling test above makes the *checkpoint* aware and the target naive,
        which exercises the read path's normalization and never reaches this
        comparison; this is the inverse case.
        """
        # UTC-03:00 rather than UTC so a naive-vs-aware mixup cannot coincidentally
        # compare equal, and a day ahead so the watermark must genuinely move.
        aware = (_BASE + timedelta(days=1)).replace(
            tzinfo=timezone(timedelta(hours=-3))
        )
        self._targets = [_target("build-1", "target-1", aware)]
        watcher, store = self._make_watcher()
        _seed(self.storage, "seed-build", _BASE)

        watcher._reconcile()

        assert store.calls == [("build-1", "target-1")]
        # Normalized to aware UTC on write: 2026-01-02T00:00-03:00 ->
        # 2026-01-02T03:00+00:00 (the same instant, one unambiguous format).
        assert _watermark(self.storage) == aware.astimezone(timezone.utc)

    def test_malformed_checkpoint_records_nothing_instead_of_raising(self):
        """A checkpoint missing ``finished_at`` turns recording off, not a crash.

        ``_reconcile`` used to index the key directly, so a hand-edited or
        partially-written ``gb_kv_pairs`` row raised ``KeyError`` out of the scan.
        Recording stays off until the key is corrected — the safe direction.
        """
        self._targets = [_target("build-1", "target-1", _BASE)]
        watcher, store = self._make_watcher()
        self.storage.kv_pair_storage.set_value(
            LINEAGE_WATCHER_CHECKPOINT_KEY, {"build_id": "seed-build"}
        )

        watcher._reconcile()

        assert store.calls == []

    def test_unparseable_checkpoint_records_nothing_instead_of_raising(self):
        """A non-ISO ``finished_at`` is reported and disables recording."""
        self._targets = [_target("build-1", "target-1", _BASE)]
        watcher, store = self._make_watcher()
        self.storage.kv_pair_storage.set_value(
            LINEAGE_WATCHER_CHECKPOINT_KEY,
            {"build_id": "seed-build", "finished_at": "not-a-timestamp"},
        )

        watcher._reconcile()

        assert store.calls == []

    def test_overflowing_aware_checkpoint_records_instead_of_wedging(self):
        """An aware ``datetime.min`` checkpoint records rather than failing forever.

        Regression: ``--base-build-id all`` anchors the checkpoint at
        ``datetime.min``, and a backend may hand that back timezone-aware with a
        positive UTC offset.
        Normalizing it shifts backwards past ``datetime.min``, raising
        ``OverflowError`` — which the read guard did not catch (only ``TypeError``
        and ``ValueError``), so it escaped to ``_run``'s blanket handler and every
        scan failed, recording nothing at all. That is the same wedge the guard
        already prevents for unparseable values, so it is handled the same way:
        the value clamps to ``datetime.min`` and the backfill proceeds.
        """
        self._targets = [_target("build-1", "target-1", _BASE)]
        watcher, store = self._make_watcher()
        _seed(
            self.storage,
            "seed-build",
            datetime.min.replace(tzinfo=timezone(timedelta(hours=5))),
        )

        # Must not raise, and must still record (a clamped datetime.min anchor is
        # a full backfill, not a disabled scan).
        watcher._reconcile()

        assert store.calls == [("build-1", "target-1")]

    def test_successful_target_records_lineage(self):
        self._targets = [_target("build-1", "target-1", _BASE)]
        watcher, store = self._make_watcher()

        watcher._reconcile()

        assert store.calls == [("build-1", "target-1")]
        assert _watermark(self.storage) == _BASE

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
        assert _watermark(self.storage) == _BASE

        # A newer target appears; the next scan picks it up and advances.
        new_at = _BASE + timedelta(seconds=30)
        self._targets.append(_target("b2", "t2", new_at))
        watcher._reconcile()

        assert ("b2", "t2") in store.calls
        assert _watermark(self.storage) == new_at

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

    def test_dropped_target_survives_restart_and_unblocks_checkpoint(self):
        """A permanently-dropped target must not come back after a restart.

        The checkpoint refuses to advance past an unrecorded target, so if the
        drop set were in-memory only, a restart would resurrect the failing
        target, block the watermark, exhaust its attempts again, and repeat
        forever — wedging every newer target behind it.
        """
        self._targets = [_target("build-p", "target-p", _BASE)]
        started_at = _BASE - timedelta(days=1)
        watcher, store = self._make_watcher(fail={"target-p"}, since=started_at)

        # Exhaust the attempts: target-p is dropped and the drop is persisted.
        for _ in range(LineageWatcher._MAX_RECORD_ATTEMPTS):
            watcher._reconcile()
        assert "target-p" in watcher._dropped
        assert self.storage.kv_pair_storage.get_value(LINEAGE_WATCHER_DROPPED_KEY) == {
            "target_ids": ["target-p"]
        }
        # It blocked the checkpoint while it was still being retried: the
        # watermark never moved off where the watcher started.
        assert _watermark(self.storage) == started_at

        # Simulate a restart: fresh watcher, same durable gb_kv_pairs. It reloads
        # the drop set the way start() does; the watermark needs no restoring
        # since it is read from the checkpoint on each scan.
        restarted = LineageWatcher()
        restarted._store = store
        restarted._load_dropped(self.storage)
        assert restarted._dropped == {"target-p"}

        # A newer target arrives. target-p stays skipped rather than being
        # retried from zero, so it no longer blocks the checkpoint: the newer
        # target records and the watermark advances past target-p.
        newer_at = _BASE + timedelta(seconds=10)
        self._targets.append(_target("build-q", "target-q", newer_at))
        restarted._reconcile()

        assert ("build-p", "target-p") not in store.calls
        assert ("build-q", "target-q") in store.calls
        assert _watermark(self.storage) == newer_at

    def test_first_scan_after_load_and_verify_records_only_the_next_target(self):
        """Starting from a seeded checkpoint, the checkpoint's own target is
        recorded by the start()-time verification (not by the scan), and the
        first ``_reconcile()`` only picks up anything newer than that.
        """
        checkpoint_at = _BASE + timedelta(seconds=1)
        self._targets = [
            _target("b1", "t1", _BASE),
            _target("b2", "t2", checkpoint_at),
        ]
        watcher, store = self._make_watcher()
        _seed(self.storage, "b2", checkpoint_at)
        watcher._verify_checkpoint(self.storage)
        # t2 is the checkpoint's target: verified and recorded at start()-time.
        assert _watermark(self.storage) == checkpoint_at
        assert ("b2", "t2") in store.calls

        store.calls.clear()
        # A genuinely newer target appears and is picked up. t1 is NOT: it
        # finished *before* the anchor (b2's oldest target, which is the
        # checkpointed t2 itself), and the anchored walk reaches back to the
        # anchor row, not past it. That is the accepted bound of this design —
        # the anchor covers stragglers between it and now, not lineage already
        # behind it, which is what the start()-time verification sweep and an
        # explicit re-seed are for.
        self._targets.append(_target("b3", "t3", _BASE + timedelta(minutes=1)))
        watcher._reconcile()
        assert {c[1] for c in store.calls} == {"t3"}


@pytest.mark.live("storage", "lineage")
class TestLineageWatcherCheckpoint:
    """``start()``'s checkpoint verification, driven directly via
    ``_verify_checkpoint`` (bypassing the background thread)."""

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
        admin_storage.kv_pair_storage = _StubKeyValuePairStorage()
        self.storage = admin_storage
        yield

    def _make_watcher(self, fail: set = None) -> tuple[LineageWatcher, _StubStore]:
        watcher = LineageWatcher()
        store = _StubStore(fail=fail)
        watcher._store = store
        return watcher, store

    def test_missing_checkpoint_is_not_seeded_and_records_nothing(self):
        """A missing checkpoint is never created implicitly.

        Even with recordable targets sitting in the admin DB, start() must leave
        the key absent and record nothing — so the following scans are no-ops
        until the key is seeded explicitly. Seeding it from the newest successful
        target would pick a starting point for the operator; that choice belongs
        to whoever seeds the key.
        """
        self._targets = [_target("b1", "t1", _BASE)]
        watcher, store = self._make_watcher()

        watcher._verify_checkpoint(self.storage)

        assert store.calls == []
        assert _watermark(self.storage) is None

    def test_checkpoint_without_build_id_skips_verification_without_raising(self):
        """A checkpoint missing ``build_id`` skips the sweep instead of failing.

        Regression: ``build_id`` was indexed directly, so a malformed
        ``gb_kv_pairs`` row raised ``KeyError`` out of ``_verify_checkpoint`` —
        and ``start()`` guards only per-target recording errors, so the watcher
        would not start at all. Skipping the start-up sweep is strictly better:
        steady-state scans still run off the watermark.
        """
        self._targets = [_target("b1", "t1", _BASE)]
        watcher, store = self._make_watcher()
        self.storage.kv_pair_storage.set_value(
            LINEAGE_WATCHER_CHECKPOINT_KEY, {"finished_at": _BASE.isoformat()}
        )

        # Must not raise; the sweep is skipped rather than half-run.
        watcher._verify_checkpoint(self.storage)

        assert store.calls == []

    def test_verify_checkpoint_already_recorded_is_a_no_op(self):
        self._targets = [_target("b1", "t1", _BASE)]
        watcher, store = self._make_watcher()
        _seed(self.storage, "b1", _BASE)
        store.calls.append(("b1", "t1"))  # Pre-record it in the stub store.
        store._recorded.add("t1")

        watcher._verify_checkpoint(self.storage)

        # Already recorded: filter_unrecorded excludes it, no duplicate call.
        assert store.calls == [("b1", "t1")]
        assert _watermark(self.storage) == _BASE

    def test_verify_checkpoint_reports_done_so_the_sweep_runs_once(self):
        """The success path must return True, not fall off the end as None.

        ``_reconcile`` gates the sweep on ``if not self._checkpoint_verified``
        and stores this return value there. A None return is falsy, so the
        build-scoped scan would re-run on every iteration of the monitoring
        loop: a wasted sink round-trip per scan forever, and a log line that
        repeats the checkpoint's own target endlessly, reading like a watcher
        wedged on one target while the steady-state scan quietly makes progress
        past it.
        """
        self._targets = [_target("b1", "t1", _BASE)]
        watcher, store = self._make_watcher()
        _seed(self.storage, "b1", _BASE)

        assert watcher._verify_checkpoint(self.storage) is True

    def test_failed_verification_leaves_the_checkpoint_intact(self):
        """A verification failure at start() must not disturb the checkpoint.

        The checkpoint is already durable, so a target that fails to re-record
        here is re-surfaced by the next scan (its finished_at sits at the
        watermark, inside the overlap window). Clearing or rewinding the key
        instead would either re-drive lineage that is already recorded or, worse,
        lose the resume point entirely. The failure is logged and swallowed.
        """
        self._targets = [_target("b1", "t1", _BASE)]
        watcher, store = self._make_watcher(fail={"t1"})
        checkpoint = {"build_id": "b1", "finished_at": _BASE.isoformat()}
        self.storage.kv_pair_storage.set_value(
            LINEAGE_WATCHER_CHECKPOINT_KEY, checkpoint
        )

        watcher._verify_checkpoint(self.storage)

        assert store.calls == []
        # Unchanged on disk, and the watermark still loads, so scans continue.
        assert (
            self.storage.kv_pair_storage.get_value(LINEAGE_WATCHER_CHECKPOINT_KEY)
            == checkpoint
        )
        assert _watermark(self.storage) == _BASE

        # The next scan retries it now that recording works. get_admin_storage is
        # patched here because this class's fixture (unlike TestLineageWatcher's)
        # does not: its other tests call _verify_checkpoint, which takes storage as
        # an argument, while _reconcile looks it up itself.
        store._fail = set()
        with patch(
            "gbserver.lineage.lineage_watcher.get_admin_storage",
            return_value=self.storage,
        ):
            watcher._reconcile()
        assert ("b1", "t1") in store.calls

    def test_verification_is_scoped_to_the_checkpoint_build(self):
        """Start-time verification only records the checkpoint's own build."""
        self._targets = [
            _target("b1", "t1", _BASE),
            _target("b2", "t2", _BASE + timedelta(seconds=5)),
        ]
        watcher, store = self._make_watcher()
        self.storage.kv_pair_storage.set_value(
            LINEAGE_WATCHER_CHECKPOINT_KEY,
            {"build_id": "b1", "finished_at": _BASE.isoformat()},
        )

        watcher._verify_checkpoint(self.storage)

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
        self.storage.kv_pair_storage.set_value(
            LINEAGE_WATCHER_CHECKPOINT_KEY,
            {"build_id": "b1", "finished_at": _BASE.isoformat()},
        )

        watcher._verify_checkpoint(self.storage)

        assert ("b1", "t1") in store.calls
        assert _watermark(self.storage) == _BASE

    def test_verify_checkpoint_not_actually_recorded_gets_recorded(self):
        self._targets = [_target("b1", "t1", _BASE)]
        watcher, store = self._make_watcher()
        # Checkpoint says t1 is done, but the store never actually recorded it
        # (e.g. a crash between recording and persisting the checkpoint).
        self.storage.kv_pair_storage.set_value(
            LINEAGE_WATCHER_CHECKPOINT_KEY,
            {"build_id": "b1", "finished_at": _BASE.isoformat()},
        )

        watcher._verify_checkpoint(self.storage)

        assert ("b1", "t1") in store.calls
        assert _watermark(self.storage) == _BASE
