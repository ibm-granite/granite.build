from datetime import datetime, timezone

from gbserver.storage.stored_target_run import (
    StoredTargetRun,
    latest_finished_target,
    latest_success_per_target,
)
from gbserver.types.status import Status

_BASE = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def _run(uuid: str, name: str, finished_at) -> StoredTargetRun:
    """A SUCCESS target run with a pinned uuid/name/finished_at."""
    return StoredTargetRun(
        uuid=uuid,
        build_id="b1",
        environment_uri="env://test",
        name=name,
        status=Status.SUCCESS,
        finished_at=finished_at,
    )


class TestLatestFinishedTarget:
    """``latest_finished_target`` picks the newest run from an unordered set."""

    def test_empty_input_is_none(self):
        assert latest_finished_target([]) is None

    def test_single_run_is_returned(self):
        assert latest_finished_target([_run("t1", "targetA", _BASE)]).uuid == "t1"

    def test_newest_wins_regardless_of_order(self):
        # Model get_by_where's undefined order: newest is not first.
        later = _BASE.replace(minute=5)
        picked = latest_finished_target(
            [_run("old", "targetA", _BASE), _run("new", "targetA", later)][::-1]
        )
        assert picked.uuid == "new"

    def test_naive_and_aware_do_not_raise(self):
        naive = datetime(2026, 1, 1, 0, 0, 0)
        aware_later = datetime(2026, 1, 1, 0, 5, 0, tzinfo=timezone.utc)
        picked = latest_finished_target(
            [_run("naive", "targetA", naive), _run("aware", "targetA", aware_later)]
        )
        assert picked.uuid == "aware"


class TestLatestSuccessPerTarget:
    """``latest_success_per_target`` keeps one SUCCESS run per target name."""

    def test_empty_input(self):
        assert latest_success_per_target([]) == []

    def test_single_run_is_kept(self):
        [kept] = latest_success_per_target([_run("t1", "targetA", _BASE)])
        assert kept.uuid == "t1"

    def test_distinct_names_are_all_kept(self):
        runs = [_run("t1", "targetA", _BASE), _run("t2", "targetB", _BASE)]
        assert {r.uuid for r in latest_success_per_target(runs)} == {"t1", "t2"}

    def test_latest_finished_wins_regardless_of_order(self):
        # Feed oldest-last to prove ordering is by finished_at, not list position.
        later = _BASE.replace(minute=5)
        runs = [
            _run("new", "targetA", later),
            _run("old", "targetA", _BASE),
        ]
        [kept] = latest_success_per_target(runs)
        assert kept.uuid == "new"

    def test_naive_and_aware_finished_at_do_not_raise(self):
        # SQLite can read finished_at back offset-naive; mixing it with an aware
        # value must not raise "can't compare offset-naive and offset-aware
        # datetimes", and the later (aware) run must win.
        naive = datetime(2026, 1, 1, 0, 0, 0)
        aware_later = datetime(2026, 1, 1, 0, 5, 0, tzinfo=timezone.utc)
        runs = [
            _run("naive", "targetA", naive),
            _run("aware", "targetA", aware_later),
        ]
        [kept] = latest_success_per_target(runs)
        assert kept.uuid == "aware"

    def test_run_with_finished_at_beats_unset(self):
        runs = [
            _run("no-ts", "targetA", None),
            _run("has-ts", "targetA", _BASE),
        ]
        [kept] = latest_success_per_target(runs)
        assert kept.uuid == "has-ts"

    def test_first_appearance_order_is_preserved(self):
        # A newest-first input stays newest-first: kept names keep their
        # first-seen position even when a later duplicate updates the winner.
        runs = [
            _run("b", "targetB", _BASE),
            _run("a", "targetA", _BASE),
            _run("a2", "targetA", _BASE.replace(minute=5)),
        ]
        assert [r.name for r in latest_success_per_target(runs)] == [
            "targetB",
            "targetA",
        ]
