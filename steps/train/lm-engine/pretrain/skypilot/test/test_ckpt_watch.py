# SPDX-License-Identifier: Apache-2.0

"""Tests for ``src/ckpt_watch.py``, the checkpoint rung watcher.

No test sleeps: ``watch()`` takes its ``sleep`` callable, and every case here
passes a fake one that both advances the fake world (writes the next checkpoint,
touches the stop file) and caps the loop, so a watcher that fails to terminate
fails the test instead of hanging the suite.
"""

from __future__ import annotations

import io
import json
import pathlib
import subprocess
import sys

import ckpt_watch
import pytest

# ── fake train_dir helpers ──────────────────────────────────────────────────────


def write_marker(train_dir: pathlib.Path, value) -> None:
    """Write the marker file the way lm-engine does: json.dump of one key."""
    (train_dir / ckpt_watch.MARKER).write_text(
        json.dumps({"latest_checkpointed_iteration": value}, indent=4)
    )


def write_raw(train_dir: pathlib.Path, text: str) -> None:
    (train_dir / ckpt_watch.MARKER).write_text(text)


def make_rung(train_dir: pathlib.Path, rung: int) -> pathlib.Path:
    path = train_dir / f"global_step{rung}"
    path.mkdir()
    return path


class FakeSleep:
    """A ``sleep`` that runs scripted side effects instead of waiting.

    ``steps`` is a list of callables, one per sleep. Each is called in turn; when
    they run out the watcher is considered stuck and the test fails, so no case
    can loop forever.
    """

    def __init__(self, *steps):
        self.steps = list(steps)
        self.calls: list[float] = []

    def __call__(self, interval):
        self.calls.append(interval)
        if not self.steps:
            raise AssertionError(
                f"watch() slept {len(self.calls)} times without returning"
            )
        self.steps.pop(0)()


def run_watch(train_dir, rungs, *, stop_file=None, interval=60, sleep=None):
    """Call ``watch()`` against a ``StringIO`` and return (emitted, lines)."""
    out = io.StringIO()
    if stop_file is None:
        stop_file = train_dir / "never-created.stop"
    emitted = ckpt_watch.watch(
        train_dir,
        rungs,
        interval,
        stop_file,
        out=out,
        sleep=sleep if sleep is not None else FakeSleep(),
    )
    return emitted, out.getvalue().splitlines()


# ── latest() ────────────────────────────────────────────────────────────────────


def test_latest_absent_marker_is_none(tmp_path):
    assert ckpt_watch.latest(tmp_path) is None


def test_latest_reads_the_iteration(tmp_path):
    write_marker(tmp_path, 400)
    assert ckpt_watch.latest(tmp_path) == 400


def test_latest_empty_file_is_none(tmp_path):
    write_raw(tmp_path, "")
    assert ckpt_watch.latest(tmp_path) is None


def test_latest_partial_write_is_none(tmp_path):
    # The real write is open("w") + json.dump, so a poll can land mid-write.
    write_raw(tmp_path, '{\n    "latest_checkpointed_i')
    assert ckpt_watch.latest(tmp_path) is None


def test_latest_empty_object_is_none(tmp_path):
    write_raw(tmp_path, "{}")
    assert ckpt_watch.latest(tmp_path) is None


def test_latest_valid_json_but_not_a_dict_is_none(tmp_path):
    for text in ("400", "null", "[400]", '"400"'):
        write_raw(tmp_path, text)
        assert ckpt_watch.latest(tmp_path) is None, text


def test_latest_accepts_a_numeric_string_value(tmp_path):
    write_marker(tmp_path, "400")
    assert ckpt_watch.latest(tmp_path) == 400


def test_latest_accepts_a_float_value(tmp_path):
    write_marker(tmp_path, 400.0)
    assert ckpt_watch.latest(tmp_path) == 400


def test_latest_non_numeric_value_is_none(tmp_path):
    for value in ("abc", None, [400], {"a": 1}):
        write_marker(tmp_path, value)
        assert ckpt_watch.latest(tmp_path) is None, value


def test_latest_does_not_swallow_a_broken_marker(tmp_path):
    # A directory where the marker should be is not "mid-write", it is broken:
    # it must raise rather than be polled forever as "not yet".
    (tmp_path / ckpt_watch.MARKER).mkdir()
    with pytest.raises(OSError):
        ckpt_watch.latest(tmp_path)


# ── complete_rungs() ────────────────────────────────────────────────────────────


def test_complete_rungs_no_marker_is_empty(tmp_path):
    make_rung(tmp_path, 2)
    assert ckpt_watch.complete_rungs(tmp_path, [2]) == []


def test_complete_rungs_marker_and_dirs(tmp_path):
    write_marker(tmp_path, 4)
    make_rung(tmp_path, 2)
    make_rung(tmp_path, 4)
    assert ckpt_watch.complete_rungs(tmp_path, [2, 4]) == [2, 4]


def test_complete_rungs_ahead_of_the_marker_is_withheld(tmp_path):
    # Rung 6 is named by the ladder but training has only reached 4.
    write_marker(tmp_path, 4)
    make_rung(tmp_path, 2)
    make_rung(tmp_path, 4)
    assert ckpt_watch.complete_rungs(tmp_path, [2, 4, 6]) == [2, 4]


def test_complete_rungs_withholds_a_dir_the_marker_has_not_confirmed(tmp_path):
    # save_checkpoint() mkdirs global_step6/ before it writes anything into it,
    # so the directory alone means nothing; the marker still says 4.
    write_marker(tmp_path, 4)
    make_rung(tmp_path, 4)
    make_rung(tmp_path, 6)
    assert ckpt_watch.complete_rungs(tmp_path, [4, 6]) == [4]


def test_complete_rungs_withholds_a_rung_with_no_dir(tmp_path):
    write_marker(tmp_path, 4)
    make_rung(tmp_path, 4)
    assert ckpt_watch.complete_rungs(tmp_path, [2, 4]) == [4]


def test_complete_rungs_ignores_a_file_named_like_a_rung(tmp_path):
    write_marker(tmp_path, 4)
    (tmp_path / "global_step4").write_text("not a directory")
    assert ckpt_watch.complete_rungs(tmp_path, [4]) == []


def test_complete_rungs_is_ascending_whatever_the_ladder_order(tmp_path):
    write_marker(tmp_path, 10)
    for rung in (2, 4, 10):
        make_rung(tmp_path, rung)
    assert ckpt_watch.complete_rungs(tmp_path, [10, 2, 4]) == [2, 4, 10]


def test_complete_rungs_empty_ladder_is_empty(tmp_path):
    write_marker(tmp_path, 4)
    make_rung(tmp_path, 4)
    assert ckpt_watch.complete_rungs(tmp_path, []) == []


# ── line() ──────────────────────────────────────────────────────────────────────


def test_line_is_the_artifact_marker_gbserver_scrapes(tmp_path):
    assert ckpt_watch.line(tmp_path, 4) == (
        f"GB_ARTIFACT_ID:checkpoint_4 GB_ARTIFACT_PATH:{tmp_path}/global_step4"
    )


# ── watch() ─────────────────────────────────────────────────────────────────────


def test_watch_emits_each_rung_once_in_order(tmp_path):
    write_marker(tmp_path, 4)
    make_rung(tmp_path, 2)
    make_rung(tmp_path, 4)
    emitted, lines = run_watch(tmp_path, [2, 4])
    assert emitted == {2, 4}
    assert lines == [ckpt_watch.line(tmp_path, 2), ckpt_watch.line(tmp_path, 4)]


def test_watch_returns_at_once_when_every_rung_is_already_there(tmp_path):
    write_marker(tmp_path, 4)
    make_rung(tmp_path, 2)
    make_rung(tmp_path, 4)
    sleep = FakeSleep()  # any sleep at all fails the test
    run_watch(tmp_path, [2, 4], sleep=sleep)
    assert sleep.calls == []


def test_watch_emits_rungs_as_they_land_across_polls(tmp_path):
    write_marker(tmp_path, 2)
    make_rung(tmp_path, 2)

    def land_4():
        make_rung(tmp_path, 4)
        write_marker(tmp_path, 4)

    sleep = FakeSleep(land_4)
    emitted, lines = run_watch(tmp_path, [2, 4], interval=30, sleep=sleep)
    assert emitted == {2, 4}
    assert lines == [ckpt_watch.line(tmp_path, 2), ckpt_watch.line(tmp_path, 4)]
    assert sleep.calls == [30]  # the interval it was given, once


def test_watch_never_emits_a_non_rung_checkpoint(tmp_path):
    # SAVE_INTERVAL writes more checkpoints than the ladder names.
    write_marker(tmp_path, 4)
    for rung in (2, 3, 4):
        make_rung(tmp_path, rung)
    emitted, lines = run_watch(tmp_path, [2, 4])
    assert emitted == {2, 4}
    assert all("global_step3" not in one for one in lines)


def test_watch_does_not_emit_an_unconfirmed_rung(tmp_path):
    write_marker(tmp_path, 4)
    make_rung(tmp_path, 4)
    make_rung(tmp_path, 6)  # mkdir'd, nothing saved into it yet
    stop = tmp_path / "stop"

    sleep = FakeSleep(stop.touch)
    emitted, lines = run_watch(tmp_path, [4, 6], stop_file=stop, sleep=sleep)
    assert emitted == {4}
    assert lines == [ckpt_watch.line(tmp_path, 4)]


def test_watch_re_emits_both_rungs_on_a_restart(tmp_path):
    # A build-level retry starts a new watcher against the same train_dir;
    # gbserver only re-dispatches a downstream export if the binding is
    # delivered again, so every complete rung must be emitted afresh.
    write_marker(tmp_path, 4)
    make_rung(tmp_path, 2)
    make_rung(tmp_path, 4)
    first_emitted, first_lines = run_watch(tmp_path, [2, 4])
    second_emitted, second_lines = run_watch(tmp_path, [2, 4])
    assert first_emitted == second_emitted == {2, 4}
    assert first_lines == second_lines


def test_watch_stops_on_the_stop_file_after_a_final_sweep(tmp_path):
    write_marker(tmp_path, 2)
    make_rung(tmp_path, 2)
    stop = tmp_path / "stop"

    def finish():
        # Training wrote its last rung and exited between two polls; the stop
        # file and the checkpoint appear in the same gap.
        make_rung(tmp_path, 4)
        write_marker(tmp_path, 4)
        stop.touch()

    sleep = FakeSleep(finish)
    emitted, lines = run_watch(tmp_path, [2, 4, 6], stop_file=stop, sleep=sleep)
    assert emitted == {2, 4}
    assert lines == [ckpt_watch.line(tmp_path, 2), ckpt_watch.line(tmp_path, 4)]


def test_watch_stop_file_present_from_the_start_still_sweeps(tmp_path):
    write_marker(tmp_path, 4)
    make_rung(tmp_path, 4)
    stop = tmp_path / "stop"
    stop.touch()
    sleep = FakeSleep()
    emitted, lines = run_watch(tmp_path, [4, 6], stop_file=stop, sleep=sleep)
    assert emitted == {4}
    assert lines == [ckpt_watch.line(tmp_path, 4)]
    assert sleep.calls == []


def test_watch_empty_ladder_returns_immediately(tmp_path):
    sleep = FakeSleep()
    emitted, lines = run_watch(tmp_path, [], sleep=sleep)
    assert emitted == set()
    assert lines == []
    assert sleep.calls == []


def test_watch_tolerates_a_duplicated_rung(tmp_path):
    write_marker(tmp_path, 4)
    make_rung(tmp_path, 4)
    emitted, lines = run_watch(tmp_path, [4, 4])
    assert emitted == {4}
    assert lines == [ckpt_watch.line(tmp_path, 4)]


def test_watch_rides_out_a_partial_marker(tmp_path):
    write_raw(tmp_path, "{")
    make_rung(tmp_path, 2)

    def complete_the_write():
        write_marker(tmp_path, 2)

    sleep = FakeSleep(complete_the_write)
    emitted, lines = run_watch(tmp_path, [2], sleep=sleep)
    assert emitted == {2}
    assert lines == [ckpt_watch.line(tmp_path, 2)]


# ── parse_rungs() ───────────────────────────────────────────────────────────────


def test_parse_rungs_sorted_and_deduped():
    assert ckpt_watch.parse_rungs("10,2,4,2") == [2, 4, 10]


def test_parse_rungs_tolerates_spaces_and_trailing_commas():
    assert ckpt_watch.parse_rungs(" 2 , 4 ,") == [2, 4]


def test_parse_rungs_empty_is_empty():
    assert ckpt_watch.parse_rungs("") == []
    assert ckpt_watch.parse_rungs("   ") == []


@pytest.mark.parametrize("text", ["abc", "2,x", "2.5", "-4", "0", "2;4"])
def test_parse_rungs_rejects_malformed(text):
    with pytest.raises(ValueError):
        ckpt_watch.parse_rungs(text)


# ── CLI ─────────────────────────────────────────────────────────────────────────


def test_main_emits_and_exits_zero(tmp_path, capsys):
    write_marker(tmp_path, 4)
    make_rung(tmp_path, 2)
    make_rung(tmp_path, 4)
    code = ckpt_watch.main(
        [
            "--train-dir",
            str(tmp_path),
            "--rungs",
            "2,4",
            "--interval",
            "60",
            "--stop-file",
            str(tmp_path / "stop"),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out.splitlines()
    assert out == [ckpt_watch.line(tmp_path, 2), ckpt_watch.line(tmp_path, 4)]


def test_main_empty_rungs_says_so_and_exits_zero(tmp_path, capsys):
    # A ladder that parses to nothing emits nothing, which would otherwise look
    # exactly like a watcher that is still waiting. Say it on stderr.
    code = ckpt_watch.main(
        [
            "--train-dir",
            str(tmp_path),
            "--rungs",
            "",
            "--stop-file",
            str(tmp_path / "stop"),
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "no checkpoint rungs" in captured.err


def test_main_rejects_malformed_rungs(tmp_path, capsys):
    code = ckpt_watch.main(
        [
            "--train-dir",
            str(tmp_path),
            "--rungs",
            "2,nope",
            "--stop-file",
            str(tmp_path / "stop"),
        ]
    )
    assert code == 2
    assert "nope" in capsys.readouterr().err


def test_main_rejects_a_missing_train_dir(tmp_path, capsys):
    code = ckpt_watch.main(
        [
            "--train-dir",
            str(tmp_path / "absent"),
            "--rungs",
            "2",
            "--stop-file",
            str(tmp_path / "stop"),
        ]
    )
    assert code == 2
    assert "train-dir" in capsys.readouterr().err


def test_main_interval_defaults(tmp_path):
    parser = ckpt_watch.build_parser()
    args = parser.parse_args(
        ["--train-dir", str(tmp_path), "--rungs", "2", "--stop-file", "s"]
    )
    assert args.interval == 60


def test_module_runs_standalone_without_torch_or_lm_engine(tmp_path):
    # It is started by run_pretrain.sh as a plain python helper, before (and
    # outside) the trainer's environment: stdlib imports only.
    src = pathlib.Path(__file__).resolve().parent.parent / "src"
    write_marker(tmp_path, 2)
    make_rung(tmp_path, 2)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(src / "ckpt_watch.py"),
            "--train-dir",
            str(tmp_path),
            "--rungs",
            "2",
            "--stop-file",
            str(tmp_path / "stop"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [ckpt_watch.line(tmp_path, 2)]
