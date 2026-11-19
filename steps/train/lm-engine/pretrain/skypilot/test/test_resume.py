# SPDX-License-Identifier: Apache-2.0

"""Tests for ``resume.py`` -- the three resume situations and the progress guard."""

import json
import pathlib
import subprocess
import sys

import pytest
import resume

MARKER = "latest_checkpointed_iteration.json"


# ── fixtures on disk ────────────────────────────────────────────────────────────


def _rung(root: pathlib.Path, iteration: int, *, ranks: int | None = 4) -> pathlib.Path:
    """A complete-looking ``global_step<N>`` with ``ranks`` rng_state shards."""
    rung = root / f"global_step{iteration}"
    rung.mkdir(parents=True, exist_ok=True)
    (rung / "metadata.json").write_text(
        json.dumps({"accelerator": "cuda", "consumed_samples": iteration * 8})
    )
    if ranks is not None:
        rng = rung / "rng_state"
        rng.mkdir(exist_ok=True)
        for rank in range(ranks):
            (rng / f"rng_state-{rank}.pt").write_bytes(b"x")
    return rung


def _mark(root: pathlib.Path, iteration: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / MARKER).write_text(json.dumps({"latest_checkpointed_iteration": iteration}))


def _saved(root: pathlib.Path, iteration: int, **kwargs) -> pathlib.Path:
    rung = _rung(root, iteration, **kwargs)
    _mark(root, iteration)
    return rung


def _decide(train_dir, *, resume_src="", world_size=4, require_progress=False, state):
    return resume.decide(
        pathlib.Path(train_dir),
        resume_src=str(resume_src),
        world_size=world_size,
        require_progress=require_progress,
        state_file=pathlib.Path(state),
    )


# ── the three situations ────────────────────────────────────────────────────────


def test_fresh(tmp_path):
    train = tmp_path / "train"
    train.mkdir()
    decision = _decide(train, state=tmp_path / "state.json")
    assert decision == {"load": None, "seeded": False, "error": None}


def test_fresh_when_train_dir_absent(tmp_path):
    decision = _decide(tmp_path / "nope", state=tmp_path / "state.json")
    assert decision["load"] is None and decision["error"] is None


def test_resume_in_place(tmp_path):
    train = tmp_path / "train"
    _saved(train, 200)
    decision = _decide(train, state=tmp_path / "state.json")
    assert decision["seeded"] is False and decision["error"] is None
    load = decision["load"]
    assert load["load_path"] == str(train)
    assert load["iteration"] is None
    assert load["load_optimizer"] is True
    assert load["load_lr_scheduler"] is True
    assert load["load_starting_iteration"] is True
    assert load["load_dataloader_state"] is False


def test_load_keys_are_exactly_what_render_config_reads(tmp_path):
    train = tmp_path / "train"
    _saved(train, 10)
    load = _decide(train, state=tmp_path / "state.json")["load"]
    assert set(load) == {
        "load_path",
        "iteration",
        "load_optimizer",
        "load_lr_scheduler",
        "load_starting_iteration",
        "load_dataloader_state",
        "load_rng_state",
    }


def test_seed_from_resume_src(tmp_path):
    src = tmp_path / "other" / "train"
    _saved(src, 100)
    _rung(src, 50)
    train = tmp_path / "train"
    decision = _decide(train, resume_src=src, state=tmp_path / "state.json")
    assert decision["seeded"] is True and decision["error"] is None
    assert decision["load"]["load_path"] == str(train)
    for iteration in (50, 100):
        link = train / f"global_step{iteration}"
        assert link.is_symlink()
        assert pathlib.Path(str(link.readlink())).is_absolute()
        assert link.resolve() == (src / f"global_step{iteration}").resolve()
    copied = train / MARKER
    assert copied.is_file() and not copied.is_symlink()
    assert json.loads(copied.read_text())["latest_checkpointed_iteration"] == 100


def test_seed_creates_train_dir(tmp_path):
    src = tmp_path / "src"
    _saved(src, 7)
    train = tmp_path / "deep" / "train"
    decision = _decide(train, resume_src=src, state=tmp_path / "state.json")
    assert decision["seeded"] is True and train.is_dir()


def test_seed_is_idempotent(tmp_path):
    src = tmp_path / "src"
    _saved(src, 7)
    train = tmp_path / "train"
    first = _decide(train, resume_src=src, state=tmp_path / "state.json")
    second = _decide(train, resume_src=src, state=tmp_path / "state.json")
    assert first["seeded"] is True
    # The second attempt finds train_dir already complete: resume in place.
    assert second["seeded"] is False and second["error"] is None
    assert second["load"]["load_path"] == str(train)


def test_seed_skips_incomplete_rungs(tmp_path):
    """A rung past the source marker is half-written; it must not be linked."""
    src = tmp_path / "src"
    _saved(src, 100)
    _rung(src, 150)
    train = tmp_path / "train"
    _decide(train, resume_src=src, state=tmp_path / "state.json")
    assert (train / "global_step100").is_symlink()
    assert not (train / "global_step150").exists()


def test_train_dir_wins_over_resume_src(tmp_path):
    src = tmp_path / "src"
    _saved(src, 900)
    train = tmp_path / "train"
    _saved(train, 10)
    decision = _decide(train, resume_src=src, state=tmp_path / "state.json")
    assert decision["seeded"] is False
    assert not (train / "global_step900").exists()


def test_resume_src_without_marker_is_fresh(tmp_path):
    src = tmp_path / "src"
    _rung(src, 100)  # directory only, no marker: never completed
    train = tmp_path / "train"
    decision = _decide(train, resume_src=src, state=tmp_path / "state.json")
    assert decision == {"load": None, "seeded": False, "error": None}
    assert not (train / "global_step100").exists()


def test_resume_src_nonexistent_is_fresh(tmp_path):
    decision = _decide(
        tmp_path / "train", resume_src=tmp_path / "gone", state=tmp_path / "state.json"
    )
    assert decision == {"load": None, "seeded": False, "error": None}


def test_resume_src_equal_to_train_dir(tmp_path):
    train = tmp_path / "train"
    _saved(train, 42)
    decision = _decide(train, resume_src=train, state=tmp_path / "state.json")
    assert decision["seeded"] is False
    assert decision["load"]["load_path"] == str(train)
    assert (train / "global_step42").is_dir() and not (
        train / "global_step42"
    ).is_symlink()


def test_resume_src_equal_to_train_dir_by_symlink(tmp_path):
    train = tmp_path / "train"
    _saved(train, 42)
    alias = tmp_path / "alias"
    alias.symlink_to(train)
    decision = _decide(train, resume_src=alias, state=tmp_path / "state.json")
    assert decision["seeded"] is False


def test_resume_src_empty_string(tmp_path):
    train = tmp_path / "train"
    _saved(train, 5)
    decision = _decide(train, resume_src="", state=tmp_path / "state.json")
    assert decision["seeded"] is False and decision["load"] is not None


# ── load_rng_state ─────────────────────────────────────────────────────────────


def test_rng_true_when_world_size_matches(tmp_path):
    train = tmp_path / "train"
    _saved(train, 10, ranks=8)
    decision = _decide(train, world_size=8, state=tmp_path / "state.json")
    assert decision["load"]["load_rng_state"] is True


def test_rng_false_when_world_size_differs(tmp_path):
    train = tmp_path / "train"
    _saved(train, 10, ranks=8)
    decision = _decide(train, world_size=16, state=tmp_path / "state.json")
    assert decision["load"]["load_rng_state"] is False


def test_rng_false_when_unknown(tmp_path):
    train = tmp_path / "train"
    _saved(train, 10, ranks=None)  # no rng_state/ at all
    decision = _decide(train, world_size=8, state=tmp_path / "state.json")
    assert decision["load"]["load_rng_state"] is False


def test_rng_false_when_shard_names_unrecognised(tmp_path):
    train = tmp_path / "train"
    rung = _saved(train, 10, ranks=4)
    (rung / "rng_state" / "rng_state-extra.pt").write_bytes(b"x")
    decision = _decide(train, world_size=4, state=tmp_path / "state.json")
    assert decision["load"]["load_rng_state"] is False


def test_rng_false_when_ranks_not_contiguous(tmp_path):
    train = tmp_path / "train"
    rung = _saved(train, 10, ranks=4)
    (rung / "rng_state" / "rng_state-3.pt").unlink()
    decision = _decide(train, world_size=4, state=tmp_path / "state.json")
    assert decision["load"]["load_rng_state"] is False


def test_rng_read_through_seeded_symlink(tmp_path):
    src = tmp_path / "src"
    _saved(src, 10, ranks=4)
    train = tmp_path / "train"
    decision = _decide(
        train, resume_src=src, world_size=4, state=tmp_path / "state.json"
    )
    assert decision["load"]["load_rng_state"] is True


def test_rng_false_when_rung_dir_missing(tmp_path):
    """Marker says 10 but the rung is gone: unknown shape, so no rng state."""
    train = tmp_path / "train"
    _mark(train, 10)
    decision = _decide(train, world_size=4, state=tmp_path / "state.json")
    assert decision["load"]["load_rng_state"] is False


# ── require_progress ───────────────────────────────────────────────────────────


def test_require_progress_errors_on_equal_iteration(tmp_path):
    train = tmp_path / "train"
    _saved(train, 300)
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"iteration": 300}))
    decision = _decide(train, require_progress=True, state=state)
    assert decision["error"] == "no progress since last attempt at step 300"


def test_require_progress_errors_on_regression(tmp_path):
    train = tmp_path / "train"
    _saved(train, 200)
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"iteration": 300}))
    decision = _decide(train, require_progress=True, state=state)
    assert decision["error"] == "no progress since last attempt at step 300"


def test_require_progress_errors_when_checkpoints_vanished(tmp_path):
    train = tmp_path / "train"
    train.mkdir()
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"iteration": 300}))
    decision = _decide(train, require_progress=True, state=state)
    assert decision["error"] == "no progress since last attempt at step 300"
    assert decision["load"] is None


def test_require_progress_ok_on_greater(tmp_path):
    train = tmp_path / "train"
    _saved(train, 400)
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"iteration": 300}))
    decision = _decide(train, require_progress=True, state=state)
    assert decision["error"] is None and decision["load"] is not None


def test_require_progress_ok_on_first_attempt(tmp_path):
    train = tmp_path / "train"
    _saved(train, 300)
    decision = _decide(train, require_progress=True, state=tmp_path / "state.json")
    assert decision["error"] is None and decision["load"] is not None


def test_require_progress_ok_on_first_attempt_fresh(tmp_path):
    decision = _decide(
        tmp_path / "train", require_progress=True, state=tmp_path / "state.json"
    )
    assert decision["error"] is None and decision["load"] is None


def test_progress_not_checked_when_off(tmp_path):
    train = tmp_path / "train"
    _saved(train, 300)
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"iteration": 300}))
    decision = _decide(train, require_progress=False, state=state)
    assert decision["error"] is None and decision["load"] is not None


@pytest.mark.parametrize(
    "body", ["", "   ", "{", "null", "[]", '{"iteration": "soon"}', '{"other": 1}', "7"]
)
def test_corrupt_state_file_does_not_block_resume(tmp_path, body):
    train = tmp_path / "train"
    _saved(train, 300)
    state = tmp_path / "state.json"
    state.write_text(body)
    decision = _decide(train, require_progress=True, state=state)
    assert decision["error"] is None and decision["load"] is not None


# ── the state file is always rewritten ─────────────────────────────────────────


def _state(path):
    return json.loads(pathlib.Path(path).read_text())["iteration"]


def test_state_file_rewritten_on_resume(tmp_path):
    train = tmp_path / "train"
    _saved(train, 300)
    state = tmp_path / "state.json"
    _decide(train, state=state)
    assert _state(state) == 300


def test_state_file_rewritten_on_fresh(tmp_path):
    state = tmp_path / "state.json"
    _decide(tmp_path / "train", state=state)
    assert _state(state) is None


def test_state_file_rewritten_on_error(tmp_path):
    train = tmp_path / "train"
    _saved(train, 300)
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"iteration": 300}))
    decision = _decide(train, require_progress=True, state=state)
    assert decision["error"]
    assert _state(state) == 300


def test_state_file_rewritten_after_seeding(tmp_path):
    src = tmp_path / "src"
    _saved(src, 120)
    state = tmp_path / "state.json"
    _decide(tmp_path / "train", resume_src=src, state=state)
    assert _state(state) == 120


def test_state_file_parent_created(tmp_path):
    train = tmp_path / "train"
    _saved(train, 1)
    state = tmp_path / "deep" / "nested" / "state.json"
    _decide(train, state=state)
    assert _state(state) == 1


def test_state_file_write_leaves_no_temp_files(tmp_path):
    train = tmp_path / "train"
    _saved(train, 1)
    state = tmp_path / "state" / "attempt.json"
    _decide(train, state=state)
    assert [p.name for p in state.parent.iterdir()] == ["attempt.json"]


# ── CLI ────────────────────────────────────────────────────────────────────────

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "resume.py"


def _cli(*args):
    return subprocess.run(
        [sys.executable, "-I", str(_SRC), *args], capture_output=True, text=True
    )


def test_cli_help_runs_on_stdlib_alone(tmp_path):
    """``python -I`` ignores PYTHONPATH and the user site: stdlib imports only."""
    done = _cli("--help")
    assert done.returncode == 0, done.stderr
    assert "--require-progress" in done.stdout


def _main(train, out, state, *extra):
    return resume.main(
        [
            "--train-dir",
            str(train),
            "--world-size",
            "4",
            "--state-file",
            str(state),
            "--out",
            str(out),
            *extra,
        ]
    )


def test_cli_writes_decision_fresh(tmp_path):
    out = tmp_path / "decision.json"
    code = _main(tmp_path / "train", out, tmp_path / "state.json")
    assert code == 0
    assert json.loads(out.read_text()) == {
        "load": None,
        "seeded": False,
        "error": None,
    }


def test_cli_writes_decision_on_resume(tmp_path):
    train = tmp_path / "train"
    _saved(train, 10, ranks=4)
    out = tmp_path / "decision.json"
    assert _main(train, out, tmp_path / "state.json") == 0
    decision = json.loads(out.read_text())
    assert decision["load"]["load_path"] == str(train)
    assert decision["load"]["load_rng_state"] is True


def test_cli_parses_require_progress_true(tmp_path):
    train = tmp_path / "train"
    _saved(train, 10)
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"iteration": 10}))
    out = tmp_path / "decision.json"
    code = _main(train, out, state, "--require-progress", "true")
    assert code == 1
    assert json.loads(out.read_text())["error"]


def test_cli_parses_require_progress_false(tmp_path):
    train = tmp_path / "train"
    _saved(train, 10)
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"iteration": 10}))
    out = tmp_path / "decision.json"
    code = _main(train, out, state, "--require-progress", "false")
    assert code == 0
    assert json.loads(out.read_text())["error"] is None


def test_cli_seeds_from_resume_src(tmp_path):
    src = tmp_path / "src"
    _saved(src, 60)
    train = tmp_path / "train"
    out = tmp_path / "decision.json"
    code = _main(train, out, tmp_path / "state.json", "--resume-src", str(src))
    assert code == 0
    assert json.loads(out.read_text())["seeded"] is True


def test_cli_accepts_empty_resume_src(tmp_path):
    train = tmp_path / "train"
    _saved(train, 60)
    out = tmp_path / "decision.json"
    code = _main(train, out, tmp_path / "state.json", "--resume-src", "")
    assert code == 0


def test_cli_writes_decision_when_train_dir_is_broken(tmp_path):
    """A marker that is a directory is not a race; the ranks must still be told."""
    train = tmp_path / "train"
    train.mkdir()
    (train / MARKER).mkdir()
    out = tmp_path / "decision.json"
    code = _main(train, out, tmp_path / "state.json")
    assert code == 1
    decision = json.loads(out.read_text())
    assert decision["load"] is None and decision["error"]


def test_cli_out_is_written_atomically(tmp_path, monkeypatch):
    """Every rank polls for ``-s $DECISION``; a partial read must be impossible."""
    seen = {}
    real_replace = resume.os.replace

    def spy(src, dst):
        seen["tmp"] = pathlib.Path(src).exists()
        return real_replace(src, dst)

    monkeypatch.setattr(resume.os, "replace", spy)
    train = tmp_path / "train"
    _saved(train, 3)
    out = tmp_path / "decision.json"
    assert _main(train, out, tmp_path / "state.json") == 0
    assert seen["tmp"] is True
    assert json.loads(out.read_text())["load"]


def test_cli_rejects_missing_args(tmp_path):
    done = _cli("--train-dir", str(tmp_path))
    assert done.returncode == 2
