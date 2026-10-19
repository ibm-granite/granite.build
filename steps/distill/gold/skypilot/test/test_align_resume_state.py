"""Tests for the resume-state rewrite (../src/align_resume_state.py).

Run from the step directory with `make test`.

The template tests (test_step_template.py) run it inside the seeding block, which
picks the checkpoint; these pin the script's own rules: what it rewrites, how, and
which values it refuses.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from gold_trainer_state import write_trainer_state

_SCRIPT = Path(__file__).resolve().parent.parent / "src" / "align_resume_state.py"


def _checkpoint(root, save_steps=1000, logging_steps=1):
    """root/checkpoint-2000 holding only a trainer_state.json on the given grid."""
    ckpt = root / "checkpoint-2000"
    ckpt.mkdir(parents=True)
    write_trainer_state(ckpt, 2000, save_steps, logging_steps)
    return ckpt


def _state(ckpt):
    """The checkpoint's trainer_state.json, parsed."""
    return json.loads((ckpt / "trainer_state.json").read_text())


def _run(ckpt, save_steps="250", logging_steps="1"):
    """Invoke the script the way the step's resume block does."""
    return subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--checkpoint",
            str(ckpt),
            "--save-steps",
            str(save_steps),
            "--logging-steps",
            str(logging_steps),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def test_it_rewrites_what_differs_and_logs_it(tmp_path):
    """Both keys take this run's values; nothing else in the state moves."""
    ckpt = _checkpoint(tmp_path)
    result = _run(ckpt, save_steps="250", logging_steps="5")
    assert result.returncode == 0, result.stderr
    state = _state(ckpt)
    assert (state["save_steps"], state["logging_steps"]) == (250, 5)
    assert state["global_step"] == 2000
    assert (
        "checkpoint-2000/trainer_state.json save_steps 1000 -> 250, logging_steps 1 -> 5"
        in result.stdout
    )


def test_the_file_is_replaced_not_edited(tmp_path):
    """The seeded state is a hardlink into the source run; the source keeps its grid."""
    ckpt = _checkpoint(tmp_path / "run")
    source = tmp_path / "source.json"
    source.hardlink_to(ckpt / "trainer_state.json")

    result = _run(ckpt)

    assert result.returncode == 0, result.stderr
    assert _state(ckpt)["save_steps"] == 250
    assert json.loads(source.read_text())["save_steps"] == 1000
    assert not (ckpt / "trainer_state.json.tmp").exists()


def test_a_matching_grid_is_left_alone(tmp_path):
    """Nothing to change: the file keeps its inode, so the hardlink survives."""
    ckpt = _checkpoint(tmp_path, save_steps=250)
    inode = (ckpt / "trainer_state.json").stat().st_ino
    result = _run(ckpt)
    assert result.returncode == 0, result.stderr
    assert (ckpt / "trainer_state.json").stat().st_ino == inode
    assert "already on this run's grid" in result.stdout


def test_a_read_only_checkpoint_dir_is_rewritten_and_keeps_its_mode(tmp_path):
    """cp -a gives an archived (555) source's mode to the seeded copy."""
    ckpt = _checkpoint(tmp_path)
    ckpt.chmod(0o555)
    try:
        result = _run(ckpt)
        assert result.returncode == 0, result.stderr
        assert _state(ckpt)["save_steps"] == 250
        assert ckpt.stat().st_mode & 0o777 == 0o555
    finally:
        ckpt.chmod(0o755)


@pytest.mark.parametrize(
    "flag, value",
    [
        ("save_steps", "${GOLD_SAVE_STEPS}"),
        ("save_steps", "0.1"),  # render_gold_config.py's type=int refuses it too
        ("save_steps", "nan"),
        ("save_steps", "inf"),
        ("logging_steps", "0"),
        ("logging_steps", "-5"),
    ],
)
def test_a_bad_step_count_is_fatal_and_writes_nothing(tmp_path, flag, value):
    """Each one names itself in a FATAL line, never a bare traceback."""
    ckpt = _checkpoint(tmp_path)
    result = _run(ckpt, **{flag: value})
    assert result.returncode == 1
    assert f"distill-gold: FATAL: {flag} must be a positive integer, got {value!r}" in (
        result.stderr
    )
    assert "Traceback" not in result.stderr
    assert _state(ckpt)["save_steps"] == 1000


@pytest.mark.parametrize("content", [None, "{not json", "[1, 2]"])
def test_an_unreadable_state_is_fatal(tmp_path, content):
    """Missing, truncated by a killed save, or not an object: FATAL, no traceback."""
    ckpt = _checkpoint(tmp_path)
    if content is None:
        (ckpt / "trainer_state.json").unlink()
    else:
        (ckpt / "trainer_state.json").write_text(content)
    result = _run(ckpt)
    assert result.returncode == 1
    assert "distill-gold: FATAL:" in result.stderr
    assert "trainer_state.json" in result.stderr
    assert "Traceback" not in result.stderr


def test_a_failed_write_is_fatal_and_leaves_no_tmp(tmp_path):
    """The write error is the one reported, and the .tmp is cleaned up."""
    ckpt = _checkpoint(tmp_path)
    # A directory where the .tmp file would go makes open(..., "w") fail.
    (ckpt / "trainer_state.json.tmp").mkdir()
    result = _run(ckpt)
    assert result.returncode == 1
    assert "cannot rewrite" in result.stderr
    assert "Is a directory" in result.stderr
    assert "Traceback" not in result.stderr
    assert _state(ckpt)["save_steps"] == 1000
