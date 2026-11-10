"""Tests for the no-progress refusal (../src/progress_check.py).

Run from the step directory with `make test`.

The template tests (test_step_template.py) run it inside the train_dir block, after
the incomplete-checkpoint move-aside; these pin the script's own rules: what counts
as the highest checkpoint, when an attempt is recorded, and when it refuses.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from gold_trainer_state import write_trainer_state

_SCRIPT = Path(__file__).resolve().parent.parent / "src" / "progress_check.py"


def _checkpoint(ckpt_dir, step, complete=True, sharded=False):
    """ckpt_dir/checkpoint-<step>; complete ones carry every completeness sentinel."""
    ckpt = ckpt_dir / f"checkpoint-{step}"
    ckpt.mkdir(parents=True)
    weights = "model.safetensors.index.json" if sharded else "model.safetensors"
    for name in (weights, "tokenizer.json"):
        (ckpt / name).write_text(name)
    if complete:
        write_trainer_state(ckpt, step, 500, 5)
    return ckpt


def _run(train_dir, ckpt_dir, attempt):
    """Invoke the script the way the step's train_dir block does."""
    return subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--train-dir",
            str(train_dir),
            "--ckpt-dir",
            str(ckpt_dir),
            "--attempt-id",
            attempt,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def _attempts(train_dir):
    """attempts.jsonl, parsed line by line."""
    path = train_dir / "attempts.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _dirs(tmp_path):
    train = tmp_path / "train"
    ckpts = train / "checkpoints" / "gold_node2"
    ckpts.mkdir(parents=True)
    return train, ckpts


def test_first_attempt_is_recorded_at_step_zero(tmp_path):
    train, ckpts = _dirs(tmp_path)
    result = _run(train, ckpts, "a1")
    assert result.returncode == 0, result.stderr
    (record,) = _attempts(train)
    assert record["attempt"] == "a1"
    assert record["start_step"] == 0
    # ISO 8601, UTC.
    assert datetime.fromisoformat(record["time"]).utcoffset().total_seconds() == 0


def test_a_missing_checkpoint_dir_is_step_zero(tmp_path):
    train = tmp_path / "train"
    train.mkdir()
    result = _run(train, train / "checkpoints" / "gold_node2", "a1")
    assert result.returncode == 0, result.stderr
    assert _attempts(train)[0]["start_step"] == 0


def test_progress_records_the_new_attempt(tmp_path):
    train, ckpts = _dirs(tmp_path)
    assert _run(train, ckpts, "a1").returncode == 0
    _checkpoint(ckpts, 500)
    result = _run(train, ckpts, "a2")
    assert result.returncode == 0, result.stderr
    assert [(r["attempt"], r["start_step"]) for r in _attempts(train)] == [
        ("a1", 0),
        ("a2", 500),
    ]


def test_no_progress_is_refused(tmp_path):
    train, ckpts = _dirs(tmp_path)
    _checkpoint(ckpts, 500)
    assert _run(train, ckpts, "a1").returncode == 0

    result = _run(train, ckpts, "a2")

    assert result.returncode == 3
    lines = result.stderr.splitlines()
    assert lines[0] == (
        "logit-distill: FATAL: no progress since attempt a1 (still at "
        "checkpoint-500); not retrying"
    )
    assert len(lines) == 2 and lines[1].strip(), "expected a one-line hint"
    # Nothing appended: the refused attempt never started.
    assert [r["attempt"] for r in _attempts(train)] == ["a1"]


def test_no_progress_from_scratch_is_refused(tmp_path):
    """An attempt that died before its first save left nothing to resume from."""
    train, ckpts = _dirs(tmp_path)
    assert _run(train, ckpts, "a1").returncode == 0
    result = _run(train, ckpts, "a2")
    assert result.returncode == 3
    assert "still at checkpoint-0" in result.stderr


def test_a_half_written_checkpoint_does_not_count(tmp_path):
    train, ckpts = _dirs(tmp_path)
    _checkpoint(ckpts, 500)
    assert _run(train, ckpts, "a1").returncode == 0
    _checkpoint(ckpts, 1000, complete=False)
    result = _run(train, ckpts, "a2")
    assert result.returncode == 3
    assert "still at checkpoint-500" in result.stderr


def test_sharded_weights_count_as_complete(tmp_path):
    train, ckpts = _dirs(tmp_path)
    assert _run(train, ckpts, "a1").returncode == 0
    _checkpoint(ckpts, 500, sharded=True)
    assert _run(train, ckpts, "a2").returncode == 0
    assert _attempts(train)[-1]["start_step"] == 500


def test_the_same_attempt_twice_is_a_relaunch_not_a_retry(tmp_path):
    """A retry_workload relaunch inside one target run keeps its attempt id: it is
    let through, even without progress, and recorded once."""
    train, ckpts = _dirs(tmp_path)
    assert _run(train, ckpts, "a1").returncode == 0
    result = _run(train, ckpts, "a1")
    assert result.returncode == 0, result.stderr
    assert len(_attempts(train)) == 1


def test_ordering_is_numeric_not_lexical(tmp_path):
    """Lexically checkpoint-500 sorts after checkpoint-1500."""
    train, ckpts = _dirs(tmp_path)
    _checkpoint(ckpts, 500)
    _checkpoint(ckpts, 1500)
    assert _run(train, ckpts, "a1").returncode == 0
    assert _attempts(train)[0]["start_step"] == 1500


def test_only_hf_checkpoint_names_count(tmp_path):
    train, ckpts = _dirs(tmp_path)
    for name in ("checkpoint-final", "checkpoint-2000-bak", "incomplete-checkpoint-9"):
        _checkpoint(ckpts, 2000).rename(ckpts / name)
    assert _run(train, ckpts, "a1").returncode == 0
    assert _attempts(train)[0]["start_step"] == 0


def test_an_unreadable_attempt_record_fails_loudly(tmp_path):
    train, ckpts = _dirs(tmp_path)
    (train / "attempts.jsonl").write_text("not json\n")
    result = _run(train, ckpts, "a1")
    assert result.returncode == 1
    assert "logit-distill: FATAL:" in result.stderr
