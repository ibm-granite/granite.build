"""Tests for the no-progress refusal (../src/progress_check.py).

Run from the step directory with `make test`.

The template tests (test_step_template.py) run it inside the train_dir block, after
the incomplete-checkpoint move-aside; these pin the script's own rules: what counts
as the highest checkpoint, when an attempt is recorded, and when it refuses: only
after two consecutive attempts made no progress, and every attempt after a refusal.
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
    assert record["status"] == "started"
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


def test_a_single_no_progress_failure_is_allowed(tmp_path):
    """One attempt that dies before its next save (an NCCL hang, a dead node) is
    common and transient: the next attempt is let through and recorded."""
    train, ckpts = _dirs(tmp_path)
    _checkpoint(ckpts, 500)
    assert _run(train, ckpts, "a1").returncode == 0

    result = _run(train, ckpts, "a2")

    assert result.returncode == 0, result.stderr
    assert [(r["attempt"], r["start_step"], r["status"]) for r in _attempts(train)] == [
        ("a1", 500, "started"),
        ("a2", 500, "started"),
    ]


def test_two_consecutive_no_progress_attempts_are_refused(tmp_path):
    train, ckpts = _dirs(tmp_path)
    _checkpoint(ckpts, 500)
    assert _run(train, ckpts, "a1").returncode == 0
    assert _run(train, ckpts, "a2").returncode == 0

    result = _run(train, ckpts, "a3")

    assert result.returncode == 3
    lines = result.stderr.splitlines()
    assert lines[0] == (
        "logit-distill: FATAL: no progress over the last 2 attempts (a1, a2; still "
        "at checkpoint-500); not retrying"
    )
    assert len(lines) == 2 and lines[1].strip(), "expected a one-line hint"
    # The refusal is recorded, so a later attempt is refused without measuring.
    assert [(r["attempt"], r["status"]) for r in _attempts(train)] == [
        ("a1", "started"),
        ("a2", "started"),
        ("a3", "refused"),
    ]
    assert _attempts(train)[-1]["start_step"] == 500


def test_no_progress_from_scratch_is_refused(tmp_path):
    """Two attempts that died before their first save left nothing to resume from."""
    train, ckpts = _dirs(tmp_path)
    assert _run(train, ckpts, "a1").returncode == 0
    assert _run(train, ckpts, "a2").returncode == 0
    result = _run(train, ckpts, "a3")
    assert result.returncode == 3
    assert "still at checkpoint-0" in result.stderr


def test_progress_in_between_resets_the_count(tmp_path):
    """a1 fails without saving, a2 saves 500 and fails, a3 fails at 500: only one
    attempt (a3) made no progress, so a4 is let through; a5 then is not."""
    train, ckpts = _dirs(tmp_path)
    assert _run(train, ckpts, "a1").returncode == 0
    assert _run(train, ckpts, "a2").returncode == 0
    _checkpoint(ckpts, 500)
    assert _run(train, ckpts, "a3").returncode == 0
    result = _run(train, ckpts, "a4")
    assert result.returncode == 0, result.stderr
    result = _run(train, ckpts, "a5")
    assert result.returncode == 3
    assert "(a3, a4; still at checkpoint-500)" in result.stderr


def test_after_a_refusal_every_later_attempt_is_refused_at_once(tmp_path):
    """Even one that would have made progress by the count: the refusal is the
    verdict on this train_dir, and only a new build clears it."""
    train, ckpts = _dirs(tmp_path)
    assert _run(train, ckpts, "a1").returncode == 0
    assert _run(train, ckpts, "a2").returncode == 0
    assert _run(train, ckpts, "a3").returncode == 3
    _checkpoint(ckpts, 500)

    result = _run(train, ckpts, "a4")

    assert result.returncode == 3
    lines = result.stderr.splitlines()
    assert lines[0].startswith("logit-distill: FATAL: attempt a3 was refused")
    assert "not retrying" in lines[0]
    assert [(r["attempt"], r["status"]) for r in _attempts(train)][-2:] == [
        ("a3", "refused"),
        ("a4", "refused"),
    ]


def test_a_half_written_checkpoint_does_not_count(tmp_path):
    train, ckpts = _dirs(tmp_path)
    _checkpoint(ckpts, 500)
    assert _run(train, ckpts, "a1").returncode == 0
    assert _run(train, ckpts, "a2").returncode == 0
    _checkpoint(ckpts, 1000, complete=False)
    result = _run(train, ckpts, "a3")
    assert result.returncode == 3
    assert "still at checkpoint-500" in result.stderr


def test_a_record_without_a_status_counts_as_started(tmp_path):
    """attempts.jsonl written before refusals were recorded carries no status."""
    train, ckpts = _dirs(tmp_path)
    _checkpoint(ckpts, 500)
    (train / "attempts.jsonl").write_text(
        '{"attempt": "a1", "start_step": 500, "time": "t"}\n'
        '{"attempt": "a2", "start_step": 500, "time": "t"}\n'
    )
    result = _run(train, ckpts, "a3")
    assert result.returncode == 3
    assert "(a1, a2;" in result.stderr


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
    assert _run(train, ckpts, "a2").returncode == 0
    for _ in range(3):
        result = _run(train, ckpts, "a2")
        assert result.returncode == 0, result.stderr
        assert "relaunched (retry_workload)" in result.stdout
    assert [r["attempt"] for r in _attempts(train)] == ["a1", "a2"]


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


def test_an_unknown_status_fails_loudly(tmp_path):
    train, ckpts = _dirs(tmp_path)
    (train / "attempts.jsonl").write_text(
        '{"attempt": "a1", "start_step": 0, "status": "odd"}\n'
    )
    result = _run(train, ckpts, "a2")
    assert result.returncode == 1
    assert "logit-distill: FATAL:" in result.stderr


def test_an_unreadable_attempt_record_fails_loudly(tmp_path):
    train, ckpts = _dirs(tmp_path)
    (train / "attempts.jsonl").write_text("not json\n")
    result = _run(train, ckpts, "a1")
    assert result.returncode == 1
    assert "logit-distill: FATAL:" in result.stderr
