#!/usr/bin/env python3
"""Refuse an attempt that would resume where the previous one did.

With ``train_dir`` set, the checkpoint directory survives a build-level retry, so a
retried run resumes from what the failed attempt wrote. A failure that recurs at the
same step (a bad batch, an OOM at a fixed sequence length, a config the trainer
rejects after loading) would then burn every retry the build has, each one holding a
multi-node allocation until it reaches the same point. This stops the second such
attempt before ``accelerate`` launches.

Every attempt that is let through appends one line to ``<train_dir>/attempts.jsonl``:
its id and the highest complete checkpoint it starts from. The next attempt is
refused (exit 3) unless it starts from a higher one. A ``retry_workload`` relaunch
inside one target run carries the same attempt id and is let through unrecorded: it
is the same attempt, relaunched on other nodes, not a new verdict on the run.

"Complete" is the step's rule: ``model.safetensors`` or
``model.safetensors.index.json``, plus ``tokenizer.json`` and ``trainer_state.json``;
only ``checkpoint-<digits>`` names, compared as numbers. The step moves a half-written
checkpoint aside before calling this, and this skips one anyway.

The step's train_dir block runs this on rank 0 only.
"""

# The annotations below are 3.10 syntax; this keeps an older $PYBIN to the FATAL
# lines instead of a TypeError at import.
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from typing import NoReturn

PREFIX = "logit-distill: progress:"
ATTEMPTS = "attempts.jsonl"
NO_PROGRESS = 3
_CHECKPOINT = re.compile(r"^checkpoint-([0-9]+)$")
_SENTINELS = ("tokenizer.json", "trainer_state.json")
_WEIGHTS = ("model.safetensors", "model.safetensors.index.json")


def fatal(message: str) -> NoReturn:
    """Exit 1 with a FATAL line on stderr, the step's convention for a refusal."""
    sys.exit(f"logit-distill: FATAL: {message}")


def is_complete(path: str) -> bool:
    """Whether the checkpoint directory at path holds every file of a finished save."""
    if not any(os.path.isfile(os.path.join(path, name)) for name in _WEIGHTS):
        return False
    return all(os.path.isfile(os.path.join(path, name)) for name in _SENTINELS)


def top_step(ckpt_dir: str) -> int:
    """The step of the highest complete checkpoint-<N> in ckpt_dir; 0 if none."""
    try:
        names = os.listdir(ckpt_dir)
    except FileNotFoundError:
        return 0
    except OSError as e:
        fatal(f"cannot list {ckpt_dir}: {e}")
    top = 0
    for name in names:
        match = _CHECKPOINT.match(name)
        if match and is_complete(os.path.join(ckpt_dir, name)):
            top = max(top, int(match.group(1)))
    return top


def last_attempt(path: str) -> dict | None:
    """The last record in the attempts file at path; None if there is none yet.

    Exits through fatal() if the file cannot be read or its last line is not a
    record: guessing would either refuse a healthy retry or wave a stuck one on.
    """
    try:
        with open(path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if line.strip()]
    except FileNotFoundError:
        return None
    except OSError as e:
        fatal(f"cannot read {path}: {e}")
    if not lines:
        return None
    try:
        record = json.loads(lines[-1])
    except ValueError as e:
        fatal(f"{path}: last line is not JSON: {e}")
    if not isinstance(record, dict) or "attempt" not in record:
        fatal(f"{path}: last line is not an attempt record: {lines[-1]!r}")
    if not isinstance(record.get("start_step"), int):
        fatal(f"{path}: last line has no integer start_step: {lines[-1]!r}")
    return record


def append_attempt(path: str, attempt: str, start_step: int) -> None:
    """Append this attempt's record to the attempts file at path."""
    record = {
        "attempt": attempt,
        "start_step": start_step,
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError as e:
        fatal(f"cannot append to {path}: {e}")


def main(argv: list[str] | None = None) -> None:
    """Compare this attempt's start against the last one's; record it or refuse."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--attempt-id", required=True)
    args = parser.parse_args(argv)

    path = os.path.join(args.train_dir, ATTEMPTS)
    top = top_step(args.ckpt_dir)
    last = last_attempt(path)

    if last is not None and last["attempt"] == args.attempt_id:
        print(
            f"{PREFIX} attempt {args.attempt_id} relaunched (retry_workload); "
            f"starting from checkpoint-{top}"
        )
        return
    if last is not None and top <= last["start_step"]:
        print(
            f"logit-distill: FATAL: no progress since attempt {last['attempt']} "
            f"(still at checkpoint-{top}); not retrying",
            file=sys.stderr,
        )
        print(
            f"  The failure recurs before the next save; read attempt "
            f"{last['attempt']}'s log, fix the cause, and start a new build "
            f"(or set require_progress: false).",
            file=sys.stderr,
        )
        sys.exit(NO_PROGRESS)

    append_attempt(path, args.attempt_id, top)
    print(f"{PREFIX} attempt {args.attempt_id} starts from checkpoint-{top}")


if __name__ == "__main__":
    main()
