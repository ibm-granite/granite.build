#!/usr/bin/env python3
"""Refuse an attempt when the two before it made no progress.

With ``train_dir`` set, the checkpoint directory survives a build-level retry, so a
retried run resumes from what the failed attempt wrote. A failure that recurs at the
same step (a bad batch, an OOM at a fixed sequence length, a config the trainer
rejects after loading) would then burn every retry the build has, each one holding a
multi-node allocation until it reaches the same point. This stops the third such
attempt before ``accelerate`` launches.

Not the second: a single attempt that dies before its next save is common and
transient here (an NCCL/IB hang, a dead node), and refusing on it would end a build
that one more attempt would have carried on. So every build-level attempt appends
one line to ``<train_dir>/attempts.jsonl``: its id, the highest complete checkpoint
it starts from, and its ``status``, ``started`` or ``refused``. An attempt is
refused (exit 3) when the two most recent ``started`` attempts both began at a step
>= the one it would start from: two consecutive attempts made no progress. Once one
attempt has been refused, every later one in the same train_dir is refused at once,
naming the refusal; only a new build (a new train_dir) clears it. A record with no
``status`` (written before refusals were recorded) counts as ``started``.

A ``retry_workload`` relaunch inside one target run carries the same attempt id as
the last ``started`` record and is let through unrecorded: it is the same attempt,
relaunched on other nodes, not a new verdict on the run.

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
# How many consecutive attempts must make no progress before the next is refused.
WINDOW = 2
STARTED = "started"
REFUSED = "refused"
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


def read_attempts(path: str) -> list[dict]:
    """Every record in the attempts file at path, oldest first; [] if none yet.

    Exits through fatal() if the file cannot be read or a line is not a record:
    guessing would either refuse a healthy retry or wave a stuck one on.
    """
    try:
        with open(path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if line.strip()]
    except FileNotFoundError:
        return []
    except OSError as e:
        fatal(f"cannot read {path}: {e}")
    records = []
    for number, line in enumerate(lines, 1):
        try:
            record = json.loads(line)
        except ValueError as e:
            fatal(f"{path}: line {number} is not JSON: {e}")
        if not isinstance(record, dict) or "attempt" not in record:
            fatal(f"{path}: line {number} is not an attempt record: {line!r}")
        if not isinstance(record.get("start_step"), int):
            fatal(f"{path}: line {number} has no integer start_step: {line!r}")
        record.setdefault("status", STARTED)
        if record["status"] not in (STARTED, REFUSED):
            fatal(f"{path}: line {number} has an unknown status: {line!r}")
        records.append(record)
    return records


def append_attempt(path: str, attempt: str, start_step: int, status: str) -> None:
    """Append this attempt's record to the attempts file at path."""
    record = {
        "attempt": attempt,
        "start_step": start_step,
        "status": status,
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError as e:
        fatal(f"cannot append to {path}: {e}")


def refuse(path: str, attempt: str, top: int, message: str, hint: str) -> NoReturn:
    """Record this attempt as refused, print the FATAL line and hint, exit 3."""
    append_attempt(path, attempt, top, REFUSED)
    print(f"logit-distill: FATAL: {message}", file=sys.stderr)
    print(f"  {hint}", file=sys.stderr)
    sys.exit(NO_PROGRESS)


def main(argv: list[str] | None = None) -> None:
    """Judge this attempt against the earlier ones; record it as started or refused."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--attempt-id", required=True)
    args = parser.parse_args(argv)

    path = os.path.join(args.train_dir, ATTEMPTS)
    top = top_step(args.ckpt_dir)
    records = read_attempts(path)
    started = [r for r in records if r["status"] == STARTED]
    refused = [r for r in records if r["status"] == REFUSED]

    # A relaunch replays the target run that last started; nothing new to judge.
    if started and started[-1]["attempt"] == args.attempt_id and not refused:
        print(
            f"{PREFIX} attempt {args.attempt_id} relaunched (retry_workload); "
            f"starting from checkpoint-{top}"
        )
        return
    if refused:
        first = refused[0]
        refuse(
            path,
            args.attempt_id,
            top,
            f"attempt {first['attempt']} was refused for no progress (at "
            f"checkpoint-{first['start_step']}); not retrying",
            f"This train_dir stays refused; read {ATTEMPTS} beside it, fix the "
            f"cause, and start a new build (or set require_progress: false).",
        )
    window = started[-WINDOW:]
    if len(window) == WINDOW and all(r["start_step"] >= top for r in window):
        names = ", ".join(r["attempt"] for r in window)
        refuse(
            path,
            args.attempt_id,
            top,
            f"no progress over the last {WINDOW} attempts ({names}; still at "
            f"checkpoint-{top}); not retrying",
            f"The failure recurs before the next save; read attempt "
            f"{window[-1]['attempt']}'s log, fix the cause, and start a new build "
            f"(or set require_progress: false).",
        )

    append_attempt(path, args.attempt_id, top, STARTED)
    print(f"{PREFIX} attempt {args.attempt_id} starts from checkpoint-{top}")


if __name__ == "__main__":
    main()
