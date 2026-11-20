# SPDX-License-Identifier: Apache-2.0

"""Emit ``GB_ARTIFACT`` lines for checkpoint rungs as lm-engine completes them.

Mid-run export and eval are the point of ``recipes/pretrain``: as soon as rung N
is complete on disk, gbserver should dispatch ``export-N`` and its evals *while
training continues*. gbserver has no mid-run checkpoint feature -- a step emits an
artifact by printing a line its monitor scrapes from the step's stdout::

    GB_ARTIFACT_ID:checkpoint_<N> GB_ARTIFACT_PATH:<path>

So ``run_pretrain.sh`` starts this on rank 0 only (the executor streams every
node's stdout into one driver log, so an unguarded marker would register the same
checkpoint once per node), in the background, for the life of the trainer.

Only ladder rungs are emitted. ``SAVE_INTERVAL`` normally writes many more
checkpoints than the ladder names, and an id the recipe did not declare as an
output is a line gbserver has nothing to bind.

COMPLETION

Both halves of the guard matter, and both are needed:

* ``latest_checkpointed_iteration.json`` says ``>= N``, and
* ``global_step<N>/`` is a directory.

lm-engine's ``save_checkpoint`` (``lm_engine/training/checkpointing/__init__.py``,
pinned at a9c6e2b8) writes that marker *last*: in the synchronous path after
``dcp.save`` and after ``ProcessGroupManager.barrier()``, and in the asynchronous
path from ``_FUTURE.add_done_callback``, i.e. once the save future has resolved.
It is therefore a completion signal, and it is the load-bearing half.

The directory check is the cheap half and it is not redundant: ``save_checkpoint``
``mkdir``s ``global_step<N>/`` *before* writing anything into it, so the directory
on its own means "started", never "finished" -- which is exactly why a rung whose
marker has not reached it is withheld even though its directory exists. The check
earns its place the other way round, catching a rung the ladder names that this
train_dir never wrote (a resumed run whose ladder is longer than the save
interval reached), where the marker alone would wave a non-existent path through.

A line emitted early hands a half-written tensor directory to an export, which
crashes or -- worse -- exports a damaged model. A line never emitted strands the
export/eval fan-out: the target never dispatches and the build waits on it. So
when the marker is unreadable this keeps polling rather than guessing: the write
is a plain ``open("w")`` + ``json.dump``, not an atomic rename, so a poll really
can see an empty or truncated file. Only the errors that mean "absent or
mid-write" are swallowed; a permissions error, or a directory where the marker
should be, is a broken train_dir and is raised.

Rungs are re-emitted by a fresh ``watch()`` (a build-level retry runs one against
the same train_dir). gbserver re-checks a downstream export only once its input
binding is delivered again; undelivered, the target is never dispatched and the
build still reports SUCCESS.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

MARKER = "latest_checkpointed_iteration.json"
KEY = "latest_checkpointed_iteration"
TAG = "global_step"
DEFAULT_INTERVAL_SECONDS = 60

# Absent, empty, truncated, or holding a value this cannot read as an iteration:
# all of these are "not yet, try the next poll".
#   FileNotFoundError  the trainer has not saved anything
#   ValueError         empty or truncated JSON (json.JSONDecodeError is a
#                      ValueError), or a value int() will not take
#   KeyError           a dict written without the key
#   TypeError          valid JSON that is not a subscriptable dict (``null``, a
#                      list, a bare number), or a value int() will not take
# Anything else -- a PermissionError, a directory where the marker should be --
# is a broken train_dir, not a race, and must not be polled away in silence.
_NOT_YET = (FileNotFoundError, KeyError, TypeError, ValueError)


def latest(train_dir: pathlib.Path) -> int | None:
    """The highest iteration lm-engine has finished saving, or None."""
    try:
        value = json.loads((train_dir / MARKER).read_text(encoding="utf-8"))[KEY]
        return int(value)
    except _NOT_YET:
        return None


def complete_rungs(train_dir: pathlib.Path, rungs: list[int]) -> list[int]:
    """The ladder rungs that are complete on disk, ascending."""
    reached = latest(train_dir)
    if reached is None:
        return []
    return [
        rung
        for rung in sorted(set(rungs))
        if rung <= reached and (train_dir / f"{TAG}{rung}").is_dir()
    ]


def line(train_dir: pathlib.Path, rung: int) -> str:
    """The artifact marker gbserver's monitor scrapes for this rung."""
    return (
        f"GB_ARTIFACT_ID:checkpoint_{rung} "
        f"GB_ARTIFACT_PATH:{train_dir / f'{TAG}{rung}'}"
    )


def watch(
    train_dir: pathlib.Path,
    rungs: list[int],
    interval: float,
    stop_file: pathlib.Path,
    out=None,
    sleep=time.sleep,
) -> set[int]:
    """Poll ``train_dir`` and emit each ladder rung once; return what was emitted.

    Returns when the stop file appears (``run_pretrain.sh`` touches it once the
    trainer exits) or when every rung has been emitted. An empty ladder returns
    at once: there is nothing to wait for, and a watcher spinning on a ladder it
    can never satisfy would outlive the trainer. The caller should simply not
    start this when emission is off; ``main`` says so on stderr, because "emitted
    nothing" and "still waiting" look identical in a log.
    """
    # Resolved here, not in the signature: a default bound at import time would
    # pin whatever sys.stdout was then, and this is called from main() after the
    # caller may have redirected it.
    if out is None:
        out = sys.stdout
    wanted = set(rungs)
    emitted: set[int] = set()

    def sweep() -> None:
        for rung in complete_rungs(train_dir, rungs):
            if rung not in emitted:
                print(line(train_dir, rung), file=out, flush=True)
                emitted.add(rung)

    while True:
        sweep()
        if emitted == wanted or stop_file.exists():
            # One last look: the trainer may have finished a rung in the gap
            # between the previous sweep and the stop file appearing.
            #
            # This narrows the window, it does not close it. A save that lands
            # *during* this sweep, or after it, is not emitted -- the trainer has
            # already exited by the time the stop file is touched, so the only
            # loss is a rung whose marker is written by an async done-callback
            # racing process teardown. The recipe's final checkpoint output (not
            # this watcher) is what guarantees the last checkpoint is delivered.
            sweep()
            return emitted
        sleep(interval)


# ── CLI ─────────────────────────────────────────────────────────────────────────


def parse_rungs(text: str) -> list[int]:
    """``"10,2,4"`` -> ``[2, 4, 10]``; empty -> ``[]``; anything else raises.

    Deliberately strict. A rung silently dropped is an export target that never
    dispatches, so a typo must stop the step, not thin the ladder.
    """
    rungs = set()
    for field in text.split(","):
        field = field.strip()
        if not field:
            continue
        if not field.isdigit() or int(field) <= 0:
            raise ValueError(f"not a checkpoint rung: {field!r}")
        rungs.add(int(field))
    return sorted(rungs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-dir", required=True, help="lm-engine save_path")
    parser.add_argument(
        "--rungs", required=True, help="ladder rungs to emit, comma-separated"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL_SECONDS,
        help="seconds between polls (default: %(default)s)",
    )
    parser.add_argument(
        "--stop-file",
        required=True,
        help="path whose appearance ends the watch",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    train_dir = pathlib.Path(args.train_dir)
    if not train_dir.is_dir():
        print(f"error: --train-dir is not a directory: {train_dir}", file=sys.stderr)
        return 2
    try:
        rungs = parse_rungs(args.rungs)
    except ValueError as error:
        print(f"error: --rungs: {error}", file=sys.stderr)
        return 2
    if not rungs:
        print("ckpt_watch: no checkpoint rungs to watch; exiting", file=sys.stderr)
        return 0

    watch(train_dir, rungs, args.interval, pathlib.Path(args.stop_file))
    return 0


if __name__ == "__main__":
    sys.exit(main())
