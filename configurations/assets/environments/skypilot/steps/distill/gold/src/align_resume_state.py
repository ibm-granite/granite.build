#!/usr/bin/env python3
"""Rewrite a seeded checkpoint's trainer_state.json to this run's save and logging grid.

HF Trainer computes ``save_steps`` and ``logging_steps`` from this run's arguments,
then, on resume, replaces the whole state with the checkpoint's ``trainer_state.json``.
The save callback reads the state, so a run seeded by ``resume_from_checkpoint_dir``
saves on the SOURCE run's grid; only a transformers warning that never reaches the
step's output says so. Build 4c758672 (A1) resumed a 250-step ladder from c2000's
1000-step grid and never wrote its 2250, 2500 or 3500 rungs.

The step's resume block runs this on rank 0, after seeding and before the trainer
starts, on the checkpoint the trainer will resume from (the block picks it, by
``get_last_checkpoint``'s rule, and checks it is complete).

Both values must be positive integers. ``render_gold_config.py`` parses them with
``type=int``, so a fraction (which HF would resolve against ``max_steps``) never
reaches the trainer, and is refused here rather than resolved. ``eval_steps`` is
left alone: the step never enables evaluation.

The file is replaced, never edited in place: it is a hardlink into the source run's
checkpoint, which must keep its own grid. ``cp -a`` copies the source directory's
mode, so a read-only source leaves a read-only copy; the directory is made writable
for the replace and its mode restored after.
"""

# The annotations below are 3.10 syntax; this keeps an older $PYBIN to the FATAL
# lines instead of a TypeError at import.
from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from typing import NoReturn

PREFIX = "distill-gold: resume:"


def fatal(message: str) -> NoReturn:
    """Exit 1 with a FATAL line on stderr, the step's convention for a refusal."""
    sys.exit(f"distill-gold: FATAL: {message}")


def step_count(name: str, raw: str) -> int:
    """raw as a positive integer, as render_gold_config.py's --<name> would take it.

    Exits through fatal() for anything else, naming the key and the value.
    """
    try:
        value = int(raw)
    except ValueError:
        fatal(f"{name} must be a positive integer, got {raw!r}")
    if value <= 0:
        fatal(f"{name} must be a positive integer, got {raw!r}")
    return value


def load_state(path: str) -> dict:
    """The parsed trainer_state.json at path; exits through fatal() if unreadable."""
    try:
        with open(path) as f:
            state = json.load(f)
    except (OSError, ValueError) as e:
        fatal(f"cannot read {path}: {e}")
    if not isinstance(state, dict):
        fatal(f"{path} is not a JSON object")
    return state


def replace_json(path: str, state: dict) -> None:
    """Write state to path through a new inode, so a hardlinked original is untouched.

    A read-only parent directory is made writable for the replace and its mode
    restored after. Exits through fatal() on any failure, naming the first error
    and leaving no .tmp file behind.
    """
    directory = os.path.dirname(path)
    tmp = path + ".tmp"
    mode = stat.S_IMODE(os.stat(directory).st_mode)
    read_only = not mode & stat.S_IWUSR
    error = None
    try:
        if read_only:
            os.chmod(directory, mode | stat.S_IWUSR)
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except OSError as e:
        error = e
        try:
            os.unlink(tmp)
        except OSError:
            pass
    if read_only:
        try:
            os.chmod(directory, mode)
        except OSError as e:
            error = error or e
    if error:
        fatal(f"cannot rewrite {path}: {error}")


def main(argv: list[str] | None = None) -> None:
    """Parse the flags, rewrite what differs, and log each change."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--save-steps", required=True)
    parser.add_argument("--logging-steps", required=True)
    args = parser.parse_args(argv)

    requested = {
        "save_steps": step_count("save_steps", args.save_steps),
        "logging_steps": step_count("logging_steps", args.logging_steps),
    }
    path = os.path.join(args.checkpoint, "trainer_state.json")
    state = load_state(path)

    changed = []
    for key, steps in requested.items():
        if state.get(key) != steps:
            changed.append(f"{key} {state.get(key)} -> {steps}")
            state[key] = steps
    name = os.path.basename(os.path.normpath(args.checkpoint))
    if changed:
        replace_json(path, state)
        print(f"{PREFIX} {name}/trainer_state.json " + ", ".join(changed))
    else:
        print(f"{PREFIX} {name}/trainer_state.json already on this run's grid")


if __name__ == "__main__":
    main()
