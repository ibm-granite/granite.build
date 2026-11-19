# SPDX-License-Identifier: Apache-2.0

"""Decide how this attempt starts: in place, seeded from another run, or fresh.

A pretraining run on Vela outlives its allocation. It gets preempted, retried by
gbserver, and sometimes deliberately continued in a *new* build, so three
situations have to be told apart before the trainer starts:

1. **Retry of this build.** ``TRAIN_DIR`` (``$RUN_DIR/train``, stable across a
   build's retries) already holds checkpoints -- resume in place.
2. **Continue another run.** The recipe passes ``RESUME_FROM_CHECKPOINT_DIR``;
   ``TRAIN_DIR`` is seeded from it (below) and then resumed as (1).
3. **Fresh.** Nothing anywhere; no ``load_args`` at all.

"Holds checkpoints" means the completion marker
``latest_checkpointed_iteration.json`` is readable -- ``ckpt_watch.latest()``,
which already handles the partial write lm-engine's non-atomic marker allows. A
``global_step<N>/`` directory on its own means "started", never "finished".

SEEDING

Seeding symlinks the source's complete ``global_step*`` directories into
``TRAIN_DIR`` and gives ``TRAIN_DIR`` a marker file of its own. That the marker
is not linked too is the point: lm-engine's rank 0 rewrites it on every save with
a plain ``open(path, "w")``, which through a symlink would follow the link and
rewrite the *source run's* marker, corrupting a run this build does not own. Nothing else is ever written to an
existing rung, so the rung links are safe to follow -- new checkpoints go to
higher, locally created ``global_step<N>`` directories.

The links are absolute. ``TRAIN_DIR`` and the source live in unrelated build
trees reached through the same PVC mount path on every node, so an absolute link
resolves identically from every rank and from any working directory, while a
relative one would have to encode a ``../..`` relationship that is not stable
and breaks outright if either tree is reached through a differently rooted mount.

The consequence to know: after seeding, older rungs in ``TRAIN_DIR`` are
*someone else's files*. If that build's post-run cleanup reaps them the links
dangle, and anything that reads an old rung afterwards (``ckpt_watch`` emits
paths for ladder rungs, which an export target then opens) fails. The newest
checkpoint is always written locally, so the run itself and its final checkpoint
are unaffected.

load_rng_state

RNG state is saved per global rank: ``global_step<N>/rng_state/rng_state-<rank>.pt``
(``_get_rng_state_path``, ``lm_engine/training/checkpointing/__init__.py`` at the
pinned a9c6e2b8), and on load each rank reads *its own* file. Restoring it under
a different world size is therefore wrong at best and a missing-file crash on the
new high ranks at worst, so it is only requested when the saved world size is
known to equal this attempt's.

lm-engine records no world size as a field. ``metadata.json`` holds
``accelerator`` plus the training loop's ``consumed_samples`` / ``commit_id`` /
``cumulative_cost_usd`` (``train.py``), and ``training_config.yml`` dumps
``distributed_args``, whose only sizes are the tensor / pipeline / context
parallel ones -- the global size is not among them. What the checkpoint *does*
record is one rng_state shard per rank, so the saved world size is the number of
those shards. That is read here, conservatively: anything unexpected (no
``rng_state/``, a name that is not ``rng_state-<rank>.pt``, a gap in the ranks, a
rung directory the marker promises but that is not there) counts as unknown, and
unknown means ``load_rng_state: false``. Losing RNG continuity costs a
pretraining run nothing it can measure; loading another topology's RNG state
costs a crash or silent divergence.

require_progress

``--state-file`` records the iteration each attempt *started from*. With
``--require-progress`` on, an attempt whose newest checkpoint has not moved past
that fails before the trainer starts, instead of burning a multi-node allocation
in a retry loop that never advances. The file is rewritten on every path --
resume, fresh, and error -- so the next attempt compares against this one.
A state file that is absent, empty or malformed is treated as "no baseline": a
corrupt bookkeeping file must never stand between a preempted run and its
checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import tempfile

try:
    import ckpt_watch
except ModuleNotFoundError:  # pragma: no cover - python -I/-P: no script dir on path
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

    import ckpt_watch

_RNG_DIR = "rng_state"
_RNG_SHARD = re.compile(r"rng_state-(\d+)\.pt\Z")


# ── world size ─────────────────────────────────────────────────────────────────


def saved_world_size(rung_dir: pathlib.Path) -> int | None:
    """The world size a rung was saved with, from its rng_state shards, or None."""
    try:
        names = [entry.name for entry in (rung_dir / _RNG_DIR).iterdir()]
    except (FileNotFoundError, NotADirectoryError):
        return None
    ranks = set()
    for name in names:
        match = _RNG_SHARD.match(name)
        if match is None:
            return None
        ranks.add(int(match.group(1)))
    if not ranks or ranks != set(range(len(ranks))):
        return None
    return len(ranks)


# ── writing by rename ──────────────────────────────────────────────────────────────────────────────────────────


def _write(path: pathlib.Path, text: str) -> None:
    """Write ``text`` to ``path`` by rename.

    Every rank polls for the decision file and reads it as soon as it is
    non-empty, so a reader must never see a half-written one. The same care is
    cheap for the state file and saves a truncated write from looking like a
    corrupt baseline.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


# ── seeding ────────────────────────────────────────────────────────────────────


def _same_dir(left: pathlib.Path, right: pathlib.Path) -> bool:
    return os.path.realpath(left) == os.path.realpath(right)


def seed(train_dir: pathlib.Path, source: pathlib.Path, reached: int) -> None:
    """Link ``source``'s complete rungs into ``train_dir`` and mark it complete.

    Rungs past ``reached`` are half-written and are skipped; a name already taken
    in ``train_dir`` is left alone, so a second attempt re-seeds harmlessly.
    """
    train_dir.mkdir(parents=True, exist_ok=True)
    for entry in sorted(source.iterdir()):
        if not entry.name.startswith(ckpt_watch.TAG) or not entry.is_dir():
            continue
        iteration = entry.name[len(ckpt_watch.TAG) :]
        if not iteration.isdigit() or int(iteration) > reached:
            continue
        link = train_dir / entry.name
        if link.exists() or link.is_symlink():
            continue
        link.symlink_to(entry.resolve())
    # Written from the value already validated, not recopied: re-reading the
    # source marker could catch the source run mid-save.
    _write(
        train_dir / ckpt_watch.MARKER,
        json.dumps({ckpt_watch.KEY: reached}, indent=4) + "\n",
    )


# ── state file ─────────────────────────────────────────────────────────────────


def previous_iteration(state_file: pathlib.Path) -> int | None:
    """What the last attempt started from, or None if there is no usable record."""
    try:
        value = json.loads(state_file.read_text(encoding="utf-8"))["iteration"]
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


# ── the decision ───────────────────────────────────────────────────────────────


def decide(
    train_dir: pathlib.Path,
    *,
    resume_src: str,
    world_size: int,
    require_progress: bool,
    state_file: pathlib.Path,
) -> dict:
    """Return ``{"load": dict|None, "seeded": bool, "error": str|None}``."""
    seeded = False
    reached = ckpt_watch.latest(train_dir)

    if reached is None and resume_src and str(resume_src).strip():
        source = pathlib.Path(resume_src)
        source_reached = ckpt_watch.latest(source)
        if source_reached is not None and not _same_dir(source, train_dir):
            seed(train_dir, source, source_reached)
            seeded = True
            reached = ckpt_watch.latest(train_dir)

    error = None
    previous = previous_iteration(state_file) if require_progress else None
    if previous is not None and (reached is None or reached <= previous):
        error = f"no progress since last attempt at step {previous}"

    load = None
    if reached is not None and error is None:
        rung = train_dir / f"{ckpt_watch.TAG}{reached}"
        load = {
            "load_path": str(train_dir),
            # None: lm-engine reads the marker itself, so the rung it loads is
            # the one it decided was complete.
            "iteration": None,
            "load_optimizer": True,
            "load_lr_scheduler": True,
            "load_starting_iteration": True,
            # The dataloader state is only resumable when the original run had
            # a data checkpoint; the recipe does not enable one.
            "load_dataloader_state": False,
            "load_rng_state": saved_world_size(rung) == world_size,
        }

    _write(state_file, json.dumps({"iteration": reached}, indent=2) + "\n")
    return {"load": load, "seeded": seeded, "error": error}


# ── CLI ─────────────────────────────────────────────────────────────────────────


def _flag(value) -> bool:
    """``true``/``false`` as the shell spells them, as ``render_config._flag`` does."""
    return str(value).strip().lower() in ("true", "yes", "on", "1", "y")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-dir", required=True, help="this build's save_path")
    parser.add_argument(
        "--resume-src", default="", help="another run's save_path to seed from"
    )
    parser.add_argument(
        "--world-size", type=int, required=True, help="nodes x gpus per node"
    )
    parser.add_argument(
        "--require-progress",
        type=_flag,
        default=False,
        help="fail when this attempt has not passed the previous one",
    )
    parser.add_argument(
        "--state-file", required=True, help="where the attempt baseline is kept"
    )
    parser.add_argument("--out", required=True, help="where to write the decision JSON")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out = pathlib.Path(args.out)
    try:
        decision = decide(
            pathlib.Path(args.train_dir),
            resume_src=args.resume_src,
            world_size=args.world_size,
            require_progress=args.require_progress,
            state_file=pathlib.Path(args.state_file),
        )
    except OSError as error:
        # A broken train_dir, not a race. The other ranks are blocked polling
        # for this file, so tell them rather than exiting into their timeout.
        decision = {"load": None, "seeded": False, "error": f"{error}"}

    _write(out, json.dumps(decision, indent=2) + "\n")
    if decision["error"]:
        print(f"error: {decision['error']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
