"""Decide what a relaunch into an existing output_dir should DO, before any GPU is touched.

WHY THIS FILE EXISTS.

`gold.py:627-629` is the entire resume mechanism today:

    if glob.glob(os.path.join(training_args.output_dir, "checkpoint-*")):
        print(f"Resuming training from checkpoint under {training_args.output_dir}")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

A bare glob answers "does a directory whose name starts with checkpoint- exist", and then
`resume_from_checkpoint=True` hands the choice to transformers' `get_last_checkpoint`, which takes
the HIGHEST step number it finds. Neither asks whether that directory holds a checkpoint. Three
distinct failures follow, and this cluster produces all three:

  1. KILLED MID-SAVE. `preemptable` preemption is requeue-and-restart-from-argv, not
     suspend/resume -- confirmed directly, printing "preflight OK" three times. transformers writes a
     checkpoint into `checkpoint-N` in place, so a kill during a save leaves a real directory with
     a partial payload. It has the highest step number, so it WINS the selection, and a complete
     `checkpoint-25` sitting beside it is passed over.

  2. ALREADY AT max_steps. A run that reached the end and then got resubmitted resumes into
     `global_step == max_steps`, trains zero steps, and re-runs the save/export path. The log looks
     like a successful run. It is a no-op that overwrites the artefact it claims to have produced.

  3. WORLD SIZE CHANGED. ZeRO-3 shards optimizer state per rank -- `bf16_zero_pp_rank_{0..N-1}`.
     Loading needs the SAME rank count that wrote it. The `_node<N>x<G>` shapes make this reachable
     by ordinary means: 2x8 and 4x4 are both 16 ranks and interchangeable, but resubmitting the
     `_node4` config (24 ranks) into a directory written by 16 would fail inside DeepSpeed, after
     the allocation and the model load.

WHAT THIS DOES INSTEAD. `preflight()` returns one of four verdicts -- FRESH, RESUME, DONE, FATAL --
and the caller acts on the verdict. Incomplete checkpoints are QUARANTINED BY RENAME, never
deleted: a rename inside one directory is atomic and reversible, and the case being handled is a
job that died unattended, so the recovery has to be one that nobody needs to be present for.
Deleting would also destroy the only evidence of what the partial save looked like.

WHY IT LIVES IN src/ AND NOT IN a companion launcher tree. Both the reference launchers and the
granite.build steps need the same answer, and the answer must not be able to differ between them.
Same reason the tracking resolver is shared: two implementations of "is this checkpoint usable"
is one more than can be kept true.

CLI (this is what the launchers call):

    python -m gb_steps_post_training.distillation.checkpoints \
        --output-dir DIR [--max-steps N] [--world-size N] [--no-quarantine]

Exit codes, chosen so a shell can branch on them without parsing text:
    0  FRESH  -- nothing to resume from; train from scratch
    0  RESUME -- a complete checkpoint is usable; stdout carries RESUME_STEP=<n>
   64  DONE   -- the requested max_steps is already reached; the caller should NOT relaunch
   65  FATAL  -- something is wrong that a rename cannot fix; do not spend an allocation
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

CKPT_RE = re.compile(r"^checkpoint-(\d+)$")

# Written by transformers itself. `trainer_state.json` is the one that carries the step, so it is
# the file whose absence makes a directory unusable rather than merely odd.
REQUIRED = ("trainer_state.json", "training_args.bin")

# ZeRO-3 shard payload. The rank-0 names are checked for presence; the full set is counted to
# recover the world size that wrote it.
OPTIM_RE = re.compile(r"^bf16_zero_pp_rank_(\d+)_mp_rank_\d+_optim_states\.pt$")
MODEL_RE = re.compile(r"^zero_pp_rank_(\d+)_mp_rank_\d+_model_states\.pt$")

QUARANTINE_DIRNAME = ".quarantined-checkpoints"


@dataclass
class Checkpoint:
    """One `checkpoint-N` directory, and everything knowable about it without loading a tensor."""

    path: Path
    step: int
    problems: list[str] = field(default_factory=list)
    state_step: int | None = None
    world_size: int | None = None

    @property
    def complete(self) -> bool:
        return not self.problems

    def __str__(self) -> str:
        ws = f", {self.world_size} ranks" if self.world_size else ""
        return f"{self.path.name} (step {self.step}{ws})"


def inspect(path: Path) -> Checkpoint:
    """Everything that can be decided from the filesystem, with no torch import and no GPU.

    Deliberately cheap: this runs on a login node and inside a preflight, so it reads one small
    JSON file and lists two directories. It cannot detect a truncated .pt file -- that needs a
    load -- but every failure observed here so far is a MISSING file or a step mismatch, because a
    kill lands between writes far more often than inside one.
    """
    m = CKPT_RE.match(path.name)
    step = int(m.group(1)) if m else -1
    ck = Checkpoint(path=path, step=step)
    if not m:
        ck.problems.append(f"name {path.name!r} is not checkpoint-<int>")
        return ck
    if not path.is_dir():
        ck.problems.append("not a directory")
        return ck

    names = {p.name for p in path.iterdir()}
    for req in REQUIRED:
        if req not in names:
            ck.problems.append(f"missing {req}")

    if "trainer_state.json" in names:
        try:
            state = json.loads((path / "trainer_state.json").read_text())
        except (OSError, json.JSONDecodeError) as exc:
            # The single most likely artefact of a kill mid-save: a JSON file that was opened and
            # partially written. It is also the cheapest thing in the tree to check.
            ck.problems.append(f"trainer_state.json will not parse ({exc.__class__.__name__}: {exc})")
        else:
            gs = state.get("global_step")
            ck.state_step = gs if isinstance(gs, int) else None
            if ck.state_step is None:
                ck.problems.append(f"trainer_state.json has no integer global_step (got {gs!r})")
            elif ck.state_step != step:
                # A directory named for one step holding state from another is not a checkpoint
                # this code will guess about.
                ck.problems.append(
                    f"trainer_state.json says global_step={ck.state_step} but the directory is "
                    f"named for step {step}"
                )

    # DeepSpeed half. `latest` is written LAST by DeepSpeed's save, which makes its absence the
    # sharpest available signal that the save did not finish.
    ds_dir = path / f"global_step{step}"
    if "latest" not in names:
        ck.problems.append("missing `latest` -- DeepSpeed writes it last, so the save did not finish")
    else:
        latest = (path / "latest").read_text().strip()
        if latest != ds_dir.name:
            ck.problems.append(f"`latest` says {latest!r}, expected {ds_dir.name!r}")
    if not ds_dir.is_dir():
        ck.problems.append(f"missing {ds_dir.name}/ -- the ZeRO shard payload")
    else:
        shard_names = {p.name for p in ds_dir.iterdir()}
        optim = sorted(int(m.group(1)) for n in shard_names if (m := OPTIM_RE.match(n)))
        model = sorted(int(m.group(1)) for n in shard_names if (m := MODEL_RE.match(n)))
        if not optim:
            ck.problems.append(f"{ds_dir.name}/ holds no *_optim_states.pt")
        elif optim != list(range(len(optim))):
            # A gap means some rank's write is missing, which is exactly what a kill during a
            # collective save produces, and it is invisible to a count-only check.
            missing = sorted(set(range(max(optim) + 1)) - set(optim))
            ck.problems.append(f"{ds_dir.name}/ optim shards are not contiguous; ranks {missing} absent")
        else:
            ck.world_size = len(optim)
        if optim and model and len(model) != len(optim):
            ck.problems.append(
                f"{ds_dir.name}/ has {len(optim)} optim shards but {len(model)} model shards"
            )
        empty = sorted(n for n in shard_names if (ds_dir / n).is_file() and (ds_dir / n).stat().st_size == 0)
        if empty:
            ck.problems.append(f"{ds_dir.name}/ holds zero-byte shard(s): {empty}")
    return ck


def survey(output_dir: Path) -> list[Checkpoint]:
    """Every checkpoint-N under output_dir, ascending by step. Highest step is last."""
    if not output_dir.is_dir():
        return []
    found = [inspect(p) for p in output_dir.iterdir() if CKPT_RE.match(p.name)]
    return sorted(found, key=lambda c: c.step)


def quarantine(ck: Checkpoint) -> Path:
    """Move an unusable checkpoint out of the glob's way. Rename only -- nothing is deleted.

    The destination is a sibling directory rather than a renamed peer, because the selector
    transformers uses is a `checkpoint-*` glob over output_dir: a name like
    `checkpoint-50.incomplete` still matches it. Under `.quarantined-checkpoints/` it cannot.

    A numeric suffix is appended if the destination is taken, so quarantining the same step twice
    (two preemptions in the same place) keeps both rather than overwriting the first.
    """
    dest_dir = ck.path.parent / QUARANTINE_DIRNAME
    dest_dir.mkdir(exist_ok=True)
    dest = dest_dir / ck.path.name
    n = 2
    while dest.exists():
        dest = dest_dir / f"{ck.path.name}.{n}"
        n += 1
    os.rename(ck.path, dest)
    (dest_dir / "README").write_text(
        "Checkpoints moved here by distillation/checkpoints.py because they were incomplete.\n"
        "Nothing in this directory is deleted automatically. It is out of the way of the\n"
        "`checkpoint-*` glob transformers uses to pick a resume point, and it is kept so the\n"
        "shape of a failed save can still be inspected. Delete it by hand when you are done.\n"
    )
    return dest


FRESH, RESUME, DONE, FATAL = "FRESH", "RESUME", "DONE", "FATAL"
EXIT = {FRESH: 0, RESUME: 0, DONE: 64, FATAL: 65}


@dataclass
class Verdict:
    kind: str
    step: int | None = None
    lines: list[str] = field(default_factory=list)


def preflight(output_dir: Path, max_steps: int | None = None, world_size: int | None = None,
              do_quarantine: bool = True) -> Verdict:
    """What a relaunch into output_dir should do.

    max_steps and world_size are OPTIONAL and their absence is not an error, because the two
    callers know different things: a launcher reads both out of the config, while a step wrapper
    may only know the directory. What is knowable is checked; what is not is stated as not
    checked, rather than assumed satisfied.
    """
    v = Verdict(kind=FRESH)
    cks = survey(output_dir)
    if not cks:
        v.lines.append(f"no checkpoint-* under {output_dir} -- training from scratch")
        return v

    bad = [c for c in cks if not c.complete]
    for c in bad:
        v.lines.append(f"INCOMPLETE {c.path.name}:")
        for p in c.problems:
            v.lines.append(f"    - {p}")
        if do_quarantine:
            dest = quarantine(c)
            v.lines.append(f"    quarantined -> {dest.relative_to(output_dir)}  (renamed, not deleted)")
        else:
            v.lines.append("    left in place (--no-quarantine): the resume selector will still "
                           "prefer it if its step is the highest")
    good = [c for c in cks if c.complete]
    if not good:
        v.kind = FATAL if not do_quarantine else FRESH
        if v.kind == FRESH:
            v.lines.append("every checkpoint was incomplete and has been quarantined -- "
                           "nothing left to resume from, so this will train from scratch")
        else:
            v.lines.append("every checkpoint is incomplete and quarantine is disabled")
        return v

    last = good[-1]
    if bad and last.step < max(c.step for c in bad):
        v.lines.append(f"note: {last.path.name} is now the resume point; the higher-numbered "
                       "quarantined checkpoint(s) would have been chosen ahead of it")

    mismatch = None
    if world_size is not None and last.world_size is not None and last.world_size != world_size:
        mismatch = (
            f"WORLD SIZE MISMATCH: {last.path.name} was written by {last.world_size} ranks and this "
            f"launch has {world_size}. ZeRO-3 shards optimizer state per rank, so the load fails "
            "inside DeepSpeed after the allocation and the model load. Relaunch on a shape with "
            f"{last.world_size} ranks (2x8 and 4x4 are both 16, and interchangeable), or point "
            "--output-dir somewhere else and start fresh."
        )

    # DONE is decided BEFORE the mismatch, and this order is a CORRECTION of the previous one.
    # The old order refused first, on the reasoning that a mismatch "cannot be fixed by not
    # relaunching". That reasoning does not survive the DONE case: a finished run never loads a
    # checkpoint, so the rank count one was written with is a fact about a load that will not
    # happen. A direct measurement showed what the old order cost -- asked at 1 GPU about an
    # output_dir already at 50/50, it answered "relaunch on a shape with 8 ranks", i.e. go and win
    # a 16-GPU allocation in order to be told there is nothing to do. On this queue that is a
    # ~30 minute PEND for a guaranteed no-op, and a caller who obeys learns nothing new.
    #
    # The mismatch is not dropped, it is DEMOTED to a note, because it still matters to the reader
    # who is about to raise max_steps and resume for real. Below DONE it still returns FATAL.
    if max_steps is not None and last.step >= max_steps:
        v.kind = DONE
        v.step = last.step
        v.lines.append(
            f"{last.path.name} is already at or past max_steps={max_steps}. Resuming would train "
            "ZERO steps and then re-run the save/export path, producing a log that reads like a "
            "successful run. Not relaunching."
        )
        if mismatch is not None:
            v.lines.append(
                f"note: {last.path.name} was written by {last.world_size} ranks and this launch has "
                f"{world_size}. Harmless as things stand, since nothing is loaded -- but raising "
                f"max_steps and resuming for real would need a shape with {last.world_size} ranks."
            )
        return v

    if mismatch is not None:
        v.kind = FATAL
        v.lines.append(mismatch)
        return v

    v.kind = RESUME
    v.step = last.step
    v.lines.append(f"resume point: {last}")
    if max_steps is not None:
        v.lines.append(f"  {max_steps - last.step} of {max_steps} steps remain")
    else:
        v.lines.append("  max_steps not supplied, so 'already finished' was NOT checked")
    if world_size is None:
        v.lines.append("  world size not supplied, so the ZeRO rank count was NOT checked")
    return v


# --------------------------------------------------------------------------- self test
def _write_ckpt(root: Path, step: int, ranks: int = 8, *, latest: bool = True,
                state_step: int | None = None, bad_json: bool = False,
                drop_optim: list[int] | None = None, empty: bool = False) -> Path:
    d = root / f"checkpoint-{step}"
    (d / f"global_step{step}").mkdir(parents=True)
    (d / "training_args.bin").write_bytes(b"x")
    if bad_json:
        (d / "trainer_state.json").write_text('{"global_step": 2')
    else:
        (d / "trainer_state.json").write_text(
            json.dumps({"global_step": step if state_step is None else state_step}))
    if latest:
        (d / "latest").write_text(f"global_step{step}\n")
    for r in range(ranks):
        if drop_optim and r in (drop_optim or []):
            continue
        f = d / f"global_step{step}" / f"bf16_zero_pp_rank_{r}_mp_rank_00_optim_states.pt"
        f.write_bytes(b"" if empty else b"x")
        (d / f"global_step{step}" / f"zero_pp_rank_{r}_mp_rank_00_model_states.pt").write_bytes(b"x")
    return d


def self_test() -> int:
    import tempfile

    bad = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal bad
        if cond:
            print(f"  OK   {name}")
        else:
            print(f"  FAIL {name}" + (f"  [{detail}]" if detail else ""))
            bad = 1

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        check("empty dir -> FRESH", preflight(root).kind == FRESH)
        check("missing dir -> FRESH", preflight(root / "nope").kind == FRESH)

        _write_ckpt(root, 25)
        v = preflight(root)
        check("one good checkpoint -> RESUME at its step", v.kind == RESUME and v.step == 25,
              f"{v.kind} {v.step}")

        # THE case this file exists for: a partial save with the HIGHEST step number.
        _write_ckpt(root, 50, latest=False)
        v = preflight(root)
        check("partial checkpoint-50 is quarantined and 25 becomes the resume point",
              v.kind == RESUME and v.step == 25 and not (root / "checkpoint-50").exists()
              and (root / QUARANTINE_DIRNAME / "checkpoint-50").is_dir(), f"{v.kind} {v.step}")
        check("quarantine is out of reach of a checkpoint-* glob",
              not list(root.glob("checkpoint-5*")))
        check("the quarantined payload still exists",
              (root / QUARANTINE_DIRNAME / "checkpoint-50" / "trainer_state.json").is_file())

        # Twice in the same place must keep both, not overwrite the first.
        _write_ckpt(root, 50, latest=False)
        preflight(root)
        check("a second quarantine of the same step does not overwrite the first",
              (root / QUARANTINE_DIRNAME / "checkpoint-50.2").is_dir())

        check("max_steps already reached -> DONE, not RESUME",
              preflight(root, max_steps=25).kind == DONE)
        check("max_steps beyond -> RESUME", preflight(root, max_steps=50).kind == RESUME)
        check("world size match -> RESUME", preflight(root, world_size=8).kind == RESUME)
        v = preflight(root, world_size=24)
        check("world size mismatch -> FATAL and says which shape wrote it",
              v.kind == FATAL and "8 ranks" in " ".join(v.lines), f"{v.kind}")
        # DONE outranks the mismatch, and both halves are asserted: the verdict flips to DONE,
        # AND the demoted mismatch is still reported. Asserting only the kind would let a version
        # that silently swallowed the rank count pass, which is the failure this demotion could
        # plausibly introduce. See the reasoning at the decision itself, confirmed directly.
        v = preflight(root, max_steps=25, world_size=24)
        check("DONE outranks the mismatch, which is demoted to a note, not dropped",
              v.kind == DONE and any("written by 8 ranks" in ln for ln in v.lines),
              f"{v.kind}: {v.lines}")
        # ... and below DONE the mismatch still refuses, since there the load really would happen.
        check("mismatch still FATAL when steps remain",
              preflight(root, max_steps=99, world_size=24).kind == FATAL)

    # Each individual defect, so a green result cannot come from one over-broad rule.
    cases = [
        ("no latest", dict(latest=False), "missing `latest`"),
        ("state step disagrees with the name", dict(state_step=7), "named for step"),
        ("unparseable trainer_state.json", dict(bad_json=True), "will not parse"),
        ("a rank's optim shard is absent", dict(drop_optim=[3]), "not contiguous"),
        ("zero-byte shards", dict(empty=True), "zero-byte"),
    ]
    for name, kw, want in cases:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_ckpt(root, 10, **kw)
            ck = inspect(root / "checkpoint-10")
            hit = any(want in p for p in ck.problems)
            check(f"detected: {name}", not ck.complete and hit, f"{ck.problems}")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write_ckpt(root, 10, latest=False)
        v = preflight(root, do_quarantine=False)
        check("--no-quarantine leaves the directory alone and refuses instead",
              v.kind == FATAL and (root / "checkpoint-10").is_dir(), f"{v.kind}")
        v = preflight(root)
        check("with quarantine, the same tree becomes FRESH rather than FATAL",
              v.kind == FRESH and not (root / "checkpoint-10").exists(), f"{v.kind}")

    if not bad:
        print("\nRESUME DECISION EXERCISED: fresh, resume, done-at-max-steps, world-size refusal, "
              "five distinct partial-save shapes, and quarantine-by-rename (twice, non-destructive)")
    return bad


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--world-size", type=int, default=None)
    ap.add_argument("--no-quarantine", action="store_true",
                    help="report and refuse instead of renaming. For an audit, not for a launcher: "
                         "a job that dies unattended needs a recovery nobody has to be present for.")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.output_dir:
        ap.error("--output-dir is required (or use --self-test)")

    # The self-test runs on EVERY invocation, because a broken decision procedure must not be
    # able to authorise an allocation -- but it is silent when it passes, so a launcher log shows
    # the verdict and not nineteen OK lines. On failure the whole transcript is released.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = self_test()
    if rc:
        sys.stdout.write(buf.getvalue())
        print("\nFATAL: checkpoints.py self-test failed; not deciding anything")
        return EXIT[FATAL]
    print("self-test: 19 resume decisions OK (silent when passing; --self-test to see them)")

    v = preflight(Path(a.output_dir), max_steps=a.max_steps, world_size=a.world_size,
                  do_quarantine=not a.no_quarantine)
    print(f"\nresume: {v.kind}")
    for line in v.lines:
        print(f"  {line}")
    if v.kind == RESUME:
        print(f"RESUME_STEP={v.step}")
    return EXIT[v.kind]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
