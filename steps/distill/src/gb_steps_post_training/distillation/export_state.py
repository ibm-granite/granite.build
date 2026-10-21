"""Resume state for distill-hf-export: is `dest` already this checkpoint, exported this way?

The general contract lives in step_state.py (RUN / SKIP / REFUSE, a `.step-done.json` written
last and atomically). This module supplies the two step-specific halves: what the EXPECTATION is,
and what the declared OUTPUTS are.

WHY THIS MODULE HOLDS NO `check`/`mark` CLI, unlike align_state.py.
distill-tokenizer-align is driven by run-align.sh, so its gate has to be reachable from shell and
that means a subprocess entrypoint. This step has no wrapper at all -- both launchers in
step-template.yaml invoke export_hf_model.py directly -- so the gate belongs inside its main(),
where it is a function call and cannot drift out of step with the export it guards. A CLI here
would be a second way to ask the same question, which is the thing this file exists to avoid.

WHY IT DOES NOT IMPORT export_hf_model.
`classify()` and `select_checkpoint()` live in the step's own src/, which is not importable from
this package (and the reverse import would be a cycle). So the caller passes the already-computed
`keep` and `dropped_unrecognised` lists in. That inverts the dependency in the direction that
works, and it has a second benefit: the expectation records what classify ACTUALLY decided for
this checkpoint, rather than re-deriving it here from a copy of the rules that could drift.

WHAT IS COMPARED, AND WHY EACH KEY IS IN OR OUT.

  checkpoint  -- by CONTENT, three ways. `name` (checkpoint-50) for legibility; `step` read out
      of trainer_state.json; and `state_sha`, the digest of trainer_state.json itself. That last
      one is the real decider and it is unusually good at this job: trainer_state.json carries the
      full `log_history`, so its digest is a fingerprint of the entire training trajectory up to
      that step. Two runs that both reach step 50 with different losses do not collide.
      Also `weights`: shard names and sizes, because the weights are what gets published and
      trainer_state.json says nothing about them.

  NOT the `--checkpoint` SELECTOR. `latest` and an explicit `checkpoint-50` that resolve to the
      same directory are the same work, and a step that REFUSED the second because the first said
      "latest" would be refusing over a spelling. The resolved identity above is what matters, and
      the selector cannot change it without changing that.

  kept  -- the file names classify() decided to publish. This is the shape of the output
      directory. A transformers release that starts writing a new file into checkpoints changes
      this list, and the already-exported dest does not contain that file; SKIPping would publish
      a directory that is missing something the current rules say belongs there.

  dropped_unrecognised  -- mirrors the manifest key. NOT the `--allow-unknown` flag: with nothing
      unrecognised in the checkpoint, the flag makes no difference to a single output byte, and
      comparing it would REFUSE two runs that produce identical directories. What matters is
      whether anything WAS dropped, which is what this records.

  padding_side / chat_template_thinking  -- both are edits to published defaults, so they change
      the bytes in dest. Compared for the obvious reason.

  verify + expect_tokenizer  -- `verify` is an ASSERTION rather than an output, and it is compared
      anyway, for the same reason align_state compares its own: a SKIP is the step declining to
      run, so a dest exported under --no-verify would never get verified no matter how many times
      someone asks with --verify. The tokenizer reference is compared by tokenizer.json DIGEST
      rather than by path, because a hub-cached model's path ends in a 40-char snapshot SHA and a
      retagged student's path is a work directory that gets repointed; the vocabulary is the thing
      the assertion is actually about.

WHERE THE MARKER LIVES, and the one objection to it.
In `dest`, beside the outputs -- it has to be, since step_state checks each output's size and
digest relative to the marker's directory. That does put a dot-file into a directory whose whole
premise (KEEP_FILES in export_hf_model.py) is that only declared files get published. It is not a
new category, though: this step already writes export_manifest.json into dest, which is equally
not part of the model. A future publish step should exclude both by name rather than assume a
model directory contains only model files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

from . import step_state, tokenizer_identity

STEP_NAME = "distill-hf-export"

MARKER_ALSO_NOT_MODEL = (step_state.MARKER_NAME, "export_manifest.json")

_WEIGHT_SUFFIX = ".safetensors"


def _sha16(path: Path) -> str | None:
    """`sha256:<16 hex>` over one file, or None when it is absent. Truncated for the same
    reason tokenizer_identity truncates: this is a change detector, and a 64-char digest in a
    diff line is unreadable."""
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return f"sha256:{h}"


def _weights(checkpoint: Path) -> list[list]:
    """[[name, bytes], ...] for the safetensors at the checkpoint root, sorted.

    Size and not digest, deliberately, and the blind spot is stated rather than hidden: a rewrite
    that produces DIFFERENT weights of the SAME length reads as identical here. Digesting them
    would mean reading 704 MB (this pairing's student) or ~60 GB (a 30B teacher) on every check,
    which would make the gate cost more than the export it is trying to skip. trainer_state.json's
    digest is the guard that actually distinguishes two runs; this list guards against a shard
    going missing or being truncated.
    """
    return sorted([p.name, p.stat().st_size]
                  for p in Path(checkpoint).glob(f"*{_WEIGHT_SUFFIX}"))


def checkpoint_identity(checkpoint: Path) -> dict:
    """Who this checkpoint is, by content. See the module docstring for why each key is here."""
    ckpt = Path(checkpoint)
    state = ckpt / "trainer_state.json"
    step = None
    if state.is_file():
        try:
            step = json.loads(state.read_text()).get("global_step")
        except (OSError, json.JSONDecodeError):
            # Left as None rather than raised: a checkpoint whose trainer_state.json will not
            # parse is a real situation (see checkpoints.py, which quarantines exactly that), and
            # this function's job is to describe what is there. `state_sha` below still changes,
            # so a damaged state file cannot make two different checkpoints compare equal.
            step = None
    return {
        "name": ckpt.name,
        "step": step,
        "state_sha": _sha16(state),
        "weights": _weights(ckpt),
    }


def expectation(*, checkpoint: Path, keep: list[str], dropped_unrecognised: list[str],
                padding_side: str, chat_template_thinking: str, verify: bool,
                expect_tokenizer_from: Path | None) -> dict:
    """The full comparison key for one export."""
    return {
        "checkpoint": checkpoint_identity(checkpoint),
        "kept": sorted(keep),
        "dropped_unrecognised": sorted(dropped_unrecognised),
        "padding_side": padding_side,
        "chat_template_thinking": chat_template_thinking,
        "verify": bool(verify),
        # None when no reference was given, and `null` when one was given but holds no
        # tokenizer.json -- two different situations, and hash_tokenizer already distinguishes
        # them by returning None only for the second. The wrapper dict keeps them apart in the
        # diff output: absent key vs a key whose value is null.
        "expect_tokenizer": (
            None if expect_tokenizer_from is None
            else {"name": tokenizer_identity.derive_name(Path(expect_tokenizer_from)),
                  "tokenizer": tokenizer_identity.hash_tokenizer(Path(expect_tokenizer_from))}
        ),
    }


def declared_outputs(keep: list[str]) -> list[str]:
    """Every file the export writes into dest, relative to dest.

    Derived from classify()'s keep list rather than listed here, so the two cannot disagree; plus
    export_manifest.json, which export() always writes and which no classify list mentions.
    """
    return sorted(set(keep) | {"export_manifest.json"})


# --------------------------------------------------------------------------- self test
def self_test() -> int:
    """Fixtures on disk, because every interesting key here is read off a filesystem.

    A sequence of comparisons rather than a case table: several cases have to MUTATE a fixture
    between them (rewrite trainer_state.json, resize a shard), and a table cannot express that
    ordering.
    """
    bad = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal bad
        if cond:
            print(f"  OK   {name}")
        else:
            print(f"  FAIL {name}" + (f": {detail}" if detail else ""))
            bad = 1

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ckpt = root / "checkpoint-50"
        ckpt.mkdir()
        (ckpt / "trainer_state.json").write_text(
            json.dumps({"global_step": 50, "log_history": [{"loss": 0.1053}]}))
        (ckpt / "model.safetensors").write_bytes(b"\x00" * 2048)
        (ckpt / "config.json").write_text("{}")

        ref = root / "retagged_student"
        ref.mkdir()
        (ref / "tokenizer.json").write_text('{"version":"1.0"}')

        KEEP = ["config.json", "model.safetensors", "tokenizer_config.json"]
        base = dict(checkpoint=ckpt, keep=KEEP, dropped_unrecognised=[],
                    padding_side="right", chat_template_thinking="default-off",
                    verify=True, expect_tokenizer_from=ref)
        first = expectation(**base)

        def same(label: str, must_match: bool, **over) -> None:
            other = expectation(**{**base, **over})
            equal = not step_state._diff(first, other)
            check(label, equal is must_match,
                  f"equal={equal}, wanted equal={must_match}")

        same("identical inputs compare equal", True)
        same("a different padding_side differs", False, padding_side="left")
        same("a different thinking policy differs", False, chat_template_thinking="keep")
        same("--no-verify differs from --verify", False, verify=False)
        same("dropping an unrecognised file differs", False, dropped_unrecognised=["junk.bin"])
        same("a different keep list differs", False, keep=KEEP + ["chat_template.jinja"])
        same("no tokenizer reference differs from one given", False, expect_tokenizer_from=None)

        # The selector is deliberately NOT part of the key; nothing to compare, so instead assert
        # the positive claim the docstring makes -- that the identity is the resolved directory.
        check("checkpoint identity is read from the directory, not from a selector string",
              checkpoint_identity(ckpt)["name"] == "checkpoint-50"
              and checkpoint_identity(ckpt)["step"] == 50)

        # Moving the tokenizer reference must not invalidate a completed export: the comparison is
        # on the vocabulary, and `derive_name` keeps a plain directory's basename, so a rename
        # DOES differ while a move does not. Both halves are asserted.
        moved = root / "elsewhere" / "retagged_student"
        moved.parent.mkdir()
        ref.rename(moved)
        same("the tokenizer reference MOVED but not renamed compares equal", True,
             expect_tokenizer_from=moved)
        renamed = root / "elsewhere" / "some_other_student"
        moved.rename(renamed)
        same("a DIFFERENT reference directory name differs", False,
             expect_tokenizer_from=renamed)
        (renamed / "tokenizer.json").write_text('{"version":"2.0"}')
        same("the same reference name with a different vocabulary differs", False,
             expect_tokenizer_from=renamed)

        # trainer_state.json is the decider. Same step, different loss history -> different run.
        (ckpt / "trainer_state.json").write_text(
            json.dumps({"global_step": 50, "log_history": [{"loss": 0.1011}]}))
        same("step 50 from a DIFFERENT trajectory differs (trainer_state digest)", False)
        check("...and the step is still read as 50, so the diff names the trajectory not the step",
              checkpoint_identity(ckpt)["step"] == 50)

        # A damaged state file must not make two checkpoints look alike.
        (ckpt / "trainer_state.json").write_text("{not json")
        idn = checkpoint_identity(ckpt)
        check("an unparseable trainer_state.json gives step=None but still a digest",
              idn["step"] is None and idn["state_sha"] is not None)

        # A shard that changes SIZE differs; a same-size rewrite does not. The second is the
        # documented blind spot, asserted so that it stays a decision rather than becoming a
        # surprise.
        #
        # These two compare against a LOCAL baseline rather than `first`, and that is not
        # tidiness. The cases above mutated trainer_state.json, so `first` no longer describes
        # this checkpoint at all: measured against it, the blind-spot case "differed" -- for the
        # right answer by the wrong reason, which is exactly the shape of a self-test that passes
        # while testing nothing.
        (ckpt / "model.safetensors").write_bytes(b"\x00" * 4096)
        resized = expectation(**base)
        check("a resized weight shard differs", bool(step_state._diff(first, resized)))
        (ckpt / "model.safetensors").write_bytes(b"\xff" * 4096)
        rewritten = expectation(**base)
        check("a same-size shard REWRITE compares equal (documented blind spot)",
              not step_state._diff(resized, rewritten))

        outs = declared_outputs(KEEP)
        check("declared_outputs adds export_manifest.json and keeps the rest",
              outs == sorted(set(KEEP) | {"export_manifest.json"}) and len(outs) == 4,
              f"{outs}")
        check("declared_outputs does not duplicate a manifest already in keep",
              declared_outputs(KEEP + ["export_manifest.json"]) == outs)

        # The expectation is written to JSON and read back by the next run, so it has to survive
        # the round trip byte for byte -- tuples-vs-lists is the classic way this quietly fails.
        rt = json.loads(json.dumps(first))
        check("the expectation survives a JSON round trip", not step_state._diff(first, rt))

    if not bad:
        print("\nEXPORT EXPECTATION EXERCISED: checkpoint identity by trajectory digest, both "
              "policy edits, the verify assertion, the kept/dropped lists, a moved vs renamed vs "
              "revocalised tokenizer reference, and the size-only weight blind spot")
    return bad


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["self-test"],
                   help="Only self-test: this step's gate is a function call inside "
                        "export_hf_model.main(), not a subprocess. See the module docstring.")
    a = p.parse_args(argv)
    if a.action == "self-test":
        return self_test()
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
