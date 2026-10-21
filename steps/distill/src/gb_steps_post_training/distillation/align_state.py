"""What "already aligned" means for distill-tokenizer-align, and nothing else.

step_state.py holds the mechanism -- the comparison, the atomic marker, the RUN/SKIP/REFUSE
vocabulary -- and deliberately holds no opinion about any step's inputs. This module is
distill-tokenizer-align's opinion: which facts about the student, the teacher and the chat
template determine the bytes the step writes, and which of its outputs are worth naming.

WHY IT IS A MODULE AND NOT A JSON BLOB IN run-align.sh. The expectation has to be computed
IDENTICALLY at check time and at mark time, or the marker written by a run describes a
different question from the one the next run asks, and the step REFUSES a directory it built
itself an hour earlier. A shell heredoc duplicated at the top and bottom of a script is the
form of that bug that looks correct in review. One function, called twice, cannot drift.

WHAT IS COMPARED, and the reasoning for each:

  student / teacher -- identified by CONTENT, not by path. This follows distill-corpus-prep,
      whose expectation() says it and says why: "a path can be repointed at a retagged
      directory with the same name, and it is the vocabulary that decides". Here it matters
      twice over, because a hub-cached model's path ends in a 40-char commit SHA that changes
      on every re-download of the same weights. Three facts per model:
        name          from tokenizer_identity.derive_name, so a mismatch reads
                      "ibm-granite/granite-4.1-3b-base", not a snapshot SHA. LEGIBILITY only;
                      the two below are what actually decide.
        tokenizer     sha256 of tokenizer.json. The teacher's is the vocabulary the student is
                      retagged ONTO, and the student's decides which of its rows can be reused,
                      moved or transplanted -- i.e. the whole retag plan.
        config        sha256 of config.json. Not redundant with the tokenizer: retag_student
                      reads vocab_size and the bos/eos/pad scheme from HERE (that is why stage
                      3 takes the teacher MODEL and not the teacher overlay), and tie_word_
                      embeddings from here decides whether the lm_head is rewritten at all.

  weights -- a STAT, not a digest: [[shard name, bytes], ...]. The retag rewrites the student's
      embedding rows, so the output genuinely depends on the input weights, and honesty
      requires saying what that costs. Digesting is 6.8 GB of sha256 per check on the real
      student, on every restart, to detect a case that has never happened; a size per shard is
      free and catches a swapped or truncated checkpoint. NOT CAUGHT: a checkpoint rewritten in
      place to exactly the same size in every shard. That is stated here rather than left for a
      reader to discover, and it is the reason `name` is compared too -- a different model is
      almost never also byte-identical in size.

  chat_template -- basename and sha256, never the full path. The template is installed into the
      retagged student verbatim, so its CONTENT is load-bearing (a template with no
      {% generation %} markers yields all-zero assistant_masks and raises at sft.py:909). Its
      LOCATION is not: templates/ is a repo directory that becomes a container path, and a
      relocation must not invalidate an aligned model. `null` when no template was installed,
      which is a different expectation from any template and compares as such.

  copy_mode -- copy vs hardlink. It does not change the bytes a reader sees, and it is compared
      anyway: a hardlinked output shares storage with the source, so "the same alignment" built
      the two ways is not the same artifact to anything that later writes to either.

  verify -- whether the post-conditions ran. Also not a property of the bytes, and also
      compared, in the one direction that matters: a --no-verify run followed by a --verify
      request must not report SKIP, because that would answer "yes, verified" about outputs
      nothing ever checked. It REFUSES and names the key, and the operator either accepts the
      unverified artifact or deletes the marker.

DECLARED OUTPUTS are a SUBSET, on purpose. step_state stats and (under its size cap) digests
every name declared here, so the list is the integrity claim, not an inventory: the three
tokenizer.json files whose agreement is the entire point of the step, the configs the retag
rewrites, the manifests, and each weight shard by size. Files an overlay copies opportunistically
(generation_config.json, special_tokens_map.json) are left out -- declaring a file that a
legitimate source can lack would turn a correct run into a REFUSE.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from gb_steps_post_training.distillation import masking, step_state, tokenizer_identity

STEP_NAME = "distill-tokenizer-align"


def _sha16(path: Path) -> str | None:
    """`sha256:<16 hex>` over a file, or None when it is absent.

    Truncated to 16 hex, matching tokenizer_identity.hash_tokenizer: this is a change detector
    between directories meant to hold the same file, and a 64-char digest makes the REFUSE
    message -- which an operator has to read at 3am -- unreadable for no gain. (The MARKER's own
    digests, written by step_state, are full-length; that is a different question, asked about
    this step's outputs rather than about its inputs.)

    None rather than a raise, so an absent config.json is a comparable FACT. A model directory
    with no config.json cannot be retagged at all, and retag_student says so with an instruction;
    this module's job is to describe, not to gate.
    """
    if not path.is_file():
        return None
    h = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return f"sha256:{h}"


def _weights(model_dir: Path) -> list[list]:
    """Every *.safetensors in `model_dir`, as [name, bytes], sorted by name.

    Sorted because a glob's order is filesystem order, and an expectation that compares unequal
    because a directory was re-created would refuse a correct run. Nested lists rather than a
    dict because JSON round-trips them identically and step_state compares parsed structures.
    """
    return sorted([p.name, p.stat().st_size] for p in model_dir.glob("*.safetensors"))


def model_identity(model_dir: Path) -> dict:
    """The facts about one model directory that decide what the retag produces."""
    d = Path(model_dir)
    return {
        "name": tokenizer_identity.derive_name(d),
        "tokenizer": tokenizer_identity.hash_tokenizer(d),
        "config": _sha16(d / "config.json"),
        "weights": _weights(d),
    }


def expectation(*, student: Path, teacher: Path, chat_template: Path | None,
                copy_mode: str, verify: bool) -> dict:
    """Everything that changes what this step writes, and nothing that does not."""
    return {
        "student": model_identity(student),
        "teacher": model_identity(teacher),
        "chat_template": (None if chat_template is None else
                          {"name": Path(chat_template).name,
                           "sha256": _sha16(Path(chat_template))}),
        "copy_mode": copy_mode,
        "verify": bool(verify),
    }


# The three files whose mutual agreement IS the step. Kept as a constant because the check and
# the mark must name the same set, and because a reader looking for "what does this step
# guarantee" should find a list rather than a comprehension.
_OVERLAY_OUTPUTS = ("tokenizer.json", "tokenizer_config.json", "overlay_manifest.json")
_RETAG_OUTPUTS = ("tokenizer.json", "tokenizer_config.json", "config.json",
                  "generation_config.json", "retag_manifest.json", "tokenizer_identity.json")


def declared_outputs(*, student: Path, chat_template: Path | None) -> list[str]:
    """Output names, relative to out_dir, in the order a reader would want them.

    The weight shards come from the STUDENT directory rather than from the output, and that is
    what makes this callable BEFORE the step runs: retag_student places one output shard per
    input shard under the same name (it rewrites the touched ones and copies the rest), so the
    student's own listing is the prediction. Reading the output instead would make the check
    circular -- it would declare exactly what happens to be there and then assert it is there.
    """
    out = [f"teacher_overlay/{n}" for n in _OVERLAY_OUTPUTS]
    out += [f"student_overlay/{n}" for n in _OVERLAY_OUTPUTS]
    out += [f"retagged_student/{n}" for n in _RETAG_OUTPUTS]
    if chat_template is not None:
        out.append("retagged_student/chat_template.jinja")
        # masking.json is derived FROM the template, so it exists exactly when the template does.
        # Conditional declarations are safe here for the same reason prep_corpus's are: the
        # condition is an ARGUMENT, identical at check time and at mark time, so the two cannot
        # disagree about which files were promised.
        out.append(f"retagged_student/{masking.MASKING_NAME}")
    out += [f"retagged_student/{name}" for name, _ in _weights(Path(student))]
    if (Path(student) / "model.safetensors.index.json").is_file():
        out.append("retagged_student/model.safetensors.index.json")
    return out


def _common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--student-model", required=True)
    ap.add_argument("--teacher-model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--chat-template", default="",
                    help="empty means no template was installed, which is its own expectation")
    ap.add_argument("--copy-mode", choices=("copy", "hardlink"), default="copy")
    ap.add_argument("--verify", dest="verify", action="store_true", default=True)
    ap.add_argument("--no-verify", dest="verify", action="store_false")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["check", "mark", "self-test"])
    _common(ap)
    if argv and argv[0] == "self-test":
        return self_test()
    a = ap.parse_args(argv)

    student, teacher = Path(a.student_model), Path(a.teacher_model)
    template = Path(a.chat_template) if a.chat_template else None
    want = expectation(student=student, teacher=teacher, chat_template=template,
                       copy_mode=a.copy_mode, verify=a.verify)
    outputs = declared_outputs(student=student, chat_template=template)
    out_dir = Path(a.out_dir)

    if a.action == "mark":
        doc = step_state.write_marker(out_dir, STEP_NAME, want, outputs)
        print(f"marked {STEP_NAME} complete in {out_dir} "
              f"({len(doc['outputs'])} output(s) recorded)")
        return 0

    v = step_state.decide(out_dir, STEP_NAME, want, outputs)
    print(f"step {STEP_NAME}: {v.kind}")
    for line in v.lines:
        print(f"  {line}")
    return step_state.EXIT[v.kind]


def self_test() -> int:
    """Exercise the expectation's SENSITIVITY, which is the only property that can be wrong.

    An expectation that never changes reports SKIP forever and a resume silently reuses an
    artifact built from different inputs; one that changes when it should not turns every restart
    into a REFUSE. Both are the same bug -- the wrong set of keys -- so each case below perturbs
    exactly one thing and states which answer it must produce.
    """
    import json
    import tempfile

    bad = 0

    def fake_model(root: Path, *, tok: str, cfg: str, shard: bytes) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        (root / "tokenizer.json").write_text(tok)
        (root / "config.json").write_text(cfg)
        (root / "model.safetensors").write_bytes(shard)
        return root

    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        student = fake_model(t / "s", tok='{"s":1}', cfg='{"vocab_size":10}', shard=b"x" * 100)
        teacher = fake_model(t / "T", tok='{"t":1}', cfg='{"vocab_size":20}', shard=b"y" * 200)
        tpl = t / "chatml.jinja"
        tpl.write_text("{% generation %}")

        base = dict(student=student, teacher=teacher, chat_template=tpl,
                    copy_mode="copy", verify=True)
        want = expectation(**base)

        # A sequence of same() calls rather than a case table, because half the cases have to
        # MUTATE a fixture on disk between them and a table cannot express that ordering.
        def same(label: str, other: dict, must_match: bool) -> None:
            nonlocal bad
            got = expectation(**other) == want
            if got != must_match:
                print(f"  FAIL self-test {label!r}: expected "
                      f"{'the same' if must_match else 'a DIFFERENT'} expectation, got the "
                      f"{'same' if got else 'different'} one")
                bad = 1

        same("identical inputs", dict(base), True)
        same("copy_mode hardlink", {**base, "copy_mode": "hardlink"}, False)
        same("verify off", {**base, "verify": False}, False)
        same("no chat template", {**base, "chat_template": None}, False)

        # A template MOVED but not changed must compare EQUAL: templates/ becomes a container
        # path, and a relocation that invalidated every aligned model would make the marker
        # useless in exactly the environment it exists for.
        moved = t / "elsewhere" / "chatml.jinja"
        moved.parent.mkdir()
        moved.write_text(tpl.read_text())
        same("template moved, same content and name", {**base, "chat_template": moved}, True)

        # ... and a template EDITED in place must compare DIFFERENT, since its content is
        # installed verbatim into the student.
        edited = t / "edited" / "chatml.jinja"
        edited.parent.mkdir()
        edited.write_text("{% generation %}{# one comment later #}")
        same("template edited", {**base, "chat_template": edited}, False)

        # The models: one perturbation each, so a missing key cannot hide behind another.
        (teacher / "tokenizer.json").write_text('{"t":2}')
        same("teacher tokenizer rewritten", dict(base), False)
        (teacher / "tokenizer.json").write_text('{"t":1}')

        (teacher / "config.json").write_text('{"vocab_size":21}')
        same("teacher config rewritten (vocab_size/special ids live here)", dict(base), False)
        (teacher / "config.json").write_text('{"vocab_size":20}')

        (student / "model.safetensors").write_bytes(b"x" * 101)
        same("student shard resized", dict(base), False)
        (student / "model.safetensors").write_bytes(b"x" * 100)

        same("back to the original inputs", dict(base), True)

        # A same-size rewrite is NOT caught, and the docstring says so. Asserting it here is
        # what keeps that sentence true: if someone later digests the weights, this case fails
        # and the docstring gets corrected instead of quietly becoming a lie.
        (student / "model.safetensors").write_bytes(b"z" * 100)
        same("student shard rewritten to the SAME size (documented blind spot)", dict(base), True)
        (student / "model.safetensors").write_bytes(b"x" * 100)

        # declared_outputs must predict from the STUDENT, before the output exists.
        outs = declared_outputs(student=student, chat_template=tpl)
        for need in ("retagged_student/tokenizer.json", "teacher_overlay/tokenizer.json",
                     "student_overlay/tokenizer.json", "retagged_student/chat_template.jinja",
                     "retagged_student/model.safetensors"):
            if need not in outs:
                print(f"  FAIL self-test declared_outputs: {need} not declared. got {outs}")
                bad = 1
        if len(set(outs)) != len(outs):
            print(f"  FAIL self-test declared_outputs: duplicate names, which would make the "
                  f"marker's output list ambiguous. got {outs}")
            bad = 1
        # The three same-named tokenizer.json files are exactly why step_state keys outputs on a
        # RELATIVE PATH; assert all three are present and distinct.
        toks = [o for o in outs if o.endswith("tokenizer.json")]
        if len(toks) != 3:
            print(f"  FAIL self-test declared_outputs: expected 3 distinct tokenizer.json "
                  f"paths, got {toks}")
            bad = 1
        # Both template-conditional outputs, in both directions. masking.json is derived from the
        # template, so declaring it without one would REFUSE a correct --no-chat-template run for
        # a file it was told not to write; NOT declaring it with one would let a resume walk past
        # an align whose masking contract was never emitted, and the gold preflight downstream
        # would then be reading a stale file or none.
        no_tmpl = declared_outputs(student=student, chat_template=None)
        for name in ("retagged_student/chat_template.jinja",
                     f"retagged_student/{masking.MASKING_NAME}"):
            if name in no_tmpl:
                print(f"  FAIL self-test declared_outputs: {name} is declared even though no "
                      "chat template was installed")
                bad = 1
            if name not in outs:
                print(f"  FAIL self-test declared_outputs: {name} is NOT declared when a template "
                      "IS installed, so a resume could skip an align that never wrote it")
                bad = 1

        # And the whole thing must survive a JSON round trip, because that is how it reaches the
        # marker: a tuple that became a list would compare unequal on the next run.
        if json.loads(json.dumps(want)) != want:
            print("  FAIL self-test: the expectation does not survive a JSON round trip")
            bad = 1

    if not bad:
        print("  OK   self-test: expectation is sensitive to both tokenizers, both configs, "
              "shard sizes, template content, copy_mode and verify; insensitive to a template "
              "MOVE; blind to a same-size weight rewrite, as documented")
        print("  OK   self-test: declared_outputs predicts the three tokenizer.json paths and "
              "the shards from the student, and declares chat_template.jinja and "
              f"{masking.MASKING_NAME} exactly when a template is installed")
        print(f"\nALIGN STATE EXERCISED: {STEP_NAME} expectation and declared outputs")
    return bad


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
