"""One answer, for every step, to "has this already been done, and with what?"

WHY THIS EXISTS.

A recipe of seven steps on a preemptable queue is restarted, not resumed, and the restart re-runs
the whole recipe from the top. Each step therefore has to answer the same question about its own
output directory, and today each one answers it differently:

  distill-corpus-prep   refuses if train.jsonl exists (prep-corpus.sh:108). Safe, but a restart
                        after a preemption in step 3 cannot get past step 1 without a human
                        deleting a directory -- and the refusal is identical whether the existing
                        corpus was built with the SAME policies (in which case there is nothing to
                        do) or with different ones (in which case continuing would be wrong).
  distill-gold-train    resumes from a checkpoint glob (see checkpoints.py).
  the rest              overwrite, or fail in whatever way the tool underneath fails.

Three behaviours for one question is two too many, and the interesting distinction -- SAME inputs
versus DIFFERENT inputs -- is the one none of them draws.

WHAT A MARKER IS. A small JSON file, `.step-done.json`, written into the output directory AFTER the
step's real outputs, holding the step's name, the EXPECTATION it was run under (every input and
policy that would change the output), and a record of the outputs it produced. Written last and
atomically, so its presence means the step finished: a marker cannot appear beside a half-written
output, which is exactly the property `checkpoint-*` lacks and the reason gold.py's bare glob needs
checkpoints.py to make up for it.

THREE ANSWERS, and the middle one is the point:

  RUN     no marker. Do the work.
  SKIP    a marker whose expectation MATCHES. The work is done; a restarted recipe walks past it.
  REFUSE  a marker whose expectation DIFFERS, or whose declared outputs are gone. Do not overwrite
          and do not silently rebuild -- say which key changed and stop.

WHY REFUSE RATHER THAN REBUILD when an output has vanished. Rebuilding is the friendlier-looking
choice and the wrong one: the corpus prep step is 0.7 h and the train step is tens of hours, and a
missing output means something happened to the tree that nobody has explained yet. Spending a day
of GPU time on that guess is worse than stopping with a message naming the file. A restart that
should rebuild says so by deleting the marker.

WHY THE EXPECTATION IS SUPPLIED BY THE CALLER and not computed here. What counts as "the same run"
is the step's own business -- prep's answer involves the tokenizer identity and eight policy flags,
export's involves a checkpoint path. This module holds no opinion about any step's inputs; it holds
the comparison, the atomicity, and the vocabulary. That is what makes it one mechanism rather than
seven.

CLI, for the shell steps:

    python step_state.py check --out-dir D --step NAME --expectation-file E.json [--output f ...]
    python step_state.py mark  --out-dir D --step NAME --expectation-file E.json [--output f ...]

Exit codes match checkpoints.py, so a step wrapper branches the same way in both places:
    0   RUN    -- proceed
   64   SKIP   -- already done with this expectation
   65   REFUSE -- something differs; the message says what
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

MARKER_NAME = ".step-done.json"
SCHEMA = 1

# Files at or below this size get a sha256 in the marker; larger ones get a size only. The cap
# exists for the multi-GB artifacts -- a 3.1 GB train.jsonl and 7 GB of safetensors would cost
# minutes per check to digest, on every check, to answer a question a size already answers well
# enough for them. 64 MiB rather than the 4 MiB this started at, because 4 MiB fell on the wrong
# side of the artifacts whose CONTENT is the point: a Granite tokenizer.json is 7.1 MB, and a
# retagged tokenizer with the right size and the wrong vocabulary is precisely the failure the
# retag step exists to prevent. 64 MiB covers every metadata artifact this pipeline writes
# (tokenizers, configs, chat templates, manifests) at ~200 ms worst case, and still skips the
# weights and the corpus.
#
# RAISING IT IS BACKWARD COMPATIBLE, and not by luck: the check at `if rec.get("sha256")` below
# compares a digest only when the MARKER recorded one. So an entry written under the old cap says
# sha256: null and is compared on size alone -- a later check cannot retroactively claim to have
# verified content that was never digested, which is the honest reading and also the one that
# avoids a spurious REFUSE on every marker written before this line changed.
DIGEST_LIMIT = 64 << 20

RUN, SKIP, REFUSE = "RUN", "SKIP", "REFUSE"
EXIT = {RUN: 0, SKIP: 64, REFUSE: 65}


@dataclass
class Decision:
    kind: str
    lines: list[str] = field(default_factory=list)
    marker: dict | None = None


def _describe(root: Path, rel: str) -> dict:
    """Describe one declared output. `rel` is the name the STEP declared, relative to out_dir.

    Recording `rel` and not `path.name` matters as soon as a step's outputs are nested, which
    distill-tokenizer-align's are: it writes teacher_overlay/tokenizer.json,
    student_overlay/tokenizer.json and retagged_student/tokenizer.json, three different files whose
    basenames are identical. Keyed on the basename they would collide in the marker, and the check
    would then stat out_dir/tokenizer.json -- a path that does not exist -- and REFUSE a directory
    that was in fact complete.
    """
    path = root / rel
    st = path.stat()
    d: dict = {"path": rel, "bytes": st.st_size}
    if st.st_size <= DIGEST_LIMIT:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        d["sha256"] = h.hexdigest()
    else:
        # Stated rather than omitted: a reader must be able to tell "this file was not digested"
        # from "this file was digested and matched".
        d["sha256"] = None
        d["digest_skipped_over_bytes"] = DIGEST_LIMIT
    return d


def write_marker(out_dir: Path, step: str, expectation: dict, outputs: list[str]) -> dict:
    """Write the marker LAST and atomically. Returns what was written.

    Atomic via os.replace within the same directory, which is a rename on every filesystem this
    runs on. The reason is the whole contract: a marker that could be observed half-written would
    be no better than the `checkpoint-*` glob it exists to improve on -- a reader would have to
    guess whether the step finished.

    An output named here but absent on disk is an error at MARK time, not a silent omission. A
    step that reports success without producing what it declared is the failure this catches
    earliest, and it costs one stat() per file.
    """
    out_dir = Path(out_dir)
    missing = [o for o in outputs if not (out_dir / o).exists()]
    if missing:
        raise FileNotFoundError(
            f"{step} declared output(s) {missing} that do not exist under {out_dir} -- "
            "refusing to write a completion marker for work that is not there")
    doc = {
        "schema": SCHEMA,
        "step": step,
        "expectation": expectation,
        "outputs": [_describe(out_dir, o) for o in outputs],
    }
    tmp = out_dir / f"{MARKER_NAME}.tmp"
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, out_dir / MARKER_NAME)
    return doc


def _diff(want: dict, got: dict) -> list[str]:
    """Key-by-key, because "the expectation differs" is useless to an operator on its own.

    Nested dicts are walked so that a changed policy reads as `policies.max_length: 4096 -> 16384`
    rather than as two opaque blobs.
    """
    lines: list[str] = []

    def walk(a, b, prefix: str) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                if k not in a:
                    lines.append(f"  {prefix}{k}: (absent before) -> {b[k]!r}")
                elif k not in b:
                    lines.append(f"  {prefix}{k}: {a[k]!r} -> (absent now)")
                else:
                    walk(a[k], b[k], f"{prefix}{k}.")
            return
        if a != b:
            lines.append(f"  {prefix.rstrip('.')}: {a!r} (recorded) -> {b!r} (requested)")

    walk(want, got, "")
    return lines


def decide(out_dir: Path, step: str, expectation: dict,
           outputs: list[str] | None = None) -> Decision:
    """RUN, SKIP or REFUSE, from the marker alone plus a stat of each declared output."""
    out_dir = Path(out_dir)
    marker = out_dir / MARKER_NAME
    if not marker.exists():
        return Decision(RUN, [f"no {MARKER_NAME} under {out_dir} -- running"])
    try:
        doc = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        # Not RUN: a marker that will not parse is a damaged directory, and the outputs beside it
        # may be anything. Naming the file and stopping is the only honest move.
        return Decision(REFUSE, [
            f"{marker} will not parse ({exc.__class__.__name__}: {exc}).",
            "Delete it by hand once you have decided whether the outputs beside it are usable."])
    if not isinstance(doc, dict):
        return Decision(REFUSE, [f"{marker} is not a JSON object"])
    if doc.get("schema") != SCHEMA:
        return Decision(REFUSE, [
            f"{marker} has schema {doc.get('schema')!r}; this code writes {SCHEMA}. "
            "The comparison below would not mean what it says, so it is not attempted."])
    if doc.get("step") != step:
        return Decision(REFUSE, [
            f"{marker} was written by step {doc.get('step')!r}, but {step!r} is asking. "
            "Two steps are sharing one output directory."])

    diff = _diff(doc.get("expectation", {}), expectation)
    if diff:
        return Decision(REFUSE, [
            f"{out_dir} already holds output from {step}, built under a DIFFERENT expectation:",
            *diff,
            "Point --out-dir somewhere else, or delete the directory if the recorded run is "
            "no longer wanted. Nothing here is overwritten."], marker=doc)

    for rec in doc.get("outputs", []):
        p = out_dir / rec["path"]
        if not p.exists():
            return Decision(REFUSE, [
                f"the marker says {step} produced {rec['path']}, and it is not there.",
                "Not rebuilding on a guess: something removed a declared output. Delete "
                f"{MARKER_NAME} to rebuild deliberately."], marker=doc)
        size = p.stat().st_size
        if size != rec["bytes"]:
            return Decision(REFUSE, [
                f"{rec['path']} is {size} bytes; the marker recorded {rec['bytes']}.",
                f"Delete {MARKER_NAME} to rebuild deliberately."], marker=doc)
        if rec.get("sha256"):
            got = _describe(out_dir, rec["path"]).get("sha256")
            if got != rec["sha256"]:
                return Decision(REFUSE, [
                    f"{rec['path']} has the recorded size but a different sha256 "
                    f"({rec['sha256'][:16]} recorded, {str(got)[:16]} now).",
                    f"Delete {MARKER_NAME} to rebuild deliberately."], marker=doc)

    # The DECLARATION itself, which until now was accepted and ignored. `outputs` was a
    # parameter this function took and never read, so a caller passing its current output list --
    # the obvious reading of which is "check these" -- got no check at all: only what the MARKER
    # recorded was ever verified, and a step that grew a new output would SKIP and leave a
    # directory missing it. Neither existing caller could reach that (both derive their declared
    # outputs from data already in the expectation, so a changed declaration shows up as a diff
    # first), which is precisely why a dead parameter is worth closing rather than leaving to be
    # discovered by the third caller.
    if outputs is not None:
        recorded = {r.get("path") for r in doc.get("outputs", [])}
        added = sorted(set(outputs) - recorded)
        removed = sorted(recorded - set(outputs))
        if added or removed:
            return Decision(REFUSE, [
                f"{out_dir} was completed by {step}, but the set of outputs {step} DECLARES has "
                "changed since:",
                *(f"  now declared, not in the marker: {a}" for a in added),
                *(f"  in the marker, no longer declared: {r}" for r in removed),
                "The recorded run is not wrong, it is answering an older question. Point the "
                f"output directory elsewhere, or delete {MARKER_NAME} to rebuild deliberately."],
                marker=doc)

    checked = sum(1 for r in doc.get("outputs", []) if r.get("sha256"))
    sized = len(doc.get("outputs", [])) - checked
    note = f"{checked} digested, {sized} size-only" if sized else f"{checked} digested"
    return Decision(SKIP, [
        f"{step} already completed in {out_dir} under this exact expectation ({note}).",
        "Nothing to do."], marker=doc)


# --------------------------------------------------------------------------- self test
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

    exp = {"dataset": "a.jsonl", "policies": {"max_length": 16384, "length_policy": "drop"}}

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        check("empty dir -> RUN", decide(d, "prep", exp).kind == RUN)

        (d / "train.jsonl").write_text("x\n")
        (d / "corpus_manifest.json").write_text("{}\n")
        write_marker(d, "prep", exp, ["train.jsonl", "corpus_manifest.json"])
        check("marker is written into the output dir", (d / MARKER_NAME).is_file())
        check("no .tmp is left behind", not (d / f"{MARKER_NAME}.tmp").exists())
        check("same expectation -> SKIP", decide(d, "prep", exp).kind == SKIP)

        # The `outputs` argument, which used to be accepted and ignored. Three cases, because the
        # interesting ones are the two DIRECTIONS of a changed declaration -- a grown declaration
        # would otherwise SKIP into a directory missing the new file, and a shrunk one means the
        # marker is vouching for something the step no longer claims to produce.
        DECL = ["train.jsonl", "corpus_manifest.json"]
        check("matching declaration -> still SKIP", decide(d, "prep", exp, DECL).kind == SKIP)
        v = decide(d, "prep", exp, DECL + ["row_ids.parquet"])
        check("a GROWN output declaration -> REFUSE, naming the new output",
              v.kind == REFUSE and any("row_ids.parquet" in l for l in v.lines), f"{v.lines}")
        v = decide(d, "prep", exp, ["train.jsonl"])
        check("a SHRUNK output declaration -> REFUSE, naming the dropped one",
              v.kind == REFUSE and any("corpus_manifest.json" in l for l in v.lines), f"{v.lines}")

        # THE distinction none of the existing steps draws.
        other = json.loads(json.dumps(exp))
        other["policies"]["max_length"] = 4096
        v = decide(d, "prep", other)
        check("a changed policy -> REFUSE, naming the key and both values",
              v.kind == REFUSE and any("policies.max_length: 16384" in l and "4096" in l
                                       for l in v.lines), f"{v.lines}")
        added = json.loads(json.dumps(exp))
        added["policies"]["think_policy"] = "strip"
        v = decide(d, "prep", added)
        check("a NEW key -> REFUSE (absent before)",
              v.kind == REFUSE and any("absent before" in l for l in v.lines), f"{v.lines}")
        removed = {"dataset": "a.jsonl"}
        v = decide(d, "prep", removed)
        check("a REMOVED key -> REFUSE (absent now)",
              v.kind == REFUSE and any("absent now" in l for l in v.lines), f"{v.lines}")

        check("another step asking about the same dir -> REFUSE",
              decide(d, "align", exp).kind == REFUSE)

        # A declared output that vanished must not trigger a silent multi-hour rebuild.
        (d / "train.jsonl").unlink()
        v = decide(d, "prep", exp)
        check("a vanished output -> REFUSE, not RUN",
              v.kind == REFUSE and any("not there" in l for l in v.lines), f"{v.lines}")
        (d / "train.jsonl").write_text("x\n")
        check("restoring it identically -> SKIP again", decide(d, "prep", exp).kind == SKIP)
        (d / "train.jsonl").write_text("xy\n")
        v = decide(d, "prep", exp)
        check("a resized output -> REFUSE", v.kind == REFUSE and any("bytes" in l for l in v.lines))
        (d / "train.jsonl").write_text("y\n")   # same size, different content
        v = decide(d, "prep", exp)
        check("same size, different content -> REFUSE via sha256",
              v.kind == REFUSE and any("sha256" in l for l in v.lines), f"{v.lines}")

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / MARKER_NAME).write_text("{not json")
        v = decide(d, "prep", exp)
        check("an unparseable marker -> REFUSE, not RUN",
              v.kind == REFUSE and any("will not parse" in l for l in v.lines), f"{v.lines}")
        (d / MARKER_NAME).write_text(json.dumps({"schema": 99, "step": "prep", "expectation": exp}))
        check("a future schema -> REFUSE rather than a comparison that means nothing",
              decide(d, "prep", exp).kind == REFUSE)

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        try:
            write_marker(d, "prep", exp, ["train.jsonl"])
        except FileNotFoundError as exc:
            check("marking a declared output that does not exist raises",
                  "refusing to write a completion marker" in str(exc), str(exc))
        else:
            check("marking a declared output that does not exist raises", False)

    # Size-only entries must be distinguishable from digested ones, or a big file would look
    # verified when only its length was compared.
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        big = d / "big.bin"
        # Sparse, via truncate: self_test() runs on EVERY invocation of this module, and now that
        # the cap is 64 MiB, writing the bytes would put 64 MB through the filesystem each time to
        # test a code path that only looks at st_size.
        with big.open("wb") as fh:
            fh.truncate(DIGEST_LIMIT + 1)
        (d / "small.txt").write_text("s")
        doc = write_marker(d, "prep", exp, ["big.bin", "small.txt"])
        recs = {r["path"]: r for r in doc["outputs"]}
        check("a file over the digest limit records sha256 None and says why",
              recs["big.bin"]["sha256"] is None
              and recs["big.bin"]["digest_skipped_over_bytes"] == DIGEST_LIMIT, f"{recs}")
        check("a small file records a real sha256", bool(recs["small.txt"]["sha256"]))
        v = decide(d, "prep", exp)
        check("the SKIP message says how many were digested and how many size-only",
              v.kind == SKIP and "1 digested, 1 size-only" in " ".join(v.lines), f"{v.lines}")

        # A marker written under a SMALLER digest cap says sha256: null for a file that today's cap
        # would digest. It must still be compared on size alone -- otherwise raising the cap would
        # make every pre-existing marker REFUSE, and the check would be claiming to have verified
        # content that was never digested.
        marker = json.loads((d / MARKER_NAME).read_text())
        for rec in marker["outputs"]:
            if rec["path"] == "small.txt":
                rec["sha256"] = None
                rec["digest_skipped_over_bytes"] = 4 << 20
        (d / MARKER_NAME).write_text(json.dumps(marker))
        check("an entry the marker did NOT digest is compared on size alone, so raising "
              "DIGEST_LIMIT cannot invalidate an older marker",
              decide(d, "prep", exp).kind == SKIP, f"{decide(d, 'prep', exp).lines}")
        (d / "small.txt").write_text("S")   # same size, different content, undigested by the marker
        check("...and such an entry therefore cannot catch a same-size rewrite, which is what "
              "recording sha256: null MEANS",
              decide(d, "prep", exp).kind == SKIP)

    # Nested outputs. distill-tokenizer-align writes three files called tokenizer.json in three
    # subdirectories, so a marker keyed on the basename would collide and then look for a file at
    # the top level that was never there.
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        for sub, body in (("teacher_overlay", "T"), ("student_overlay", "S"), ("retagged_student", "R")):
            (d / sub).mkdir()
            (d / sub / "tokenizer.json").write_text(body)
        decl = [f"{s}/tokenizer.json" for s in ("teacher_overlay", "student_overlay", "retagged_student")]
        doc = write_marker(d, "align", exp, decl)
        check("nested outputs keep their declared relative paths, so three same-named files stay "
              "three entries", sorted(r["path"] for r in doc["outputs"]) == sorted(decl),
              f"{[r['path'] for r in doc['outputs']]}")
        check("three distinct digests, not one repeated",
              len({r["sha256"] for r in doc["outputs"]}) == 3)
        check("a nested marker verifies", decide(d, "align", exp).kind == SKIP,
              f"{decide(d, 'align', exp).lines}")
        (d / "retagged_student" / "tokenizer.json").write_text("X")
        v = decide(d, "align", exp)
        check("a rewritten nested output is caught, and named with its subdirectory",
              v.kind == REFUSE and any("retagged_student/tokenizer.json" in l for l in v.lines),
              f"{v.lines}")

    if not bad:
        print("\nSTEP COMPLETION EXERCISED: run/skip/refuse, changed+added+removed expectation "
              "keys, wrong step, vanished/resized/rewritten output, damaged and future markers, "
              "and the digest-limit distinction")
    return bad


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["check", "mark", "self-test"])
    ap.add_argument("--out-dir")
    ap.add_argument("--step")
    ap.add_argument("--expectation-file",
                    help="JSON file holding the expectation. A FILE and not a --expectation string, "
                         "because these hold paths and templates that shell quoting mangles.")
    ap.add_argument("--output", action="append", default=[],
                    help="repeatable; a filename relative to --out-dir")
    a = ap.parse_args(argv)

    if a.action == "self-test":
        return self_test()
    for req in ("out_dir", "step", "expectation_file"):
        if not getattr(a, req):
            ap.error(f"--{req.replace('_', '-')} is required for {a.action}")
    expectation = json.loads(Path(a.expectation_file).read_text())

    if a.action == "mark":
        doc = write_marker(Path(a.out_dir), a.step, expectation, a.output)
        print(f"marked {a.step} complete in {a.out_dir} "
              f"({len(doc['outputs'])} output(s) recorded)")
        return 0

    v = decide(Path(a.out_dir), a.step, expectation, a.output)
    print(f"step {a.step}: {v.kind}")
    for line in v.lines:
        print(f"  {line}")
    return EXIT[v.kind]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
