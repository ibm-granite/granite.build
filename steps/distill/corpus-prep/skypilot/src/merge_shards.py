#
# PORTED, not authored here. Upstream source of truth:
#   repo   github.ibm.com/Herbert-Woisetschlaeger/gb-steps-collection-post-training
#   path   steps/distill-corpus-prep/src/merge_shards.py
#   commit 70c1550a171aa8e09a9ad9047a5bf763c39e8579
#
# Verbatim apart from `black`/`isort` reflow, which CI requires repo-wide. Keep it that
# way so re-syncing upstream stays a three-way merge; behaviour changes belong upstream.
# The ONE intentional divergence is the empty-corpus guard at the top of merge(): upstream
# divides by the kept count with no zero check, so a shard set that dropped every row
# crashed with ZeroDivisionError after the full lockstep pass.
#
# It imports gb_steps_post_training.distillation at module scope, which is delivered at
# RUN time from the checkout named by code_config (see step-template.yaml). That is why
# the tests for this file are gated on GB_DISTILL_CODE_DIR — see test/conftest.py.
#
"""Merge K sharded prep_corpus runs into one corpus, byte-identically.

WHY BYTE-IDENTICAL IS THE CONTRACT AND NOT JUST A NICETY. A merged corpus is only useful if
"prepped sharded" and "prepped in one process" mean the same thing -- otherwise sharding
introduces a second corpus definition, and every count in the manifest, every row id, and
every downstream provenance claim acquires a "which way was it prepped?" caveat that nobody
will remember to ask. Stating equality as the contract makes the whole question a diff, which
a test can answer, rather than an argument about distributions, which it cannot.

It is achievable because assignment is modulo over the input stream: shard s holds input
records s, s+K, s+2K, ..., each in input order. So reading the K per-shard sidecars in
LOCKSTEP -- one record from shard 0, one from shard 1, ..., one from shard K-1, repeat --
reconstructs the global input order exactly, with no index and no sort. The sidecar has one
entry per assigned record (prep_corpus asserts this before it publishes), so the lockstep
never has to guess whether a shard skipped a row.

The kept rows then follow from the sidecar rather than being matched to it: within a shard,
train.jsonl holds exactly the kept records in input order, so walking that shard's sidecar and
pulling the next line from its train.jsonl on each `disposition == "kept"` entry pairs them up
without reading a single record body. That matters at scale -- the merge stays O(1) in memory
over a 24 GB corpus, and it never parses the rows it is copying.

WHAT IS RECOMPUTED RATHER THAN COPIED, and it is the part worth reviewing:
  - `kept_index` is SHARD-LOCAL in the inputs and must become global, or two rows from
    different shards claim the same index.
  - the means in `token_stats` are recomputed from the summed totals and the summed kept
    count. Averaging K rounded per-shard means would be wrong twice over -- unequal shard
    sizes, and each input already rounded to one decimal.
  - `max_tokens` is the max of the maxes, which is the one aggregate that composes trivially.
  - `drop_reasons`, `transformations` and the counts are summed per key.
  - `template_errors` keeps the FIRST example per error type in shard order. It is a sample
    for a human to read, not a count, so an arbitrary-but-deterministic pick is honest; the
    counts that decisions get made on live in `drop_reasons`.

WHAT IS REFUSED. Anything where a merge would have to invent an answer: a shard set that does
not agree on tokenizer identity, chat template, dataset or policies (those are not shards of
one corpus, they are different corpora); a missing or duplicated shard index; a nonzero
eval_fraction (prep_corpus refuses to draw one under sharding, so seeing one here means a
shard was prepped by a different code path); and an output directory that already holds a
corpus, because a half-overwritten merge is indistinguishable from a complete one.

WHAT IS SKIPPED, which is the one thing that USED to be refused. Re-merging the SAME shards into
a directory a previous merge marked complete exits 0 without rewriting a byte. The old code
refused that too, and refusing it meant a recipe preempted during TRAINING could not restart
without a human deleting a correct 24 GB corpus -- the identical defect prep_corpus's AlreadyDone
was introduced to fix one file over. The distinction is drawn from `.step-done.json` and the
shard digests recorded in it (see expectation()), never from the mere presence of train.jsonl,
and with no marker present the old refusal stands unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The shared step-completion contract. Vendored into this step's image already -- prep_corpus
# imports it from the same place (Dockerfile:45-46), so this adds no build surface.
from gb_steps_post_training.distillation import step_state  # noqa: E402
from prep_corpus import MANIFEST_NAME, ROWS_NAME, AlreadyDone, PrepError  # noqa: E402

# Every key that has to agree across shards for them to be shards of ONE corpus. Kept as data
# rather than an if-chain so that a key added to the manifest is a one-line change here, and so
# the error can name exactly what disagreed.
MUST_AGREE = (
    "tokenizer_identity",
    "chat_template_source",
    "template_renders_documents",
    "format",
    "tokenized",
    "dataset",
    "dataset_split",
    "dataset_config",
    "policies",
    "seed",
    "eval_fraction",
)


STEP_NAME = "distill-corpus-prep/merge"

# The three files a merge writes. No eval.jsonl: a merge refuses a nonzero eval_fraction, so a
# merged corpus never has one, and declaring a file that can never exist would refuse every run.
MERGE_OUTPUTS = ["train.jsonl", ROWS_NAME, MANIFEST_NAME]


def expectation(mans: list[dict]) -> dict:
    """What "this merge has already been done" means. Content, not paths.

    WHY THIS EXISTS AT ALL, given that the exists-check below already refused an occupied
    out-dir. That refusal is correct about the danger and wrong about the common case, and it is
    the same mistake prep_corpus's AlreadyDone was introduced to fix one file over: it gives the
    SAME answer to "this identical merge is already here" and to "a different corpus is here, do
    not touch it". On the preemptable queue the first case is the normal one -- a recipe
    restarted after a preemption in a LATER step re-runs the merge from the top -- and answering
    it with exit 1 means the pipeline cannot restart without a human deleting a correct 24 GB
    corpus. So: SKIP when the shards and the merge are the same, REFUSE when they are not, and
    keep the old refusal for the case with no marker at all.

    WHAT IDENTIFIES A MERGE. It is a deterministic function of the shard BYTES, so:

      shards -- per shard, ordered by index: the index, the sidecar's sha256, its entry count,
          and train.jsonl's size. `rows.sha256` is the strong one and it is FREE: prep_corpus
          already computed it (prep_corpus.py:806-809) over every record's id and disposition,
          so it fingerprints exactly what the lockstep merge reads. train.jsonl's size is the
          cheap complement, because the sidecar covers dispositions and ids but not the row
          BODIES the merge copies. NOT digested: that is 24 GB of sha256 per restart to catch a
          shard rewritten in place to exactly the same length, and this is stated rather than
          left for a reader to infer.

      corpus -- tokenizer_identity, dataset, policies and seed from the reference shard.
          REDUNDANT with the digests above and included anyway, for the same reason align_state
          carries a model `name`: when this refuses, an operator needs to read "a different
          dataset" rather than "shard 3's sidecar digest moved". Legibility is a real
          requirement of a refusal, not a decoration.

      the shard COUNT, via the per-shard list's length. Merging 8 shards and merging the same
          8 plus a 9th are different merges even when the first 8 are byte-identical.

    WHAT IS OUT:

      --force -- a permission, not an input. Two merges of the same shards produce the same
          bytes whether or not force was passed, so a forced re-run of an identical merge should
          SKIP rather than redo 24 GB. Force still does its real job: it overrides the
          no-marker exists-check below.

      the shard DIRECTORY PATHS -- recorded in the manifest's merged_from for provenance and
          deliberately not compared. Shards get staged, copied between filesystems and renamed;
          the bytes are what decide the output.
    """
    ref = mans[0]
    return {
        # Keys in SORTED order, which is not cosmetic. step_state renders a refusal as
        # "(recorded) -> (requested)", the recorded side comes back through JSON with its keys
        # sorted and the requested side renders in insertion order -- so an unsorted literal makes
        # two IDENTICAL shards print with their keys in different positions, and the reader has to
        # diff four fields by eye to find the one changed hex. Job 1161745's leg 4c showed exactly
        # that. An operator reads this message under preemption pressure.
        "shards": [
            {
                "index": m["shard"]["index"],
                "rows_entries": m["rows"]["entries"],
                "rows_sha256": m["rows"]["sha256"],
                "train_bytes": (m["_dir"] / "train.jsonl").stat().st_size,
            }
            for m in mans
        ],
        "corpus": {
            "tokenizer_identity": ref.get("tokenizer_identity"),
            "dataset": ref.get("dataset"),
            "policies": ref.get("policies"),
            "seed": ref.get("seed"),
        },
    }


def load_manifests(shard_dirs: list[Path]) -> list[dict]:
    """Read and cross-validate the shard manifests. Returns them ordered BY SHARD INDEX, which
    is what the lockstep merge depends on -- not by the order the paths were given."""
    mans = []
    for d in shard_dirs:
        f = d / MANIFEST_NAME
        if not f.is_file():
            raise PrepError(
                f"{d} has no {MANIFEST_NAME} -- an unfinished shard cannot be "
                "merged, and a merge that skipped it would look complete"
            )
        m = json.loads(f.read_text())
        m["_dir"] = d
        mans.append(m)

    k = len(mans)
    seen: dict[int, Path] = {}
    for m in mans:
        sh = m.get("shard")
        if not isinstance(sh, dict) or "index" not in sh or "count" not in sh:
            raise PrepError(
                f"{m['_dir']}/{MANIFEST_NAME} has no `shard` block. It was written "
                "by a prep_corpus without sharding support, so its rows cannot be "
                "placed in the global order."
            )
        if sh["count"] != k:
            raise PrepError(
                f"{m['_dir']} says shard count {sh['count']} but {k} directories "
                "were given. Either a shard is missing or one belongs to a "
                "different run -- both would silently drop rows."
            )
        if sh["index"] in seen:
            raise PrepError(
                f"shard index {sh['index']} appears twice: {seen[sh['index']]} and "
                f"{m['_dir']}. Merging both would duplicate those rows."
            )
        seen[sh["index"]] = m["_dir"]
    missing = sorted(set(range(k)) - set(seen))
    if missing:
        raise PrepError(
            f"shard indices {missing} are missing from the {k} directories given"
        )

    ref = mans[0]
    for m in mans[1:]:
        for key in MUST_AGREE:
            if m.get(key) != ref.get(key):
                raise PrepError(
                    f"shards disagree on {key!r}: {ref['_dir'].name} has "
                    f"{json.dumps(ref.get(key))[:120]}, {m['_dir'].name} has "
                    f"{json.dumps(m.get(key))[:120]}. These are not shards of one corpus."
                )
    if ref.get("eval_fraction"):
        raise PrepError(
            f"eval_fraction is {ref['eval_fraction']} -- prep_corpus refuses to draw an eval "
            "split under sharding, so this manifest came from a different code path and the "
            "merge cannot know which rows were held out."
        )
    return sorted(mans, key=lambda m: m["shard"]["index"])


def verify_shards(mans: list[dict]) -> list[str]:
    """Each shard's sidecar must match the digest its OWN manifest records. Returns one line per
    shard for the log; raises PrepError on a mismatch.

    WHY THIS IS NOT REDUNDANT WITH THE RESUME GATE, which is the interesting part. The gate's
    expectation keys on `rows.sha256` read out of each shard's manifest -- a SELF-REPORT. Job
    1161579 measured what that means: perturbing a byte in shard 1's corpus_rows.jsonl changed
    nothing the gate could see, because the manifest still claimed the old digest, so the merge
    skipped and reported a corpus built from a file that no longer existed in that form. The gate
    was working exactly as designed; the design trusted the wrong thing.

    So the claim is CHECKED here, before `expectation()` is computed, and the ordering matters:
    an expectation must never be built from a self-report already known to be false.

    Cost is small and worth naming, because "digest the inputs" is the kind of line that quietly
    becomes an hour. The sidecar is ~90 bytes/row -- ~9 MB per shard at the full corpus's 811,172
    rows over 8 shards, ~73 MB total, seconds. train.jsonl is deliberately NOT digested: it is
    24 GB, and its size is already in the expectation, which catches every truncation and every
    append. A same-length rewrite of train.jsonl remains a known blind spot, stated rather than
    papered over.
    """
    lines = []
    for m in mans:
        d = m["_dir"]
        claimed = (m.get("rows") or {}).get("sha256")
        entries = (m.get("rows") or {}).get("entries")
        f = d / ROWS_NAME
        if claimed is None:
            raise PrepError(
                f"{d}/{MANIFEST_NAME} records no rows.sha256, so its sidecar cannot be checked "
                "and the merge would key its resume marker on an unverifiable claim. Re-prep this "
                "shard with a prep_corpus that emits one."
            )
        if not f.is_file():
            raise PrepError(
                f"{d} has a manifest but no {ROWS_NAME} -- the per-datapoint "
                "provenance the manifest claims is not there"
            )
        h = hashlib.sha256()
        with f.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        got = h.hexdigest()
        if got != claimed:
            raise PrepError(
                f"{f} does not match the digest its own manifest records: sha256 {got[:16]} on "
                f"disk, {claimed[:16]} claimed. The file changed after prep wrote it. Nothing "
                "downstream can tell which version the merged corpus_rows.jsonl came from, so "
                f"re-prep shard {m['shard']['index']} rather than merging this."
            )
        lines.append(
            f"shard {m['shard']['index']}: {ROWS_NAME} verified, {entries} entries, "
            f"sha256 {got[:12]}"
        )
    return lines


def merge(shard_dirs: list[Path], out_dir: Path, *, force: bool = False) -> dict:
    mans = load_manifests(shard_dirs)
    k = len(mans)

    # Before the lockstep pass, not after it: the shards already say how many rows they kept,
    # and an empty corpus is a prep outcome to report, not a division to crash on.
    if sum(m["counts"]["kept"] for m in mans) == 0:
        dropped = {}
        for m in mans:
            for reason, n in (m["counts"].get("drop_reasons") or {}).items():
                dropped[reason] = dropped.get(reason, 0) + n
        raise PrepError(
            f"no rows kept across {k} shards ({sum(m['counts']['input'] for m in mans)} "
            f"input records); drop reasons: {json.dumps(dropped, sort_keys=True)}. "
            "There is no corpus to merge."
        )

    # BEFORE expectation(), so the resume marker is never keyed on a claim known to be false.
    for line in verify_shards(mans):
        print(f"  {line}")

    out_dir.mkdir(parents=True, exist_ok=True)
    train_out, rows_out = out_dir / "train.jsonl", out_dir / ROWS_NAME

    # ---- resume. Asked BEFORE the exists-check below, because the marker can distinguish the
    # two cases that check conflates, and asked AFTER load_manifests because the shard digests
    # are what "this merge" means.
    want = expectation(mans)
    verdict = step_state.decide(out_dir, STEP_NAME, want, MERGE_OUTPUTS)
    if verdict.kind == step_state.SKIP:
        # AlreadyDone, not a return: merge() returns a freshly built manifest and a caller cannot
        # tell "built" from "found" by looking at one. prep_corpus draws the same distinction
        # with the same exception for the same reason, so a reader meets one idiom, not two.
        raise AlreadyDone(
            json.loads((out_dir / MANIFEST_NAME).read_text()), verdict.lines
        )
    if verdict.kind == step_state.REFUSE:
        raise PrepError("\n  ".join(verdict.lines))

    # An existing corpus is not overwritten by default. A merge that half-replaced one would
    # leave a directory that looks finished, with rows from two runs and a manifest describing
    # neither -- and row ids that a training run may already have recorded.
    #
    # STILL HERE, and it is not made redundant by the gate above. decide() returns RUN when there
    # is no marker at all, which is exactly the state of every corpus merged before the marker
    # existed and of any directory a human assembled by hand. Deleting this check would turn the
    # resume feature into a silent overwrite of precisely the output it was added to protect.
    for f in (train_out, rows_out, out_dir / MANIFEST_NAME):
        if f.exists() and not force:
            raise PrepError(
                f"{f} already exists, and there is no {step_state.MARKER_NAME} "
                "recording which merge produced it. Refusing to overwrite: a run may "
                "already reference these rows. Use --force only if you know it does "
                "not."
            )

    sidecars = [(m["_dir"] / ROWS_NAME).open() for m in mans]
    trains = [(m["_dir"] / "train.jsonl").open() for m in mans]
    n_rows = n_kept = 0
    try:
        with rows_out.open("w") as rfh, train_out.open("w") as tfh:
            # Lockstep. `exhausted` tracks which shards are finished: shard sizes differ by at
            # most one (the input length need not be a multiple of K), so the loop must keep
            # draining the others rather than stopping at the first short shard.
            exhausted = [False] * k
            while not all(exhausted):
                for s in range(k):
                    if exhausted[s]:
                        continue
                    line = sidecars[s].readline()
                    if not line:
                        exhausted[s] = True
                        continue
                    entry = json.loads(line)
                    if entry.get("disposition") == "kept":
                        row = trains[s].readline()
                        if not row:
                            raise PrepError(
                                f"shard {s}'s sidecar says a record was kept but its "
                                f"train.jsonl is exhausted after {n_kept} rows. The pair was "
                                "not written by the same run."
                            )
                        tfh.write(row if row.endswith("\n") else row + "\n")
                        # Shard-local -> global. Every consumer of the sidecar keys on this.
                        entry["kept_index"] = n_kept
                        n_kept += 1
                    rfh.write(
                        json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    n_rows += 1
            for s in range(k):
                if trains[s].readline():
                    raise PrepError(
                        f"shard {s}'s train.jsonl has rows its sidecar does not account for. "
                        "The merged corpus would be missing them, so it is not published."
                    )
    finally:
        for fh in sidecars + trains:
            fh.close()

    ref = mans[0]
    tot_in = sum(m["counts"]["input"] for m in mans)
    if n_rows != tot_in:
        raise PrepError(
            f"merged sidecar has {n_rows} entries for {tot_in} input records "
            "across the shards -- rows were lost in the merge"
        )
    tot_kept = sum(m["counts"]["kept"] for m in mans)
    if n_kept != tot_kept:
        raise PrepError(f"merged {n_kept} kept rows, shards report {tot_kept}")

    def sum_key(section: str, key: str) -> int:
        return sum(m[section].get(key, 0) for m in mans)

    def sum_dicts(section: str, key: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for m in mans:
            for kk, vv in (m[section].get(key) or {}).items():
                out[kk] = out.get(kk, 0) + vv
        return dict(sorted(out.items()))

    def first_of(section: str, key: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for m in mans:  # shard order -> deterministic pick
            for kk, vv in (m[section].get(key) or {}).items():
                out.setdefault(kk, vv)
        return dict(sorted(out.items()))

    total_tokens = sum_key("token_stats", "total_tokens")
    total_target = sum_key("token_stats", "total_target_tokens")
    manifest = {kk: ref[kk] for kk in MUST_AGREE}
    manifest["tokenizer_path"] = ref["tokenizer_path"]
    # The merged corpus is whole, so it is shard 0 of 1 -- the same thing an unsharded run
    # writes. `merged_from` keeps the fact that it was produced in pieces, without letting
    # `shard` claim this is a piece.
    manifest["shard"] = {"index": 0, "count": 1}
    manifest["merged_from"] = {
        "shards": k,
        "dirs": [str(m["_dir"].resolve()) for m in mans],
        "row_ids_unchanged": True,
    }
    manifest["counts"] = {
        "input": tot_in,
        "rendered": sum_key("counts", "rendered"),
        "kept": tot_kept,
        "truncated": sum_key("counts", "truncated"),
        "dropped": sum_key("counts", "dropped"),
        "drop_reasons": sum_dicts("counts", "drop_reasons"),
    }
    for name, fn in (("transformations", sum_dicts), ("template_errors", first_of)):
        got = fn("counts", name)
        if got:
            manifest["counts"][name] = got
    manifest["token_stats"] = {
        "total_tokens": total_tokens,
        "total_target_tokens": total_target,
        "max_tokens": max(m["token_stats"]["max_tokens"] for m in mans),
        # Recomputed from the totals, NOT averaged from the shards' rounded means.
        "mean_tokens": round(total_tokens / tot_kept, 1),
        "mean_target_tokens": round(total_target / tot_kept, 1),
        "total_mask_tokens_all_assistant": sum_key(
            "token_stats", "total_mask_tokens_all_assistant"
        ),
    }
    manifest["splits"] = {
        "train": {"path": str(train_out.resolve()), "examples": n_kept}
    }

    h = hashlib.sha256()
    with rows_out.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    manifest["rows"] = {
        "path": str(rows_out.resolve()),
        "sha256": h.hexdigest(),
        "entries": n_rows,
        "row_id_scheme": ref["rows"]["row_id_scheme"],
        "id_field": ref["rows"]["id_field"],
        "training_rows": n_kept,
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    # LAST, and atomically, for the same reason prep_corpus marks last: the marker is what a
    # restarted recipe reads to decide whether to walk past this merge, so it must not be able to
    # exist beside a half-written train.jsonl. It goes after the manifest, which itself goes after
    # the sidecar and train.jsonl.
    step_state.write_marker(out_dir, STEP_NAME, want, MERGE_OUTPUTS)
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--shard-dir",
        action="append",
        default=[],
        help="a shard's out-dir. Repeat once per shard. Order is irrelevant: the "
        "merge orders by each manifest's shard.index.",
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--force", action="store_true", help="overwrite an existing corpus in --out-dir"
    )
    args = p.parse_args(argv)

    dirs = [Path(d) for d in args.shard_dir]
    if len(dirs) < 2:
        print(
            f"FATAL --shard-dir given {len(dirs)} time(s); a merge needs at least 2",
            file=sys.stderr,
        )
        return 2
    try:
        m = merge(dirs, Path(args.out_dir), force=args.force)
    except AlreadyDone as exc:
        # rc 0. A recipe restarted after a preemption in a later step must be able to walk past
        # this one, and "the merge you asked for is already here" is a success, not a failure.
        # The summary below is the same shape as the built-it case so a log reader compares them
        # without having to notice which branch ran.
        print(f"=== already merged -> {args.out_dir}")
        for line in exc.lines:
            print(f"  {line}")
        c = exc.manifest["counts"]
        print(f"  input {c['input']:,}  kept {c['kept']:,}  dropped {c['dropped']:,}")
        print(
            f"  sidecar {exc.manifest['rows']['entries']:,} entries, sha256 "
            f"{exc.manifest['rows']['sha256'][:16]}"
        )
        print(
            f"  delete {Path(args.out_dir) / step_state.MARKER_NAME} to re-merge deliberately"
        )
        return 0
    except PrepError as exc:
        print(f"FATAL {exc}", file=sys.stderr)
        return 1
    c = m["counts"]
    print(f"merged {m['merged_from']['shards']} shards -> {args.out_dir}")
    print(
        f"  input {c['input']:,}  kept {c['kept']:,}  dropped {c['dropped']:,} "
        f"{c['drop_reasons'] or ''}"
    )
    print(
        f"  mean tokens {m['token_stats']['mean_tokens']}  max {m['token_stats']['max_tokens']:,}"
    )
    print(
        f"  sidecar {m['rows']['entries']:,} entries, sha256 {m['rows']['sha256'][:16]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
