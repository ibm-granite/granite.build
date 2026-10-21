"""Run the teacher ONCE over a corpus and keep its top-K logits per assistant token.

WHAT THIS IS FOR. Off-policy KD needs the teacher's distribution at every token the student is
trained on. Holding a 30B teacher in memory next to the student for a multi-day run costs GPUs
that could be running the student, and it recomputes the same forward pass every epoch. This step
pays for it once: a forward pass over the corpus, `top_k` logits and their token ids per assistant
position, written as raw binary shards plus an index that maps a corpus row to (shard, offset).

WHAT IT IS NOT. It is not a route to on-policy GOLD and cannot become one. On-policy means the
STUDENT generates and the teacher scores text that does not exist until training time; there is
nothing to precompute. This is an optimization of the OFF-POLICY path only.

ITS CONSUMER is sft.py's KD path (`--precomputed-logits-dir`), which
  * reads meta.json and REFUSES if `top_k` != its own `kd_top_k` (sft.py:1403),
  * loads meta.json's `tokenizer_name_or_path` and runs verify_tokenizer_consistency against the
    training tokenizer (sft.py:1407-1424) -- so that field must name the tokenizer this pass
    ACTUALLY used, which is why --teacher-tokenizer exists (see below),
  * reads index.jsonl and drops `skipped` rows,
  * re-tokenizes each row and RAISES if the assistant-token count disagrees with the index's
    `num_assistant_tokens` (sft.py:1063) -- the cross-check that catches a template change
    between this pass and training,
  * memmaps `shards/indices_%06d.bin` and `shards/logits_%06d.bin` at
    (`shard_offset`, `assistant_logits_used`).
Every field name below that the consumer reads is therefore FIXED by that code, not by taste.

HOW IT DIFFERS FROM an earlier exploratory version of this same script, which is what this
was ported from. Each difference is a defect found while porting, not a preference:

  1. --teacher-tokenizer, separate from --teacher-model. The original took one --teacher_model and
     used it for both the weights and the tokenizer, then wrote `tokenizer_name_or_path:
     args.teacher_model` into meta.json. That is the `tokenizer_class` trap of
     docs/tokenizer_mismatch.md in the most expensive possible place: a whole corpus mis-segmented,
     discovered days later at training time. distill-tokenizer-align exists to produce the
     tokenizer this pass should use (its `teacher_overlay`), and now it can be handed one.
     meta.json records the tokenizer that was really used.

  2. RESUME TRUNCATES THE OPEN SHARD. This is the real bug the port found. The writer appends
     `n_assist` rows to `shards/*.bin` and THEN writes the index line naming
     (shard_id, shard_offset). Killed between the two -- which on `preemptable` is not a corner
     case -- the .bin files carry rows the index does not mention. The original then resumed by
     deriving `shard_offset` from the INDEX and reopening the files in "ab" mode, i.e. writing at
     end-of-file. Every subsequent row in that shard is then stored at a byte offset the index
     disagrees with, by exactly the orphan row count: the trainer reads a different token's
     teacher distribution for the rest of that shard, silently, and the loss still looks fine.
     _reconcile_shards() truncates each of this node's shard files to the length the index
     implies before the first append, and refuses if a file is SHORTER than the index claims
     (that direction is not recoverable and means something else is wrong).

  3. A COMPLETENESS POST-CONDITION. The output of a partial pass is structurally valid: shards +
     index + meta describe whatever was written, so a pass that covered a tenth of the corpus
     still loads and still trains. Nothing in the original noticed. After the merge, every corpus
     row must appear in the index exactly once -- as precomputed or as explicitly skipped -- and
     the skip fraction must be under --max-skip-fraction. Both are checked, and the shard
     inventory is verified against the index (byte lengths, contiguity, no gaps or overlaps).
     verify_output() is also reachable on its own via --verify-only, so a restarted recipe can
     re-check a directory it is about to skip instead of trusting a marker.

  4. AN EXPECTATION FILE. `.precompute-expectation.json` in the output dir records the inputs and
     policies that determine the artifact (corpus md5, teacher, tokenizer hash, top_k, max_length,
     dtype, response template, ignore_documents). Resuming into a directory whose expectation
     differs REFUSES and names the key. Without it, resume is "append to whatever is there", and
     pointing a second corpus at the same output dir produces one index over two corpora with no
     error anywhere. --emit-expectation writes it without running anything, so the launcher can
     compute the corpus md5 ONCE and reuse it for its own step_state gate.
     `batch_size` and `seed` are deliberately NOT in it: after an OOM you want to resume the same
     artifact at a smaller batch size. (Strictly, batched padding can perturb attention numerics
     at the last bit; that is accepted here and recorded so nobody has to rediscover the choice.)

  5. Only this node's slice of the corpus is held in memory. The original loaded the whole JSONL
     on every rank: 8 ranks x a multi-GB corpus per node, of which each node uses 1/num_nodes.

  6. datetime.utcnow() -> timezone-aware now(); git provenance via code_provenance.describe()
     (which never raises and records a dirty tree, which a bare `git rev-parse HEAD` does not).

WHAT WAS KEPT DELIBERATELY, because it is right and non-obvious:
  * TP within a node, DP across nodes, over a 2-D device mesh. transformers extracts mesh["tp"]
    and only uses that submesh, so the "dp" axis costs nothing and lets each node own a data slice.
  * The lm_head PROBE. Under tp_plan="auto" a column-sharded lm_head would give each rank a slice
    of the vocabulary, and topk over a slice returns rank-local ids that mean nothing. The probe
    compares the projection's output width against vocab_size and refuses rather than writing
    plausible garbage.
  * Capturing lm_head's INPUT with a pre-hook and projecting only at assistant positions, in
    chunks. The [B, T, V] logits tensor is never materialized; at V=100k and T=8192 it would not
    fit, and 99% of its rows would be discarded anyway.
  * `del model` before the cross-node barrier. init_device_mesh only sets up subgroup metadata;
    the world communicator's buffers are allocated on FIRST USE, and with ~40 GB/rank of teacher
    resident that allocation OOMs.
  * Deleting the teacher does NOT happen before the per-node writes, so the writer path still has
    the model's lm_head. That ordering is load-bearing and easy to "clean up" wrongly.

SHARD IDS are `node_id * MAX_SHARDS_PER_NODE + local_idx`, which caps this at MAX_SHARDS_PER_NODE
shards per node and 999 nodes at the current constant. Both are real ceilings and both raise.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import socket
import sys

# code_provenance and tokenizer_identity are stdlib-only, so build_expectation() and
# verify_output() stay importable with nothing installed. utils is NOT: it pulls torch,
# transformers and accelerate at module level, so its two verifiers are imported inside
# run_precompute() with the rest of the heavy stack.
from gb_steps_post_training.distillation import code_provenance, tokenizer_identity

STEP_NAME = "distill-logit-precompute"

# Per-node shard-id namespace. The id encodes the node so that nodes never contend for a file
# name, which is what lets each node write its own shards with no coordination at all.
MAX_SHARDS_PER_NODE = 1000

# The stored dtypes. CONSTANTS, not --dtype: --dtype is the dtype the TEACHER is loaded in.
# meta.json records both, because a config key called `dtype` sitting next to a stored dtype that
# ignores it is exactly the kind of thing that gets misread once and believed forever.
LOGITS_DTYPE = "float16"
LOGITS_ITEMSIZE = 2
INDICES_DTYPE = "int32"
INDICES_ITEMSIZE = 4

EXPECTATION_NAME = ".precompute-expectation.json"

# lm_head is applied to this many assistant positions at a time. Bounds the transient
# [chunk, vocab] logits tensor, which at vocab 100k and float32 accumulation is the largest
# allocation in the loop.
LM_HEAD_CHUNK = 4096

# Named here, resolved to torch objects inside run_precompute(): torch is NOT imported at module
# level, so that --emit-expectation and --verify-only -- both pure file work, and both run in the
# launcher's preflight where a CUDA init would be actively unwelcome -- cost neither the import nor
# a GPU context. Same reason export_hf_model.py defers its transformers import.
DTYPE_NAMES = ("bfloat16", "float16", "float32")


# --------------------------------------------------------------------------- args


def get_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Hyphenated flags, unlike the earlier version's underscored ones: this is what every
    # other entrypoint in this collection accepts, and the step launcher passes them through
    # unchanged instead of translating -- one fewer place for a name to drift.
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--teacher-model", required=True, help="Directory of teacher WEIGHTS.")
    p.add_argument(
        "--teacher-tokenizer",
        default="",
        help="Directory of the teacher TOKENIZER; defaults to --teacher-model. Pass "
             "distill-tokenizer-align's teacher_overlay. Recorded in meta.json as "
             "tokenizer_name_or_path, which is what the training-time consumer compares against.",
    )
    p.add_argument("--top-k", type=int, default=256)
    p.add_argument(
        "--max-length", type=int, default=8192,
        help="Rows whose rendered length exceeds this are SKIPPED, not truncated (they appear in "
             "index.jsonl with skipped=true). A skip is a silent corpus reduction, so see "
             "--max-skip-fraction.",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--dtype", choices=sorted(DTYPE_NAMES), default="bfloat16",
                   help=f"Dtype the TEACHER is loaded in. Stored logits are always "
                        f"{LOGITS_DTYPE} and stored ids {INDICES_DTYPE}.")
    p.add_argument("--shard-target-tokens", type=int, default=4_000_000,
                   help="Soft cap on rows per shard file. Affects file count, nothing else.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ignore-documents", action=argparse.BooleanOptionalAction, default=False,
                   help="Drop each row's `documents` before rendering (RAG-free rendering).")
    p.add_argument(
        "--response-template", default="<|im_start|>assistant\n",
        help="Fallback assistant-span marker, used only when the chat template carries no "
             "{% generation %} markers. Recorded in meta.json with its token ids.",
    )
    p.add_argument(
        "--max-skip-fraction", type=float, default=0.05,
        help="Refuse if more than this fraction of corpus rows were skipped. A skipped row is "
             "one the trainer will not see: silently training on 60%% of a corpus is the failure "
             "mode this number exists to make loud.",
    )
    p.add_argument(
        "--allow-tokenizer-mismatch", action=argparse.BooleanOptionalAction, default=False,
        help="Skip the check that --teacher-tokenizer agrees with the tokenizer shipped in "
             "--teacher-model. Only for a deliberately re-tokenized teacher.",
    )
    p.add_argument(
        "--expectation-file", default="",
        help="A previously emitted expectation JSON, reused instead of recomputing the corpus "
             "md5. Compared against the output dir's stored expectation before anything is "
             "appended.",
    )
    p.add_argument(
        "--emit-expectation", default="",
        help="Write the expectation JSON to this path and exit. No CUDA, no model load.",
    )
    p.add_argument(
        "--shard-count", type=int, default=0,
        help="Total number of INDEPENDENT passes over this corpus. 0 (default) derives the DP "
             "layout from the launcher's world size, which is the historical behaviour. With "
             "--shard-index, this job precomputes one residue class in a SINGLE-NODE job, so N "
             "such jobs occupy N x gpus_per_node GPUs with no rendezvous. The parts MUST then be "
             "merged with a companion tool and certified with --verify-only: a "
             "shard job cannot check completeness, because completeness is a property of the whole.",
    )
    p.add_argument(
        "--shard-index", type=int, default=-1,
        help="Which residue class this job owns, 0 <= --shard-index < --shard-count.",
    )
    p.add_argument(
        "--verify-only", action="store_true",
        help="Re-check an existing output directory (index vs corpus, shards vs index) and exit. "
             "No CUDA, no model load.",
    )
    a = p.parse_args(argv)
    if not a.teacher_tokenizer:
        a.teacher_tokenizer = a.teacher_model
    return a


# --------------------------------------------------------------------- corpus io


def scan_jsonl(path, keep=None):
    """Return (rows, n_total). `keep(i) -> bool` selects which rows are MATERIALIZED; every row is
    counted either way, because n_total is what the completeness post-condition compares against.

    A dict keyed by the GLOBAL index, not a list: with DP across nodes each node holds 1/num_nodes
    of the corpus, and every id written to the index must still be the global one.
    """
    rows = {}
    n_total = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            i = n_total
            n_total += 1
            if keep is None or keep(i):
                rows[i] = json.loads(line)
    return rows, n_total


def file_md5(path, block_size=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(block_size):
            h.update(chunk)
    return h.hexdigest()


def count_lines(path):
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def normalize_tools(tools_value):
    """The index carries `tools` forward for the consumer's per-row tool filters, so it has to
    survive whichever of the two shapes the corpus uses: a JSON string or an already-parsed list.
    """
    if tools_value is None:
        return None
    if isinstance(tools_value, str):
        s = tools_value.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None
    if isinstance(tools_value, list):
        return tools_value or None
    return None


def parse_tools_for_template(tools_value):
    """`tools` as apply_chat_template wants it: a list of dicts, or None."""
    parsed = normalize_tools(tools_value)
    return parsed if parsed else None


def serialize_tools_for_index(tools_value):
    """`tools` as the INDEX must carry it: a JSON string, or None. Not a parsed list.

    WHY THE INDEX AND THE TEMPLATE NEED DIFFERENT SHAPES. apply_chat_template wants parsed objects,
    so parse_tools_for_template exists for that. But the index is read back by
    sft.py::_load_dataset via `Dataset.from_list`, which hands the rows to pyarrow -- and pyarrow
    must infer ONE struct schema for the whole column. Tool definitions are JSON Schema, so
    `function.parameters.properties.<name>.default` is arbitrary JSON: measured on the blend index,
    `limit.type` is a str in 676 tools and a list in 1, `page.default` is int in 185, str in 182 and
    float in 1, `season.default` is str in 22 and int in 12. There is no single Arrow schema for
    that, and the loader dies with:

        pyarrow.lib.ArrowInvalid: cannot mix struct and non-struct, non-null values

    That failure is where this requirement comes from -- the blend arm reached the trainer and died loading its own index.
    It surfaced only at row 16,561 of 20,000, because the first tool-bearing rows happened to agree.

    THE CORPUS ALREADY SOLVED THIS. The deliverable corpus stores `tools` as a JSON STRING (measured:
    all 3,451 tool-bearing rows in blend-subset20k are str), which is why sft.py's trainer is
    documented as one that "handles tools stored as JSON strings" and calls json.loads on the way to
    the template. A string is one Arrow type no matter what the schema inside it looks like. So the
    index matches the corpus convention rather than inventing a second one.

    Round-trip safe: the consumer parses it back before rendering, so the template -- and therefore
    the assistant mask and num_assistant_tokens this index is cross-checked against -- is unchanged.
    """
    parsed = normalize_tools(tools_value)
    if not parsed:
        return None
    return json.dumps(parsed, ensure_ascii=False)


def _scan_response_template(input_ids, response_template_ids, eos_token_id):
    """Assistant mask by scanning for the response-template token sequence, for chat templates
    that carry no {% generation %} markers. Each match opens a span; eos closes it. Mirrors
    sft.py's fallback so a row masked here and re-masked there agrees.
    """
    n = len(input_ids)
    m = len(response_template_ids)
    if m == 0 or n < m:
        return None
    mask = [0] * n
    i = 0
    found = False
    while i <= n - m:
        if input_ids[i:i + m] == response_template_ids:
            found = True
            j = i + m
            while j < n:
                mask[j] = 1
                if eos_token_id is not None and input_ids[j] == eos_token_id:
                    j += 1
                    break
                j += 1
            i = j
        else:
            i += 1
    return mask if found else None


# ------------------------------------------------------------------ expectation


def build_expectation(args):
    """The identity of the artifact: everything that changes what gets written.

    Deliberately EXCLUDED: batch_size and seed (performance, not content -- see the module
    docstring), output_dir (an artifact is not identified by where it sits), and max_skip_fraction
    (a post-condition threshold; loosening it does not change a byte that was already written).
    """
    tok_dir = os.path.abspath(args.teacher_tokenizer)
    return {
        "step": STEP_NAME,
        "teacher_model": os.path.abspath(args.teacher_model),
        "teacher_tokenizer": tok_dir,
        # Content hash of the tokenizer, not just its path: the overlay is a directory in this
        # tree, and an in-place edit of tokenizer.json would otherwise be invisible.
        "teacher_tokenizer_sha": tokenizer_identity.hash_tokenizer(tok_dir),
        "top_k": int(args.top_k),
        "max_length": int(args.max_length),
        "dtype": args.dtype,
        "shard_target_tokens": int(args.shard_target_tokens),
        "ignore_documents": bool(args.ignore_documents),
        "response_template": args.response_template,
        "source_jsonl": os.path.abspath(args.input_jsonl),
        "source_jsonl_md5": file_md5(args.input_jsonl),
        "source_rows": count_lines(args.input_jsonl),
    }


def _expectation_diff(stored, current):
    lines = []
    for key in sorted(set(stored) | set(current)):
        was, now = stored.get(key, "<absent>"), current.get(key, "<absent>")
        if was != now:
            lines.append(f"  {key}:\n    stored : {was!r}\n    current: {now!r}")
    return lines


def install_expectation(output_dir, expectation):
    """Compare against the output dir's stored expectation, or store it. Refuse on a difference.

    Called before the first append. Resuming into a directory built from other inputs would append
    to one index over two corpora, and every consumer of that index would be quietly wrong.
    """
    path = os.path.join(output_dir, EXPECTATION_NAME)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            stored = json.load(f)
        diff = _expectation_diff(stored, expectation)
        if diff:
            raise RuntimeError(
                "this output dir was started under a DIFFERENT expectation, so resuming into it "
                "would mix two precomputes in one index:\n"
                + "\n".join(diff)
                + f"\n  stored in: {path}\n"
                "  Point --output-dir somewhere else, or delete that directory to start over."
            )
        return path
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(expectation, f, indent=2, sort_keys=True)
    os.replace(tmp, path)
    return path


# ------------------------------------------------------------------------ shards


def shard_paths(shards_dir, shard_id):
    return (
        os.path.join(shards_dir, f"indices_{shard_id:06d}.bin"),
        os.path.join(shards_dir, f"logits_{shard_id:06d}.bin"),
    )


def read_done_state(index_part_path, node_id):
    """Recover this node's progress from its index part.

    Returns (done_source_indices, current_local_idx, current_shard_token_count, rows_by_shard),
    where rows_by_shard maps shard_id -> row count the INDEX accounts for. That last one is what
    _reconcile_shards needs: the index is the authority on what a shard contains, and the .bin
    files can only ever be ahead of it (see the module docstring, difference 2).
    """
    done = set()
    rows_by_shard = {}
    max_local_idx = -1
    last_offset_end = 0
    if not os.path.exists(index_part_path):
        return done, 0, 0, rows_by_shard
    with open(index_part_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                # A torn LAST line is the expected shape of a kill mid-write, and everything before
                # it is still good. Anything earlier is corruption we must not paper over.
                raise RuntimeError(
                    f"{index_part_path}:{lineno} will not parse ({exc}). If this is the LAST line "
                    "it is a torn write from a kill; delete that line and resume. If it is not, "
                    "this file has been damaged and the output dir should be rebuilt."
                ) from exc
            done.add(row["source_idx"])
            if row.get("skipped"):
                continue
            shard_id = row["shard_id"]
            end = row["shard_offset"] + row["num_assistant_tokens"]
            rows_by_shard[shard_id] = max(rows_by_shard.get(shard_id, 0), end)
            local_idx = shard_id - node_id * MAX_SHARDS_PER_NODE
            if local_idx > max_local_idx:
                max_local_idx, last_offset_end = local_idx, end
            elif local_idx == max_local_idx:
                last_offset_end = max(last_offset_end, end)
    if max_local_idx < 0:
        return done, 0, 0, rows_by_shard
    return done, max_local_idx, last_offset_end, rows_by_shard


def _reconcile_shards(shards_dir, node_id, rows_by_shard, top_k, emit=print):
    """Make every shard file this node owns exactly as long as the index says it is.

    THE BUG THIS FIXES. Rows are appended to the .bin files and THEN named in the index. Killed
    between the two, the files carry rows the index does not mention; the original resumed with
    "ab" (write at end-of-file) while deriving shard_offset from the index, so every later row in
    that shard was stored one orphan-block away from where the index says it is. Nothing fails --
    the trainer just reads another token's teacher distribution for the rest of the shard.

    Truncation is the correct repair because the index is the authority and the orphan rows will
    simply be recomputed: their source rows are not in `done`.

    A file SHORTER than the index claims is the other direction and is NOT repaired: it means an
    index line exists for data that does not, which the writer's ordering cannot produce.
    """
    if not os.path.isdir(shards_dir):
        return
    lo = node_id * MAX_SHARDS_PER_NODE
    hi = lo + MAX_SHARDS_PER_NODE
    for name in sorted(os.listdir(shards_dir)):
        if not (name.startswith("indices_") and name.endswith(".bin")):
            continue
        try:
            shard_id = int(name[len("indices_"):-len(".bin")])
        except ValueError:
            continue
        if not (lo <= shard_id < hi):
            continue          # another node's shard; not ours to touch
        want_rows = rows_by_shard.get(shard_id, 0)
        for path, itemsize, label in (
            (shard_paths(shards_dir, shard_id)[0], INDICES_ITEMSIZE, "indices"),
            (shard_paths(shards_dir, shard_id)[1], LOGITS_ITEMSIZE, "logits"),
        ):
            if not os.path.exists(path):
                if want_rows:
                    raise RuntimeError(
                        f"{path} is missing but the index accounts for {want_rows} row(s) in "
                        f"shard {shard_id}. The index cannot be ahead of the data; this output "
                        "dir is damaged."
                    )
                continue
            want_bytes = want_rows * top_k * itemsize
            have_bytes = os.path.getsize(path)
            if have_bytes == want_bytes:
                continue
            if have_bytes < want_bytes:
                raise RuntimeError(
                    f"{path} holds {have_bytes} bytes but the index accounts for {want_rows} "
                    f"row(s) = {want_bytes} bytes ({label}, top_k={top_k}). The index is AHEAD of "
                    "the data, which the writer's ordering cannot produce -- something else "
                    "modified this directory. Refusing to resume."
                )
            emit(
                f"[resume] truncating {os.path.basename(path)}: {have_bytes} -> {want_bytes} bytes "
                f"({(have_bytes - want_bytes) // (top_k * itemsize)} orphan row(s) written after "
                "the last index line, i.e. a kill between the two writes)"
            )
            with open(path, "r+b") as fh:
                fh.truncate(want_bytes)


# ------------------------------------------------------------------ verification


def verify_output(output_dir, *, top_k=None, n_source=None, max_skip_fraction=None, emit=print):
    """Post-condition: this directory is a COMPLETE, self-consistent precompute of its corpus.

    Everything here is checkable from files alone -- no GPU, no teacher -- which is what lets a
    restarted recipe re-verify a directory it is about to skip instead of trusting a marker.

    1. Every corpus row appears exactly once, as precomputed or as explicitly skipped. The failure
       this catches has no downstream symptom: a shard set is VALID WHEN INCOMPLETE.
    2. The skip fraction is under the threshold. A skip is a row the trainer never sees.
    3. Each shard's byte length equals rows * top_k * itemsize, for both files.
    4. Within a shard the (offset, count) spans TILE [0, rows): no gap, no overlap. This is what
       proves the index and the bytes actually correspond, rather than merely having equal totals.
    """
    meta_path = os.path.join(output_dir, "meta.json")
    index_path = os.path.join(output_dir, "index.jsonl")
    for path in (meta_path, index_path):
        if not os.path.exists(path):
            raise RuntimeError(f"{path} is missing; this is not a finished precompute.")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    if top_k is None:
        top_k = int(meta["top_k"])
    if n_source is None:
        n_source = int(meta["n_source"])
    if max_skip_fraction is None:
        max_skip_fraction = float(meta.get("max_skip_fraction", 1.0))

    seen = set()
    spans = {}          # shard_id -> list[(offset, count)]
    skips = {}
    n_kept = 0
    with open(index_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            idx = row["source_idx"]
            if idx in seen:
                raise RuntimeError(f"{index_path}:{lineno}: duplicate source_idx={idx}.")
            seen.add(idx)
            if row.get("skipped"):
                skips[row.get("skip_reason") or "unknown"] = (
                    skips.get(row.get("skip_reason") or "unknown", 0) + 1
                )
                continue
            n_kept += 1
            n = int(row["num_assistant_tokens"])
            if n <= 0:
                raise RuntimeError(
                    f"{index_path}:{lineno}: a non-skipped row claims {n} assistant tokens."
                )
            spans.setdefault(int(row["shard_id"]), []).append((int(row["shard_offset"]), n))

    missing = sorted(set(range(n_source)) - seen)
    extra = sorted(i for i in seen if not 0 <= i < n_source)
    if missing or extra:
        detail = []
        if missing:
            detail.append(
                f"  {len(missing)} of {n_source} corpus row(s) are absent from the index, first: "
                f"{missing[:10]}"
            )
        if extra:
            detail.append(f"  {len(extra)} index row(s) name a source_idx outside the corpus: "
                          f"{extra[:10]}")
        raise RuntimeError(
            "INCOMPLETE precompute. A shard set is valid when incomplete -- it loads, it memmaps, "
            "and it trains -- so this is checked here because nothing downstream can notice it.\n"
            + "\n".join(detail)
            + "\n  Re-run the step against the same --output-dir: resume skips what is already "
            "done and computes only the rest."
        )

    n_skipped = sum(skips.values())
    frac = (n_skipped / n_source) if n_source else 0.0
    if frac > max_skip_fraction:
        raise RuntimeError(
            f"{n_skipped} of {n_source} rows ({frac:.1%}) were SKIPPED, above "
            f"--max-skip-fraction {max_skip_fraction:.1%}: "
            + ", ".join(f"{k}={v}" for k, v in sorted(skips.items()))
            + f"\n  A skipped row is one the trainer never sees, so this is a silent corpus "
            f"reduction. `too_long` means the row exceeded --max-length ({meta.get('max_length')}) "
            "-- raise it (rows are skipped, never truncated). `no_assistant` means no assistant "
            "span could be derived, which points at the chat template or --response-template.\n"
            "  Raise --max-skip-fraction only to accept the reduction deliberately."
        )

    shards_dir = os.path.join(output_dir, "shards")
    total_rows = 0
    for shard_id, sp in sorted(spans.items()):
        sp.sort()
        cursor = 0
        for offset, n in sp:
            if offset != cursor:
                raise RuntimeError(
                    f"shard {shard_id}: rows do not tile -- expected the next row at offset "
                    f"{cursor}, index says {offset}. A gap means the .bin holds rows nothing "
                    "reads; an overlap means two rows read the same bytes."
                )
            cursor += n
        total_rows += cursor
        for path, itemsize, label in (
            (shard_paths(shards_dir, shard_id)[0], INDICES_ITEMSIZE, INDICES_DTYPE),
            (shard_paths(shards_dir, shard_id)[1], LOGITS_ITEMSIZE, LOGITS_DTYPE),
        ):
            if not os.path.exists(path):
                raise RuntimeError(f"{path} is missing but the index accounts for {cursor} rows.")
            want = cursor * top_k * itemsize
            have = os.path.getsize(path)
            if have != want:
                raise RuntimeError(
                    f"{path}: {have} bytes, but the index accounts for {cursor} row(s) x "
                    f"top_k={top_k} x {itemsize} ({label}) = {want} bytes. "
                    + ("The file is LONGER than the index: orphan rows from a kill between the "
                       "shard write and the index write. A fresh run of the step reconciles this; "
                       "it is only fatal in a directory that claims to be finished."
                       if have > want else
                       "The file is SHORTER than the index claims, so some row's teacher logits "
                       "do not exist.")
                )
    emit(
        f"[verify] {output_dir}: {len(seen)} index rows over {n_source} corpus rows, "
        f"{n_kept} with logits, {n_skipped} skipped ({frac:.2%}"
        + ("".join(f", {k}={v}" for k, v in sorted(skips.items())) if skips else "")
        + f"), {len(spans)} shard(s), {total_rows} teacher rows x top_k {top_k}"
    )
    return {
        "n_index_rows": len(seen), "n_source": n_source, "n_kept": n_kept,
        "n_skipped": n_skipped, "skips": skips, "n_shards": len(spans),
        "total_teacher_rows": total_rows,
    }


# ------------------------------------------------------------------------- main


def main(argv=None):
    args = get_args(argv)

    # ---- Modes that must not touch CUDA, handled before PartialState() is constructed.
    if args.emit_expectation:
        exp = build_expectation(args)
        tmp = args.emit_expectation + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(exp, f, indent=2, sort_keys=True)
        os.replace(tmp, args.emit_expectation)
        print(f"wrote expectation: {args.emit_expectation}")
        print(f"  corpus  : {exp['source_jsonl']}")
        print(f"  rows    : {exp['source_rows']}  md5={exp['source_jsonl_md5']}")
        print(f"  teacher : {exp['teacher_model']}")
        print(f"  tokenizer: {exp['teacher_tokenizer']}  sha={exp['teacher_tokenizer_sha']}")
        return 0

    if args.verify_only:
        verify_output(args.output_dir, max_skip_fraction=args.max_skip_fraction)
        return 0

    return run_precompute(args)


def _ensure_process_group(torch, world_size, *, env=None, emit=print):
    """Make a single-rank launch look like a launched one, to torch AND to everything downstream.

    `accelerate launch --num_processes 1` uses accelerate's SIMPLE launcher: it runs the module as
    a plain subprocess and sets none of RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR. Two separate
    things then break, and this step's first two real runs found them one after the other:

      * init_device_mesh("cuda", (1, 1)) dies in torch's env:// rendezvous with "environment
        variable RANK expected, but not set", confirmed by direct measurement;
      * with the group initialized but the environment still bare, transformers'
        initialize_tensor_parallelism() indexes os.environ["LOCAL_RANK"] directly and raises
        KeyError, also confirmed by direct measurement.

    Every rank count above 1 goes through accelerate's multiprocessing launcher, which sets all of
    them -- which is why an earlier version, run only at 8 and 16 ranks, never met either.

    The second failure is why this sets the whole launcher environment rather than just the two
    variables torch's rendezvous reads: transformers is not the only library that reads LOCAL_RANK
    out of os.environ, and satisfying them one KeyError at a time is a losing game. A 1x1 mesh is a
    LEGAL configuration of this step -- `gpus_per_node: 1` with a teacher that fits one GPU -- so
    it has to be indistinguishable from a launched one, not merely far enough along to start.

    The six values are ASSIGNED, not defaulted: for a single-rank group the truth is known exactly
    (rank 0 of 1, local rank 0 of 1), and an inherited value that contradicts it -- a MASTER_PORT
    left in the environment by some other launcher, say -- would be wrong rather than a useful
    hint, and would surface as a rendezvous hang rather than an error.

    REFUSES, rather than helping, when world_size > 1. A multi-rank launch arriving here without a
    process group means the launcher never set the rendezvous up, and quietly forming a 1-rank
    group would leave each process writing its own COMPLETE-looking artifact over 1/world_size of
    the corpus into one directory -- the valid-looking partial precompute that every post-condition
    in this module exists to prevent. Returns True if it initialized, False if it found a group.
    """
    if torch.distributed.is_initialized():
        return False
    if world_size != 1:
        raise RuntimeError(
            f"world_size={world_size} but torch.distributed is not initialized, so the launcher "
            "did not set up a rendezvous. Refusing to fabricate a single-rank group: each rank "
            "would write a separate, complete-looking artifact over its own slice of the corpus "
            "into the same output dir, and every index would be internally consistent. Launch "
            f"with `accelerate launch --num_processes {world_size} --num_machines ...`."
        )
    env = os.environ if env is None else env
    # A free port rather than the conventional 29500: compute nodes here are shared, and a collision
    # surfaces as a rendezvous hang rather than an error.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env["RANK"] = "0"
    env["WORLD_SIZE"] = "1"
    env["LOCAL_RANK"] = "0"
    env["LOCAL_WORLD_SIZE"] = "1"
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(port)
    emit(
        "[rank 0] the launcher provided no process group (world_size=1, accelerate's simple "
        f"launcher); setting RANK/LOCAL_RANK/WORLD_SIZE/LOCAL_WORLD_SIZE and initializing a "
        f"single-rank nccl group on 127.0.0.1:{port}"
    )
    # Before the group, and before the mesh: DeviceMesh otherwise warns that it is guessing the
    # device via global_rank % num_devices_per_host, and nccl's communicator wants the device set.
    torch.cuda.set_device(0)
    torch.distributed.init_process_group(backend="nccl")
    return True


def run_precompute(args):
    """The GPU pass. Imports torch et al. HERE; see DTYPE_NAMES for why."""
    import torch
    import torch.nn.functional as F
    from accelerate import PartialState
    from torch.distributed.device_mesh import init_device_mesh
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    from gb_steps_post_training.distillation.utils import (
        verify_fast_tokenizer,
        verify_tokenizer_consistency,
    )

    dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    set_seed(args.seed)   # nothing here samples; called so the flag is not a lie

    state = PartialState()
    rank = state.process_index
    world_size = state.num_processes

    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size % local_world_size != 0:
        raise RuntimeError(
            f"world_size={world_size} is not divisible by local_world_size={local_world_size}; "
            "heterogeneous nodes are not supported."
        )
    num_nodes = world_size // local_world_size
    node_id = rank // local_world_size
    is_node_writer = local_rank == 0

    # ---- SHARD MODE: the row split is not the mesh.
    # Without --shard-count these two stay derived from the launcher's world size, exactly as
    # before. With it, this job is one of N INDEPENDENT single-node passes: it owns residue class
    # --shard-index for row selection, shard-id namespacing and its index_part name, while the
    # device mesh stays (1, tp) because that is all this job actually launched. That is sound
    # precisely because transformers only ever collectives over mesh["tp"] (see the mesh comment
    # below) -- the "dp" dimension carries no traffic, it only decides which rows are mine.
    # The single-node assertion is the guard that makes this safe to combine with a real
    # multi-node launch later: shard mode is for jobs that did NOT rendezvous.
    mesh_dp = num_nodes
    if args.shard_count:
        if not 0 <= args.shard_index < args.shard_count:
            raise RuntimeError(
                f"--shard-index {args.shard_index} is outside [0, {args.shard_count}). Every "
                f"residue class must be run exactly once or the merged index is incomplete."
            )
        if world_size != local_world_size:
            raise RuntimeError(
                f"shard mode expects a SINGLE-NODE job (world_size={world_size} != "
                f"local_world_size={local_world_size}). It exists to avoid a multi-node "
                f"rendezvous; combining it with one would make two layers disagree about which "
                f"rows this job owns."
            )
        num_nodes = args.shard_count
        node_id = args.shard_index
        mesh_dp = 1
        if rank == 0:
            print(f"SHARD MODE: residue class {node_id} of {num_nodes} "
                  f"(rows where i % {num_nodes} == {node_id}); mesh stays (1, {local_world_size}); "
                  f"NO merge and NO completeness check here -- merge the parts and run "
                  f"--verify-only.", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    shards_dir = os.path.join(args.output_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)
    index_part_path = os.path.join(args.output_dir, f"index_part_{node_id:04d}.jsonl")

    # ---- Expectation gate, before a single byte is appended.
    if args.expectation_file:
        # Supplied by the launcher, which already paid for the corpus md5 to run its own
        # step_state gate. One md5 per run, not three.
        with open(args.expectation_file, encoding="utf-8") as f:
            expectation = json.load(f)
    else:
        expectation = build_expectation(args)
    if is_node_writer:
        install_expectation(args.output_dir, expectation)
    n_source_declared = int(expectation["source_rows"])

    done_source_indices, current_local_idx, current_shard_token_count, rows_by_shard = (
        read_done_state(index_part_path, node_id)
    )
    if is_node_writer:
        # Before any append, and only on the rank that appends.
        _reconcile_shards(shards_dir, node_id, rows_by_shard, args.top_k)
    print(
        f"[rank {rank}] read_done_state: node={node_id} already_done={len(done_source_indices)} "
        f"current_local_idx={current_local_idx} shard_rows={current_shard_token_count}",
        flush=True,
    )
    if current_shard_token_count >= args.shard_target_tokens:
        current_local_idx += 1
        current_shard_token_count = 0
    if current_local_idx >= MAX_SHARDS_PER_NODE:
        raise RuntimeError(
            f"node {node_id} has used all {MAX_SHARDS_PER_NODE} shard slots; raise "
            "--shard-target-tokens (or MAX_SHARDS_PER_NODE, which also lowers the node ceiling)."
        )
    current_shard_id = node_id * MAX_SHARDS_PER_NODE + current_local_idx

    if rank == 0:
        print(code_provenance.format_block(code_provenance.describe()), flush=True)
        print(f"World size: {world_size} (num_nodes={num_nodes}, tp={local_world_size})")
        print(f"Corpus: {args.input_jsonl} ({n_source_declared} rows)")

    # ---- Tokenizer. SEPARATE from the weights; see difference 1 in the module docstring.
    if rank == 0:
        print(f"Loading tokenizer: {args.teacher_tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_tokenizer, trust_remote_code=True)
    verify_fast_tokenizer(tokenizer, args.teacher_tokenizer, source_label="precompute-teacher")
    if not args.allow_tokenizer_mismatch and os.path.abspath(
        args.teacher_tokenizer
    ) != os.path.abspath(args.teacher_model):
        # The overlay is BUILT from the teacher, so the two must agree on token ids. What this
        # catches is handing this step the STUDENT's tokenizer -- which would mis-segment the whole
        # corpus, and which nothing else in the pass would notice.
        weights_tok = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
        verify_tokenizer_consistency(
            tokenizer, weights_tok,
            train_source=args.teacher_tokenizer, ref_source=args.teacher_model,
            context="precompute (teacher tokenizer vs the tokenizer shipped with the weights)",
        )
        del weights_tok
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    response_template_ids = None
    if args.response_template:
        response_template_ids = tokenizer.encode(args.response_template, add_special_tokens=False)
        if not response_template_ids:
            response_template_ids = None

    torch_dtype = dtypes[args.dtype]

    # ---- Teacher. 2-D mesh ("dp", "tp"): transformers uses mesh["tp"] for its collectives and
    # never touches "dp", which is what lets each node process its own slice independently.
    if rank == 0:
        print(f"Loading model: {args.teacher_model}")
    _ensure_process_group(torch, world_size)
    mesh = init_device_mesh("cuda", (mesh_dp, local_world_size), mesh_dim_names=("dp", "tp"))
    model_kwargs = dict(
        torch_dtype=torch_dtype, trust_remote_code=True, tp_plan="auto", device_mesh=mesh,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.teacher_model, attn_implementation="flash_attention_2", **model_kwargs
        )
    except (ValueError, TypeError, RuntimeError, ImportError) as e:
        # Long context cannot tolerate eager/SDPA: the [B, heads, T, T] score matrix is infeasible.
        if args.max_length > 16384:
            raise RuntimeError(
                f"flash_attention_2 unavailable and max_length={args.max_length} > 16384; "
                "eager/SDPA attention's [B, heads, T, T] matrix is infeasible at this length. "
                f"Original error: {type(e).__name__}: {e}"
            ) from e
        # Loud, and recorded in meta.json. On this account FA2 comes from the kernels hub rather
        # than the absent flash_attn package, so the
        # usual cause of landing here is a cold kernel cache under HF_HUB_OFFLINE=1 -- which the
        # launcher preflights precisely so this fallback stays theoretical.
        if rank == 0:
            print(f"WARNING: flash_attention_2 unavailable, falling back to the transformers "
                  f"default: {type(e).__name__}: {e}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(args.teacher_model, **model_kwargs)
    model = model.eval()   # NOT .to(device): tp_plan="auto" has already placed the shards
    attn_impl = getattr(model.config, "_attn_implementation", "unknown")

    vocab_size = int(model.config.vocab_size)
    if args.top_k > vocab_size:
        raise ValueError(f"top_k={args.top_k} exceeds vocab_size={vocab_size}")

    lm_head = model.get_output_embeddings()
    if lm_head is None:
        lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise RuntimeError(
            "Could not locate lm_head: model.get_output_embeddings() returned None and "
            "model.lm_head is not set."
        )
    lm_head_bias = getattr(lm_head, "bias", None)
    logits_scaling = getattr(model.config, "logits_scaling", 1.0)

    # The probe. A column-sharded lm_head under tp_plan="auto" gives each rank a slice of the
    # vocabulary, so topk over it returns rank-local ids -- plausible numbers, wrong tokens.
    hidden_size = int(model.config.hidden_size)
    with torch.inference_mode():
        probe_in = torch.zeros(1, hidden_size, dtype=torch_dtype, device=state.device)
        probe_width = int(F.linear(probe_in, lm_head.weight, lm_head_bias).size(-1))
        del probe_in
    if probe_width != vocab_size:
        raise RuntimeError(
            f"lm_head output width {probe_width} != vocab_size {vocab_size}. lm_head appears to be "
            "column-sharded by TP; the chunked topk path would return rank-local token ids and "
            "requires a dist.all_gather along the last dim before topk."
        )
    print(
        f"[rank {rank}] model loaded: attn={attn_impl} vocab_size={vocab_size} "
        f"lm_head_probe_width={probe_width} logits_scaling={logits_scaling}",
        flush=True,
    )

    # ---- This node's slice: rows where (i % num_nodes) == node_id, minus what is already done.
    source_data, n_source = scan_jsonl(
        args.input_jsonl, keep=lambda i: (i % num_nodes) == node_id
    )
    if n_source != n_source_declared:
        raise RuntimeError(
            f"the corpus has {n_source} rows but the expectation recorded "
            f"{n_source_declared}. {args.input_jsonl} changed under the run."
        )
    node_indices = [i for i in sorted(source_data) if i not in done_source_indices]
    print(
        f"[rank {rank}] source loaded: node={node_id} n_source={n_source} "
        f"held={len(source_data)} pending={len(node_indices)}",
        flush=True,
    )

    pad_token_id = tokenizer.pad_token_id
    n_processed = 0
    n_skipped_no_assist = 0
    n_skipped_too_long = 0

    def write_index_row(row):
        with open(index_part_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    pbar = tqdm(
        range(0, len(node_indices), args.batch_size),
        desc=f"rank {rank}", disable=not is_node_writer,
    )
    for batch_start in pbar:
        batch_idx_list = node_indices[batch_start:batch_start + args.batch_size]

        tokenized = []   # (src_idx, input_ids, logit_positions, skip_reason, n_assist)
        for src_idx in batch_idx_list:
            record = source_data[src_idx]
            messages = record.get("messages", [])
            if not messages:
                tokenized.append((src_idx, None, None, "no_assistant", 0))
                continue

            tools = parse_tools_for_template(record.get("tools"))
            documents = [] if args.ignore_documents else record.get("documents", []) or []
            chat_template_kwargs = {"documents": documents} if documents else {}

            try:
                processed = tokenizer.apply_chat_template(
                    messages, return_dict=True, tokenize=True,
                    return_assistant_tokens_mask=True, tools=tools, **chat_template_kwargs,
                )
            except Exception:
                tokenized.append((src_idx, None, None, "no_assistant", 0))
                continue

            input_ids = processed["input_ids"]
            assistant_masks = processed.get("assistant_masks", None)
            if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
                input_ids = input_ids[0]
                if assistant_masks is not None:
                    assistant_masks = assistant_masks[0]

            # Templates without {% generation %} return either None or an all-zero mask; both mean
            # "no markers", and both are rescued by scanning for the response template. Same
            # two-way test as sft.py, deliberately.
            if assistant_masks is None or 1 not in assistant_masks:
                if response_template_ids is not None:
                    assistant_masks = _scan_response_template(
                        input_ids, response_template_ids, tokenizer.eos_token_id
                    )

            if assistant_masks is None or 1 not in assistant_masks:
                tokenized.append((src_idx, None, None, "no_assistant", 0))
                continue

            # The logit that PREDICTS an assistant token sits one position earlier, so position 0
            # can never be scored. This is the definition the consumer mirrors when it recomputes
            # assistant_logits_used as sum(mask[1:]).
            logit_positions = [i - 1 for i, m in enumerate(assistant_masks) if m == 1 and i >= 1]
            if not logit_positions:
                tokenized.append((src_idx, None, None, "no_assistant", 0))
                continue
            if len(input_ids) > args.max_length:
                tokenized.append((src_idx, None, None, "too_long", 0))
                continue
            tokenized.append((src_idx, input_ids, logit_positions, None, len(logit_positions)))

        forward_batch = [t for t in tokenized if t[3] is None]

        if forward_batch:
            input_ids_list = [t[1] for t in forward_batch]
            max_len = max(len(x) for x in input_ids_list)
            B = len(input_ids_list)
            padded_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
            attn = torch.zeros((B, max_len), dtype=torch.long)
            for i, ids in enumerate(input_ids_list):
                padded_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
                attn[i, :len(ids)] = 1
            padded_ids = padded_ids.to(state.device)
            attn = attn.to(state.device)

            # Capture lm_head's INPUT and make its forward the identity, so the CausalLM path never
            # materializes [B, T, V] (nor applies logits_scaling -- we do that below, on the rows
            # we keep).
            captured = {}

            def _capture_hook(module, args_, kwargs_):
                captured["h"] = args_[0]
                return args_, kwargs_

            hook = lm_head.register_forward_pre_hook(_capture_hook, with_kwargs=True)
            orig_lm_head_forward = lm_head.forward
            lm_head.forward = lambda x: x
            try:
                with torch.inference_mode():
                    model(input_ids=padded_ids, attention_mask=attn, use_cache=False)
            finally:
                hook.remove()
                lm_head.forward = orig_lm_head_forward

            hidden = captured["h"]     # [B, T, H]
            captured.clear()

            if is_node_writer:
                per_example_hidden = []
                per_example_meta = []
                for b_idx, (src_idx, _, logit_positions, _, n_assist) in enumerate(forward_batch):
                    pos = torch.tensor(logit_positions, dtype=torch.long, device=hidden.device)
                    per_example_hidden.append(hidden[b_idx].index_select(0, pos))
                    per_example_meta.append((src_idx, n_assist))
                hidden_assist = torch.cat(per_example_hidden, dim=0)   # [N, H]
                del per_example_hidden, hidden

                topk_vals_chunks, topk_idx_chunks = [], []
                with torch.inference_mode():
                    for s in range(0, hidden_assist.size(0), LM_HEAD_CHUNK):
                        chunk_logits = F.linear(
                            hidden_assist[s:s + LM_HEAD_CHUNK], lm_head.weight, lm_head_bias
                        )
                        if logits_scaling != 1.0:
                            chunk_logits = chunk_logits / logits_scaling
                        tv, ti = torch.topk(chunk_logits, k=args.top_k, dim=-1)
                        topk_vals_chunks.append(tv)
                        topk_idx_chunks.append(ti)
                        del chunk_logits
                topk_vals_all = torch.cat(topk_vals_chunks, dim=0)
                topk_idx_all = torch.cat(topk_idx_chunks, dim=0)
                del topk_vals_chunks, topk_idx_chunks, hidden_assist

                cursor = 0
                for src_idx, n_assist in per_example_meta:
                    topk_vals = topk_vals_all[cursor:cursor + n_assist]
                    topk_idx = topk_idx_all[cursor:cursor + n_assist]
                    cursor += n_assist
                    topk_vals_np = topk_vals.detach().to(torch.float16).cpu().numpy()
                    topk_idx_np = topk_idx.detach().to(torch.int32).cpu().numpy()

                    if (current_shard_token_count > 0
                            and current_shard_token_count + n_assist > args.shard_target_tokens):
                        current_local_idx += 1
                        if current_local_idx >= MAX_SHARDS_PER_NODE:
                            raise RuntimeError(
                                f"node {node_id} exhausted {MAX_SHARDS_PER_NODE} shard slots."
                            )
                        current_shard_id = node_id * MAX_SHARDS_PER_NODE + current_local_idx
                        current_shard_token_count = 0

                    indices_path, logits_path = shard_paths(shards_dir, current_shard_id)
                    # Bytes first, index line second. A kill between them leaves orphan bytes,
                    # which _reconcile_shards truncates on the next start. The other order would
                    # leave an index row pointing at data that does not exist, which is NOT
                    # repairable -- so this ordering is deliberate, not incidental.
                    with open(indices_path, "ab") as f:
                        f.write(topk_idx_np.tobytes())
                    with open(logits_path, "ab") as f:
                        f.write(topk_vals_np.tobytes())
                    write_index_row({
                        "source_idx": src_idx,
                        "messages": source_data[src_idx].get("messages"),
                        "tools": serialize_tools_for_index(source_data[src_idx].get("tools")),
                        "documents": source_data[src_idx].get("documents", []) or [],
                        "skipped": False,
                        "skip_reason": None,
                        "num_assistant_tokens": n_assist,
                        "shard_id": current_shard_id,
                        "shard_offset": current_shard_token_count,
                    })
                    current_shard_token_count += n_assist
                    n_processed += 1
                del topk_vals_all, topk_idx_all

        if is_node_writer:
            for src_idx, _, _, skip_reason, _ in tokenized:
                if skip_reason is None:
                    continue
                write_index_row({
                    "source_idx": src_idx,
                    "messages": source_data[src_idx].get("messages"),
                    "tools": serialize_tools_for_index(source_data[src_idx].get("tools")),
                    "documents": source_data[src_idx].get("documents", []) or [],
                    "skipped": True,
                    "skip_reason": skip_reason,
                    "num_assistant_tokens": 0,
                    "shard_id": -1,
                    "shard_offset": -1,
                })
                if skip_reason == "no_assistant":
                    n_skipped_no_assist += 1
                elif skip_reason == "too_long":
                    n_skipped_too_long += 1

            pbar.set_postfix(
                processed=n_processed, skip_noassist=n_skipped_no_assist,
                skip_toolong=n_skipped_too_long, shard=current_shard_id,
            )
            n_batches = (len(node_indices) + args.batch_size - 1) // args.batch_size
            if ((batch_start // args.batch_size) % 50 == 0
                    or batch_start + args.batch_size >= len(node_indices)):
                print(
                    f"[rank {rank}] heartbeat: node={node_id} "
                    f"batch={batch_start // args.batch_size + 1}/{n_batches} "
                    f"processed={n_processed} skip_noassist={n_skipped_no_assist} "
                    f"skip_toolong={n_skipped_too_long} shard={current_shard_id}",
                    flush=True,
                )

    print(
        f"[rank {rank}] done: processed={n_processed} skip_noassist={n_skipped_no_assist} "
        f"skip_toolong={n_skipped_too_long}",
        flush=True,
    )
    if torch.cuda.is_available():
        print(f"[rank {rank}] peak alloc = {torch.cuda.max_memory_allocated() / 1e9:.2f} GB",
              flush=True)
    sys.stdout.flush()

    # Free the teacher BEFORE the cross-node barrier. init_device_mesh only sets up subgroup
    # metadata; the world communicator's CUDA buffers are allocated on first use, and with ~40 GB
    # per rank of model resident that allocation OOMs.
    del model
    lm_head = None
    torch.cuda.empty_cache()

    state.wait_for_everyone()

    rc = 0
    if args.shard_count and rank == 0:
        print(f"SHARD MODE: wrote index_part_{node_id:04d}.jsonl and STOPPING. The other "
              f"{num_nodes - 1} part(s) belong to other jobs, so merging here would silently "
              f"emit a partial index. Run the companion merge tool once every "
              f"shard is DONE, then --verify-only.", flush=True)
    if rank == 0 and not args.shard_count:
        print("Merging index parts...", flush=True)
        all_rows = []
        for n in range(num_nodes):
            part_path = os.path.join(args.output_dir, f"index_part_{n:04d}.jsonl")
            if not os.path.exists(part_path):
                # No longer merely noted: with DP by `i % num_nodes` a missing part is a whole
                # residue class of the corpus, and the completeness check below is what turns that
                # into a failure instead of a smaller index.
                print(f"  MISSING: {part_path}", flush=True)
                continue
            n_before = len(all_rows)
            with open(part_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        all_rows.append(json.loads(line))
            print(f"  read {part_path}: +{len(all_rows) - n_before} rows "
                  f"(total={len(all_rows)})", flush=True)

        all_rows.sort(key=lambda x: x["source_idx"])
        index_path = os.path.join(args.output_dir, "index.jsonl")
        tmp_index = index_path + ".tmp"
        with open(tmp_index, "w", encoding="utf-8") as f:
            for row in all_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp_index, index_path)

        counts = {}
        for row in all_rows:
            if row.get("skipped"):
                counts[row.get("skip_reason") or "unknown"] = (
                    counts.get(row.get("skip_reason") or "unknown", 0) + 1
                )
        n_kept = sum(1 for r in all_rows if not r.get("skipped"))

        meta = {
            "step": STEP_NAME,
            "teacher_model": args.teacher_model,
            # The tokenizer that was ACTUALLY used. sft.py loads this path and compares it with the
            # training tokenizer, so a wrong value here is a silent mis-segmentation, and the whole
            # reason --teacher-tokenizer exists.
            "tokenizer_name_or_path": args.teacher_tokenizer,
            "top_k": args.top_k,
            "logits_dtype": LOGITS_DTYPE,
            "indices_dtype": INDICES_DTYPE,
            "teacher_load_dtype": args.dtype,
            "attn_implementation": attn_impl,
            "vocab_size": vocab_size,
            "source_jsonl": os.path.abspath(args.input_jsonl),
            "source_jsonl_md5": expectation["source_jsonl_md5"],
            "n_source": n_source,
            "n_kept": n_kept,
            "n_skipped": len(all_rows) - n_kept,
            "skips": counts,
            "max_skip_fraction": args.max_skip_fraction,
            "max_length": args.max_length,
            "ignore_documents": bool(args.ignore_documents),
            "shard_target_tokens": args.shard_target_tokens,
            "world_size_at_precompute": world_size,
            "tp_world_size": local_world_size,
            "num_nodes": num_nodes,
            "dp_world_size": num_nodes,
            "code_provenance": code_provenance.describe(),
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "response_template": args.response_template,
            "response_template_ids": response_template_ids,
            "expectation": expectation,
        }
        tmp_meta = os.path.join(args.output_dir, "meta.json.tmp")
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        os.replace(tmp_meta, os.path.join(args.output_dir, "meta.json"))
        print(f"Wrote {index_path} ({len(all_rows)} rows) and meta.json", flush=True)

        # The post-condition runs LAST and against the files on disk, not against the counters --
        # the counters cannot see a shard the writer never flushed. A failure here leaves
        # meta.json in place on purpose: the directory is resumable and the message says so.
        try:
            verify_output(
                args.output_dir, top_k=args.top_k, n_source=n_source,
                max_skip_fraction=args.max_skip_fraction,
            )
        except RuntimeError as exc:
            print(f"\nPOST-CONDITION FAILED\n{exc}", file=sys.stderr, flush=True)
            rc = 4

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        # rc lives on rank 0 only, and every rank must agree on the exit code or the launcher sees
        # a zero from a run whose post-condition failed.
        rc_t = torch.tensor([rc], device=state.device)
        torch.distributed.broadcast(rc_t, src=0)
        rc = int(rc_t.item())
        torch.distributed.destroy_process_group()
    sys.stdout.flush()
    sys.stderr.flush()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
