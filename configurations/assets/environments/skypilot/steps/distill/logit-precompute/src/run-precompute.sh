#!/bin/bash
#
# PORTED, not authored here. Upstream source of truth:
#   repo   github.ibm.com/Herbert-Woisetschlaeger/gb-steps-collection-post-training
#   path   steps/distill-logit-precompute/src/run-precompute.sh
#   commit 70c1550a171aa8e09a9ad9047a5bf763c39e8579
#
# The ONLY intentional divergence is the artifact marker prefix: LLMB_ARTIFACT_* ->
# GB_ARTIFACT_*. The monitor accepts both; GB_ is what every other step here prints.
#
# This script owns the marker, and the step template deliberately does NOT print it a
# second time — upstream's template did, which would register two NEWARTIFACT events for
# one id.
#
# Two things the launcher must hand it, both learned on distill-sft: PRECOMPUTE_SRC
# and LIB_DIR (exporting CHECKOUT_ROOT is discarded, because this file computes it
# unconditionally from STEP_HOME), and /stage/.venv/bin on PATH for the bare `accelerate`.
#
# Keep this a near-verbatim copy so re-syncing upstream stays a three-way merge.
#
# distill-logit-precompute step entrypoint, INSIDE the container.
#
# WHAT THIS STEP IS FOR. It runs the teacher ONCE over the corpus and stores its top-K logits per
# assistant token, so the off-policy KD arm can train without a 30B teacher resident beside the
# student and without recomputing the same forward pass every epoch. Its consumer is
# distill-sft's --precomputed-logits-dir, which turns that step from the SFT CONTROL into
# a forward-KL distillation arm -- so a recipe that sets both is no longer running a control.
#
# WHAT IT IS NOT. It is not a route to on-policy GOLD. On-policy means the STUDENT generates and
# the teacher scores text that does not exist until training time; there is nothing to precompute.
#
# WHY THE POST-CONDITIONS ARE THE POINT OF THIS FILE. The output of a PARTIAL precompute is
# structurally valid: shards/ + index.jsonl + meta.json describe whatever was actually written, so
# a run that covered a tenth of the corpus, or skipped every row over --max-length, or ran against
# the wrong tokenizer, produces a directory that loads, memmaps, and trains. The failure surfaces
# as a slightly worse student, weeks later, with no error anywhere. So this launcher does not
# finish when the pass exits 0 -- it finishes when the artifact has been VERIFIED against the
# corpus, and on the SKIP path it re-verifies rather than trusting a marker.
#
# This is a deliberate near-twin of distill-sft/src/run-sft.sh: same part location, same
# kernel preflight, same allocation assertion, same nodes>1 refusal. Where they differ the
# difference is named. Two differences are worth reading before the code:
#   - the module is launched with `accelerate launch -m`, not by cd-ing to its directory and
#     naming a file. precompute_logits.py imports from gb_steps_post_training.distillation at
#     module scope, and a flat launch puts the flat directory on sys.path and the package root
#     NOWHERE -- jobs 1162604 and 1162625, 49 s of allocation each and zero work done. sft.py is
#     launched flat for historical reasons and needs the PYTHONPATH workaround below; this one
#     does not, and is launched the way align_state and run_divergence already are.
#   - the resume gate is asked TWICE, of two different questions. See "resume" below.
set -uo pipefail

# ---------------------------------------------------------------- locating our own parts
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STEP_HOME="${STEP_HOME:-$HERE}"
CHECKOUT_ROOT="$(cd -- "$STEP_HOME/../../.." 2>/dev/null && pwd || echo "")"

first_dir() { local c; for c in "$@"; do [[ -d "$c" ]] && { (cd "$c" && pwd); return 0; }; done; return 1; }

if [[ -z "${PRECOMPUTE_SRC:-}" ]]; then
  PRECOMPUTE_SRC="$(first_dir \
    "$STEP_HOME/vendor/gb_steps_post_training/distillation" \
    "${CHECKOUT_ROOT:-/nonexistent}/src/gb_steps_post_training/distillation" \
  )" || PRECOMPUTE_SRC="$STEP_HOME/vendor/gb_steps_post_training/distillation"
fi
if [[ -z "${LIB_DIR:-}" ]]; then
  LIB_DIR="$(first_dir \
    "$STEP_HOME/lib" \
    "${CHECKOUT_ROOT:-/nonexistent}/scripts/bluevela/lib" \
  )" || LIB_DIR="$STEP_HOME/lib"
fi
# The package root goes on PYTHONPATH because we launch `-m gb_steps_post_training.distillation.…`
# and that name has to be importable. Unlike run-sft.sh this is the ONLY thing it is for -- there
# is no flat launch here to work around.
PKG_ROOT="$(cd -- "$PRECOMPUTE_SRC/../.." 2>/dev/null && pwd || echo "")"
[[ -n "$PKG_ROOT" ]] && export PYTHONPATH="$PKG_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PKG="gb_steps_post_training.distillation"

PYBIN="${PYBIN:-python}"
ACCELERATE="${ACCELERATE:-accelerate}"

# Defaults mirror step-template.yaml; every one is overridable by a flag.
CORPUS="" ; TEACHER_MODEL="" ; TEACHER_TOKENIZER="" ; OUTPUT_DIR="output"
TOP_K="256" ; DTYPE="bfloat16" ; MAX_LENGTH="8192" ; SHARD_TARGET_TOKENS="4000000"
BATCH_SIZE="4" ; SEED="0" ; IGNORE_DOCUMENTS="0"
RESPONSE_TEMPLATE="<|im_start|>assistant
"
MAX_SKIP_FRACTION="0.05" ; ALLOW_TOKENIZER_MISMATCH="0"
GPN="8" ; NODES="1" ; HF_HOME_ARG=""
# Shard mode: unset by default, so the historical single-pass behaviour is byte-identical.
SHARD_INDEX="" ; SHARD_COUNT=""

while (( $# )); do
  case "$1" in
    --corpus-path)            CORPUS="$2"; shift 2 ;;
    --teacher-model-path)     TEACHER_MODEL="$2"; shift 2 ;;
    --teacher-tokenizer-path) TEACHER_TOKENIZER="$2"; shift 2 ;;
    --output-dir)             OUTPUT_DIR="$2"; shift 2 ;;
    --top-k)                  TOP_K="$2"; shift 2 ;;
    --dtype)                  DTYPE="$2"; shift 2 ;;
    --max-length)             MAX_LENGTH="$2"; shift 2 ;;
    --shard-target-tokens)    SHARD_TARGET_TOKENS="$2"; shift 2 ;;
    --batch-size)             BATCH_SIZE="$2"; shift 2 ;;
    --seed)                   SEED="$2"; shift 2 ;;
    --response-template)      RESPONSE_TEMPLATE="$2"; shift 2 ;;
    --max-skip-fraction)      MAX_SKIP_FRACTION="$2"; shift 2 ;;
    # Flag PAIRS, no value. step-template.yaml renders
    # `{% if %}--flag{% else %}--no-flag{% endif %}` because jinja writes a YAML boolean with
    # PYTHON casing, so `--flag {{ x }}` arrives as the literal string "False" and silently does
    # nothing (job 1138651).
    --ignore-documents)             IGNORE_DOCUMENTS="1"; shift ;;
    --no-ignore-documents)          IGNORE_DOCUMENTS="0"; shift ;;
    --allow-tokenizer-mismatch)     ALLOW_TOKENIZER_MISMATCH="1"; shift ;;
    --no-allow-tokenizer-mismatch)  ALLOW_TOKENIZER_MISMATCH="0"; shift ;;
    --gpus-per-node)          GPN="$2"; shift 2 ;;
    --nodes)                  NODES="$2"; shift 2 ;;
    --shard-index)            SHARD_INDEX="$2"; shift 2 ;;
    --shard-count)            SHARD_COUNT="$2"; shift 2 ;;
    --hf-home)                HF_HOME_ARG="$2"; shift 2 ;;
    *) echo "FATAL: unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -z "$TEACHER_TOKENIZER" ]] && TEACHER_TOKENIZER="$TEACHER_MODEL"

echo "=== distill-logit-precompute ==="
date; hostname
echo "step home : $STEP_HOME"
echo "src       : $PRECOMPUTE_SRC"
echo "lib dir   : $LIB_DIR"
echo "pkg root  : ${PKG_ROOT:-<unresolved>}"
echo
echo "corpus    : $CORPUS"
echo "teacher   : $TEACHER_MODEL"
# Printed on its own line even when it equals the teacher, because "which tokenizer segmented this
# corpus" is the question a reader of this log most often needs answered, and meta.json's
# tokenizer_name_or_path is what the training-time consumer will compare against.
if [[ "$TEACHER_TOKENIZER" == "$TEACHER_MODEL" ]]; then
  echo "tokenizer : $TEACHER_TOKENIZER   (the teacher's own -- pass distill-tokenizer-align's"
  echo "            teacher_overlay if the two should differ)"
else
  echo "tokenizer : $TEACHER_TOKENIZER   (SEPARATE from the weights)"
fi
echo "output    : $OUTPUT_DIR"
echo "top_k     : $TOP_K   max_length: $MAX_LENGTH   dtype: $DTYPE (teacher load; stored logits are float16)"

for req in "CORPUS:--corpus-path" "TEACHER_MODEL:--teacher-model-path"; do
  var="${req%%:*}"; flag="${req#*:}"
  if [[ -z "${!var}" ]]; then
    echo "FATAL: $flag is required." >&2; exit 2
  fi
done
# PRECOMPUTE_DRY_RUN=1 STOPS HERE: every argument parsed and every argument-only precondition
# checked, and nothing touched on this machine. The twin of SFT_DRY_RUN in run-sft.sh, with the same
# reasons (see that file's comment: the flag surface is a contract two checks measure, so the lever
# is an env var and not a flag; and 0 rather than 2, because 2 already means "unknown argument" and
# run-all.sh maps it to UNMEASURED).
#
# WHY EXACTLY HERE, between the two kinds of precondition above and below. The required-argument
# loop just above asks a question about the ARGUMENT VECTOR -- "did the caller pass a --corpus at
# all" -- and a drift check wants that answered, since a step template that stops passing a required
# flag is the same defect class as one that passes a flag this launcher dropped. The three tests
# immediately below ask questions about this MACHINE -- does that corpus file exist here, is the
# 30B teacher staged on this filesystem -- which is not what a drift check is asking and not
# something a login node or a CPU-only suite can satisfy. Putting the lever on this line is what
# lets checks/precompute-drift.sh assert the template's flags without a 60 GB teacher on disk.
if [[ "${PRECOMPUTE_DRY_RUN:-0}" == "1" ]]; then
  echo "PRECOMPUTE_DRY_RUN=1: arguments parsed, nothing executed."
  echo "dry-run: all arguments accepted"
  exit 0
fi

[[ -f "$CORPUS" ]] || { echo "FATAL: corpus not found: $CORPUS" >&2; exit 2; }
[[ -d "$TEACHER_MODEL" ]] || { echo "FATAL: teacher model dir not found: $TEACHER_MODEL" >&2; exit 2; }
[[ -d "$TEACHER_TOKENIZER" ]] || { echo "FATAL: teacher tokenizer dir not found: $TEACHER_TOKENIZER" >&2; exit 2; }

if [[ -n "$HF_HOME_ARG" ]]; then
  export HF_HOME="$HF_HOME_ARG"
  echo "HF_HOME   : $HF_HOME"
fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# Same precondition as run-sft.sh and run-gold.sh: the teacher is a granite hybrid model whose
# Mamba2 hub kernels resolve at IMPORT time, so with HF_HUB_OFFLINE=1 a cold cache is fatal AFTER
# the GPUs are allocated (job 1136209).
#
# The flash-attn2 repo is asked for here for the same reason run-sft.sh asks for it and run-gold.sh
# does not: this step requests attn_implementation="flash_attention_2" explicitly. On this account
# FA2 comes from the kernels hub, not the absent `flash_attn` package (unbuildable: system CUDA
# 13.1 vs torch's cu128), and transformers resolves it at MODEL-LOAD time -- offline, after the
# allocation. precompute_logits.py does have a fallback, but the fallback is a WARNING that
# silently changes attention numerics for a whole corpus, so the preflight exists to keep it
# theoretical. checks/fa2-alias.sh asserts the repo id by reading it out of transformers.
if [[ -f "$LIB_DIR/gold-kernels.sh" ]]; then
  source "$LIB_DIR/gold-kernels.sh"
  require_hub_kernels "$PYBIN" kernels-community/flash-attn2 || exit 1
fi

mkdir -p "$OUTPUT_DIR" || { echo "FATAL: cannot create output_dir: $OUTPUT_DIR" >&2; exit 1; }

# The one place the artifact paths are announced, called from BOTH the SKIP path and the success
# path. run-align.sh's rule, and it matters as much here: a resumed recipe whose precompute says
# "already done" and then publishes nothing has broken the training step downstream.
publish_artifacts() {
  echo "GB_ARTIFACT_ID:teacher_logits GB_ARTIFACT_PATH:${OUTPUT_DIR}"
}

IGNORE_DOCS_FLAG=(--no-ignore-documents)
[[ "$IGNORE_DOCUMENTS" == "1" ]] && IGNORE_DOCS_FLAG=(--ignore-documents)
TOK_MISMATCH_FLAG=(--no-allow-tokenizer-mismatch)
[[ "$ALLOW_TOKENIZER_MISMATCH" == "1" ]] && TOK_MISMATCH_FLAG=(--allow-tokenizer-mismatch)

# Every argument that describes the ARTIFACT, in one array, used by all three invocations below.
# One array rather than three copies: the expectation the resume gate is asked about has to be the
# expectation the run then honours, and two argument lists that must agree by hand eventually do
# not -- which here would mean a SKIP decided about a different artifact than the one on disk.
PC_ARGS=(
  --input-jsonl "$CORPUS"
  --output-dir "$OUTPUT_DIR"
  --teacher-model "$TEACHER_MODEL"
  --teacher-tokenizer "$TEACHER_TOKENIZER"
  --top-k "$TOP_K"
  --max-length "$MAX_LENGTH"
  --dtype "$DTYPE"
  --shard-target-tokens "$SHARD_TARGET_TOKENS"
  --batch-size "$BATCH_SIZE"
  --seed "$SEED"
  --response-template "$RESPONSE_TEMPLATE"
  --max-skip-fraction "$MAX_SKIP_FRACTION"
  "${IGNORE_DOCS_FLAG[@]}"
  "${TOK_MISMATCH_FLAG[@]}"
)

# Shard mode is opt-in and BOTH halves are required: a count with no index (or the reverse) would
# otherwise fall back to deriving the split from the world size, which is the silent-residue-class
# failure this whole mode exists to make explicit.
if [[ -n "$SHARD_COUNT" || -n "$SHARD_INDEX" ]]; then
  if [[ -z "$SHARD_COUNT" || -z "$SHARD_INDEX" ]]; then
    echo "FATAL: --shard-index and --shard-count must be given together." >&2
    exit 2
  fi
  PC_ARGS+=( --shard-count "$SHARD_COUNT" --shard-index "$SHARD_INDEX" )
  echo "shard     : residue class $SHARD_INDEX of $SHARD_COUNT (single-node, no rendezvous)"
  echo "            THIS JOB DOES NOT MERGE. Run merge-index-parts.py + --verify-only after."
fi

# ------------------------------------------------------------------- resume
# THE GATE IS ASKED TWICE, of two genuinely different questions. Conflating them is how a resume
# path ends up either redoing finished work or appending to a mismatched artifact:
#
#   Q1, HERE, via step_state: "is this output dir a COMPLETE precompute of exactly this input?"
#       0=RUN / 64=SKIP / 65=REFUSE. This is the only question a marker can answer, because a
#       marker is written after the outputs and says nothing about a run that was killed.
#   Q2, INSIDE the module, via .precompute-expectation.json: "is the PARTIAL progress in this dir
#       progress on this same input?" A resume appends to existing shards, so it must refuse a
#       directory started from another corpus, top_k or tokenizer -- otherwise one index ends up
#       describing two precomputes and nothing anywhere raises.
#
# Both compare the SAME expectation document, and --emit-expectation computes it once. That is not
# only tidiness: the document contains the corpus md5, and this corpus is tens of GB.
EXPECTATION_FILE="$OUTPUT_DIR/.precompute-expectation.json"
echo
echo "--- [0/3] expectation (one corpus md5, reused by the resume gate and the run)"
"$PYBIN" -m "${PKG}.precompute_logits" "${PC_ARGS[@]}" \
  --emit-expectation "$OUTPUT_DIR/.expectation-current.json" || exit $?

STATE_ARGS=(
  --out-dir "$OUTPUT_DIR"
  --step distill-logit-precompute
  --expectation-file "$OUTPUT_DIR/.expectation-current.json"
  --output index.jsonl
  --output meta.json
)

echo
echo "--- [1/3] resume check"
# `|| STATE_RC=$?` rather than a bare call: 64 is not an error, it is the answer.
STATE_RC=0
"$PYBIN" -m "${PKG}.step_state" check "${STATE_ARGS[@]}" || STATE_RC=$?
case "$STATE_RC" in
  0)  ;;
  64) echo
      echo "marker says this precompute is already complete under this exact expectation."
      # AND THEN WE CHECK ANYWAY. The marker attests to what a previous process believed when it
      # finished; it cannot attest to what is on disk now. Re-verification is stat-only -- index
      # rows against the corpus row count, shard byte lengths against the index -- so it costs
      # seconds against a multi-hour pass, and it is the difference between "a marker exists" and
      # "the artifact this recipe is about to train on is intact".
      echo "re-verifying the artifact rather than trusting the marker:"
      if ! "$PYBIN" -m "${PKG}.precompute_logits" "${PC_ARGS[@]}" --verify-only; then
        echo
        echo "ERROR: the marker claims this precompute is complete, but the artifact does not" >&2
        echo "       verify (see above). Nothing was overwritten. Either restore the output dir" >&2
        echo "       or delete it and re-run: resume recomputes only what is missing." >&2
        exit 1
      fi
      echo
      echo "nothing to do."
      publish_artifacts
      exit 0 ;;
  65) echo
      echo "ERROR: refusing to touch ${OUTPUT_DIR}. See the key named above." >&2
      echo "       This directory was completed under a DIFFERENT expectation. Its shards are" >&2
      echo "       what a training run will read as teacher truth, so it is not silently" >&2
      echo "       rebuilt. Point --output-dir elsewhere, or delete it deliberately." >&2
      exit 1 ;;
  *)  echo
      echo "ERROR: the resume check itself failed (exit ${STATE_RC}). A step that cannot tell" >&2
      echo "       whether it has already run must not guess." >&2
      exit 1 ;;
esac

# ------------------------------------------------------------- allocation assertion
# run-sft.sh's rule verbatim, and the consequence of getting it wrong is different but no less
# quiet: gpus_per_node sets the TP width, so a pass that runs at 2 GPUs while the config claims 8
# does not fail -- it either OOMs much later on a long row or produces a correct-looking artifact
# from a differently sharded forward. CUDA_VISIBLE_DEVICES wins over nvidia-smi when set, because
# a scheduler granting 2 of 8 GPUs restricts it that way and `nvidia-smi -L` reports all eight.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  VISIBLE=$(awk -F, '{n=0; for(i=1;i<=NF;i++) if(length($i)) n++; print n}' <<<"$CUDA_VISIBLE_DEVICES")
  VISIBLE_SRC="CUDA_VISIBLE_DEVICES"
elif command -v nvidia-smi >/dev/null 2>&1; then
  VISIBLE=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)
  VISIBLE_SRC="nvidia-smi -L"
else
  VISIBLE=""
  VISIBLE_SRC=""
fi

if [[ -n "$VISIBLE" ]]; then
  echo "allocation: $VISIBLE GPU(s) visible (via $VISIBLE_SRC), gpus_per_node=$GPN"
  if (( VISIBLE != GPN )); then
    echo "FATAL: gpus_per_node=$GPN but $VISIBLE GPU(s) are visible (via $VISIBLE_SRC)." >&2
    echo "       gpus_per_node is the TP width here, so this is not a throughput question:" >&2
    echo "       the teacher is sharded across exactly that many ranks. Fix the profile's" >&2
    echo "       ACCELERATORS or the recipe's GPUS_PER_NODE so the two agree." >&2
    exit 1
  fi
else
  echo "WARNING: cannot count GPUs (no CUDA_VISIBLE_DEVICES, no nvidia-smi)." >&2
  echo "         gpus_per_node=$GPN is UNVERIFIED against the real allocation." >&2
fi

# ------------------------------------------------------------------ launch
if (( NODES > 1 )); then
  # D3, and it bites HARDER here than in run-sft.sh. This module's design is TP within a node and
  # DP ACROSS nodes over a 2-D device mesh, so multi-node is not a throughput option -- it is how
  # the corpus is divided. Launching one node while NODES claims more would not fail: each node
  # takes rows where `i % num_nodes == node_id`, so a single node started with num_nodes derived
  # from a world size that never materialized silently precomputes a RESIDUE CLASS of the corpus.
  # The completeness post-condition is what turns that into an error rather than a smaller shard
  # set -- but refusing here is cheaper than discovering it after a multi-hour pass.
  echo "FATAL: nodes=$NODES needs a multi-node rendezvous (main_process_ip/port and a" >&2
  echo "       machine_rank per host) that a step is not handed. Tracked as D3 -- see" >&2
  echo "       docs/planning/distillation-steps-plan.md. Use nodes=1 here, or run the" >&2
  echo "       multi-node shape from scripts/bluevela/ inside an LSF allocation where" >&2
  echo "       LSB_HOSTS supplies the host list." >&2
  exit 1
fi

NUM_PROCESSES=$(( NODES * GPN ))
echo "topology  : TP $GPN within 1 node, DP across $NODES node(s)"
echo
echo "--- [2/3] teacher forward pass"
# `-m`, and from no particular directory: the module imports from the package at module scope, so
# a flat launch would put the flat dir on sys.path and the package root nowhere -- jobs 1162604 /
# 1162625. Nothing here needs a cd.
"$ACCELERATE" launch \
  --num_processes "$NUM_PROCESSES" \
  --num_machines "$NODES" \
  -m "${PKG}.precompute_logits" "${PC_ARGS[@]}"
RC=$?

if (( RC != 0 )); then
  echo
  echo "precompute rc=$RC -- NO marker written, so a re-run resumes from the index." >&2
  echo "  The output dir is left in place on purpose: read_done_state() picks up every row" >&2
  echo "  already indexed, and the next start truncates any shard bytes written after the" >&2
  echo "  last index line. Deleting it discards finished work." >&2
  date
  echo "=== done (rc=$RC) ==="
  exit "$RC"
fi

# ------------------------------------------------------------------ mark complete
# AFTER the real outputs and after the module's own post-condition passed (rc=0 already implies
# it: precompute_logits.py runs verify_output() as its last act and exits 4 if it fails). The
# marker is written last so that a kill anywhere earlier leaves a directory that resumes rather
# than one that claims to be finished.
echo
echo "--- [3/3] recording completion"
"$PYBIN" -m "${PKG}.step_state" mark "${STATE_ARGS[@]}" || {
  echo "WARNING: the artifact is complete and verified, but the completion marker could not be" >&2
  echo "         written. A re-run will redo the pass instead of skipping it. Nothing is wrong" >&2
  echo "         with the output." >&2
}
# Now that it is recorded, the transient current-expectation file is the stored one.
rm -f "$OUTPUT_DIR/.expectation-current.json"

echo
publish_artifacts
echo "precompute rc=$RC  output=$OUTPUT_DIR"
date
echo "=== done (rc=$RC) ==="
exit "$RC"
