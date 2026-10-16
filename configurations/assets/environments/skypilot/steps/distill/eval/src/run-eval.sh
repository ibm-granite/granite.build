#!/usr/bin/env bash
#
# PORTED, not authored here. Upstream source of truth:
#   repo   github.ibm.com/Herbert-Woisetschlaeger/gb-steps-collection-post-training
#   path   steps/distill-eval/src/run-eval.sh
#   commit 70c1550a171aa8e09a9ad9047a5bf763c39e8579
#
# The ONLY intentional divergence is the artifact marker prefix: LLMB_ARTIFACT_* ->
# GB_ARTIFACT_*. The monitor accepts both; GB_ is what every other step here prints.
#
# NOTE on where the marker lives: this script owns it, via publish_artifacts(), which it
# calls on both the success path and the already-measured SKIP path. Upstream's
# step-template echoed the same marker AGAIN after the script returned; the port drops
# that duplicate rather than registering one artifact id twice.
#
# Keep this a near-verbatim copy so re-syncing upstream stays a three-way merge.
#
# distill-eval: measure how far the student's distribution sits from the teacher's on a
# held-out corpus, and how peaked the student's own predictions are.
#
# WHY THIS STEP EXISTS ALONGSIDE steps/bfcl-eval, WHICH ALREADY EVALUATES MODELS.
# BFCL scores tool-calling accuracy against ground truth: it answers "is the student
# right?". It structurally cannot answer "did the student move toward the TEACHER?",
# because that is not a property of one model's outputs -- it is a distance between two
# models' distributions over the same inputs, and BFCL sees only one model and a label.
# For a distillation recipe that distance is the direct read on whether transfer happened,
# and it needs no labelled data at all. So the recipe wires BOTH: bfcl-eval for capability,
# this for transfer. The plan doc asked for that finding to be recorded rather than routed
# around, and this header is where it lives.
#
# ONE PROCESS, ALL METRICS. Every metric is a reduction over the same pair of logit
# tensors, so --metrics jsd,kld,rkld,entropy costs one forward pass over the corpus, not
# four. The scratchpad ran compute_jsd.py and compute_entropy.py as separate invocations,
# which reloaded a 30B teacher per metric.
set -euo pipefail

STUDENT_MODEL=""
TEACHER_MODEL=""
CORPUS=""
OUT_DIR=""
METRICS="jsd,entropy"
MAX_SAMPLES="256"
MAX_LENGTH="4096"
BATCH_SIZE="4"
SEED="42"
DTYPE="bfloat16"
MAX_INCOMPLETE_FRACTION="0.25"
ALLOW_TOKENIZER_MISMATCH="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --student-model)   STUDENT_MODEL="$2"; shift 2 ;;
    --teacher-model)   TEACHER_MODEL="$2"; shift 2 ;;
    --corpus)          CORPUS="$2"; shift 2 ;;
    --out-dir)         OUT_DIR="$2"; shift 2 ;;
    --metrics)         METRICS="$2"; shift 2 ;;
    --max-samples)     MAX_SAMPLES="$2"; shift 2 ;;
    --max-length)      MAX_LENGTH="$2"; shift 2 ;;
    --batch-size)      BATCH_SIZE="$2"; shift 2 ;;
    --seed)            SEED="$2"; shift 2 ;;
    --dtype)           DTYPE="$2"; shift 2 ;;
    --max-incomplete-fraction) MAX_INCOMPLETE_FRACTION="$2"; shift 2 ;;
    --allow-tokenizer-mismatch)    ALLOW_TOKENIZER_MISMATCH="true"; shift ;;
    --no-allow-tokenizer-mismatch) ALLOW_TOKENIZER_MISMATCH="false"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

for name in STUDENT_MODEL CORPUS OUT_DIR; do
  if [[ -z "${!name}" ]]; then
    echo "ERROR: --${name,,} is required" | tr '_' '-' >&2
    exit 2
  fi
done

# PYBIN is not hardcoded to `python`, and that is not defensive style -- it is the fix for
# a defect that executing the rendered command caught in distill-corpus-prep (LSF job
# 1137785): a launcher that says `python` works in the image and fails everywhere the step
# is verified outside it, which is where these steps are actually exercised first.
PYBIN="${PYBIN:-python}"
PKG="gb_steps_post_training.distillation"

echo "=== distill-eval ==="
echo "  student : ${STUDENT_MODEL}"
echo "  teacher : ${TEACHER_MODEL:-<none: single-model metrics only>}"
echo "  corpus  : ${CORPUS}"
echo "  out     : ${OUT_DIR}"
echo "  metrics : ${METRICS}   max_samples: ${MAX_SAMPLES}   max_length: ${MAX_LENGTH}"

# Printed from ONE place, because a step that reports SKIP still has to hand its consumer the
# path. A resumed recipe whose eval says "already measured" and then publishes nothing has
# broken whatever reads eval_metrics, which is a worse failure than re-measuring.
publish_artifacts() {
  echo
  echo "GB_ARTIFACT_ID:eval_metrics GB_ARTIFACT_PATH:${OUT_DIR}"
}

# ------------------------------------------------------------------- resume
# A recipe on the preemptable queue is RESTARTED, not resumed: every step re-runs from the top,
# so each answers "has this already been done, and with what?" for itself. eval_state.py is this
# step's answer, and its docstring carries the reasoning that matters most here -- this step's
# output is a NUMBER ABOUT A MODEL THAT LIVES AT A FIXED PATH, so a marker keyed on
# --student-model would report the PREVIOUS student's divergence as the new one's. Identity is
# the student's and teacher's content plus the sampling and length policy.
#
#   0  RUN     nothing recorded here; measure
#  64  SKIP    recorded under this EXACT expectation; publish and exit
#  65  REFUSE  recorded under a different one, or metrics.json / per_sample.jsonl is gone. The
#              message names the key. Nothing is overwritten, because a silently rebuilt
#              metrics.json is indistinguishable from the one an arm was accepted on.
#
# Every forwarded flag is one eval_state compares EXCEPT --batch-size, which it accepts and
# ignores on purpose (see its _common docstring). It is forwarded anyway so that this list and
# run_divergence's differ in no line, which is how a divergence between them stays visible.
STATE_ARGS=(
  --student-model "$STUDENT_MODEL"
  --corpus "$CORPUS"
  --out-dir "$OUT_DIR"
  --metrics "$METRICS"
  --max-samples "$MAX_SAMPLES"
  --max-length "$MAX_LENGTH"
  --max-incomplete-fraction "$MAX_INCOMPLETE_FRACTION"
  --batch-size "$BATCH_SIZE"
  --seed "$SEED"
  --dtype "$DTYPE"
)
# Omitted, not blanked -- the same reasoning as the --teacher-model forwarding below: an empty
# teacher is "no teacher", which is a different expectation from any teacher, and passing "" would
# make it a nonexistent path instead.
if [[ -n "$TEACHER_MODEL" ]]; then
  STATE_ARGS+=(--teacher-model "$TEACHER_MODEL")
fi
if [[ "$ALLOW_TOKENIZER_MISMATCH" == "true" ]]; then
  STATE_ARGS+=(--allow-tokenizer-mismatch)
else
  STATE_ARGS+=(--no-allow-tokenizer-mismatch)
fi

echo
echo "--- [0/2] resume check"
# `|| STATE_RC=$?` rather than a bare call: `set -e` would abort on 64, which is not an error but
# the whole point of asking.
STATE_RC=0
"$PYBIN" -m "${PKG}.eval_state" check "${STATE_ARGS[@]}" || STATE_RC=$?
case "$STATE_RC" in
  0)  ;;
  64) echo
      echo "nothing to do: ${OUT_DIR} already holds this measurement."
      publish_artifacts
      exit 0 ;;
  65) echo
      echo "ERROR: refusing to overwrite ${OUT_DIR}. See the key named above." >&2
      exit 1 ;;
  *)  echo
      echo "ERROR: the resume check itself failed (exit ${STATE_RC}). Fix that before" >&2
      echo "       running the step: a step that cannot tell whether it has already run" >&2
      echo "       must not guess." >&2
      exit 1 ;;
esac

ARGS=(
  --student-model "$STUDENT_MODEL"
  --corpus "$CORPUS"
  --out "$OUT_DIR"
  --metrics "$METRICS"
  --max-samples "$MAX_SAMPLES"
  --max-length "$MAX_LENGTH"
  --batch-size "$BATCH_SIZE"
  --seed "$SEED"
  --dtype "$DTYPE"
  --max-incomplete-fraction "$MAX_INCOMPLETE_FRACTION"
)
# Passed only when non-empty. An empty --teacher-model would be a path that does not exist
# rather than an absent teacher, and the difference decides whether a divergence is even
# well-defined -- so the flag is omitted, not blanked.
if [[ -n "$TEACHER_MODEL" ]]; then
  ARGS+=(--teacher-model "$TEACHER_MODEL")
fi
if [[ "$ALLOW_TOKENIZER_MISMATCH" == "true" ]]; then
  # Loud, because the resulting numbers are not comparable to anything. The module refuses
  # by default for that reason; this echo makes the override visible in the step's own log
  # rather than only in metrics.json.
  echo "  WARNING: --allow-tokenizer-mismatch is set. If the two tokenizers disagree on" >&2
  echo "  token ids, the divergence is arithmetic over mismatched vocabularies and cannot" >&2
  echo "  be compared across runs or models." >&2
  ARGS+=(--allow-tokenizer-mismatch)
fi

"$PYBIN" -m "${PKG}.run_divergence" "${ARGS[@]}"

# Post-condition on the step's own output. The module writes metrics.json and exits 0, but
# "exited 0" and "produced a usable metric" are different claims, and this step's whole
# value is the number -- an empty or NaN mean that reaches a recipe as a green step is the
# failure mode worth spending a check on.
"$PYBIN" - "$OUT_DIR" <<'PYCHECK'
import json, math, sys
from pathlib import Path
metrics_file = Path(sys.argv[1]) / "metrics.json"
if not metrics_file.is_file():
    print(f"ERROR: {metrics_file} was not written", file=sys.stderr)
    raise SystemExit(1)
payload = json.loads(metrics_file.read_text())
summaries = payload.get("metrics") or {}
if not summaries:
    print("ERROR: metrics.json contains no metric summaries", file=sys.stderr)
    raise SystemExit(1)
bad = []
for name, s in summaries.items():
    if s.get("n_samples", 0) < 1:
        bad.append(f"{name}: n_samples={s.get('n_samples')}")
    mean = s.get("mean")
    if mean is None or math.isnan(mean) or math.isinf(mean):
        bad.append(f"{name}: mean={mean}")
    # A divergence is non-negative by definition and so is an entropy. A negative value
    # means the reduction is wrong, which is not something a caller should have to notice.
    elif mean < 0:
        bad.append(f"{name}: mean={mean} is negative")
    # JSD with a natural log is bounded above by ln 2 REGARDLESS of the two models. This is
    # the only bound here that does not depend on the vocabulary, so it is the one check
    # that catches a broken mixture term rather than an unexpected model.
    if name == "jsd" and mean is not None and mean > math.log(2) + 1e-6:
        bad.append(f"jsd: mean={mean} exceeds ln2={math.log(2):.6f}")
if bad:
    print("ERROR: metrics.json failed its post-condition:", file=sys.stderr)
    for b in bad:
        print(f"  {b}", file=sys.stderr)
    raise SystemExit(1)
for name, s in sorted(summaries.items()):
    print(f"  verified: {name} mean={s['mean']:.6f} over {s['n_samples']} samples")
# Reported, not merely tolerated. An operator reading only the means cannot tell whether
# they cover whole completions or the first 2048 tokens of them, and the truncated case is
# the one that leaves no other trace: it does not reduce n_samples (job 1138147 scored 16 of
# 16 records at --max-length 2048 while cutting roughly half of every answer).
counts = payload.get("counts", {})
if counts.get("skipped_no_assistant_span"):
    print(f"  NOTE: {counts['skipped_no_assistant_span']} sampled record(s) had no measurable "
          f"assistant span and were dropped; the reported mean is over the remainder, which "
          f"is shorter than the corpus.")
if counts.get("truncated_at_max_length"):
    print(f"  NOTE: {counts['truncated_at_max_length']} sampled record(s) were TRUNCATED at "
          f"max_length; {counts.get('truncated_tokens_dropped', 0)} completion token(s) were "
          f"never scored. The means above are over measured prefixes, not whole answers -- "
          f"raise max_length to compare against a full-length training run.")
PYCHECK

# The marker goes LAST, after the post-condition above, and that ordering is what makes it
# trustworthy: written before the check, it would record a NaN or negative mean as a completed
# measurement and the next restart would walk straight past it. step_state writes it atomically,
# so it cannot appear beside a half-written per_sample.jsonl.
echo
echo "--- marking complete"
"$PYBIN" -m "${PKG}.eval_state" mark "${STATE_ARGS[@]}"

publish_artifacts
