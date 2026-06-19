#!/usr/bin/env bash
# Launch 26 evals (25 sage + 1 BFCL) via granite.build on AWS L40S spot instances.
#
# This is the granite.build equivalent of gbansible/skypilot/run_all_evals.sh.
# It checks S3 for completed evals and only launches missing ones.
#
# NOTE: BigCodeBench is NOT included here — launch it separately:
#   gb build start -f recipes/granite4-350m/aws/bcb-eval/build.yaml \
#     --param NAME=<experiment> --param MODEL_S3=<s3_path>
#
# Usage:
#   ./scripts/run_all_evals.sh <s3_checkpoint_subpath> <experiment>
#
# Example:
#   export GBSERVER_SECRET_SKYPILOT_DOCKER_PASSWORD=$(aws ecr get-login-password --region us-east-2)
#   ./scripts/run_all_evals.sh sft/v0-20260614_093520-hf/step_hf_10000 eval-l40s-350m-002
#
# Controls:
#   DRY_RUN=1  — check state only, don't launch anything
#   FORCE=1    — re-run all evals regardless of state
#   GROUPED=1  — use grouped mode (5 instances) instead of individual (22 instances)
#
# Prerequisites:
#   - gbserver standalone running with GBSERVER_SECRET_SKYPILOT_DOCKER_PASSWORD set
#   - AWS credentials configured
#   - S3 buckets accessible

set -euo pipefail

CHECKPOINT_PATH=${1:?"Usage: $0 <s3_checkpoint_subpath> <experiment>"}
EXPERIMENT=${2:?"Usage: $0 <s3_checkpoint_subpath> <experiment>"}

DRY_RUN="${DRY_RUN:-0}"
GROUPED="${GROUPED:-0}"

MODEL_S3="s3://granite-build-checkpoints/${CHECKPOINT_PATH}"
S3_RESULTS_PREFIX="s3://granite-build-eval-results/sage/${EXPERIMENT}"

# ─── Eval tracking ───────────────────────────────────────────────────────────
EVAL_STATE_DIR=".eval_runs"
EVAL_STATE_FILE="${EVAL_STATE_DIR}/${EXPERIMENT}.completed"
mkdir -p "$EVAL_STATE_DIR"
touch "$EVAL_STATE_FILE"

# Status counters
declare -A EVAL_STATUS_MAP
EVALS_COMPLETED=0
EVALS_RUNNING=0
EVALS_PENDING=0
EVALS_INCOMPLETE=0

MULTILINGUAL_INDIVIDUAL_EVALS=(
  "multilingual-global-mmlu"
  "multilingual-mgsm"
  "multilingual-include-ar-de-es-fr"
  "multilingual-include-hi-bn-ta-te"
  "multilingual-include-it-ja-ko-nl-pt-zh"
)

# ─── Check if an eval is already completed ───────────────────────────────────
# Returns 0 (skip) if completed, 1 (launch) if not
should_skip_eval() {
  local eval_name="$1"

  if [[ "${FORCE:-0}" == "1" ]]; then
    EVAL_STATUS_MAP["$eval_name"]="pending (forced)"
    ((EVALS_PENDING++)) || true
    return 1
  fi

  # Check local state file
  if grep -qx "$eval_name" "$EVAL_STATE_FILE" 2>/dev/null; then
    EVAL_STATUS_MAP["$eval_name"]="completed (local)"
    ((EVALS_COMPLETED++)) || true
    return 0
  fi

  # Check S3 for .done marker
  if [[ "$eval_name" == "bfcl" ]]; then
    local bfcl_s3="s3://granite-build-eval-results/bfcl/${EXPERIMENT}/code-bfclv3/bfcl.done"
    if aws s3 ls "$bfcl_s3" --region us-east-2 2>/dev/null | grep -q .; then
      EVAL_STATUS_MAP["$eval_name"]="completed (S3)"
      ((EVALS_COMPLETED++)) || true
      echo "$eval_name" >> "$EVAL_STATE_FILE"
      return 0
    elif aws s3 ls "s3://granite-build-eval-results/bfcl/${EXPERIMENT}/code-bfclv3/bfcl.log" --region us-east-2 2>/dev/null | grep -q .; then
      EVAL_STATUS_MAP["$eval_name"]="incomplete (preempted)"
      ((EVALS_INCOMPLETE++)) || true
      return 1
    fi
  else
    if aws s3 ls "${S3_RESULTS_PREFIX}/${eval_name}.done" --region us-east-2 2>/dev/null | grep -q .; then
      EVAL_STATUS_MAP["$eval_name"]="completed (S3)"
      ((EVALS_COMPLETED++)) || true
      echo "$eval_name" >> "$EVAL_STATE_FILE"
      return 0
    elif aws s3 ls "${S3_RESULTS_PREFIX}/${eval_name}.log" --region us-east-2 2>/dev/null | grep -q .; then
      EVAL_STATUS_MAP["$eval_name"]="incomplete (preempted)"
      ((EVALS_INCOMPLETE++)) || true
      return 1
    fi
  fi

  EVAL_STATUS_MAP["$eval_name"]="pending"
  ((EVALS_PENDING++)) || true
  return 1
}

# ─── Check grouped eval (all sub-evals must be done) ─────────────────────────
should_skip_grouped_eval() {
  local group_name="$1"
  shift
  local -a individual_evals=("$@")

  if [[ "${FORCE:-0}" == "1" ]]; then
    for e in "${individual_evals[@]}"; do
      EVAL_STATUS_MAP["$e"]="pending (forced)"
      ((EVALS_PENDING++)) || true
    done
    return 1
  fi

  # Check local state
  if grep -qx "$group_name" "$EVAL_STATE_FILE" 2>/dev/null; then
    for e in "${individual_evals[@]}"; do
      EVAL_STATUS_MAP["$e"]="completed (local)"
      ((EVALS_COMPLETED++)) || true
    done
    return 0
  fi

  # Check S3 for each sub-eval
  local done_count=0
  for e in "${individual_evals[@]}"; do
    if aws s3 ls "${S3_RESULTS_PREFIX}/${e}.done" --region us-east-2 2>/dev/null | grep -q .; then
      EVAL_STATUS_MAP["$e"]="completed (S3)"
      ((EVALS_COMPLETED++)) || true
      ((done_count++)) || true
    elif aws s3 ls "${S3_RESULTS_PREFIX}/${e}.log" --region us-east-2 2>/dev/null | grep -q .; then
      EVAL_STATUS_MAP["$e"]="incomplete (preempted)"
      ((EVALS_INCOMPLETE++)) || true
    else
      EVAL_STATUS_MAP["$e"]="pending"
      ((EVALS_PENDING++)) || true
    fi
  done

  if [[ $done_count -eq ${#individual_evals[@]} ]]; then
    echo "$group_name" >> "$EVAL_STATE_FILE"
    return 0
  fi
  return 1
}

# ─── Launch a gb build ────────────────────────────────────────────────────────
launch_eval() {
  local name="$1"
  local build_file="$2"

  if [[ "$DRY_RUN" == "1" ]]; then
    return
  fi

  echo "[$(date)] Launching: $name"
  gb build start -f "$build_file" \
    --param NAME="${EXPERIMENT}" \
    --param MODEL_S3="${MODEL_S3}"
}

# ═════════════════════════════════════════════════════════════════════════════════
echo "============================================================"
echo " granite.build Eval Suite — 26 evals"
echo " Checkpoint: ${MODEL_S3}"
echo " Experiment: ${EXPERIMENT}"
echo " Mode:       $(if [[ "$GROUPED" == "1" ]]; then echo "GROUPED (5 instances)"; else echo "INDIVIDUAL (22 instances)"; fi)"
[[ "$DRY_RUN" == "1" ]] && echo " DRY RUN — no launches"
[[ "${FORCE:-0}" == "1" ]] && echo " FORCE — re-run all"
echo " NOTE: BigCodeBench launched separately"
echo "============================================================"
echo ""

if [[ "$GROUPED" == "1" ]]; then
  # ─── GROUPED MODE: 5 instances ──────────────────────────────────────────────

  # OLMES (11 evals)
  OLMES_EVALS=(
    "code-olmes-cruxeval" "general-olmes-agi-eval" "general-olmes-bbh"
    "general-olmes-mmlu-pro" "general-olmes-ifeval" "general-olmes-mmlu-mc"
    "math-olmes-deepmind-math" "math-olmes-gpqa" "math-olmes-gsm8k"
    "math-olmes-gsm-symbolic" "math-olmes-minerva-math"
  )
  if ! should_skip_grouped_eval "olmes-grouped" "${OLMES_EVALS[@]}"; then
    launch_eval "olmes-grouped (11 evals)" "recipes/granite4-350m/aws/olmes-eval/build.yaml"
  fi

  # CODE (7 evals)
  CODE_EVALS=(
    "code-evalplus-humaneval" "code-evalplus-mbpp"
    "code-multiple-sh" "code-multiple-cpp" "code-multiple-java"
    "code-multiple-js" "code-multiple-rs"
  )
  if ! should_skip_grouped_eval "code-grouped" "${CODE_EVALS[@]}"; then
    launch_eval "code-grouped (7 evals)" "recipes/granite4-350m/aws/code-eval/build.yaml"
  fi

  # SAFETY (2 evals)
  SAFETY_EVALS=("safety-attaq" "safety-salad-bench")
  if ! should_skip_grouped_eval "safety-grouped" "${SAFETY_EVALS[@]}"; then
    launch_eval "safety-grouped (2 evals)" "recipes/granite4-350m/aws/safety-eval/build.yaml"
  fi

  # MULTILINGUAL (5 evals)
  if ! should_skip_grouped_eval "multilingual-grouped" "${MULTILINGUAL_INDIVIDUAL_EVALS[@]}"; then
    launch_eval "multilingual-grouped (5 evals)" "recipes/granite4-350m/aws/multilingual-eval/build.yaml"
  fi

  # BFCL (1 eval)
  if ! should_skip_eval "bfcl"; then
    launch_eval "bfcl" "recipes/granite4-350m/aws/bfcl-eval/build.yaml"
  fi

else
  # ─── INDIVIDUAL MODE: 22 instances ─────────────────────────────────────────

  # OLMES evals (11) — each on its own instance
  declare -A OLMES_SCRIPTS=(
    ["code-olmes-cruxeval"]="code_olmes_cruxeval_slim.sh"
    ["general-olmes-agi-eval"]="general_olmes_agi_eval_slim.sh"
    ["general-olmes-bbh"]="general_olmes_bbh_slim.sh"
    ["general-olmes-mmlu-pro"]="general_olmes_mmlu_pro_slim.sh"
    ["general-olmes-ifeval"]="general_olmes_ifeval_slim.sh"
    ["general-olmes-mmlu-mc"]="general_olmes_mmlu_mc_slim.sh"
    ["math-olmes-deepmind-math"]="math_olmes_deepmind_math_slim.sh"
    ["math-olmes-gpqa"]="math_olmes_gpqa_slim.sh"
    ["math-olmes-gsm8k"]="math_olmes_gsm8k_slim.sh"
    ["math-olmes-gsm-symbolic"]="math_olmes_gsm_symbolic_slim.sh"
    ["math-olmes-minerva-math"]="math_olmes_minerva_math_slim.sh"
  )

  OLMES_PENDING=()
  for eval_name in "${!OLMES_SCRIPTS[@]}"; do
    should_skip_eval "$eval_name" || OLMES_PENDING+=("$eval_name")
  done

  if [[ ${#OLMES_PENDING[@]} -gt 0 && "$DRY_RUN" != "1" ]]; then
    echo "[$(date)] Launching ${#OLMES_PENDING[@]} OLMES evals via run-all-evals build"
    # Use the full run-all-evals build which launches all targets in parallel
    # For partial re-runs, we launch individually
    for eval_name in "${OLMES_PENDING[@]}"; do
      echo "[$(date)]   Starting: $eval_name (${OLMES_SCRIPTS[$eval_name]})"
      gb build start -f recipes/granite4-350m/aws/full-eval/build.yaml \
        --param NAME="${EXPERIMENT}" \
        --param MODEL_S3="${MODEL_S3}" \
        --target "${eval_name}" 2>/dev/null || \
      echo "    (--target not supported; use run-all-evals build for full suite)"
      break  # Launch full build once (all targets run in parallel)
    done
  fi

  # CODE evals (7)
  CODE_PENDING=()
  declare -A CODE_SCRIPTS=(
    ["code-evalplus-humaneval"]="code_evalplus_humaneval_slim.sh"
    ["code-evalplus-mbpp"]="code_evalplus_mbpp_slim.sh"
    ["code-multiple-sh"]="code_multiple_slim.sh"
    ["code-multiple-cpp"]="code_multiple_slim.sh"
    ["code-multiple-java"]="code_multiple_slim.sh"
    ["code-multiple-js"]="code_multiple_slim.sh"
    ["code-multiple-rs"]="code_multiple_slim.sh"
  )
  for eval_name in "${!CODE_SCRIPTS[@]}"; do
    should_skip_eval "$eval_name" || CODE_PENDING+=("$eval_name")
  done

  # SAFETY evals (2)
  SAFETY_PENDING=()
  declare -A SAFETY_SCRIPTS=(
    ["safety-attaq"]="safety_attaq_slim.sh"
    ["safety-salad-bench"]="safety_salad_bench_slim.sh"
  )
  for eval_name in "${!SAFETY_SCRIPTS[@]}"; do
    should_skip_eval "$eval_name" || SAFETY_PENDING+=("$eval_name")
  done

  # MULTILINGUAL (5 grouped)
  ML_SKIP=0
  should_skip_grouped_eval "multilingual-grouped" "${MULTILINGUAL_INDIVIDUAL_EVALS[@]}" && ML_SKIP=1

  # BFCL (1)
  BFCL_SKIP=0
  should_skip_eval "bfcl" && BFCL_SKIP=1

  # If there are any pending evals across groups, launch the full build
  TOTAL_PENDING=$((${#OLMES_PENDING[@]} + ${#CODE_PENDING[@]} + ${#SAFETY_PENDING[@]} + (1 - ML_SKIP) + (1 - BFCL_SKIP)))

  if [[ $TOTAL_PENDING -gt 0 && "$DRY_RUN" != "1" ]]; then
    echo ""
    echo "[$(date)] Launching full eval build (${TOTAL_PENDING} eval groups pending)"
    echo "  OLMES pending: ${#OLMES_PENDING[@]}/11"
    echo "  CODE pending:  ${#CODE_PENDING[@]}/7"
    echo "  SAFETY pending: ${#SAFETY_PENDING[@]}/2"
    echo "  MULTILINGUAL:  $([[ $ML_SKIP -eq 0 ]] && echo "pending" || echo "done")"
    echo "  BFCL:          $([[ $BFCL_SKIP -eq 0 ]] && echo "pending" || echo "done")"
    echo ""
    gb build start -f recipes/granite4-350m/aws/full-eval/build.yaml \
      --param NAME="${EXPERIMENT}" \
      --param MODEL_S3="${MODEL_S3}"
  fi
fi

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Eval Status Summary — ${EXPERIMENT}"
echo "============================================================"
printf "  %-45s %s\n" "EVAL" "STATUS"
printf "  %-45s %s\n" "----" "------"
for eval_name in $(echo "${!EVAL_STATUS_MAP[@]}" | tr ' ' '\n' | sort); do
  printf "  %-45s %s\n" "$eval_name" "${EVAL_STATUS_MAP[$eval_name]}"
done
echo "------------------------------------------------------------"
printf "  Completed: %d | Running: %d | Incomplete: %d | Pending: %d | Total: %d\n" \
  "$EVALS_COMPLETED" "$EVALS_RUNNING" "$EVALS_INCOMPLETE" "$EVALS_PENDING" \
  $((EVALS_COMPLETED + EVALS_RUNNING + EVALS_INCOMPLETE + EVALS_PENDING))
echo "============================================================"

if [[ "$DRY_RUN" == "1" ]]; then
  echo ""
  echo "  DRY RUN — no builds launched."
  echo "  To launch: $0 $CHECKPOINT_PATH $EXPERIMENT"
  echo "  Grouped:   GROUPED=1 $0 $CHECKPOINT_PATH $EXPERIMENT"
else
  echo ""
  echo "  Monitor: gb build list"
  echo "  Clusters: sky status"
  echo "  Results: aws s3 ls s3://granite-build-eval-results/sage/${EXPERIMENT}/"
  echo ""
  echo "  BigCodeBench (requires evaluator sidecar — launch separately):"
  echo "    gb build start -f recipes/granite4-350m/aws/bcb-eval/build.yaml \\"
  echo "      --param NAME=${EXPERIMENT} --param MODEL_S3=${MODEL_S3}"
fi
echo "============================================================"
