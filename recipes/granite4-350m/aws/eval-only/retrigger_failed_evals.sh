#!/usr/bin/env bash
######
# retrigger_failed_evals.sh — Find failed eval steps and retrigger only those.
#
# Identifies which specific eval types (multilingual, bcb, bfcl) failed for
# which checkpoints and launches only the failed combinations.
#
# Usage:
#   ./recipes/granite4-350m/aws/eval-only/retrigger_failed_evals.sh [LOG_FILE] [EXPERIMENT_PREFIX]
#
# Examples:
#   DRY_RUN=1 ./recipes/granite4-350m/aws/eval-only/retrigger_failed_evals.sh
#   ./recipes/granite4-350m/aws/eval-only/retrigger_failed_evals.sh /tmp/standalone.log test1-l40s-4
######
set -euo pipefail

LOG_FILE="${1:-/tmp/standalone.log}"
EXPERIMENT_PREFIX="${2:-}"
SCRIPT_DIR="recipes/granite4-350m/aws/eval-only"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "$LOG_FILE" ]]; then
    echo "ERROR: Log file not found: $LOG_FILE"
    exit 1
fi

if [[ ! -d "$SCRIPT_DIR" ]]; then
    echo "ERROR: Eval recipes not found at: $SCRIPT_DIR"
    echo "Run from the granite.build root directory."
    exit 1
fi

echo "Scanning $LOG_FILE for failed eval steps..."
echo ""

# Step 1: Identify which eval step types failed
FAILED_STEP_URIS=$(grep "stored_step_run.*status=<Status.FAILED" "$LOG_FILE" \
    | grep -E "sage-eval|bfcl-eval" \
    | sed -n "s/.*definition_uri='//p" \
    | sed "s/'.*//" \
    | sort -u)

if [[ -z "$FAILED_STEP_URIS" ]]; then
    echo "No failed eval steps found."
    exit 0
fi

echo "Failed eval types:"
echo "$FAILED_STEP_URIS" | while read -r uri; do
    case "$uri" in
        *sage-eval-multilingual*) echo "  - multilingual" ;;
        *sage-eval-bcb*) echo "  - bigcodebench (bcb)" ;;
        *bfcl-eval*) echo "  - bfcl" ;;
        *) echo "  - $uri" ;;
    esac
done
echo ""

# Step 2: Find all checkpoint paths that were evaluated (from rendered FULL CONFIG lines)
ALL_EVAL_PATHS=$(grep "FULL CONFIG" "$LOG_FILE" \
    | grep "model_source_s3" \
    | sed -n "s/.*model_source_s3': '//p" \
    | sed "s/'.*//" \
    | grep "^s3://.*-hf" \
    | sort -u)

if [[ -z "$ALL_EVAL_PATHS" ]]; then
    echo "No rendered checkpoint paths found in logs."
    echo "Tip: Use the build template directly:"
    echo "  gb build start -f $SCRIPT_DIR/build.yaml --param NAME=<name> --param MODEL_S3=<s3-path>"
    exit 0
fi

# Step 3: Build retrigger list — each failed step type × each checkpoint
RETRIGGER_LIST=""
while read -r step_uri; do
    while read -r model_s3; do
        RETRIGGER_LIST="${RETRIGGER_LIST}${step_uri}|${model_s3}"$'\n'
    done <<< "$ALL_EVAL_PATHS"
done <<< "$FAILED_STEP_URIS"

# Deduplicate
RETRIGGER_LIST=$(echo "$RETRIGGER_LIST" | grep -v "^$" | sort -u)

COUNT=$(echo "$RETRIGGER_LIST" | wc -l | tr -d ' ')
echo "Found $COUNT failed eval(s) to retrigger:"
echo "$RETRIGGER_LIST" | while IFS='|' read -r step_uri model_s3; do
    case "$step_uri" in
        *sage-eval-multilingual*) eval_name="multilingual" ;;
        *sage-eval-bcb*) eval_name="bcb" ;;
        *bfcl-eval*) eval_name="bfcl" ;;
        *) eval_name="$step_uri" ;;
    esac
    echo "  $eval_name → $model_s3"
done
echo ""

# Step 3: Launch each failed eval
echo "$RETRIGGER_LIST" | while IFS='|' read -r step_uri model_s3; do
    [[ -z "$step_uri" || -z "$model_s3" ]] && continue

    # Map step URI to build template and name
    case "$step_uri" in
        *sage-eval-multilingual*)
            build_yaml="$SCRIPT_DIR/build-multilingual.yaml"
            eval_name="multilingual" ;;
        *sage-eval-bcb*)
            build_yaml="$SCRIPT_DIR/build-bcb.yaml"
            eval_name="bcb" ;;
        *bfcl-eval*)
            build_yaml="$SCRIPT_DIR/build-bfcl.yaml"
            eval_name="bfcl" ;;
        *)
            echo "WARNING: Unknown step URI $step_uri, skipping"
            continue ;;
    esac

    if [[ ! -f "$build_yaml" ]]; then
        echo "WARNING: Build template not found: $build_yaml, skipping"
        continue
    fi

    # Use the original experiment name (the first path segment after the bucket)
    # e.g., s3://granite-build-checkpoints/test1-l40s-4/v0-... → test1-l40s-4
    experiment_part=$(echo "$model_s3" | sed 's|.*granite-build-checkpoints/||' | awk -F/ '{print $1}')
    NAME="${experiment_part}"

    # Filter by experiment prefix if specified
    if [[ -n "$EXPERIMENT_PREFIX" ]] && [[ "$experiment_part" != "$EXPERIMENT_PREFIX"* ]]; then
        continue
    fi

    echo "Triggering $eval_name eval for: $model_s3"
    echo "  NAME=$NAME"
    echo "  BUILD=$build_yaml"

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "  [DRY RUN] gb build start -f $build_yaml --param NAME=$NAME --param MODEL_S3=$model_s3"
    else
        gb build start -f "$build_yaml" --param "NAME=$NAME" --param "MODEL_S3=$model_s3" &
        echo "  Launched (pid=$!)"
    fi
    echo ""
done

echo "Done. $COUNT eval(s) to retrigger."
if [[ "$DRY_RUN" == "1" ]]; then
    echo "(Dry run — nothing was actually launched. Unset DRY_RUN to execute.)"
fi
