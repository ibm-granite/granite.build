#!/usr/bin/env bash
######
# build_status.sh — Show build progress: SFT training, checkpoints, evals, and retries.
#
# Usage:
#   ./recipes/granite4-350m/aws/eval-only/build_status.sh [LOG_FILE]
#   watch -n 10 ./recipes/granite4-350m/aws/eval-only/build_status.sh
######
set -uo pipefail

LOG_FILE="${1:-/tmp/standalone.log}"

if [[ ! -f "$LOG_FILE" ]]; then
    echo "ERROR: Log file not found: $LOG_FILE"
    exit 1
fi

# Strip ANSI escape codes for reliable grep matching
CLEAN_LOG=$(mktemp)
trap "rm -f $CLEAN_LOG" EXIT
sed 's/\x1b\[[0-9;]*m//g' "$LOG_FILE" > "$CLEAN_LOG"
LOG_FILE="$CLEAN_LOG"

echo "════════════════════════════════════════════════════════════════"
echo "  Build Status ($(date '+%H:%M:%S'))"
echo "  Log: $LOG_FILE"
echo "════════════════════════════════════════════════════════════════"
echo ""

# --- SFT Training ---
echo "┌─ SFT Training ─────────────────────────────────────────────"
SFT_STATUS=$(grep "stored_step_run.*openinstruct-sft.*status=<Status" "$LOG_FILE" \
    | tail -1 \
    | sed -n 's/.*status=<Status\.\([A-Z]*\).*/\1/p')
SFT_STATUS=${SFT_STATUS:-"NOT STARTED"}

# Get latest tqdm progress bar line
PROGRESS_BAR=$(grep "get_events_from_log_line.*it/s\|get_events_from_log_line.*it\]" "$LOG_FILE" \
    | tail -1 \
    | sed -n 's/.*] Running get_events_from_log_line for line [0-9]*: *//p' \
    | sed 's/[0-9\/]* [0-9:,]* - INFO - __main__ -   Step:.*//' \
    | sed 's/[[:space:]]*$//')
if [[ -z "$PROGRESS_BAR" ]]; then
    LAST_STEP=$(grep "JSON Log Event.*Step:" "$LOG_FILE" \
        | tail -1 \
        | sed -n "s/.*'message': 'Step: //p" \
        | sed "s/'.*//")
    [[ -n "$LAST_STEP" ]] && PROGRESS_BAR="Step $LAST_STEP"
fi

# Get cluster name
SFT_CLUSTER=$(grep "Started live log stream for\|Launching SkyPilot cluster" "$LOG_FILE" \
    | grep "openinstruct\|sft" \
    | tail -1 \
    | sed -n 's/.*stream for \([^ ]*\).*/\1/p')
if [[ -z "$SFT_CLUSTER" ]]; then
    SFT_CLUSTER=$(grep "SkyPilot cluster.*launched" "$LOG_FILE" | tail -1 | sed -n 's/.*cluster \([^ ]*\).*/\1/p')
fi

echo "│  Status: $SFT_STATUS"
[[ -n "$PROGRESS_BAR" ]] && echo "│  Progress: $PROGRESS_BAR"
[[ -n "$SFT_CLUSTER" ]] && echo "│  Cluster: $SFT_CLUSTER"
echo "└────────────────────────────────────────────────────────────"
echo ""

# --- Checkpoints ---
echo "┌─ Checkpoints Emitted ──────────────────────────────────────"
CHECKPOINTS=$(grep "newartifact_in_environment_event.*ArtifactEventPayload" "$LOG_FILE" \
    | grep "binding_id='checkpoint'" \
    | sed -n "s/.*'path': '//p" \
    | sed "s/'}.*//" \
    | sort -u)

FINAL=$(grep "newartifact_in_environment_event.*ArtifactEventPayload" "$LOG_FILE" \
    | grep "binding_id='final'" \
    | sed -n "s/.*'path': '//p" \
    | sed "s/'}.*//" \
    | sort -u)

if [[ -n "$CHECKPOINTS" ]]; then
    echo "$CHECKPOINTS" | while read -r ckpt; do
        basename=$(echo "$ckpt" | awk -F/ '{print $NF}')
        short=$(echo "$basename" | sed 's/epoch_hf_/e/;s/step_hf_/s/')
        echo "│  ✓ checkpoint: $short  ($ckpt)"
    done
else
    echo "│  (none yet)"
fi

if [[ -n "$FINAL" ]]; then
    echo "│  ✓ final: $FINAL"
fi
echo "└────────────────────────────────────────────────────────────"
echo ""

# --- Eval Targets ---
echo "┌─ Eval Targets ─────────────────────────────────────────────"

# Get all eval step runs with their status
EVAL_STEPS=$(grep "stored_step_run.*status=<Status" "$LOG_FILE" \
    | grep -E "sage-eval|bfcl-eval" \
    | sed "s/.*target_name='//;s/'.*definition_uri='/ /;s/'.*status=<Status\./ /;s/[:'].*//;s/  */ /g" \
    | sort -u)

if [[ -n "$EVAL_STEPS" ]]; then
    # Get final status per target_id (last status wins), include checkpoint path
    grep "stored_step_run.*status=<Status" "$LOG_FILE" \
        | grep -E "sage-eval|bfcl-eval" \
        | while IFS= read -r line; do
            target_name=$(echo "$line" | sed -n "s/.*target_name='//p" | sed "s/'.*//")
            step_uri=$(echo "$line" | sed -n "s/.*definition_uri='//p" | sed "s/'.*//")
            status=$(echo "$line" | sed -n "s/.*status=<Status\.//p" | sed "s/[:'].*//")
            target_id=$(echo "$line" | sed -n "s/.*target_id='//p" | sed "s/'.*//")
            # Extract experiment name from config (contains checkpoint info)
            experiment=$(echo "$line" | sed -n "s/.*'experiment': '//p" | sed "s/'.*//")
            # Also get uuid (targetsteprun_id) for FULL CONFIG lookup
            step_run_id=$(echo "$line" | sed -n "s/.*uuid='//p" | sed "s/'.*//")
            echo "$target_id|$target_name|$step_uri|$status|$experiment|$step_run_id"
        done | sort -t'|' -k1,1 -k4,4 | awk -F'|' '
            {latest[$1]=$0}
            END {for (id in latest) print latest[id]}
        ' | sort -t'|' -k5,5 -k2,2 | while IFS='|' read -r tid tname suri status experiment step_run_id; do
            case "$suri" in
                *sage-eval-multilingual*) eval_type="multilingual" ;;
                *sage-eval-bcb*) eval_type="bcb" ;;
                *bfcl-eval*) eval_type="bfcl" ;;
                *) eval_type="$suri" ;;
            esac
            case "$status" in
                SUCCESS) icon="✅" ;;
                FAILED) icon="❌" ;;
                RUNNING) icon="⚡" ;;
                PENDING) icon="🔵" ;;
                *) icon="?" ;;
            esac
            # Get checkpoint from rendered model_source_s3 in FULL CONFIG
            # Try step_run_id first (appears in targetsteprun logs), then target_id
            ckpt=""
            for lookup_id in "$step_run_id" "$tid"; do
                [[ -z "$lookup_id" ]] && continue
                model_path=$(grep "$lookup_id" "$LOG_FILE" \
                    | grep "FULL CONFIG\|model_source_s3" \
                    | sed -n "s/.*model_source_s3': '//p" \
                    | sed "s/'.*//" \
                    | grep "^s3://" \
                    | head -1)
                if [[ -n "$model_path" ]]; then
                    raw_ckpt=$(echo "$model_path" | awk -F/ '{print $NF}')
                    ckpt=$(echo "$raw_ckpt" | sed 's/epoch_hf_/e/;s/step_hf_/s/')
                    break
                fi
            done
            # Fallback: try experiment name
            if [[ -z "$ckpt" && -n "$experiment" ]]; then
                ckpt=$(echo "$experiment" | sed 's/.*-\(e[0-9]*\)$/\1/;s/.*-\(s[0-9]*\)$/\1/;s/.*-\(final\)$/\1/')
                [[ "$ckpt" == "$experiment" ]] && ckpt=""
            fi
            if [[ -n "$ckpt" ]]; then
                echo "│  $icon $eval_type [$ckpt] — $status"
            else
                echo "│  $icon $eval_type — $status"
            fi
        done
else
    echo "│  (no evals triggered yet)"
fi
echo "└────────────────────────────────────────────────────────────"
echo ""

# --- Retries ---
echo "┌─ Retries ───────────────────────────────────────────────────"

# Count retries per target by looking for retry-related log entries with launch_ids
# Each "Waiting X seconds before retry" or "Detected WORKLOAD_STATUS_EVENT FAILED"
# is associated with a launch_id which maps to a target
RETRY_LINES=$(grep "Waiting.*seconds before retry\|AnyFailureRetryStrategy.*Detected\|Transient provision failure" "$LOG_FILE" || true)

if [[ -n "$RETRY_LINES" ]]; then
    # Extract launch_ids from retry lines and cross-reference with target names
    # Group retries by the step that triggered them
    echo "$RETRY_LINES" | while IFS= read -r line; do
        ts=$(echo "$line" | sed -n 's/.*\[\([0-9-]* [0-9:,]*\)\].*/\1/p')
        launch_id=$(echo "$line" | sed -n 's/.*launch_id \([^ ]*\).*/\1/p')
        if [[ -z "$launch_id" ]]; then
            launch_id=$(echo "$line" | sed -n 's/.*launch_id=\([^ ]*\).*/\1/p')
        fi
        echo "│    [$ts] launch=$launch_id"
    done | tail -10

    # Count provision retries (inner) vs handler retries (outer)
    PROVISION_RETRIES=$(grep -c "Transient provision failure" "$LOG_FILE" || true)
    HANDLER_RETRIES=$(grep -c "Waiting.*seconds before retry\|AnyFailureRetryStrategy.*Detected" "$LOG_FILE" || true)

    echo "│"
    echo "│  Provision retries (inner): ${PROVISION_RETRIES:-0}"
    echo "│  Handler retries (outer):   ${HANDLER_RETRIES:-0}"

    # Show last few retry events with timestamps
    echo "$RETRY_LINES" | tail -5 | while IFS= read -r line; do
        ts=$(echo "$line" | sed -n 's/.*\[\([0-9-]* [0-9:,]*\)\].*/\1/p')
        if echo "$line" | grep -q "Transient provision failure"; then
            cluster=$(echo "$line" | sed -n 's/.*failure for \([^ ]*\).*/\1/p')
            attempt=$(echo "$line" | sed -n 's/.*(attempt \([0-9]*\)).*/\1/p')
            echo "│    [$ts] provision attempt $attempt ($cluster)"
        else
            echo "│    [$ts] handler retry"
        fi
    done

    TOTAL_RETRIES=$((${PROVISION_RETRIES:-0} + ${HANDLER_RETRIES:-0}))
    echo "│  Total: ${TOTAL_RETRIES:-0}"
else
    echo "│  (no retries)"
fi
echo "└────────────────────────────────────────────────────────────"
echo ""

# --- Summary ---
TOTAL_EVALS=$(grep "stored_step_run.*status=<Status" "$LOG_FILE" | grep -E "sage-eval|bfcl-eval" | sed -n "s/.*target_id='//p" | sed "s/'.*//" | sort -u | wc -l | tr -d ' ')
SUCCEEDED=$(grep "stored_step_run.*status=<Status.SUCCESS" "$LOG_FILE" | grep -E "sage-eval|bfcl-eval" | sed -n "s/.*target_id='//p" | sed "s/'.*//" | sort -u | wc -l | tr -d ' ')
FAILED=$(grep "stored_step_run.*status=<Status.FAILED" "$LOG_FILE" | grep -E "sage-eval|bfcl-eval" | sed -n "s/.*target_id='//p" | sed "s/'.*//" | sort -u | wc -l | tr -d ' ')
RUNNING=$(grep "stored_step_run.*status=<Status.RUNNING" "$LOG_FILE" | grep -E "sage-eval|bfcl-eval" | sed -n "s/.*target_id='//p" | sed "s/'.*//" | sort -u | wc -l | tr -d ' ')

echo "┌─ Summary ──────────────────────────────────────────────────"
CKPT_COUNT=0
[[ -n "$CHECKPOINTS" ]] && CKPT_COUNT=$(echo "$CHECKPOINTS" | wc -l | tr -d ' ')
FINAL_COUNT=0
[[ -n "$FINAL" ]] && FINAL_COUNT=$(echo "$FINAL" | wc -l | tr -d ' ')

echo "│  SFT: $SFT_STATUS"
echo "│  Checkpoints: $CKPT_COUNT epoch + $FINAL_COUNT final"
TOTAL_RETRIES=$(grep -c "Waiting.*seconds before retry\|AnyFailureRetryStrategy.*Detected\|Transient provision failure" "$LOG_FILE" 2>/dev/null || true)
TOTAL_RETRIES=${TOTAL_RETRIES:-0}
echo "│  Evals: $SUCCEEDED succeeded, $FAILED failed (of $TOTAL_EVALS targets)"
echo "│  Retries: $TOTAL_RETRIES"
echo "└────────────────────────────────────────────────────────────"
