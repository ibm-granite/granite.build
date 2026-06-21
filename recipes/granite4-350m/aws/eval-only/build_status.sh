#!/usr/bin/env bash
######
# build_status.sh — Show build progress: SFT training, checkpoints, evals, and retries.
#
# Uses incremental processing: maintains a state file with byte offset so
# subsequent runs only read new log lines appended since the last check.
#
# Usage:
#   ./recipes/granite4-350m/aws/eval-only/build_status.sh [--reset] [LOG_FILE]
#   watch -n 10 ./recipes/granite4-350m/aws/eval-only/build_status.sh
######
set -uo pipefail

# Handle --reset flag
if [[ "${1:-}" == "--reset" ]]; then
    shift
    LOG_FILE="${1:-/tmp/standalone.log}"
    rm -f "${LOG_FILE}.build_status"
    echo "State reset. Next run will reprocess from beginning."
    exit 0
fi

LOG_FILE="${1:-/tmp/standalone.log}"

if [[ ! -f "$LOG_FILE" ]]; then
    echo "ERROR: Log file not found: $LOG_FILE"
    exit 1
fi

STATE_FILE="${LOG_FILE}.build_status"

# --- Load previous state ---
OFFSET=0
SFT_STATUS="NOT STARTED"
SFT_CLUSTER=""
PROGRESS_BAR=""
CHECKPOINTS=""
FINAL=""
PROVISION_RETRIES=0
HANDLER_RETRIES=0
EVAL_STATE=""

if [[ -f "$STATE_FILE" ]]; then
    source "$STATE_FILE"
fi

# --- Detect log rotation and read only new bytes ---
FILE_SIZE=$(wc -c < "$LOG_FILE" | tr -d ' ')
if [[ $FILE_SIZE -lt $OFFSET ]]; then
    OFFSET=0  # log was truncated/rotated
fi

if [[ $FILE_SIZE -eq $OFFSET ]]; then
    NEW_DATA=""
else
    NEW_DATA=$(tail -c +$((OFFSET + 1)) "$LOG_FILE" | sed 's/\x1b\[[0-9;]*m//g')
fi
OFFSET=$FILE_SIZE

# --- Process new data incrementally ---
if [[ -n "$NEW_DATA" ]]; then
    # SFT status (latest wins)
    new_sft=$(echo "$NEW_DATA" | grep "stored_step_run.*openinstruct-sft.*status=<Status" \
        | tail -1 | sed -n 's/.*status=<Status\.\([A-Z]*\).*/\1/p')
    [[ -n "$new_sft" ]] && SFT_STATUS="$new_sft"

    # Progress bar (latest wins)
    new_progress=$(echo "$NEW_DATA" \
        | grep "get_events_from_log_line.*it/s\|get_events_from_log_line.*it\]" \
        | tail -1 \
        | sed -n 's/.*] Running get_events_from_log_line for line [0-9]*: *//p' \
        | sed 's/[0-9\/]* [0-9:,]* - INFO - __main__ -   Step:.*//' \
        | sed 's/[[:space:]]*$//')
    [[ -n "$new_progress" ]] && PROGRESS_BAR="$new_progress"

    # Fallback progress from Step events
    if [[ -z "$PROGRESS_BAR" ]]; then
        last_step=$(echo "$NEW_DATA" | grep "JSON Log Event.*Step:" \
            | tail -1 | sed -n "s/.*'message': 'Step: //p" | sed "s/'.*//")
        [[ -n "$last_step" ]] && PROGRESS_BAR="Step $last_step"
    fi

    # Cluster name
    new_cluster=$(echo "$NEW_DATA" | grep "Started live log stream for" \
        | tail -1 | sed -n 's/.*stream for \([^ ]*\).*/\1/p')
    if [[ -z "$new_cluster" ]]; then
        new_cluster=$(echo "$NEW_DATA" | grep "SkyPilot cluster.*launched" \
            | tail -1 | sed -n 's/.*cluster \([^ ]*\).*/\1/p')
    fi
    [[ -n "$new_cluster" ]] && SFT_CLUSTER="$new_cluster"

    # Checkpoints (append new with timestamp, dedup by path)
    new_ckpts=$(echo "$NEW_DATA" | grep "newartifact_in_environment_event.*ArtifactEventPayload" \
        | grep "binding_id='checkpoint'" \
        | while IFS= read -r line; do
            ts=$(echo "$line" | sed -n 's/.*\[\([0-9-]* [0-9:]*\)[,].*/\1/p')
            path=$(echo "$line" | sed -n "s/.*'path': '//p" | sed "s/'}.*//" )
            [[ -n "$path" ]] && echo "${ts}|${path}"
        done)
    if [[ -n "$new_ckpts" ]]; then
        CHECKPOINTS=$(printf "%s\n%s" "$CHECKPOINTS" "$new_ckpts" | grep -v "^$" | sort -t'|' -k2 -u)
    fi

    # Final checkpoint
    new_final=$(echo "$NEW_DATA" | grep "newartifact_in_environment_event.*ArtifactEventPayload" \
        | grep "binding_id='final'" \
        | while IFS= read -r line; do
            ts=$(echo "$line" | sed -n 's/.*\[\([0-9-]* [0-9:]*\)[,].*/\1/p')
            path=$(echo "$line" | sed -n "s/.*'path': '//p" | sed "s/'}.*//" )
            [[ -n "$path" ]] && echo "${ts}|${path}"
        done)
    if [[ -n "$new_final" ]]; then
        FINAL=$(printf "%s\n%s" "$FINAL" "$new_final" | grep -v "^$" | sort -t'|' -k2 -u)
    fi

    # Eval statuses (accumulate: target_id|name|uri|status|experiment|step_run_id)
    new_evals=$(echo "$NEW_DATA" | grep "stored_step_run.*status=<Status" \
        | grep -E "sage-eval|bfcl-eval" \
        | while IFS= read -r line; do
            tid=$(echo "$line" | sed -n "s/.*target_id='//p" | sed "s/'.*//")
            tname=$(echo "$line" | sed -n "s/.*target_name='//p" | sed "s/'.*//")
            suri=$(echo "$line" | sed -n "s/.*definition_uri='//p" | sed "s/'.*//")
            status=$(echo "$line" | sed -n "s/.*status=<Status\.//p" | sed "s/[:'].*//")
            srid=$(echo "$line" | sed -n "s/.*uuid='//p" | sed "s/'.*//")
            ts=$(echo "$line" | sed -n 's/.*\[\([0-9-]* [0-9:]*\)[,].*/\1/p')
            echo "$tid|$tname|$suri|$status|$ts|$srid"
        done)
    if [[ -n "$new_evals" ]]; then
        EVAL_STATE=$(printf "%s\n%s" "$EVAL_STATE" "$new_evals" | grep -v "^$")
    fi


    # Retry counts (increment)
    new_provision=$(echo "$NEW_DATA" | grep -c "Transient provision failure" || true)
    new_handler=$(echo "$NEW_DATA" | grep -c "Waiting.*seconds before retry\|AnyFailureRetryStrategy.*Detected" || true)
    PROVISION_RETRIES=$((PROVISION_RETRIES + new_provision))
    HANDLER_RETRIES=$((HANDLER_RETRIES + new_handler))
fi

# --- Save state ---
cat > "$STATE_FILE" <<EOF
OFFSET=$OFFSET
SFT_STATUS="$SFT_STATUS"
SFT_CLUSTER="$SFT_CLUSTER"
PROGRESS_BAR="$PROGRESS_BAR"
CHECKPOINTS="$CHECKPOINTS"
FINAL="$FINAL"
PROVISION_RETRIES=$PROVISION_RETRIES
HANDLER_RETRIES=$HANDLER_RETRIES
EVAL_STATE="$EVAL_STATE"
EOF

# --- Display ---
echo "════════════════════════════════════════════════════════════════"
echo "  Build Status ($(date '+%H:%M:%S'))"
echo "════════════════════════════════════════════════════════════════"
echo ""

echo "┌─ SFT Training ─────────────────────────────────────────────"
echo "│  Status: $SFT_STATUS"
[[ -n "$PROGRESS_BAR" ]] && echo "│  Progress: $PROGRESS_BAR"
[[ -n "$SFT_CLUSTER" ]] && echo "│  Cluster: $SFT_CLUSTER"
echo "└────────────────────────────────────────────────────────────"
echo ""

echo "┌─ Checkpoints Emitted ──────────────────────────────────────"
if [[ -n "$CHECKPOINTS" ]]; then
    echo "$CHECKPOINTS" | while read -r entry; do
        [[ -z "$entry" ]] && continue
        ts=$(echo "$entry" | cut -d'|' -f1)
        ckpt=$(echo "$entry" | cut -d'|' -f2)
        basename=$(echo "$ckpt" | awk -F/ '{print $NF}')
        short=$(echo "$basename" | sed 's/epoch_hf_/e/;s/step_hf_/s/')
        echo "│  ✓ $short [$ts]  $ckpt"
    done
else
    echo "│  (none yet)"
fi
if [[ -n "$FINAL" ]]; then
    echo "$FINAL" | while read -r entry; do
        [[ -z "$entry" ]] && continue
        ts=$(echo "$entry" | cut -d'|' -f1)
        f=$(echo "$entry" | cut -d'|' -f2)
        echo "│  ✓ final [$ts]  $f"
    done
fi
echo "└────────────────────────────────────────────────────────────"
echo ""

echo "┌─ Eval Targets ─────────────────────────────────────────────"
if [[ -n "$EVAL_STATE" ]]; then
    # Deduplicate: keep last status per target_id
    echo "$EVAL_STATE" | sort -t'|' -k1,1 -k4,4 | awk -F'|' '
        {latest[$1]=$0}
        END {for (id in latest) print latest[id]}
    ' | sort -t'|' -k5,5 -k2,2 | while IFS='|' read -r tid tname suri status ts srid; do
        [[ -z "$tid" ]] && continue
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
        # Show timestamp (just time portion for brevity)
        short_ts=""
        [[ -n "$ts" ]] && short_ts=$(echo "$ts" | awk '{print $2}')
        if [[ -n "$short_ts" ]]; then
            echo "│  $icon $eval_type [$short_ts] — $status"
        else
            echo "│  $icon $eval_type — $status"
        fi
    done
else
    echo "│  (no evals triggered yet)"
fi
echo "└────────────────────────────────────────────────────────────"
echo ""

echo "┌─ Retries ───────────────────────────────────────────────────"
TOTAL_RETRIES=$((PROVISION_RETRIES + HANDLER_RETRIES))
if [[ $TOTAL_RETRIES -gt 0 ]]; then
    # Show per-cluster (per-eval) provision attempts
    # Map cluster → target_name from MESSAGE_EVENT "SkyPilot job on <cluster>" lines
    # or from "Launching SkyPilot cluster: name=<cluster> target=<target>"
    if [[ $PROVISION_RETRIES -gt 0 ]]; then
        # Build cluster→target map from log
        CLUSTER_MAP=$(sed 's/\x1b\[[0-9;]*m//g' "$LOG_FILE" \
            | grep "SkyPilot job.*on gb-\|Launching SkyPilot cluster.*target=" \
            | sed -n 's/.*target_name": "\([^"]*\)".*on \(gb-[^ :"]*\).*/\2|\1/p; s/.*name=\([^ ]*\) target=\([^ ]*\).*/\1|\2/p' \
            | sort -u)

        sed 's/\x1b\[[0-9;]*m//g' "$LOG_FILE" \
            | grep "Transient provision failure" \
            | sed -n 's/.*\[\([0-9-]* [0-9:]*\)[,].*failure for \([^ ]*\) (attempt \([0-9]*\)).*/\1|\2|\3/p' \
            | awk -F'|' '{if($3+0 > max[$2]+0) {max[$2]=$3; ts[$2]=$1}} END{for(c in max) print c"|"max[c]"|"ts[c]}' \
            | sort | while IFS='|' read -r cluster attempts last_ts; do
                # Look up target name from cluster map
                etype=$(echo "$CLUSTER_MAP" | grep "^$cluster|" | head -1 | cut -d'|' -f2)
                etype=${etype:-"$cluster"}
                short_ts=$(echo "$last_ts" | awk '{print $2}')
                echo "│  ⟳ $etype ($cluster): attempt $attempts/10 [$short_ts]"
            done
    fi
    if [[ $HANDLER_RETRIES -gt 0 ]]; then
        echo "│  Handler retries: $HANDLER_RETRIES"
    fi
else
    echo "│  (no retries)"
fi
echo "└────────────────────────────────────────────────────────────"
echo ""

# --- Summary ---
CKPT_COUNT=0
[[ -n "$CHECKPOINTS" ]] && CKPT_COUNT=$(echo "$CHECKPOINTS" | grep -c "." || true)
FINAL_COUNT=0
[[ -n "$FINAL" ]] && FINAL_COUNT=$(echo "$FINAL" | grep -c "." || true)

TOTAL_EVALS=0
SUCCEEDED=0
FAILED=0
if [[ -n "$EVAL_STATE" ]]; then
    TOTAL_EVALS=$(echo "$EVAL_STATE" | sort -t'|' -k1,1 | awk -F'|' '{seen[$1]=1} END{print length(seen)}')
    SUCCEEDED=$(echo "$EVAL_STATE" | sort -t'|' -k1,1 -k4,4 | awk -F'|' '{latest[$1]=$4} END{n=0; for(k in latest) if(latest[k]=="SUCCESS") n++; print n}')
    FAILED=$(echo "$EVAL_STATE" | sort -t'|' -k1,1 -k4,4 | awk -F'|' '{latest[$1]=$4} END{n=0; for(k in latest) if(latest[k]=="FAILED") n++; print n}')
fi

echo "┌─ Summary ──────────────────────────────────────────────────"
echo "│  SFT: $SFT_STATUS"
echo "│  Checkpoints: $CKPT_COUNT epoch + $FINAL_COUNT final"
echo "│  Evals: $SUCCEEDED succeeded, $FAILED failed (of $TOTAL_EVALS targets)"
echo "│  Retries: $TOTAL_RETRIES"
echo "└────────────────────────────────────────────────────────────"
