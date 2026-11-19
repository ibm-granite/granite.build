#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# The lm-engine-pretrain entry point: one node's share of one pretraining job.
#
# Every input arrives as an environment variable, exported by the step template's
# `run:` block from `config` (see step-template.yaml); nothing is parsed from argv,
# so the same script serves the data-check phase, a single-node run and a
# multi-node torchrun. SkyPilot runs it once per node, with its own
# SKYPILOT_NODE_RANK.
#
# Read from the template:
#   TRAIN_DIR             the build-stable training directory (config.train_dir)
#   LME_PHASE             "train" | "data-check"
#   LME_KNOBS_JSON        config.lme, as JSON
#   LME_RUNGS             config.ckpt_rungs, comma-separated
#   LME_EMIT              "true" to run the rank-0 checkpoint watcher
#   LME_WATCH_INTERVAL    the watcher's poll interval, in seconds
#   LME_RESUME_SRC        config.resume_from_checkpoint_dir ("" = auto-detect)
#   LME_REQUIRE_PROGRESS  "true" to fail an attempt that trained nothing
#   GB_ATTEMPT_TOKEN      a token that CHANGES on every attempt of the build
#
# Read from the environment, with a default:
#   SKYPILOT_NODE_RANK / _NUM_NODES / _NUM_GPUS_PER_NODE / _NODE_IPS  the topology
#   NODE_NAME                   the Kubernetes node, for the log file name
#   LME_DECISION_WAIT_SECONDS   how long a worker waits for rank 0's resume
#                               decision (default 600). An operations and test
#                               knob; the template does not set it.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# -I: the run's working directory holds the step's file mounts, and an isolated
# interpreter cannot import a module out of it. resume.py puts its own directory
# back on sys.path for the one sibling import it needs.
PY=(python -I)

RANK="${SKYPILOT_NODE_RANK:-0}"
NNODES="${SKYPILOT_NUM_NODES:-1}"
GPUS="${SKYPILOT_NUM_GPUS_PER_NODE:-1}"
# SKYPILOT_NODE_IPS is newline-joined and rank-ordered, so its first line is
# rank 0's address, already an IP.
MADDR="$(printf '%s\n' "${SKYPILOT_NODE_IPS:-127.0.0.1}" | head -n1)"
TOKEN="${GB_ATTEMPT_TOKEN:?GB_ATTEMPT_TOKEN is exported by the step template}"

echo "lm-engine: node_rank=$RANK nnodes=$NNODES gpus=$GPUS master=$MADDR k8s_node=${NODE_NAME:-?} host=$(hostname)"

mkdir -p "$TRAIN_DIR/logs"

if [ "$LME_PHASE" = data-check ]; then
  exec "${PY[@]}" "$HERE/data_check.py" --knobs "$LME_KNOBS_JSON"
fi

# ── the resume decision ───────────────────────────────────────────────────────
# Rank 0 decides and writes the decision file (atomically); every rank then reads
# it, so no two ranks can disagree about whether this attempt resumes — a split
# decision hangs in the first collective instead of failing.
#
# The decision file is per ATTEMPT: retry_workload relaunches into the same
# builds/<build>/runs/<targetrun> workdir, and this file lives under the
# build-stable train_dir, so a fixed name would let a previous attempt's decision
# be read as this one's.
#
# The state file is per BUILD, deliberately: resume.py compares this attempt's
# progress against the previous attempt's (--require-progress), so its baseline
# must outlive an attempt. It sits in train_dir because that is the lineage it
# describes — a recipe pointed at a fresh train_dir is a fresh run and must start
# with no baseline, and a train_dir that is deleted takes its baseline with it.
DECISION="$TRAIN_DIR/.resume-decision.$TOKEN.json"
STATE="$TRAIN_DIR/.resume-state.json"
if [ "$RANK" = 0 ]; then
  "${PY[@]}" "$HERE/resume.py" --train-dir "$TRAIN_DIR" --resume-src "$LME_RESUME_SRC" \
    --world-size "$((NNODES * GPUS))" --require-progress "$LME_REQUIRE_PROGRESS" \
    --state-file "$STATE" --out "$DECISION"
fi
WAIT_SECONDS="${LME_DECISION_WAIT_SECONDS:-600}"
waited=0
while [ ! -s "$DECISION" ] && [ "$waited" -lt "$WAIT_SECONDS" ]; do
  sleep 1
  waited=$((waited + 1))
done
if [ ! -s "$DECISION" ]; then
  echo "lm-engine: no resume decision at $DECISION after ${WAIT_SECONDS}s" >&2
  exit 1
fi

# ── the lm-engine config ──────────────────────────────────────────────────────
# Per node, not one shared file: render_config.py writes with a plain truncating
# open(), so every node writing one path on the shared filesystem would race.
# The inputs are identical, so the rendered files are too; they are kept for
# diagnosis. A non-empty "error" in the decision is fatal here, on every rank.
CONFIG="$TRAIN_DIR/configs/lme_config.$TOKEN.node$RANK.yml"
mkdir -p "$TRAIN_DIR/configs"
"${PY[@]}" "$HERE/render_config.py" --knobs "$LME_KNOBS_JSON" --train-dir "$TRAIN_DIR" \
  --num-nodes "$NNODES" --gpus-per-node "$GPUS" --decision "$DECISION" --out "$CONFIG"
echo "===== $CONFIG ====="
cat "$CONFIG"

# ── the checkpoint watcher ────────────────────────────────────────────────────
# Rank 0 only, and only when the recipe asked for mid-run artifacts: the watcher
# prints a GB_ARTIFACT_ID line per rung on stdout, which is what makes gbserver
# dispatch the matching export target while training continues.
#
# The watcher stops when the stop file appears. Touching it only on the success
# path would leave it polling until the job was killed, so it is released from an
# EXIT trap as well: every exit path, including a failed trainer and a SIGTERM.
# The watcher is waited for, never killed — its last sweep is what emits a rung
# that completed just before the trainer exited.
STOP="$TRAIN_DIR/.train-exited.$TOKEN"
WATCH=""
release_watcher() {
  [ -n "$WATCH" ] || return 0
  : > "$STOP" 2>/dev/null || true
  wait "$WATCH" 2>/dev/null || true
  WATCH=""
}
trap release_watcher EXIT

if [ "$RANK" = 0 ] && [ "$LME_EMIT" = true ]; then
  "${PY[@]}" "$HERE/ckpt_watch.py" --train-dir "$TRAIN_DIR" --rungs "$LME_RUNGS" \
    --interval "$LME_WATCH_INTERVAL" --stop-file "$STOP" &
  WATCH=$!
  echo "lm-engine: watching $TRAIN_DIR for rungs '$LME_RUNGS' every ${LME_WATCH_INTERVAL}s (pid $WATCH)"
fi

# ── the trainer ───────────────────────────────────────────────────────────────
set +e
torchrun --nnodes="$NNODES" --node_rank="$RANK" --nproc_per_node="$GPUS" \
  --rdzv_backend=c10d --rdzv_endpoint="$MADDR:29500" \
  -m lm_engine.training.train --config "$CONFIG" \
  2>&1 | tee "$TRAIN_DIR/logs/node${RANK}-${NODE_NAME:-$(hostname)}.log"
rc=${PIPESTATUS[0]}
set -e
release_watcher

[ "$rc" = 0 ] || exit "$rc"
# `if`, not `[ "$RANK" = 0 ] && echo`: as the last command of a `set -e` script
# the test's own status would become the script's, so every worker would "fail".
if [ "$RANK" = 0 ]; then
  echo "GB_ARTIFACT_ID:checkpoint GB_ARTIFACT_PATH:$TRAIN_DIR"
fi
