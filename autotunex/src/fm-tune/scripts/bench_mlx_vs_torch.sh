#!/usr/bin/env bash
# Benchmark the SAME tuning run on both backends and report wall-clock time.
#
#   Run 1:  --backend torch   (HuggingFace/PyTorch driver → MPS on Apple Silicon)
#   Run 2:  --backend mlx      (Apple Silicon MLX driver)
#
# Everything else is held identical: same config file, same dataset, same model,
# same tuning algorithm, same output-per-run. Only --backend changes, so the time
# difference is attributable to the backend.
#
# Run from repo root:   bash scripts/bench_mlx_vs_torch.sh
# Requires Apple Silicon + the [mlx] extra:   uv pip install -e ".[mlx]"
#
# Everything is tunable via environment variables (defaults in parens):
#   MODEL      HF model id / path         (HuggingFaceTB/SmolLM2-135M-Instruct)
#   ALGO       tuning algo: lora|sft      (lora)  -- must run on BOTH backends
#   CONFIG     YAML config file           (autotune/configs/autotune_mlx.yaml)
#   TRAIN_FILE training jsonl             (datasets/finance_train.jsonl)
#   VAL_FILE   validation jsonl           (datasets/finance_validation.jsonl)
#   TRAIN_N    subset: first N train rows (256; set 0 to use the whole file)
#   VAL_N      subset: first N val rows   (32;  set 0 to use the whole file)
#   AUTOTUNE   1 = full HPO sweep, 0 = single run (--no_autotune)  (0)
#   OUT        base output dir            (/tmp/fmtune_bench)
#
# Example — compare a full HPO sweep on the whole dataset:
#   AUTOTUNE=1 TRAIN_N=0 VAL_N=0 bash scripts/bench_mlx_vs_torch.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${BASH_VERSION:-}" ]; then
  echo "ERROR: run with bash, not sh:   bash scripts/bench_mlx_vs_torch.sh" >&2
  exit 1
fi

PY=.venv/bin/python
[ -x "$PY" ] || PY=python

MODEL="${MODEL:-HuggingFaceTB/SmolLM2-135M-Instruct}"
ALGO="${ALGO:-lora}"
CONFIG="${CONFIG:-autotune/configs/autotune_mlx.yaml}"
TRAIN_FILE="${TRAIN_FILE:-datasets/finance_train.jsonl}"
VAL_FILE="${VAL_FILE:-datasets/finance_validation.jsonl}"
TRAIN_N="${TRAIN_N:-256}"
VAL_N="${VAL_N:-32}"
AUTOTUNE="${AUTOTUNE:-0}"
OUT="${OUT:-/tmp/fmtune_bench}"

AUTOTUNE_FLAG="--no_autotune"
MODE="single run (--no_autotune)"
if [ "$AUTOTUNE" = "1" ]; then AUTOTUNE_FLAG=""; MODE="full HPO sweep"; fi

# ---- sanity checks -----------------------------------------------------------
if [ "$(uname -m)" != "arm64" ]; then
  echo "ERROR: this benchmark requires Apple Silicon (arm64); got $(uname -m)." >&2
  exit 1
fi
if ! "$PY" -c 'import mlx.core, mlx_lm' 2>/dev/null; then
  echo "ERROR: the [mlx] extra is not installed. Run:  uv pip install -e \".[mlx]\"" >&2
  exit 1
fi
for f in "$TRAIN_FILE" "$VAL_FILE" "$CONFIG"; do
  [ -f "$f" ] || { echo "ERROR: not found: $f" >&2; exit 1; }
done

# ---- build the (optional) dataset subset — SHARED by both runs ---------------
mkdir -p "$OUT"
TRAIN_USE="$TRAIN_FILE"; VAL_USE="$VAL_FILE"
if [ "$TRAIN_N" != "0" ]; then
  TRAIN_USE="$OUT/train_subset.jsonl"; head -n "$TRAIN_N" "$TRAIN_FILE" > "$TRAIN_USE"
fi
if [ "$VAL_N" != "0" ]; then
  VAL_USE="$OUT/val_subset.jsonl"; head -n "$VAL_N" "$VAL_FILE" > "$VAL_USE"
fi
TRAIN_ROWS=$(wc -l < "$TRAIN_USE" | tr -d ' ')
VAL_ROWS=$(wc -l < "$VAL_USE" | tr -d ' ')

echo "=============================================================="
echo " fm-tune backend benchmark — torch (MPS)  vs  mlx"
echo "--------------------------------------------------------------"
echo " model      : $MODEL"
echo " algo       : $ALGO"
echo " config     : $CONFIG"
echo " mode       : $MODE"
echo " train/val  : $TRAIN_ROWS / $VAL_ROWS rows  ($TRAIN_USE)"
echo " output     : $OUT"
echo "=============================================================="

fmt () {  # seconds -> "Xm Ys"
  local s="$1"; printf '%dm %02ds' $(( s / 60 )) $(( s % 60 ))
}

# ---- warm the MLX conversion cache (one-time HF->MLX cost, excluded from run) -
# This also downloads the HF snapshot into the shared ~/.cache/huggingface, so
# neither timed run below pays a first-time model download.
echo
echo "==> Warming caches (HF download + HF->MLX convert; measured separately)..."
CONV_SECS=0
SECONDS=0
set +e
"$PY" - "$MODEL" "$ALGO" <<'EOF' 2>&1 | sed 's/^/    /'
import sys
from autotune import mlx_backend
model, algo = sys.argv[1], sys.argv[2]
path = mlx_backend.ensure_mlx_model(model, quantize=(algo == "qlora"))
print(f"MLX model ready at: {path}")
EOF
WARM_RC=$?
set -e
CONV_SECS=$SECONDS
[ "$WARM_RC" -eq 0 ] || { echo "ERROR: MLX conversion/warmup failed (rc=$WARM_RC)." >&2; exit 1; }
echo "    one-time convert+download: $(fmt "$CONV_SECS")"

# ---- run one backend, timed --------------------------------------------------
run_backend () {
  local backend="$1"
  local outdir="$OUT/$backend"
  local logf="$OUT/${backend}.log"
  rm -rf "$outdir"; mkdir -p "$outdir"
  # fm-tune pins Ray's temp dir to /tmp/_ray/job_0 (LSB_JOBID is unset off-LSF, so
  # every run collides here). Stale sessions/sockets left by a prior killed run in
  # that dir make ray.init HANG at startup. Clear it (and any procs still holding
  # it) before each run so a bad prior run can't poison this one.
  pkill -f "/tmp/_ray/job_0" 2>/dev/null || true
  rm -rf /tmp/_ray/job_0
  echo
  echo "==> Running backend=$backend ...  (streaming below; also saved to $logf)"
  echo "    (first Ray startup takes ~20-40s with little output — that's normal, not a hang)"
  SECONDS=0
  set +e
  # Stream to the terminal AND the log via tee; PYTHONUNBUFFERED keeps output live.
  PYTHONUNBUFFERED=1 "$PY" main.py \
    --config_file "$CONFIG" \
    --train_file "$TRAIN_USE" \
    --validation_file "$VAL_USE" \
    --model_name_or_path "$MODEL" \
    --tuning_algo "$ALGO" \
    --backend "$backend" \
    --output_dir "$outdir" \
    --output_model_name "bench-$backend" \
    --run_name "bench-$backend" \
    $AUTOTUNE_FLAG \
    2>&1 | tee "$logf"
  local rc=${PIPESTATUS[0]}
  set -e
  local secs=$SECONDS
  if [ "$rc" -eq 0 ]; then
    echo "    backend=$backend finished in $(fmt "$secs")  (${secs}s)"
  else
    echo "    backend=$backend FAILED (rc=$rc) after $(fmt "$secs"); tail of log:"
    tail -n 25 "$logf" | sed 's/^/      | /'
  fi
  # export results to globals keyed by backend
  eval "SECS_${backend}=$secs"
  eval "RC_${backend}=$rc"
}

# torch first, then mlx (order noted in the summary; rerun to check thermal drift)
run_backend torch
run_backend mlx

# ---- summary -----------------------------------------------------------------
echo
echo "=============================================================="
echo " RESULTS  ($MODE, $TRAIN_ROWS train rows, algo=$ALGO)"
echo "--------------------------------------------------------------"
printf ' %-14s %-12s %s\n' "backend" "wall-clock" "status"
printf ' %-14s %-12s %s\n' "torch (MPS)" "$(fmt "$SECS_torch")" "$([ "$RC_torch" -eq 0 ] && echo OK || echo "FAILED(rc=$RC_torch)")"
printf ' %-14s %-12s %s\n' "mlx"         "$(fmt "$SECS_mlx")"   "$([ "$RC_mlx" -eq 0 ] && echo OK || echo "FAILED(rc=$RC_mlx)")"
echo "--------------------------------------------------------------"
if [ "$RC_torch" -eq 0 ] && [ "$RC_mlx" -eq 0 ] && [ "$SECS_mlx" -gt 0 ]; then
  awk -v t="$SECS_torch" -v m="$SECS_mlx" 'BEGIN{
    printf " mlx speedup vs torch: %.2fx  (torch %ds / mlx %ds)\n", t/m, t, m
  }'
fi
echo " one-time HF->MLX convert (excluded above): $(fmt "$CONV_SECS")"
echo " logs: $OUT/torch.log  $OUT/mlx.log"
echo "=============================================================="
