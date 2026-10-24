#!/usr/bin/env bash
# Smoke-test the MLX backend end to end. Run from repo root: bash scripts/test_mlx.sh
# Requires Apple Silicon + the [mlx] extra:  uv pip install -e ".[mlx]"
#
# You supply the data: TRAIN_FILE and VAL_FILE are required SFT jsonl files
# (see docs/dataset-sft.md). autotune/tools/build_*_dataset.py or any HF dataset
# exported to jsonl can produce them. Only the first 96 / 32 rows are used.
#   TRAIN_FILE=path/to/train.jsonl VAL_FILE=path/to/val.jsonl bash scripts/test_mlx.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${TRAIN_FILE:-}" ] || [ -z "${VAL_FILE:-}" ]; then
  echo "ERROR: TRAIN_FILE and VAL_FILE must be set." >&2
  echo "usage: TRAIN_FILE=path/to/train.jsonl VAL_FILE=path/to/val.jsonl bash scripts/test_mlx.sh" >&2
  exit 1
fi
PY=.venv/bin/python
[ -x "$PY" ] || PY=python

echo "==> 1/3  Check MLX availability"
"$PY" - <<'EOF'
import mlx.core as mx, mlx_lm, platform
print(f"mlx {mx.__version__} | mlx_lm {mlx_lm.__version__} | arch {platform.machine()}")
EOF

echo "==> 2/3  Unit tests (mlx mocked)"
"$PY" -m pytest tests/test_mlx_backend.py tests/test_driver_single_mlx.py \
  tests/test_device.py tests/test_optimizer_routing.py -q

echo "==> 3/3  LoRA fine-tune on MLX (SmolLM2-135M, tiny subset)"
head -96 "$TRAIN_FILE" > /tmp/fin_train_small.jsonl
head -32 "$VAL_FILE"   > /tmp/fin_val_small.jsonl
rm -rf /tmp/fmtune_mlx_test
"$PY" main.py \
  --config_file autotune/configs/autotune_mlx.yaml \
  --train_file /tmp/fin_train_small.jsonl \
  --validation_file /tmp/fin_val_small.jsonl \
  --model_name_or_path HuggingFaceTB/SmolLM2-135M-Instruct \
  --tuning_algo lora --backend mlx \
  --output_dir /tmp/fmtune_mlx_test --output_model_name smollm2-mlx \
  --run_name mlx-test --no_autotune

ADAPTER=/tmp/fmtune_mlx_test/models/smollm2-mlx/adapters.safetensors
if [ -f "$ADAPTER" ]; then echo "PASS — MLX adapter saved:"; ls -lh "$ADAPTER"
else echo "FAIL — MLX adapter not found."; exit 1; fi

# For QLoRA: rerun with --tuning_algo qlora (4-bit MLX base).
# Negative check (expect clear rejection, non-zero exit, before Ray starts):
#   "$PY" main.py ... --tuning_algo vera --backend mlx ... ; echo "exit=$?"
