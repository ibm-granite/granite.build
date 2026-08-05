#!/usr/bin/env bash
# Smoke-test Apple Silicon (MPS) support for fm-tune.
#
# Run from the repo root:   bash test_mps.sh
#
# Assumes deps are installed into .venv:   uv pip install -e ".[core]"
# (falls back to whatever `python` is active if .venv is absent).
#
# This is a scratch helper — it is not tracked by git; delete it anytime.
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
[ -x "$PY" ] || PY=python

echo "==> 1/4  Check MPS availability"
"$PY" - <<'EOF'
import platform, torch
print(f"torch {torch.__version__} | arch {platform.machine()} | macOS {platform.mac_ver()[0]}")
print(f"mps available: {torch.backends.mps.is_available()} | built: {torch.backends.mps.is_built()}")
assert torch.backends.mps.is_available(), "MPS unavailable — need Apple Silicon + torch built with MPS."
EOF

echo
echo "==> 2/4  Run the MPS-relevant unit tests (no GPU needed; all monkeypatched)"
"$PY" -m pytest \
  tests/test_device.py \
  tests/test_cluster.py \
  tests/test_optimizer_routing.py \
  tests/test_pipeline.py \
  tests/test_template_utils.py -q

echo
echo "==> 3/4  Build a fast run: 1 epoch + a small data subset (keeps it ~1-2 min)"
head -96 datasets/finance_train.jsonl      > /tmp/fin_train_small.jsonl
head -32 datasets/finance_validation.jsonl > /tmp/fin_val_small.jsonl
"$PY" - <<'EOF'
import yaml
src = open("autotune/configs/autotune_mac.yaml").read()
def set_default(text, key, val):
    i = text.index(f"\n  {key}:"); j = text.index("default:", i); k = text.index("\n", j)
    return text[:j] + f"default: {val}" + text[k:]
src = set_default(src, "num_train_epochs", 1)
open("/tmp/autotune_mac_fast.yaml", "w").write(src)
yaml.safe_load(open("/tmp/autotune_mac_fast.yaml"))   # validate it still parses
print("wrote /tmp/autotune_mac_fast.yaml (num_train_epochs=1)")
EOF

echo
echo "==> 4/4  LoRA fine-tune on MPS (downloads SmolLM2-135M on first run)"
rm -rf /tmp/fmtune_mps_test
PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false "$PY" main.py \
  --config_file /tmp/autotune_mac_fast.yaml \
  --train_file /tmp/fin_train_small.jsonl \
  --validation_file /tmp/fin_val_small.jsonl \
  --model_name_or_path HuggingFaceTB/SmolLM2-135M-Instruct \
  --tuning_algo lora \
  --output_dir /tmp/fmtune_mps_test \
  --output_model_name smollm2-lora \
  --run_name mps-test \
  --no_autotune

echo
echo "==> Result"
ADAPTER=/tmp/fmtune_mps_test/models/smollm2-lora/adapter_model.safetensors
if [ -f "$ADAPTER" ]; then
  echo "PASS — trained on MPS and saved a LoRA adapter:"
  ls -lh "$ADAPTER"
else
  echo "FAIL — adapter not found. Re-run and inspect the log above."
  exit 1
fi

# ---------------------------------------------------------------------------
# Optional: prove the guard rails reject an impossible-on-Metal config.
# Expect a clear ValueError ("QLoRA ... requires CUDA") and a non-zero exit,
# BEFORE any Ray cluster starts:
#
#   PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python main.py \
#     --config_file /tmp/autotune_mac_fast.yaml \
#     --train_file /tmp/fin_train_small.jsonl --validation_file /tmp/fin_val_small.jsonl \
#     --model_name_or_path HuggingFaceTB/SmolLM2-135M-Instruct \
#     --tuning_algo qlora --output_dir /tmp/fmtune_guard --output_model_name x \
#     --run_name guard-test --no_autotune ; echo "exit=$?"
#
# For a full (slow) run instead of the fast subset, use the shipped preset and
# the full dataset — note autotune_mac.yaml defaults to 10 epochs:
#
#   .venv/bin/python main.py --config_file autotune/configs/autotune_mac.yaml \
#     --train_file datasets/finance_train.jsonl \
#     --validation_file datasets/finance_validation.jsonl \
#     --model_name_or_path HuggingFaceTB/SmolLM2-135M-Instruct \
#     --tuning_algo lora --output_dir /tmp/fmtune_full \
#     --output_model_name smollm2-lora --run_name mps-full --no_autotune
# ---------------------------------------------------------------------------
