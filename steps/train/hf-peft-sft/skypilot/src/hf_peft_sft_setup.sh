#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Setup-phase sanity check for the hf-peft-sft step (baked into the image at
# /opt/hf-peft-sft/). Everything is preinstalled, so this only proves the image
# is the one the step expects before any GPU time: the training script is there
# and its venv imports the whole stack.
#
# CONTRACT
#   hf_peft_sft_setup.sh            (no arguments)
#   env HF_PEFT_SFT_HOME    default /opt/hf-peft-sft
#   env HF_PEFT_SFT_PYTHON  default $HF_PEFT_SFT_HOME/.venv/bin/python
set -euo pipefail

home="${HF_PEFT_SFT_HOME:-/opt/hf-peft-sft}"
py="${HF_PEFT_SFT_PYTHON:-${home}/.venv/bin/python}"

if [ ! -f "${home}/hf_peft_sft_train.py" ]; then
  echo "hf-peft-sft: ERROR ${home}/hf_peft_sft_train.py not found; the image is not the hf-peft-sft image." >&2
  exit 1
fi
code='import accelerate, bitsandbytes, clearml, datasets, deepspeed, peft, torch, transformers, trl
print(f"peft {peft.__version__} trl {trl.__version__} transformers {transformers.__version__} torch {torch.__version__} cuda {torch.version.cuda}")'
if ! out="$("$py" -c "$code" 2>&1)"; then
  echo "hf-peft-sft: ERROR cannot import the training stack from ${py}:" >&2
  printf '%s\n' "$out" >&2
  exit 1
fi
echo "hf-peft-sft: ${out##*$'\n'}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
  || echo "hf-peft-sft: WARNING nvidia-smi not available"
