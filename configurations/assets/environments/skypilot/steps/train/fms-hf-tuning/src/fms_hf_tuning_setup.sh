#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Setup-phase sanity check for the fms-hf-tuning step (baked into the image at
# /opt/fms-hf-tuning/). Everything is preinstalled, so this only proves the image
# is the one the step expects before any GPU time: the step's scripts and
# accelerate configs are there and its venv imports the whole stack, flash-attn
# included (upstream turns use_flash_attn on by default).
#
# CONTRACT
#   fms_hf_tuning_setup.sh            (no arguments)
#   env FMS_HF_TUNING_HOME    default /opt/fms-hf-tuning
#   env FMS_HF_TUNING_PYTHON  default $FMS_HF_TUNING_HOME/.venv/bin/python
set -euo pipefail

home="${FMS_HF_TUNING_HOME:-/opt/fms-hf-tuning}"
py="${FMS_HF_TUNING_PYTHON:-${home}/.venv/bin/python}"

for f in fms_hf_tuning_entry.py fms_hf_tuning_job.py fms_hf_tuning_run.sh \
         accelerate/single_gpu.yaml accelerate/ddp.yaml accelerate/fsdp.yaml; do
  if [ ! -f "${home}/${f}" ]; then
    echo "fms-hf-tuning: ERROR ${home}/${f} not found; the image is not the fms-hf-tuning image." >&2
    exit 1
  fi
done
code='import accelerate, clearml, datasets, flash_attn, peft, torch, transformers, trl, tuning
from importlib.metadata import version
print("fms-hf-tuning", version("fms-hf-tuning"), "transformers", transformers.__version__, "trl", trl.__version__, "peft", peft.__version__, "flash-attn", flash_attn.__version__, "torch", torch.__version__, "cuda", torch.version.cuda)'
if ! out="$("$py" -c "$code" 2>&1)"; then
  echo "fms-hf-tuning: ERROR cannot import the training stack from ${py}:" >&2
  printf '%s\n' "$out" >&2
  exit 1
fi
echo "fms-hf-tuning: ${out##*$'\n'}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
  || echo "fms-hf-tuning: WARNING nvidia-smi not available"
