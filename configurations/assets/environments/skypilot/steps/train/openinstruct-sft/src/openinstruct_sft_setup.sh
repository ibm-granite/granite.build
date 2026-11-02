#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Setup-phase sanity check for the openinstruct-sft step (baked into the image at
# /opt/openinstruct-sft/). Everything is preinstalled in the image, so this only
# proves the image is the one the step expects before any GPU time: finetune.py
# is where it should be and its venv imports the training stack.
#
# CONTRACT
#   openinstruct_sft_setup.sh            (no arguments)
#   env OPEN_INSTRUCT_HOME    default /opt/open-instruct
#   env OPEN_INSTRUCT_PYTHON  default $OPEN_INSTRUCT_HOME/.venv/bin/python
set -euo pipefail

oi_home="${OPEN_INSTRUCT_HOME:-/opt/open-instruct}"
oi_python="${OPEN_INSTRUCT_PYTHON:-${oi_home}/.venv/bin/python}"

if [ ! -f "${oi_home}/open_instruct/finetune.py" ]; then
  echo "openinstruct-sft: ERROR ${oi_home}/open_instruct/finetune.py not found; the image is not the openinstruct-sft image." >&2
  exit 1
fi
if ! out="$(cd "$oi_home" && "$oi_python" -c 'import accelerate, deepspeed, transformers, open_instruct; print("open_instruct ok")' 2>&1)"; then
  echo "openinstruct-sft: ERROR cannot import the training stack from ${oi_python}:" >&2
  printf '%s\n' "$out" >&2
  exit 1
fi
echo "openinstruct-sft: ${out##*$'\n'} (commit ${OPEN_INSTRUCT_COMMIT:-unknown})"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "openinstruct-sft: WARNING nvidia-smi not available"
