#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Setup-phase entrypoint for the nemo-rl step, invoked ONCE per cluster from the
# generated step.yaml's `setup:` block (after the guard) and shipped to the node
# by `file_mounts: {src: src}`.
#
# Approach A runs the public NGC *release* image, which bundles the NeMo-RL
# source (examples/) and prefetched venvs. This script verifies that assumption
# and prepares the step's output directories. It does NOT install clearml (that
# is `uv run --with clearml` at run time) and does NOT bootstrap uv (the image
# already provides it).
#
# CONTRACT
#   nemo_rl_setup.sh --nemo-rl-root <root>
set -euo pipefail

nemo_rl_root=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --nemo-rl-root) nemo_rl_root="$2"; shift 2 ;;
    --)             shift; break ;;
    *)              break ;;
  esac
done
: "${nemo_rl_root:?--nemo-rl-root is required}"

# Fail LOUDLY here rather than letting the run phase die confusingly: a base
# image without the bundled source has no examples/run_*.py to launch.
if [ ! -d "${nemo_rl_root}/examples" ]; then
  echo "nemo-rl: ERROR the image does not bundle the NeMo-RL source:" >&2
  echo "nemo-rl: ${nemo_rl_root}/examples not found." >&2
  echo "nemo-rl: use the NGC *release* image (bundled source + prefetched venvs)," >&2
  echo "nemo-rl: e.g. nvcr.io/nvidia/nemo-rl:<release>, or set nemo_rl_config.nemo_rl_root." >&2
  exit 1
fi

# Output locations the run phase writes into (results for checkpoints, logs for
# TensorBoard). Created once; run reuses them. hf_output is deliberately NOT
# created here: convert_dcp_to_hf.py raises FileExistsError if --hf-ckpt-path
# already exists, so the converter must create it itself at run time.
mkdir -p ./results ./logs
