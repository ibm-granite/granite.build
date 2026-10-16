#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Required-config guard for the nemo-rl step, invoked from the TOP of both the
# generated step.yaml's `setup:` and `run:` blocks and shipped to the node by
# `file_mounts: {src: src}`.
#
# WHY IT RUNS IN BOTH PHASES
# `setup` is the expensive one: the launcher prepends `hf download` for every
# hf:// input binding into it. Guarding only in `run` would pay for the download
# before refusing a bad build. `run` guards too because it can be reached on a
# warm cluster without a fresh setup.
#
# CONTRACT
#   nemo_rl_guard.sh --algorithm <a> --config-name <c> --output <name> \
#                    --num-nodes <n> --base-model <path> \
#                    --dataset <path> --dataset-key <key>
#
#   Every option is REQUIRED on the command line but MAY be empty — that is what
#   is being checked.
set -euo pipefail

# ASCII-exact matching below; a UTF-8 locale would widen bracket classes.
export LC_ALL=C

algorithm=""
config_name=""
output=""
num_nodes=""
base_model=""
dataset=""
dataset_key=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --algorithm)   algorithm="$2";   shift 2 ;;
    --config-name) config_name="$2"; shift 2 ;;
    --output)      output="$2";      shift 2 ;;
    --num-nodes)   num_nodes="$2";   shift 2 ;;
    --base-model)  base_model="$2";  shift 2 ;;
    --dataset)     dataset="$2";     shift 2 ;;
    --dataset-key) dataset_key="$2"; shift 2 ;;
    --)            shift; break ;;
    *)             break ;;
  esac
done

# --- algorithm / config_name ----------------------------------------------------
if [ -z "$algorithm" ]; then
  echo "nemo-rl: ERROR nemo_rl_config.algorithm is required." >&2
  echo "nemo-rl: it selects examples/run_<algorithm>.py in the image, e.g. sft." >&2
  exit 1
fi
if [ -z "$config_name" ]; then
  echo "nemo-rl: ERROR nemo_rl_config.config_name is required." >&2
  echo "nemo-rl: it names a YAML under examples/configs/, e.g. sft.yaml." >&2
  exit 1
fi

# --- output (artifact id) -------------------------------------------------------
# The marker is space-delimited and its path is interpolated into the monitor's
# JSON, so whitespace or a double quote in the id would silently misbind. Only
# emptiness and those two characters are checkable here (declared outputs are not
# in the runtime render context). Checked pre-flight so a bad name never reaches
# the end of a run.
if [ -z "$output" ]; then
  echo "nemo-rl: ERROR nemo_rl_config.output is required." >&2
  echo "nemo-rl: it must name one of this target's declared model outputs, and" >&2
  echo "nemo-rl: becomes the artifact id the step registers." >&2
  exit 1
fi
case $output in
  *[[:space:]]*|*'"'*)
    echo "nemo-rl: ERROR artifact id contains whitespace or a double quote: '${output}'" >&2
    echo "nemo-rl: the artifact marker is space-delimited; rename the declared output." >&2
    exit 1 ;;
esac

# --- num_nodes ------------------------------------------------------------------
# Iteration 1 is single-node only. Multi-node needs NeMo-RL's Ray ray.sub model,
# which is an impedance mismatch with SkyPilot's per-node RANK/MASTER_ADDR launch
# and is deferred to its own iteration.
if [ "$num_nodes" != "1" ]; then
  echo "nemo-rl: ERROR compute_config.num_nodes must be 1 (got '${num_nodes}')." >&2
  echo "nemo-rl: this step is single-node, multi-GPU only in iteration 1." >&2
  exit 1
fi

# --- base_model -----------------------------------------------------------------
if [ -z "$base_model" ]; then
  echo "nemo-rl: ERROR no base model: bind the target input 'base_model' to a model" >&2
  echo "nemo-rl: (hf:// uri or a prior-step binding)." >&2
  exit 1
fi
# A misspelled binding name is PRESERVED, not raised, at render time, so it
# arrives here as literal "{{ ... }}". A real staged path never contains braces.
case $base_model in
  *'{{'*|*'{%'*)
    echo "nemo-rl: ERROR base_model still contains an unrendered Jinja expression:" >&2
    echo "nemo-rl: '${base_model}' — the binding name is probably misspelled." >&2
    exit 1 ;;
esac

# --- dataset (optional) ---------------------------------------------------------
# Empty dataset => use the config's built-in dataset. A BOUND dataset needs a
# Hydra key to inject its path under, because NeMo-RL data configs have no single
# universal path field. Refusing a bound-but-unkeyed dataset avoids silently
# training on the config's built-in data while a dataset was supplied.
if [ -n "$dataset" ]; then
  case $dataset in
    *'{{'*|*'{%'*)
      echo "nemo-rl: ERROR dataset still contains an unrendered Jinja expression:" >&2
      echo "nemo-rl: '${dataset}' — the binding name is probably misspelled." >&2
      exit 1 ;;
  esac
  if [ -z "$dataset_key" ]; then
    echo "nemo-rl: ERROR a dataset is bound but nemo_rl_config.dataset_override_key is unset." >&2
    echo "nemo-rl: set it to the Hydra key your config reads the dataset path from" >&2
    echo "nemo-rl: (e.g. data.train.input_file), or omit the dataset to use the" >&2
    echo "nemo-rl: config's built-in dataset." >&2
    exit 1
  fi
fi
