#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Binding-path guard for the nemo-rl step, invoked from the generated step.yaml's
# `setup:` and `run:` blocks and shipped to the node by `file_mounts: {src: src}`.
#
# WHAT IS CHECKED WHERE
# Checks that depend only on config (algorithm, config_name, output, num_nodes)
# are `raise_error`s in step-template.yaml: they fail the strict launcher-config
# fill at step creation, before anything is provisioned. That matters because the
# launcher PREPENDS the inline `hf download` to `setup:`, so this script runs only
# after the node is up and the inputs are staged. What is left here is what is
# only known on the node: the staged binding paths. It runs in `run` too because
# that phase can be reached on a warm cluster without a fresh setup.
#
# CONTRACT
#   nemo_rl_guard.sh --base-model <path> --dataset <path> --dataset-key <key>
#
#   Every option is REQUIRED on the command line but MAY be empty — that is what
#   is being checked.
set -euo pipefail

# ASCII-exact matching below; a UTF-8 locale would widen bracket classes.
export LC_ALL=C

base_model=""
dataset=""
dataset_key=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --base-model)  base_model="$2";  shift 2 ;;
    --dataset)     dataset="$2";     shift 2 ;;
    --dataset-key) dataset_key="$2"; shift 2 ;;
    *)
      echo "nemo-rl: ERROR nemo_rl_guard.sh: unknown option '$1'." >&2
      exit 1 ;;
  esac
done

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
