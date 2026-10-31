#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Staged-binding-path guard for the hf-peft-sft step, called from step.yaml's
# setup: and run: blocks (baked into the image at /opt/hf-peft-sft/).
#
# Config-only checks are raise_error()s in step-template.yaml (they fail at step
# creation, before provisioning). What is left is only knowable on the node: the
# staged input paths. It runs in run: too, which a warm cluster reaches without
# a fresh setup.
#
# CONTRACT
#   hf_peft_sft_guard.sh --base-model <path> --dataset <path-or-empty>
set -euo pipefail
export LC_ALL=C

base_model=""
dataset=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --base-model) base_model="$2"; shift 2 ;;
    --dataset)    dataset="$2";    shift 2 ;;
    *) echo "hf-peft-sft: ERROR hf_peft_sft_guard.sh: unknown option '$1'." >&2; exit 1 ;;
  esac
done

fail() { echo "hf-peft-sft: ERROR $*" >&2; exit 1; }

# A misspelled binding name is preserved, not raised, at render time, so it
# arrives as literal "{{ ... }}". A real staged path never contains braces.
check_rendered() {
  case $2 in
    *'{{'*|*'{%'*) fail "$1 still contains an unrendered Jinja expression: '$2' (misspelled binding name?)." ;;
  esac
}

[ -n "$base_model" ] || fail "no base model: bind the target input 'base_model' (hf:// uri or a prior-step binding)."
check_rendered base_model "$base_model"
[ -d "$base_model" ] || fail "base_model '${base_model}' is not a directory on this node."
[ -f "$base_model/config.json" ] || fail "base_model '${base_model}' has no config.json; it is not a Hugging Face model directory."

if [ -n "$dataset" ]; then
  check_rendered dataset "$dataset"
  [ -e "$dataset" ] || fail "dataset '${dataset}' does not exist on this node."
fi
