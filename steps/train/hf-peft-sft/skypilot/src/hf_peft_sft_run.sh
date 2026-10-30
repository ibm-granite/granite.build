#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Run-phase entrypoint for the hf-peft-sft step (baked into the image at
# /opt/hf-peft-sft/). step.yaml computes the VALUES with Jinja and passes them as
# argv; this script does the shell: GPU count, the accelerate config, a fresh
# staging dir, ClearML on/off, `accelerate launch hf_peft_sft_train.py`, the swap
# into output_path, and the step's stdout markers.
#
# CONTRACT
#   hf_peft_sft_run.sh --base-model <dir> --output-id <id> --output-path <dir>
#       --method <lora|qlora|alora> --distributed <ddp|fsdp|deepspeed>
#       --mixed-precision <bf16|fp16|no>
#       --clearml-mode <auto|on|off> --clearml-project <p> --clearml-task <t>
#       -- <hf_peft_sft_train.py args...>
#
#   --output-path  where the adapter ends up; relative paths are from the step's
#                  working dir. Replaced on a re-run, so it must be absent, empty,
#                  or a previous adapter (adapter_config.json at its root).
#   env HF_PEFT_SFT_HOME (default /opt/hf-peft-sft): scripts, venv, configs.
#   env SKYPILOT_NUM_GPUS_PER_NODE: GPUs to use; falls back to `nvidia-smi -L`.
set -euo pipefail

fail() { echo "hf-peft-sft: ERROR $*" >&2; exit 1; }

base_model="" output_id="" output_path="" method="" distributed="" mixed_precision=""
clearml_mode="" clearml_project="" clearml_task=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --base-model|--output-id|--output-path|--method|--distributed|--mixed-precision|--clearml-mode|--clearml-project|--clearml-task)
      [ "$#" -ge 2 ] || fail "$1 needs a value." ;;
  esac
  case "$1" in
    --base-model)      base_model="$2";      shift 2 ;;
    --output-id)       output_id="$2";       shift 2 ;;
    --output-path)     output_path="$2";     shift 2 ;;
    --method)          method="$2";          shift 2 ;;
    --distributed)     distributed="$2";     shift 2 ;;
    --mixed-precision) mixed_precision="$2"; shift 2 ;;
    --clearml-mode)    clearml_mode="$2";    shift 2 ;;
    --clearml-project) clearml_project="$2"; shift 2 ;;
    --clearml-task)    clearml_task="$2";    shift 2 ;;
    --)                shift; break ;;
    *) echo "hf-peft-sft: ERROR hf_peft_sft_run.sh: unknown option '$1'." >&2; exit 1 ;;
  esac
done

[ -n "$base_model" ] || fail "--base-model is required."
[ -n "$output_id" ] || fail "--output-id is required."
[ -n "$output_path" ] || fail "--output-path is required."
case $method in lora|qlora|alora) ;; *) fail "--method must be lora, qlora or alora, got '${method}'." ;; esac
case $distributed in ddp|fsdp|deepspeed) ;; *) fail "--distributed must be ddp, fsdp or deepspeed, got '${distributed}'." ;; esac
case $mixed_precision in bf16|fp16|no) ;; *) fail "--mixed-precision must be bf16, fp16 or no, got '${mixed_precision}'." ;; esac
case $clearml_mode in auto|on|off) ;; *) fail "--clearml-mode must be auto, on or off, got '${clearml_mode}'." ;; esac
case $output_id in *[[:space:]]*|*'"'*) fail "output id contains whitespace or a double quote: '${output_id}'." ;; esac

home="${HF_PEFT_SFT_HOME:-/opt/hf-peft-sft}"

# --- GPUs ------------------------------------------------------------------------
num_gpus="${SKYPILOT_NUM_GPUS_PER_NODE:-}"
if [ -z "$num_gpus" ]; then
  num_gpus="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
fi
case $num_gpus in ''|*[!0-9]*) num_gpus=0 ;; esac
[ "$num_gpus" -ge 1 ] || fail "no GPUs on this node. Request them in the build with launcher_config.resources.accelerators (e.g. \"A10G:1\")."
if [ "$distributed" != ddp ] && [ "$num_gpus" -lt 2 ]; then
  fail "distributed: ${distributed} needs at least 2 GPUs; this node has ${num_gpus}. Use distributed: ddp, or request more GPUs."
fi

# --- accelerate config -----------------------------------------------------------
launch=(--num_machines 1 --num_processes "$num_gpus" --mixed_precision "$mixed_precision")
case $distributed in
  ddp)       if [ "$num_gpus" -gt 1 ]; then launch=(--multi_gpu "${launch[@]}"); fi ;;
  fsdp)      cfg="fsdp.yaml"
             if [ "$method" = qlora ]; then cfg="fsdp_qlora.yaml"; fi
             launch=(--config_file "${home}/accelerate/${cfg}" "${launch[@]}") ;;
  deepspeed) launch=(--config_file "${home}/accelerate/deepspeed_z3.yaml" "${launch[@]}") ;;
esac

# --- output locations ------------------------------------------------------------
# Absolute, because the marker is read off-node. Train into a fresh <out>.partial
# sibling (same filesystem, so the final move is a rename).
given_output_path="$output_path"
case $output_path in /*) ;; *) output_path="$(pwd)/${output_path}" ;; esac
output_path="${output_path%/}"
# `/`, `.`, `..` would make out_dir an existing parent (e.g. the workdir), which
# the post-training `rm -rf "$out_dir"` would delete.
case ${output_path##*/} in ''|.|..)
  fail "output path '${given_output_path}' does not name a directory to create; use a path like ./adapter_output or /shared/adapters/run1." ;;
esac
mkdir -p "$(dirname "$output_path")"
out_dir="$(cd "$(dirname "$output_path")" && pwd)/$(basename "$output_path")"
stage="${out_dir}.partial"
case $out_dir in *'"'*) fail "output path contains a double quote: '${out_dir}'." ;; esac

physical() {
  local p="$1" rest="" d
  while [ ! -d "$p" ]; do
    rest="/$(basename "$p")${rest}"
    p="$(dirname "$p")"
  done
  d="$(cd "$p" && pwd -P)"
  d="${d%/}${rest}"
  printf '%s\n' "${d:-/}"
}
overlaps() {
  case "$1/" in "$2/"*) return 0 ;; esac
  case "$2/" in "$1/"*) return 0 ;; esac
  return 1
}
base_phys="$(physical "$base_model")"
for p in "$out_dir" "$stage"; do
  if overlaps "$(physical "$p")" "$base_phys"; then
    fail "output path ${p} overlaps the base_model ${base_model}; replacing it would delete the input. Point output_path at a different directory."
  fi
done
if [ -e "$out_dir" ] && [ ! -d "$out_dir" ]; then
  fail "output path ${out_dir} exists and is not a directory."
fi
if [ -d "$out_dir" ] && [ -n "$(ls -A "$out_dir")" ] && [ ! -f "$out_dir/adapter_config.json" ]; then
  fail "output path ${out_dir} is non-empty and not a previous adapter (no adapter_config.json); point output_path at a new or empty directory."
fi
rm -rf "$stage"

# Caches in the (writable) step workdir: /opt is read-only for Pyxis's job user.
export HF_DATASETS_CACHE="$(pwd)/hf_cache/datasets"
export HF_HUB_DISABLE_TELEMETRY=1

# --- tracking --------------------------------------------------------------------
clearml_active=0
case $clearml_mode in
  on) clearml_active=1 ;;
  auto)
    if [ -n "${CLEARML_API_HOST:-}" ] && [ -n "${CLEARML_API_ACCESS_KEY:-}" ] && [ -n "${CLEARML_API_SECRET_KEY:-}" ]; then
      clearml_active=1
    fi ;;
esac
report_to=none
if [ "$clearml_active" = 1 ]; then
  report_to=clearml
  export CLEARML_PROJECT="${clearml_project:-hf-peft-sft}"
  export CLEARML_TASK="${clearml_task:-sft-${method}}"
  # transformers' ClearMLCallback uploads every save by default; the adapter is
  # already a GB artifact. Override with launcher_config.envs.CLEARML_LOG_MODEL.
  export CLEARML_LOG_MODEL="${CLEARML_LOG_MODEL:-false}"
  echo "hf-peft-sft: ClearML tracking on (project '${CLEARML_PROJECT}', task '${CLEARML_TASK}')"
else
  echo "hf-peft-sft: ClearML tracking off (clearml_enabled=${clearml_mode})"
fi

# --- train -----------------------------------------------------------------------
bf16=false fp16=false
case $mixed_precision in bf16) bf16=true ;; fp16) fp16=true ;; esac
train=(
  --model_name_or_path "$base_model"
  --output_dir "$stage"
  --method "$method"
  --distributed "$distributed"
  --mixed_precision "$mixed_precision"
  --bf16 "$bf16"
  --fp16 "$fp16"
  --report_to "$report_to"
  --save_strategy no
  --push_to_hub false
)
echo "hf-peft-sft: ${method} on ${num_gpus} GPU(s), distributed ${distributed}, into ${stage}"
accelerate launch "${launch[@]}" "${home}/hf_peft_sft_train.py" "${train[@]}" "$@"

# --- register --------------------------------------------------------------------
if [ ! -f "$stage/adapter_config.json" ] || [ ! -f "$stage/adapter_model.safetensors" ]; then
  fail "training finished but ${stage} has no adapter (adapter_config.json + adapter_model.safetensors); nothing registered."
fi
rm -rf "$out_dir"
mv "$stage" "$out_dir"
read -r peft_v trl_v < <("${home}/.venv/bin/python" -c 'import peft, trl; print(peft.__version__, trl.__version__)' 2>/dev/null || echo "unknown unknown")
echo "GB_STEP_METADATA_KEY:method GB_STEP_METADATA_VALUE:${method}"
echo "GB_STEP_METADATA_KEY:peft_version GB_STEP_METADATA_VALUE:${peft_v}"
echo "GB_STEP_METADATA_KEY:trl_version GB_STEP_METADATA_VALUE:${trl_v}"
# Last line: GB_ is the prefix space://monitors/skypilot matches.
echo "GB_ARTIFACT_ID:${output_id} GB_ARTIFACT_PATH:${out_dir}"
