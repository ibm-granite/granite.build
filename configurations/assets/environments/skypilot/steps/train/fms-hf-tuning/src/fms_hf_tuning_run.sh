#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Run-phase entrypoint for the fms-hf-tuning step (baked into the image at
# /opt/fms-hf-tuning/). step.yaml computes the VALUES with Jinja and passes them
# as argv; this script does the shell: GPU count, the accelerate config, the
# job JSON (fms_hf_tuning_job.py build), ClearML on/off, `accelerate launch
# fms_hf_tuning_entry.py`, the output check, the swap into output_path, and the
# step's stdout markers.
#
# CONTRACT
#   fms_hf_tuning_run.sh --base-model <dir> --output-id <id> --output-path <dir>
#       --peft-method <none|lora|alora> --distributed <ddp|fsdp>
#       --data-path <path-or-hf-id-or-empty> --data-config-b64 <b64 json>
#       --args-b64 <b64 json> --base-model-id <id-or-empty>
#       --clearml-mode <auto|on|off> --clearml-project <p> --clearml-task <t>
#
#   --output-path  where the model ends up; relative paths are from the step's
#                  working dir. Replaced on a re-run, so it must be absent, empty,
#                  or a previous output (config.json or adapter_config.json).
#   env FMS_HF_TUNING_HOME (default /opt/fms-hf-tuning): scripts, venv, configs.
#   env SKYPILOT_NUM_GPUS_PER_NODE: GPUs to use; falls back to `nvidia-smi -L`.
set -euo pipefail

fail() { echo "fms-hf-tuning: ERROR $*" >&2; exit 1; }

base_model="" output_id="" output_path="" peft_method="" distributed=""
data_path="" data_config_b64="" args_b64="" base_model_id=""
clearml_mode="" clearml_project="" clearml_task=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --base-model|--output-id|--output-path|--peft-method|--distributed|--data-path|--data-config-b64|--args-b64|--base-model-id|--clearml-mode|--clearml-project|--clearml-task)
      [ "$#" -ge 2 ] || fail "$1 needs a value." ;;
  esac
  case "$1" in
    --base-model)      base_model="$2";      shift 2 ;;
    --output-id)       output_id="$2";       shift 2 ;;
    --output-path)     output_path="$2";     shift 2 ;;
    --peft-method)     peft_method="$2";     shift 2 ;;
    --distributed)     distributed="$2";     shift 2 ;;
    --data-path)       data_path="$2";       shift 2 ;;
    --data-config-b64) data_config_b64="$2"; shift 2 ;;
    --args-b64)        args_b64="$2";        shift 2 ;;
    --base-model-id)   base_model_id="$2";   shift 2 ;;
    --clearml-mode)    clearml_mode="$2";    shift 2 ;;
    --clearml-project) clearml_project="$2"; shift 2 ;;
    --clearml-task)    clearml_task="$2";    shift 2 ;;
    *) fail "fms_hf_tuning_run.sh: unknown option '$1'." ;;
  esac
done

[ -n "$base_model" ] || fail "--base-model is required."
[ -n "$output_id" ] || fail "--output-id is required."
[ -n "$output_path" ] || fail "--output-path is required."
[ -n "$args_b64" ] || fail "--args-b64 is required."
[ -n "$data_config_b64" ] || fail "--data-config-b64 is required."
case $peft_method in none|lora|alora) ;; *) fail "--peft-method must be none, lora or alora, got '${peft_method}'." ;; esac
case $distributed in ddp|fsdp) ;; *) fail "--distributed must be ddp or fsdp, got '${distributed}'." ;; esac
case $clearml_mode in auto|on|off) ;; *) fail "--clearml-mode must be auto, on or off, got '${clearml_mode}'." ;; esac
case $output_id in *[[:space:]]*|*'"'*) fail "output id contains whitespace or a double quote: '${output_id}'." ;; esac

home="${FMS_HF_TUNING_HOME:-/opt/fms-hf-tuning}"
py="${home}/.venv/bin/python"

# --- GPUs ------------------------------------------------------------------------
visible="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
case $visible in ''|*[!0-9]*) visible=0 ;; esac
num_gpus="${SKYPILOT_NUM_GPUS_PER_NODE:-$visible}"
case $num_gpus in ''|*[!0-9]*) num_gpus=0 ;; esac
[ "$num_gpus" -ge 1 ] || fail "no GPUs on this node. Request them in the build with launcher_config.resources.accelerators (e.g. \"A10G:1\")."
# One process per allocated GPU; refuse when the container sees fewer (#604).
if [ "$visible" -ge 1 ] && [ "$visible" -lt "$num_gpus" ]; then
  fail "${num_gpus} GPU(s) were allocated (SKYPILOT_NUM_GPUS_PER_NODE) but this container can access only ${visible} (nvidia-smi -L lists ${visible}; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}). On SLURM this is ibm-granite/granite.build#604; rerun, or report the node."
fi
if [ "$distributed" = fsdp ] && [ "$num_gpus" -lt 2 ]; then
  fail "distributed: fsdp needs at least 2 GPUs; this node has ${num_gpus}. Use distributed: ddp, or request more GPUs."
fi

# --- accelerate config -----------------------------------------------------------
# Always a step-owned --config_file, so a user's ~/.cache/huggingface/accelerate
# default (shared home dirs on SLURM) is never read.
case $distributed in
  ddp)  cfg="single_gpu.yaml"; if [ "$num_gpus" -gt 1 ]; then cfg="ddp.yaml"; fi ;;
  fsdp) cfg="fsdp.yaml" ;;
esac
launch=(--config_file "${home}/accelerate/${cfg}" --num_machines 1 --num_processes "$num_gpus")
# Port 0: a free rendezvous port (Pyxis containers share the host network).
if [ "$num_gpus" -gt 1 ]; then launch+=(--main_process_port 0); fi

# --- output locations ------------------------------------------------------------
given_output_path="$output_path"
case $output_path in /*) ;; *) output_path="$(pwd)/${output_path}" ;; esac
output_path="${output_path%/}"
case ${output_path##*/} in ''|.|..)
  fail "output path '${given_output_path}' does not name a directory to create; use a path like ./model_output or /shared/models/run1." ;;
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
if [ -d "$out_dir" ] && [ -n "$(ls -A "$out_dir")" ] \
   && [ ! -f "$out_dir/config.json" ] && [ ! -f "$out_dir/adapter_config.json" ]; then
  fail "output path ${out_dir} is non-empty and not a previous model (no config.json or adapter_config.json); point output_path at a new or empty directory."
fi
rm -rf "$stage"

# Scratch, job files and caches in the (writable) step workdir: /opt is
# read-only for Pyxis's job user.
work="$(pwd)"
scratch="${work}/fms_hf_tuning_scratch"
rm -rf "$scratch"
export HF_DATASETS_CACHE="${work}/hf_cache/datasets"
export HF_HUB_DISABLE_TELEMETRY=1

# --- tracking --------------------------------------------------------------------
clearml_active=off
case $clearml_mode in
  on) clearml_active=on ;;
  auto)
    if [ -n "${CLEARML_API_HOST:-}" ] && [ -n "${CLEARML_API_ACCESS_KEY:-}" ] && [ -n "${CLEARML_API_SECRET_KEY:-}" ]; then
      clearml_active=on
    fi ;;
esac
if [ "$clearml_active" = on ]; then
  # transformers' ClearMLCallback uploads every save by default; the model is
  # already a GB artifact. Override with launcher_config.envs.CLEARML_LOG_MODEL.
  export CLEARML_LOG_MODEL="${CLEARML_LOG_MODEL:-false}"
  echo "fms-hf-tuning: ClearML tracking on (project '${clearml_project:-fms-hf-tuning}', task '${clearml_task:-sft-${peft_method}}')"
else
  echo "fms-hf-tuning: ClearML tracking off (clearml_enabled=${clearml_mode})"
fi

# --- job JSON --------------------------------------------------------------------
job="${work}/fms_hf_tuning_job.json"
"$py" "${home}/fms_hf_tuning_job.py" build \
  --args-b64 "$args_b64" --data-config-b64 "$data_config_b64" \
  --data-path "$data_path" --base-model "$base_model" --peft-method "$peft_method" \
  --output-dir "$scratch" --save-model-dir "$stage" \
  --clearml "$clearml_active" --clearml-project "$clearml_project" --clearml-task "$clearml_task" \
  --job-out "$job" --data-config-out "${work}/fms_hf_tuning_data_config.json"

# --- train -----------------------------------------------------------------------
errlog="${work}/fms_hf_tuning_error.log"
rm -f "$errlog"
echo "fms-hf-tuning: peft_method ${peft_method} on ${num_gpus} GPU(s), distributed ${distributed}, into ${stage}"
rc=0
SFT_TRAINER_CONFIG_JSON_PATH="$job" TERMINATION_LOG_FILE="$errlog" \
  accelerate launch "${launch[@]}" "${home}/fms_hf_tuning_entry.py" || rc=$?
if [ "$rc" -ne 0 ]; then
  case $rc in
    1)   kind="an input error (USER_ERROR_EXIT_CODE 1)" ;;
    203) kind="an internal error (INTERNAL_ERROR_EXIT_CODE 203)" ;;
    *)   kind="exit code ${rc}" ;;
  esac
  if [ -s "$errlog" ]; then
    echo "fms-hf-tuning: fms-hf-tuning's termination log:" >&2
    cat "$errlog" >&2
    echo >&2
  fi
  fail "training failed with ${kind}; nothing registered."
fi

# --- check, swap, register -------------------------------------------------------
post="$("$py" "${home}/fms_hf_tuning_job.py" check-output "$stage" "$peft_method" --base-model-id "$base_model_id")" \
  || fail "the saved output did not pass the check; nothing registered."
fms_v=unknown tf_v=unknown
while read -r tag a b; do
  if [ "$tag" = fms-hf-tuning-versions: ] && [ -n "$b" ]; then fms_v="$a" tf_v="$b"; fi
done <<<"$post"
rm -rf "$out_dir"
mv "$stage" "$out_dir"
rm -rf "$scratch"
echo "GB_STEP_METADATA_KEY:peft_method GB_STEP_METADATA_VALUE:${peft_method}"
echo "GB_STEP_METADATA_KEY:distributed GB_STEP_METADATA_VALUE:${distributed}"
echo "GB_STEP_METADATA_KEY:fms_hf_tuning_version GB_STEP_METADATA_VALUE:${fms_v}"
echo "GB_STEP_METADATA_KEY:transformers_version GB_STEP_METADATA_VALUE:${tf_v}"
# Last line: GB_ is the prefix space://monitors/skypilot matches.
echo "GB_ARTIFACT_ID:${output_id} GB_ARTIFACT_PATH:${out_dir}"
