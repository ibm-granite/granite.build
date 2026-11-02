#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Run-phase entrypoint for the openinstruct-sft step (baked into the image at
# /opt/openinstruct-sft/). step.yaml computes the VALUES with Jinja and passes
# them as argv; this script does the shell: GPU count, a fresh staging dir,
# ClearML on/off, `accelerate launch finetune.py`, the swap into output_path, and
# the step's two stdout markers.
#
# CONTRACT
#   openinstruct_sft_run.sh --base-model <dir> --output-id <id> --output-path <dir>
#       --mixed-precision <mp> --deepspeed-config <file-or-empty>
#       --clearml-mode <auto|on|off> --clearml-project <p-or-empty>
#       --clearml-task <t-or-empty> -- <finetune.py args...>
#
#   --output-path  where the HF model ends up; relative paths are taken from the
#                  step's working dir. Replaced on a re-run, so it must be absent,
#                  empty, or a previous model (config.json at its root).
#   --             everything after it goes to finetune.py verbatim.
#   env OPEN_INSTRUCT_HOME (default /opt/open-instruct): the source tree; cwd for
#                  the launch so configs/ds_configs/ resolves.
#   env SKYPILOT_NUM_GPUS_PER_NODE: GPUs to use (set by SkyPilot from the build's
#                  resources.accelerators); falls back to `nvidia-smi -L`.
set -euo pipefail

base_model=""
output_id=""
output_path=""
mixed_precision=""
deepspeed_config=""
clearml_mode=""
clearml_project=""
clearml_task=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --base-model)       base_model="$2";       shift 2 ;;
    --output-id)        output_id="$2";        shift 2 ;;
    --output-path)      output_path="$2";      shift 2 ;;
    --mixed-precision)  mixed_precision="$2";  shift 2 ;;
    --deepspeed-config) deepspeed_config="$2"; shift 2 ;;
    --clearml-mode)     clearml_mode="$2";     shift 2 ;;
    --clearml-project)  clearml_project="$2";  shift 2 ;;
    --clearml-task)     clearml_task="$2";     shift 2 ;;
    --)                 shift; break ;;
    *) echo "openinstruct-sft: ERROR openinstruct_sft_run.sh: unknown option '$1'." >&2; exit 1 ;;
  esac
done

fail() { echo "openinstruct-sft: ERROR $*" >&2; exit 1; }

# Set-but-empty is invisible to `set -u`, so check the required values explicitly.
: "${base_model:?--base-model is required}"
: "${output_id:?--output-id is required}"
: "${output_path:?--output-path is required}"
: "${mixed_precision:?--mixed-precision is required}"
case $clearml_mode in auto|on|off) ;; *) fail "--clearml-mode must be auto, on or off, got '${clearml_mode}'." ;; esac
# The marker is space-delimited, so the ID must be one word.
case $output_id in *[[:space:]]*|*'"'*) fail "output id contains whitespace or a double quote: '${output_id}'." ;; esac

oi_home="${OPEN_INSTRUCT_HOME:-/opt/open-instruct}"

# --- GPUs ------------------------------------------------------------------------
num_gpus="${SKYPILOT_NUM_GPUS_PER_NODE:-}"
if [ -z "$num_gpus" ]; then
  num_gpus="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
fi
case $num_gpus in ''|*[!0-9]*) num_gpus=0 ;; esac
if [ "$num_gpus" -lt 1 ]; then
  fail "no GPUs on this node. Request them in the build with launcher_config.resources.accelerators (e.g. \"A10G:1\")."
fi

# --- output locations --------------------------------------------------------------
# Absolute, because the marker is read off-node. finetune.py writes into a fresh
# <output_path>.partial sibling (same filesystem, so the final move is a rename):
# it auto-resumes from any step_*/epoch_* left in its output_dir, so a fresh dir
# per run keeps a warm-cluster re-run from silently resuming an old run.
case $output_path in /*) ;; *) output_path="$(pwd)/${output_path}" ;; esac
output_path="${output_path%/}"
mkdir -p "$(dirname "$output_path")"
out_dir="$(cd "$(dirname "$output_path")" && pwd)/$(basename "$output_path")"
stage="${out_dir}.partial"
# finetune.py caches the tokenized dataset under --dataset_local_cache_dir, default
# ./local_dataset_cache relative to its cwd (/opt/open-instruct, read-only for the
# job user under Pyxis). Keep it next to the output, in the step's working dir.
cache_dir="$(pwd)/local_dataset_cache"
case $out_dir in *'"'*) fail "output path contains a double quote: '${out_dir}'." ;; esac

# Symlink-free absolute path; for a path that does not exist yet, its nearest
# existing ancestor's plus the rest.
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
# True when one path is the other or contains it.
overlaps() {
  case "$1/" in "$2/"*) return 0 ;; esac
  case "$2/" in "$1/"*) return 0 ;; esac
  return 1
}
# The base model has a config.json, so it would pass the previous-model check
# below and be deleted by the swap (or by clearing the staging dir).
base_phys="$(physical "$base_model")"
for p in "$out_dir" "$stage"; do
  if overlaps "$(physical "$p")" "$base_phys"; then
    fail "output path ${p} overlaps the base_model ${base_model}; replacing it would delete the input. Point output_path at a different directory."
  fi
done

# Replacing output_path deletes it: refuse anything but nothing / an empty dir /
# a previous model, before any GPU time.
if [ -e "$out_dir" ] && [ ! -d "$out_dir" ]; then
  fail "output path ${out_dir} exists and is not a directory."
fi
if [ -d "$out_dir" ] && [ -n "$(ls -A "$out_dir")" ] && [ ! -f "$out_dir/config.json" ]; then
  fail "output path ${out_dir} is non-empty and not a previous model (no config.json); point output_path at a new or empty directory."
fi
rm -rf "$stage"

# --- tracking ----------------------------------------------------------------------
# finetune.py reads wandb's run URL whenever tracking is on, and calls wandb.login()
# when no --wandb_entity is given. So ClearML runs as accelerate's clearml tracker
# NEXT TO a disabled wandb tracker: WANDB_MODE=disabled makes every wandb call a
# no-op (run.url is None) and --wandb_entity skips the login.
export WANDB_MODE=disabled
clearml_active=0
case $clearml_mode in
  on) clearml_active=1 ;;
  auto)
    if [ -n "${CLEARML_API_HOST:-}" ] && [ -n "${CLEARML_API_ACCESS_KEY:-}" ] && [ -n "${CLEARML_API_SECRET_KEY:-}" ]; then
      clearml_active=1
    fi ;;
esac
tracking=()
if [ "$clearml_active" = 1 ]; then
  export CLEARML_PROJECT="${clearml_project:-openinstruct-sft}"
  export CLEARML_TASK="${clearml_task:-finetune}"
  tracking=(--with_tracking --report_to wandb clearml --wandb_entity none)
  echo "openinstruct-sft: ClearML tracking on (project '${CLEARML_PROJECT}', task '${CLEARML_TASK}')"
else
  echo "openinstruct-sft: ClearML tracking off (clearml_enabled=${clearml_mode})"
fi

# --- train -------------------------------------------------------------------------
launch=(--mixed_precision "$mixed_precision" --num_machines 1 --num_processes "$num_gpus")
if [ -n "$deepspeed_config" ]; then
  launch+=(--use_deepspeed --deepspeed_config_file "configs/ds_configs/${deepspeed_config}")
fi
# Step-owned flags first, then the build's (template-validated) args.
ft=(
  --model_name_or_path "$base_model"
  --output_dir "$stage"
  --do_not_randomize_output_dir
  --push_to_hub false
  --try_launch_beaker_eval_jobs false
  --try_auto_save_to_beaker false
  --dataset_local_cache_dir "$cache_dir"
)
echo "openinstruct-sft: training on ${num_gpus} GPU(s) into ${stage}"
(
  cd "$oi_home"
  accelerate launch "${launch[@]}" open_instruct/finetune.py "${ft[@]}" ${tracking[@]+"${tracking[@]}"} "$@"
)

# --- register ----------------------------------------------------------------------
# A run that saved no model must never register an artifact.
[ -f "$stage/config.json" ] || fail "training finished but ${stage} has no config.json; no model was saved."
rm -rf "$out_dir"
mv "$stage" "$out_dir"
echo "GB_STEP_METADATA_KEY:open_instruct_commit GB_STEP_METADATA_VALUE:${OPEN_INSTRUCT_COMMIT:-unknown}"
# Last line: GB_ is the prefix space://monitors/skypilot matches.
echo "GB_ARTIFACT_ID:${output_id} GB_ARTIFACT_PATH:${out_dir}"
