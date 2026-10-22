#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Run-phase entrypoint for the nemo-rl step, invoked from the generated
# step.yaml's `run:` block and shipped to the node by `file_mounts: {src: src}`.
#
# WHY A SCRIPT, NOT INLINE YAML
# The step.yaml computes VALUES with Jinja (algorithm, config name, binding
# paths, the overrides map) and this script does the SHELL: resolving the driver
# and converter inside the image, assembling the Hydra argv, running training in
# the ClearML wrapper, converting the final DCP checkpoint to HF, and emitting
# the artifact marker. Keeping the shell in a file makes it shellcheck-able,
# `bash -n`-able and unit-testable rather than only renderable.
#
# CONTRACT
#   nemo_rl_run.sh --algorithm <a> --config-name <c> --nemo-rl-root <root> \
#                  --base-model <path> [--dataset <path> --dataset-key <key>] \
#                  [--num-gpus-per-node <n>] \
#                  --artifact-id <id> --output-path <dir> \
#                  --clearml-mode <auto|on|off> \
#                  [--clearml-project <p>] [--clearml-task <t>] \
#                  [--hf-output-repo <repo> [--hf-output-private <true|false>]] \
#                  [--] [hydra overrides...]
#
#   --hf-output-private  create a MISSING Hub repo private (default true; `hf
#                 upload` would otherwise create it public). Ignored by the Hub
#                 for a repo that already exists.
#   --output-path where the converted HF model is written and registered from.
#                 May be relative (to the step's working dir). Replaced on a
#                 re-run, so it must be absent, empty, or a previous HF model.
#   --            everything after it is a Hydra override passed to the driver
#                 verbatim, one argv word each.
set -euo pipefail

algorithm=""
config_name=""
nemo_rl_root=""
base_model=""
dataset=""
dataset_key=""
num_gpus_per_node=""
artifact_id=""
output_path=""
clearml_mode=""
clearml_project=""
clearml_task=""
hf_output_repo=""
hf_output_private="true"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --algorithm)       algorithm="$2";       shift 2 ;;
    --config-name)     config_name="$2";     shift 2 ;;
    --nemo-rl-root)    nemo_rl_root="$2";    shift 2 ;;
    --base-model)      base_model="$2";      shift 2 ;;
    --dataset)         dataset="$2";         shift 2 ;;
    --dataset-key)     dataset_key="$2";     shift 2 ;;
    --num-gpus-per-node) num_gpus_per_node="$2"; shift 2 ;;
    --artifact-id)     artifact_id="$2";     shift 2 ;;
    --output-path)     output_path="$2";     shift 2 ;;
    --clearml-mode)    clearml_mode="$2";    shift 2 ;;
    --clearml-project) clearml_project="$2"; shift 2 ;;
    --clearml-task)    clearml_task="$2";    shift 2 ;;
    --hf-output-repo)  hf_output_repo="$2";  shift 2 ;;
    --hf-output-private) hf_output_private="$2"; shift 2 ;;
    --)                shift; break ;;
    *)                 break ;;
  esac
done

# Set-but-empty is invisible to `set -u`, so check the required values explicitly.
: "${algorithm:?--algorithm is required}"
: "${config_name:?--config-name is required}"
: "${nemo_rl_root:?--nemo-rl-root is required}"
: "${base_model:?--base-model is required}"
: "${artifact_id:?--artifact-id is required}"
: "${output_path:?--output-path is required}"
: "${clearml_mode:?--clearml-mode is required}"
case $hf_output_private in
  true|false) ;;
  *)
    echo "nemo-rl: ERROR --hf-output-private must be true or false, got '${hf_output_private}'" >&2
    exit 1 ;;
esac

# MARKER-VALUE GUARD. The template already rejects such an `output` at render
# time; this script prints the marker, so it keeps its own contract as well (it
# is also invoked directly, e.g. by its unit tests).
case $artifact_id in
  *[[:space:]]*|*'"'*)
    echo "nemo-rl: ERROR artifact id contains whitespace or a double quote:" \
         "'${artifact_id}'" >&2
    exit 1 ;;
esac

script_dir="$(cd "$(dirname "$0")" && pwd)"

# --- resolve the driver and the named config inside the image -------------------
driver="${nemo_rl_root}/examples/run_${algorithm}.py"
config="${nemo_rl_root}/examples/configs/${config_name}"
if [ ! -f "$driver" ]; then
  echo "nemo-rl: ERROR no driver for algorithm '${algorithm}': ${driver} not found." >&2
  echo "nemo-rl: available drivers in the image:" >&2
  # shellcheck disable=SC2012
  ls -1 "${nemo_rl_root}/examples/" 2>/dev/null | sed -n 's/^run_\(.*\)\.py$/nemo-rl:   \1/p' >&2 || true
  exit 1
fi
if [ ! -f "$config" ]; then
  echo "nemo-rl: ERROR config '${config_name}' not found at ${config}." >&2
  echo "nemo-rl: available configs in the image:" >&2
  # shellcheck disable=SC2012
  ls -1 "${nemo_rl_root}/examples/configs/" 2>/dev/null | sed 's/^/nemo-rl:   /' >&2 || true
  exit 1
fi

# --- prepare absolute, writable output locations --------------------------------
# `uv run` is executed with cwd inside the image's project tree so it resolves the
# prefetched env, so every path the driver writes must be absolute and point back
# into the step's own working directory (the image tree may be read-only).
results_dir="$(pwd)/results/${algorithm}"
logs_dir="$(pwd)/logs"
# Start every run from an EMPTY checkpoint dir. The working dir survives across
# runs on a warm cluster, and both NeMo-RL (auto-resume from the latest step_N in
# checkpoint_dir) and the discovery below pick the highest step_N, so a previous
# run's step_20 would be resumed from, or registered instead of this run's
# step_3. Resuming an interrupted run is not a supported mode of this step.
rm -rf "$results_dir"
mkdir -p "$results_dir" "$logs_dir"
# The HF model goes to output_path, which a build points at shared storage when a
# downstream target reads it (the per-target workdir is removed at teardown).
# Absolutized because the marker is consumed off-node. Only the PARENT is created:
# convert_dcp_to_hf.py raises FileExistsError if --hf-ckpt-path already exists,
# so the converter writes a fresh `<output_path>.partial` sibling (same FS, so
# the final move is a rename) that then replaces output_path. That also makes a
# re-run of this phase on a warm cluster replace the previous conversion instead
# of failing on it.
case $output_path in
  /*) ;;
  *) output_path="$(pwd)/${output_path}" ;;
esac
output_path="${output_path%/}"
mkdir -p "$(dirname "$output_path")"
hf_dir="$(cd "$(dirname "$output_path")" && pwd)/$(basename "$output_path")"
hf_stage="${hf_dir}.partial"
case $hf_dir in
  *'"'*)
    echo "nemo-rl: ERROR HF output path contains a double quote: '${hf_dir}'" >&2
    exit 1 ;;
esac
# Replacing output_path deletes it, which is only safe when it holds nothing or a
# previous HF model (a config.json at its root). Refuse anything else up front,
# before any GPU time, rather than wiping a mistyped directory after training.
if [ -e "$hf_dir" ] && [ ! -d "$hf_dir" ]; then
  echo "nemo-rl: ERROR output path ${hf_dir} exists and is not a directory." >&2
  exit 1
fi
if [ -d "$hf_dir" ] && [ -n "$(ls -A "$hf_dir")" ] && [ ! -f "$hf_dir/config.json" ]; then
  echo "nemo-rl: ERROR output path ${hf_dir} is non-empty and not a previous model output" >&2
  echo "nemo-rl: (no config.json). Point output_path at a new or empty directory." >&2
  exit 1
fi

# --- assemble the Hydra override argv -------------------------------------------
# Injected first, user overrides ("$@") next, forced tail last. Hydra applies
# overrides left-to-right, so the forced tail always wins: tensorboard must be on
# for ClearML to mirror, and the checkpoint dir must be where discovery looks.
#
# wandb is disabled as an OVERRIDABLE default (injected before "$@"): the shipped
# configs default logger.wandb_enabled=True, which makes NeMo-RL call wandb.init()
# and abort with "No API key configured" — the step tracks via tensorboard+ClearML
# and threads no WANDB_API_KEY. A user who brings a key can re-enable it by passing
# logger.wandb_enabled=true, which lands after this and wins.
overrides=("policy.model_name=${base_model}")
if [ -n "$dataset" ]; then
  overrides+=("${dataset_key}=${dataset}")
fi
overrides+=("logger.wandb_enabled=false")
# `"$@"` may be empty; the `+` guard keeps `set -u` happy on bash 3.2 too.
overrides+=(${@+"$@"})
overrides+=(
  "logger.tensorboard_enabled=true"
  "checkpointing.checkpoint_dir=${results_dir}"
  "logger.log_dir=${logs_dir}"
)
# The V2 DTensor worker's checkpoints need ++checkpointing.model_save_format=
# torch_save for convert_dcp_to_hf.py, and a V1 worker rejects that override.
# Which one applies depends on the RESOLVED config (`defaults:` inheritance plus
# these overrides), so clearml_launch.py decides it in the driver's own process
# (--v2-save-format, below) with NeMo-RL's loader, and appends the override
# itself. See clearml_launch.py.
#
# Pin NeMo-RL's cluster view to the node's GPU count. The shipped configs
# default cluster.gpus_per_node=1 (sft.yaml), so without this a multi-GPU node
# would still train on one GPU. Forced (after "$@") so it stays locked to the
# same num_gpus_per_node that sized the SkyPilot accelerators request.
if [ -n "$num_gpus_per_node" ]; then
  overrides+=("cluster.gpus_per_node=${num_gpus_per_node}")
fi

# --- ClearML enablement, mirrored from clearml_launch.clearml_enabled -----------
clearml_active=0
case "$clearml_mode" in
  on) clearml_active=1 ;;
  off) clearml_active=0 ;;
  auto)
    if [ -n "${CLEARML_API_HOST:-}" ] \
       && [ -n "${CLEARML_API_ACCESS_KEY:-}" ] \
       && [ -n "${CLEARML_API_SECRET_KEY:-}" ]; then
      clearml_active=1
    fi ;;
esac
uv_with=()
if [ "$clearml_active" = 1 ]; then
  uv_with=(--with clearml)
fi

cl_opts=(--clearml-mode "$clearml_mode" --v2-save-format)
if [ -n "$clearml_project" ]; then cl_opts+=(--clearml-project "$clearml_project"); fi
if [ -n "$clearml_task" ]; then cl_opts+=(--clearml-task "$clearml_task"); fi

# --- train ----------------------------------------------------------------------
# Single driver process: clearml_launch.py arms Task.init() (when active) then
# runs the driver in-process so ClearML's auto-capture sees it.
(
  cd "$nemo_rl_root"
  uv run ${uv_with[@]+"${uv_with[@]}"} python "${script_dir}/clearml_launch.py" \
    "${cl_opts[@]}" \
    --driver "$driver" \
    -- --config "$config" "${overrides[@]}"
)

# --- find the final checkpoint --------------------------------------------------
latest=""
latest_n=-1
for d in "$results_dir"/step_*; do
  [ -d "$d" ] || continue
  n="${d##*/step_}"
  case $n in
    ''|*[!0-9]*) continue ;;
  esac
  if [ "$n" -gt "$latest_n" ]; then
    latest_n="$n"
    latest="$d"
  fi
done
if [ -z "$latest" ]; then
  echo "nemo-rl: ERROR no checkpoint found under ${results_dir} after training." >&2
  echo "nemo-rl: expected a step_<N>/ directory; check that checkpointing is enabled." >&2
  exit 1
fi

# --- convert the DCP checkpoint to HF format ------------------------------------
# Into the staging sibling (cleared first: a conversion that died mid-write leaves
# it behind), then swapped into place only once it is non-empty.
rm -rf "$hf_stage"
(
  cd "$nemo_rl_root"
  uv run python "${nemo_rl_root}/examples/converters/convert_dcp_to_hf.py" \
    --config "${latest}/config.yaml" \
    --dcp-ckpt-path "${latest}/policy/weights" \
    --hf-ckpt-path "$hf_stage"
)

# A half-finished conversion must never register a bogus artifact.
if [ -z "$(ls -A "$hf_stage" 2>/dev/null)" ]; then
  echo "nemo-rl: ERROR HF conversion produced an empty directory: ${hf_stage}" >&2
  exit 1
fi
rm -rf "$hf_dir"
mv "$hf_stage" "$hf_dir"

# --- optional push to the Hub ---------------------------------------------------
if [ -n "$hf_output_repo" ]; then
  # `hf upload` creates a missing repo PUBLIC unless --private is passed; trained
  # weights default to private so a mistyped repo name never publishes them.
  upload_opts=()
  if [ "$hf_output_private" = true ]; then upload_opts+=(--private); fi
  (
    cd "$nemo_rl_root"
    # `hf`, not `huggingface-cli`: the image's huggingface_hub removed the old
    # CLI ("deprecated and no longer works"). "." uploads to the repo root.
    uv run hf upload "$hf_output_repo" "$hf_dir" . ${upload_opts[@]+"${upload_opts[@]}"}
  )
fi

# --- register the model artifact ------------------------------------------------
# Line-start, space-delimited, emitted only now that the HF dir is non-empty.
# GB_ is the standard marker prefix the skypilot monitor (space://monitors/skypilot)
# matches; LLMB_ is accepted only as a legacy alias. A prefix the monitor does not
# match registers no artifact while the build still reports SUCCESS.
echo "GB_ARTIFACT_ID:${artifact_id} GB_ARTIFACT_PATH:${hf_dir}"
