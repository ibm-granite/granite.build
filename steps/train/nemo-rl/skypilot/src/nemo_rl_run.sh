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
#                  --artifact-id <id> --clearml-mode <auto|on|off> \
#                  [--clearml-project <p>] [--clearml-task <t>] \
#                  [--hf-output-repo <repo>] [--] [hydra overrides...]
#
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
clearml_mode=""
clearml_project=""
clearml_task=""
hf_output_repo=""

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
    --clearml-mode)    clearml_mode="$2";    shift 2 ;;
    --clearml-project) clearml_project="$2"; shift 2 ;;
    --clearml-task)    clearml_task="$2";    shift 2 ;;
    --hf-output-repo)  hf_output_repo="$2";  shift 2 ;;
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
: "${clearml_mode:?--clearml-mode is required}"

# MARKER-VALUE GUARD, pre-flight (the guard script checks this too, but this
# script is the one that prints the marker, so it refuses here as well).
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
mkdir -p "$results_dir" "$logs_dir"
# convert_dcp_to_hf.py raises FileExistsError if --hf-ckpt-path already exists,
# so hf_output must NOT be pre-created — the converter creates it itself. Its
# parent (the working dir) already exists, so `$(pwd)/hf_output` is a canonical
# absolute path without needing to `cd` into the (deliberately absent) leaf.
hf_dir="$(pwd)/hf_output"
case $hf_dir in
  *'"'*)
    echo "nemo-rl: ERROR HF output path contains a double quote: '${hf_dir}'" >&2
    exit 1 ;;
esac

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
# The V2 DTensor worker (policy.dtensor_cfg._v2: true — the default in sft.yaml,
# grpo_math_1B.yaml, ppo_math_1B.yaml) defaults model_save_format=safetensors,
# which writes sharded HF safetensors under weights/model/ with NO DCP
# .metadata. But we always run convert_dcp_to_hf.py, which requires a .metadata.
# Force torch_save so the V2 worker uses the DCP writer that emits .metadata —
# exactly what NeMo-RL's own converter CI does for a V2 checkpoint. The key is
# absent from the resolved config, so it is force-added with Hydra's `++`.
#
# This is gated on the config declaring `_v2: true`: for a V1 worker (dpo/rm,
# where _v2 is false or omitted) lm_policy raises "model_save_format must be
# None or omitted if using DTensorPolicyWorker (_v2=False)", and V1 already
# writes DCP natively, so the converter works unchanged there.
if grep -Eq '^[[:space:]]*_v2:[[:space:]]*true([[:space:]]|$)' "$config"; then
  overrides+=("++checkpointing.model_save_format=torch_save")
fi
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

cl_opts=(--clearml-mode "$clearml_mode")
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
(
  cd "$nemo_rl_root"
  uv run python "${nemo_rl_root}/examples/converters/convert_dcp_to_hf.py" \
    --config "${latest}/config.yaml" \
    --dcp-ckpt-path "${latest}/policy/weights" \
    --hf-ckpt-path "$hf_dir"
)

# A half-finished conversion must never register a bogus artifact.
if [ -z "$(ls -A "$hf_dir" 2>/dev/null)" ]; then
  echo "nemo-rl: ERROR HF conversion produced an empty directory: ${hf_dir}" >&2
  exit 1
fi

# --- optional push to the Hub ---------------------------------------------------
if [ -n "$hf_output_repo" ]; then
  (
    cd "$nemo_rl_root"
    # `hf`, not `huggingface-cli`: the image's huggingface_hub removed the old
    # CLI ("deprecated and no longer works"). "." uploads to the repo root.
    uv run hf upload "$hf_output_repo" "$hf_dir" .
  )
fi

# --- register the model artifact ------------------------------------------------
# Line-start, space-delimited, emitted only now that the HF dir is non-empty.
echo "GB_ARTIFACT_ID:${artifact_id} GB_ARTIFACT_PATH:${hf_dir}"
