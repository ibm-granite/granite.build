#!/usr/bin/env bash
set -euo pipefail

model_path=""
model_id=""
test_categories="all"
num_gpus_generate=1
num_gpus_evaluate=1
gpu_memory_utilization="0.9"
output_dir=""
vllm_port="8000"
# 900 s was not enough on 2026-09-04. Jobs 1418147 (kd20) and 1418148 (kd40) were both placed on
# p3-r28-n4 at the same instant, and in BOTH the vLLM EngineCore subprocess emitted not one log line
# for 16 minutes before dying on its OWN internal 600 s limit
# (VLLM_ENGINE_READY_TIMEOUT_S) -- so this outer wait and that inner one have to move together, and
# neither was reachable from the submitter. Overridable now; the default is unchanged.
vllm_ready_timeout="${VLLM_READY_TIMEOUT_S:-900}"
sample_fraction=""
sample_seed="42"
exclude_categories=""
num_shards=""
shard_index=""
skip_evaluate="false"
evaluate_only="false"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --model-path)             model_path="$2";             shift 2 ;;
    --model-id)               model_id="$2";               shift 2 ;;
    --test-categories)        test_categories="$2";        shift 2 ;;
    --num-gpus-generate)      num_gpus_generate="$2";       shift 2 ;;
    --num-gpus-evaluate)      num_gpus_evaluate="$2";       shift 2 ;;
    --gpu-memory-utilization) gpu_memory_utilization="$2";  shift 2 ;;
    --output-dir)             output_dir="$2";              shift 2 ;;
    --vllm-port)              vllm_port="$2";               shift 2 ;;
    --sample-fraction)        sample_fraction="$2";         shift 2 ;;
    --sample-seed)            sample_seed="$2";              shift 2 ;;
    --exclude-categories)     exclude_categories="$2";      shift 2 ;;
    --num-shards)             num_shards="$2";               shift 2 ;;
    --shard-index)            shard_index="$2";               shift 2 ;;
    --skip-evaluate)          skip_evaluate="true";          shift 1 ;;
    --evaluate-only)          evaluate_only="true";          shift 1 ;;
    *) echo "run-bfcl.sh: unknown argument $1" >&2; exit 1 ;;
  esac
done

if [ -n "$sample_fraction" ] && [ -n "$num_shards$shard_index" ]; then
  echo "run-bfcl: --sample-fraction and --num-shards/--shard-index are mutually exclusive" >&2
  exit 1
fi

if [ -n "$num_shards" ] || [ -n "$shard_index" ]; then
  : "${num_shards:?--num-shards is required when --shard-index is set}"
  : "${shard_index:?--shard-index is required when --num-shards is set}"
  # A lone shard's output only covers its own id slice per category --
  # scoring it alone is meaningless, so sharding always implies skipping
  # evaluate; the real evaluate happens once, later, on the merged tree.
  skip_evaluate="true"
fi

if [ "$evaluate_only" = "true" ] && [ -n "$num_shards$shard_index$sample_fraction" ]; then
  echo "run-bfcl: --evaluate-only cannot be combined with --num-shards/--shard-index/--sample-fraction (those only affect generate)" >&2
  exit 1
fi

: "${model_id:?--model-id is required}"
: "${output_dir:?--output-dir is required}"
if [ "$evaluate_only" != "true" ]; then
  : "${model_path:?--model-path is required}"
fi

mkdir -p "$output_dir"
# bfcl_eval writes result/ and score/ trees relative to BFCL_PROJECT_ROOT.
export BFCL_PROJECT_ROOT="$output_dir"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# bfcl-shim registers real Granite-4 checkpoints (e.g.
# ibm-granite/granite-4.2-30b-fp8) that bfcl_eval has no MODEL_CONFIG_MAPPING
# entry for, so --model-id above can name the actual checkpoint instead of
# borrowing an unrelated size's entry -- see src/bfcl-shim and README.md.
bfcl_bin="${BFCL_BIN:-$script_dir/bfcl-shim}"
# Invoke the venv's own interpreter explicitly rather than bare `python3` --
# PATH isn't guaranteed to put the venv's bin/ first in every environment
# this script runs in (e.g. enroot doesn't reliably apply a Docker image's
# baked-in ENV PATH to the container's runtime environment), and a bare
# `python3` falling back to a system interpreter without pyyaml etc.
# installed is exactly how this broke on BlueVela.
#
# The venv sits in different places relative to this script depending on
# context: in the built image, the Dockerfile COPYs src/'s contents directly
# into /opt/bfcl-eval/, flattening it, so .venv is a sibling of this script
# ($script_dir/.venv). In the local/test tree, this script still lives under
# src/ while `uv sync` creates .venv in the step root one level up
# ($script_dir/../.venv). Check both rather than hardcoding one.
if [ -x "$script_dir/.venv/bin/python3" ]; then
  venv_bin_dir="$script_dir/.venv/bin"
elif [ -x "$script_dir/../.venv/bin/python3" ]; then
  venv_bin_dir="$script_dir/../.venv/bin"
else
  venv_bin_dir=""
fi
python_bin="${PYTHON_BIN:-${venv_bin_dir:+$venv_bin_dir/python3}}"
python_bin="${python_bin:-python3}"

vllm_pid=""
cleanup() {
  if [ -n "$vllm_pid" ] && kill -0 "$vllm_pid" 2>/dev/null; then
    echo "run-bfcl: stopping standalone vLLM server (pid ${vllm_pid})"
    kill "$vllm_pid" 2>/dev/null || true
    wait "$vllm_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

generate_args=()

if [ "$evaluate_only" != "true" ]; then
  # Some published checkpoints (e.g. ibm-granite/granite-4.2-30b-fp8) ship an
  # llm-compressor recipe.yaml recording a quantization scheme but no matching
  # config.json quantization_config, which makes vLLM either fail to
  # autodetect a loader or (if forced via --quantization) pick the wrong one
  # and crash on the checkpoint's real weight_scale tensors. Patch it here so
  # every use of $model_path below gets a working path; see
  # src/patch_quant_config.py and README.md for the full failure mode.
  if ! model_path="$("$python_bin" "$script_dir/patch_quant_config.py" "$model_path" "$output_dir")"; then
    echo "run-bfcl: quantization_config patch failed; see above" >&2
    exit 1
  fi

  # Standalone vLLM server, from the same uv-managed venv as bfcl_eval itself
  # (see pyproject.toml/uv.lock and the Dockerfile) -- bfcl_eval never imports
  # vllm in-process, so there's no version conflict requiring a separate venv.
  # bfcl_eval is pointed at this server below via
  # REMOTE_OPENAI_BASE_URL/REMOTE_OPENAI_TOKENIZER_PATH + --skip-server-setup,
  # instead of letting bfcl_eval manage its own vllm subprocess.
  vllm_bin="${VLLM_SERVE_BIN:-${venv_bin_dir:+$venv_bin_dir/vllm}}"
  vllm_bin="${vllm_bin:-vllm}"
  # shellcheck disable=SC2206 # word-splitting is intentional here
  vllm_extra_args=(${VLLM_EXTRA_ARGS:-})
  # Expanded below through ${arr[@]+...}: bash < 4.4 (macOS's /bin/bash is 3.2)
  # treats "${arr[@]}" of an EMPTY array as unbound under `set -u`, which is the
  # default case here, since VLLM_EXTRA_ARGS is usually unset.

  echo "run-bfcl: starting standalone vLLM server (model_path=${model_path} port=${vllm_port} tensor_parallel_size=${num_gpus_generate})"
  "$vllm_bin" serve "$model_path" \
    --port "$vllm_port" \
    --tensor-parallel-size "$num_gpus_generate" \
    --gpu-memory-utilization "$gpu_memory_utilization" \
    --trust-remote-code \
    ${vllm_extra_args[@]+"${vllm_extra_args[@]}"} \
    > "${output_dir}/vllm-server.log" 2>&1 &
  vllm_pid=$!

  vllm_base_url="http://127.0.0.1:${vllm_port}/v1"
  echo "run-bfcl: waiting for vLLM server at ${vllm_base_url} (timeout ${vllm_ready_timeout}s, log: ${output_dir}/vllm-server.log)"
  start_ts=$(date +%s)
  until "$python_bin" -c "import urllib.request; urllib.request.urlopen('${vllm_base_url}/models', timeout=2)" >/dev/null 2>&1; do
    if ! kill -0 "$vllm_pid" 2>/dev/null; then
      echo "run-bfcl: vLLM server exited before becoming ready; see ${output_dir}/vllm-server.log" >&2
      exit 1
    fi
    if [ "$(( $(date +%s) - start_ts ))" -ge "$vllm_ready_timeout" ]; then
      echo "run-bfcl: timed out after ${vllm_ready_timeout}s waiting for vLLM server; see ${output_dir}/vllm-server.log" >&2
      exit 1
    fi
    sleep 2
  done
  echo "run-bfcl: vLLM server is ready"

  export REMOTE_OPENAI_BASE_URL="$vllm_base_url"
  export REMOTE_OPENAI_TOKENIZER_PATH="$model_path"

  generate_args=(--model "$model_id" --test-category "$test_categories" --skip-server-setup --local-model-path "$model_path")

  if [ -n "$sample_fraction" ]; then
    # bfcl_eval's `generate` has no native sampling/limit flag -- the only
    # lever it exposes for running less than a full test-category is
    # `--run-ids`, which reads exactly test_case_ids_to_generate.json (under
    # BFCL_PROJECT_ROOT, i.e. $output_dir) and ignores --test-category
    # entirely. See src/sample_test_ids.py for how that file gets built and
    # why memory categories are sampled by whole scenario, not by individual
    # id.
    echo "run-bfcl: sampling ${sample_fraction} of each category (seed=${sample_seed}) into ${output_dir}/test_case_ids_to_generate.json"
    "$python_bin" "$script_dir/sample_test_ids.py" \
      --output "${output_dir}/test_case_ids_to_generate.json" \
      --fraction "$sample_fraction" \
      --seed "$sample_seed"
    generate_args+=(--run-ids)
  elif [ -n "$num_shards" ]; then
    # Same --run-ids lever as --sample-fraction above, but shard_test_ids.py
    # partitions the *entire* corpus across --num-shards jobs instead of
    # sampling a fraction -- see that script and README.md for why shards
    # need isolated --output-dirs (bfcl_eval's --run-ids write path isn't
    # safe for concurrent writers sharing one output tree).
    echo "run-bfcl: sharding (shard ${shard_index}/${num_shards}, exclude=${exclude_categories:-none}) into ${output_dir}/test_case_ids_to_generate.json"
    shard_args=(--output "${output_dir}/test_case_ids_to_generate.json" --num-shards "$num_shards" --shard-index "$shard_index")
    if [ -n "$exclude_categories" ]; then
      # shellcheck disable=SC2206 # word-splitting is intentional here
      shard_args+=(--exclude-categories ${exclude_categories})
    fi
    "$python_bin" "$script_dir/shard_test_ids.py" "${shard_args[@]}"
    generate_args+=(--run-ids)
  fi

  echo "run-bfcl: generating responses (model_id=${model_id} categories=${test_categories} sample_fraction=${sample_fraction:-1.0})"
  "$bfcl_bin" generate "${generate_args[@]}"
fi

if [ "$skip_evaluate" = "true" ]; then
  echo "run-bfcl: --skip-evaluate set, not scoring (this shard's output only covers its own id slice)"
  echo "run-bfcl: done"
  exit 0
fi

evaluate_args=(--model "$model_id")
if [ -n "$exclude_categories" ]; then
  # A sampled/sharded output tree may exclude whole categories (e.g.
  # web_search) -- passing --test-category all/"$test_categories" verbatim
  # here would have evaluate try to score a category with no generated
  # results at all. Resolve the explicit included-category list the same
  # way generate's side did.
  # shellcheck disable=SC2207 # splitting a space-separated category list is intentional here
  resolved_categories=($("$python_bin" "$script_dir/resolve_test_categories.py" \
    --test-categories "$test_categories" \
    --exclude-categories "$exclude_categories"))
  # `evaluate`'s --test-category is a typer Option that only consumes ONE
  # value per occurrence -- `--test-category cat1 cat2 cat3` fails with
  # "unexpected extra argument(s)" for cat2/cat3. Its handle_multiple_input
  # callback joins whatever it receives with "," and re-splits on ",", so a
  # single comma-joined value is the correct way to pass a multi-category list
  # (confirmed against bfcl_eval/__main__.py's handle_multiple_input).
  resolved_categories_csv=$(IFS=,; echo "${resolved_categories[*]}")
  evaluate_args+=(--test-category "$resolved_categories_csv")
else
  evaluate_args+=(--test-category "$test_categories")
fi

if [ -n "$sample_fraction" ]; then
  # A sampled run only has ground-truth entries for the ids it generated;
  # --partial-eval tells `evaluate` to score what's present instead of
  # raising on every id it didn't generate.
  evaluate_args+=(--partial-eval)
fi

# --num-gpus-evaluate is accepted for config-surface compatibility with the
# legacy step but currently unused: current bfcl_eval's `evaluate` subcommand
# scores generated responses on CPU and takes no GPU-related flags.
echo "run-bfcl: evaluating responses (num_gpus_evaluate=${num_gpus_evaluate}, currently unused by bfcl_eval)"
"$bfcl_bin" evaluate "${evaluate_args[@]}"

echo "run-bfcl: done"
