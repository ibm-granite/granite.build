#!/usr/bin/env bash
# Exemplar eval entrypoint for the custom-image step, baked into the image at
# /opt/eval/eval.sh (see ../Dockerfile) and invoked by the generated step.yaml.
#
# It performs a PLACEHOLDER "evaluation": it reads the same parameters the step
# passes (from config.eval_config) and writes a single results file to a fixed,
# well-known path (<output-dir>/results.json). Like a real evaluator it does NOT
# print the Granite.build LLMB_ARTIFACT_ID line: the output path is known to the
# step, so the step.yaml run/command block registers it (keeping this script free
# of the artifact convention).
#
# This is deliberately minimal so the exemplar needs no Python/pip in the image.
# When eval is implemented for real, replace the body below with a real harness
# (and give the image a suitable runtime + dependencies); the flag contract and
# the fixed results.json path are what the step depends on.
set -euo pipefail

model_path=""
tasks=""
output_dir=""
batch_size=8

# Parse the flags the step passes; unknown flags are ignored for forward-compat.
while [ "$#" -gt 0 ]; do
  case "$1" in
    --model-path) model_path="$2"; shift 2 ;;
    --tasks)      tasks="$2";      shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --batch-size) batch_size="$2"; shift 2 ;;
    *) shift ;;
  esac
done

: "${model_path:?--model-path is required}"
: "${output_dir:?--output-dir is required}"
# An empty --tasks yields a single "placeholder" task (mirrors the prior default).
[ -n "$tasks" ] || tasks="placeholder"

mkdir -p "$output_dir"
results_path="$(cd "$output_dir" && pwd)/results.json"
echo "eval: starting on model=${model_path} tasks=${tasks}"
# Real evaluation would go here; we emit a well-formed placeholder artifact.
cat > "$results_path" <<JSON
{
  "model_path": "${model_path}",
  "tasks": "${tasks}",
  "batch_size": ${batch_size},
  "status": "placeholder"
}
JSON
echo "eval: wrote results to ${results_path}"
