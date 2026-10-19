#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Runs lm_eval for the eval/lm-eval step and registers its results. Called from the
# generated step.yaml's `run:` block (shipped by `file_mounts: {src: src}`).
#
# CONTRACT
#   lm_eval_run.sh --venv <dir|""> --output-dir <dir> --artifact-id <id> -- <lm_eval run argv>...
#
#   --venv         the venv setup created; "" when lm_eval comes from the image
#                  (lm_eval_config.install: false), so `lm_eval` must be on PATH.
#   --output-dir   where lm_eval writes results (--output_path); made absolute here,
#                  because the server reads the artifact marker off-node.
#   --artifact-id  the step output the results directory is registered as.
#   --             the rest is the `lm_eval run` argv, already quoted by the template,
#                  forwarded untouched. --output_path is appended here.
#
# The marker is printed only after lm_eval exits 0 and a results file exists, so a
# failed or empty evaluation registers nothing.
set -euo pipefail

P="eval/lm-eval"
venv=""
output_dir=""
artifact_id=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --venv)        venv="$2";        shift 2 ;;
    --output-dir)  output_dir="$2";  shift 2 ;;
    --artifact-id) artifact_id="$2"; shift 2 ;;
    --)            shift; break ;;
    *) echo "$P: ERROR internal: unknown run option '$1'" >&2; exit 2 ;;
  esac
done

: "${output_dir:?--output-dir is required}"
: "${artifact_id:?--artifact-id is required}"

if [ -n "$venv" ]; then
  # shellcheck source=/dev/null
  . "$venv/bin/activate"
fi
command -v lm_eval >/dev/null 2>&1 \
  || { echo "$P: ERROR lm_eval not found on PATH (install: false needs an image that provides it)" >&2; exit 1; }

# lm_eval downloads its task datasets through the HF cache; keep that inside the
# step's workdir rather than the node user's home.
export HF_HOME="$PWD/.hf-cache"

mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"

echo "$P: lm_eval run $* --output_path $output_dir"
lm_eval run "$@" --output_path "$output_dir"

# lm_eval writes <output_dir>/<model path with / -> __>/results_<timestamp>.json.
results_file="$(find "$output_dir" -type f -name 'results_*.json' | sort | tail -n 1)"
if [ -z "$results_file" ]; then
  echo "$P: ERROR lm_eval exited 0 but wrote no results_*.json under $output_dir" >&2
  exit 1
fi
echo "$P: results file $results_file"

version="$(python -c 'import importlib.metadata as m; print(m.version("lm_eval"))' 2>/dev/null || true)"
[ -n "$version" ] && echo "GB_STEP_METADATA_KEY:lm_eval_version GB_STEP_METADATA_VALUE:$version"

echo "GB_ARTIFACT_ID:${artifact_id} GB_ARTIFACT_PATH:${output_dir}"
