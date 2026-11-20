#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# The lm-engine-export entry point: one checkpoint rung, one CPU pod.
#
# Every input arrives as an environment variable, exported by the step template's
# `run:` block from `config` (see step-template.yaml); nothing is parsed from argv.
# This script's whole job is to turn that environment into one `src/export.py`
# invocation and hand back its exit code unchanged. export.py owns every decision
# about the export itself.
#
# Read from the template:
#   LME_TRAIN_DIR          config.train_dir       -> --train-dir
#   LME_OUTPUT_DIR         config.output_dir      -> --output-dir
#   LME_TOKENIZER_PATH     config.tokenizer_path  -> --tokenizer-path
#   LME_MODEL_TYPE         config.model_type      -> --model-type
#   LME_ITERATION          config.iteration       -> --iteration (omitted when empty)
#   LME_PARITY_TOLERANCE   config.parity_tolerance -> --parity-tolerance
#   LME_IMAGE              config.image           -> --image (manifest provenance only)
#   LME_EXPECTED_JSON      config.expected, as JSON text -> written to a file,
#                          passed as --expected (see below)
#
# NO torchrun, and no process group. `lm_engine.training.unshard` is single-process
# by design -- upstream's own launcher is `python -m lm_engine.training.unshard
# --config $1` -- and export.py spawns it itself. One pod, one `python export.py`.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

# export.py imports `tools.model_conversion`, which is NOT an installed package: it
# lives in the image's lm-engine source tree, which the image already puts on
# PYTHONPATH (/app/src). This script's own directory -- the file mount holding
# export.py and export_checks.py -- is PREPENDED, so both are importable. Extending
# rather than assigning is the point: replacing PYTHONPATH loses `tools`, and the
# conversion then fails after the unshard has already run.
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"

# config.expected has no inline form: export.py reads it from a JSON file so that an
# unreadable path is a hard error (exit 2) instead of a silently empty expectation
# that turns the whole config check off. The file is kept, not cleaned up -- it is
# the evidence for what a failed check was comparing against.
args=(
  --train-dir "$LME_TRAIN_DIR"
  --output-dir "$LME_OUTPUT_DIR"
  --tokenizer-path "$LME_TOKENIZER_PATH"
  --model-type "$LME_MODEL_TYPE"
  --parity-tolerance "$LME_PARITY_TOLERANCE"
)
[ -z "${LME_ITERATION:-}" ] || args+=(--iteration "$LME_ITERATION")
[ -z "${LME_IMAGE:-}" ] || args+=(--image "$LME_IMAGE")
if [ -n "${LME_EXPECTED_JSON:-}" ] && [ "${LME_EXPECTED_JSON}" != "{}" ]; then
  expected_file="$(mktemp "${TMPDIR:-/tmp}/lm-engine-export-expected.XXXXXX")"
  printf '%s\n' "$LME_EXPECTED_JSON" > "$expected_file"
  args+=(--expected "$expected_file")
  echo "lm-engine-export: expectations written to $expected_file" >&2
fi

echo "lm-engine-export: iteration='${LME_ITERATION:-latest}' model_type='$LME_MODEL_TYPE' host=$(hostname)"

# Not piped and not tee'd, deliberately. export.py prints the one
# `GB_ARTIFACT_ID:hf_model GB_ARTIFACT_PATH:<dir>` line on stdout, exactly once and
# only after every check passed; a pipeline here could duplicate it, and a
# `pipefail` mistake could swallow the exit code that keeps an unfaithful export
# from being published. `|| rc=$?` is what keeps `set -e` from aborting before the
# code can be propagated.
rc=0
python "$HERE/export.py" "${args[@]}" || rc=$?
exit "$rc"
