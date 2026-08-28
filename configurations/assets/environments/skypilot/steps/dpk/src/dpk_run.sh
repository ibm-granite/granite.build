#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Transform-mode entrypoint for the dpk step, invoked from the generated
# step.yaml's `run:` block and shipped to the node by `file_mounts: {src: src}`.
#
# WHY THIS IS A SCRIPT AND NOT INLINE YAML
# The step.yaml computes VALUES with Jinja (which module to run, where to read
# and write, the transform's argv) and this script does the SHELL. Keeping the
# shell here means it is a real file: shellcheck-able, `bash -n`-able, and
# testable directly rather than only through a rendered string. That matters
# because every bug this step has had was a shell/quoting bug invisible until a
# cluster run failed — a trailing "\" that swallowed the artifact marker, and
# single-quote escaping that broke a transform's ast.literal_eval'd value.
#
# It also collapses the escaping. Inline, a value passed Jinja -> YAML -> shell
# and needed an expression like `replace("'", "'\"'\"'")` to survive. Here the
# transform's flags arrive as real argv after a `--` separator, already split by
# the caller, so no shell re-quoting happens at all.
#
# Command mode (config.dpk_config.command) deliberately stays INLINE in the
# step.yaml: it is user-supplied shell injected verbatim, and routing it through
# a script argument would add exactly the quoting layer this script removes.
#
# CONTRACT
#   dpk_run.sh --module <mod> --input-path <dir> --output-path <dir> \
#              --artifact-id <id> [--] [transform flags...]
#
#   --module       python module to run with `python -m` (a DPK
#                  PythonTransformLauncher accepting --data_local_config).
#   --input-path   directory the transform reads. The caller resolves this from
#                  the declared input's staged path ($LLMB_INPUT_<name>).
#   --output-path  directory the transform writes. May be RELATIVE (the step's
#                  default is ./output); it is created and absolutized here.
#   --artifact-id  the declared output's name, used in the artifact marker.
#   --             everything after it is passed through to the transform
#                  verbatim as argv. Optional, but required if any transform flag
#                  could otherwise look like one of the options above.
#
# The caller is responsible for activating the venv (bare-node mode) and for
# exporting $LLMB_INPUT_<name> for each declared input.
set -euo pipefail

module=""
input_path=""
output_path=""
artifact_id=""

# Parse only this script's own options, then hand the rest to the transform.
# An explicit `--` ends option parsing; so does the first unrecognized token, so
# a caller that omits `--` still works as long as transform flags come last.
while [ "$#" -gt 0 ]; do
  case "$1" in
    --module)       module="$2";       shift 2 ;;
    --input-path)   input_path="$2";   shift 2 ;;
    --output-path)  output_path="$2";  shift 2 ;;
    --artifact-id)  artifact_id="$2";  shift 2 ;;
    --)             shift; break ;;
    *)              break ;;
  esac
done

# Fail loudly on a missing required value rather than invoking python with an
# empty module or writing to the filesystem root. `set -u` does not catch these
# because they are set-but-empty.
: "${module:?--module is required}"
: "${input_path:?--input-path is required}"
: "${output_path:?--output-path is required}"
: "${artifact_id:?--artifact-id is required}"

# Create the output directory, then resolve it to an ABSOLUTE path: the artifact
# marker below is consumed by the server off-node, so a relative path would be
# meaningless there. mkdir must come first (cd needs the directory to exist), and
# the cd runs in a subshell so this script's own working directory is unchanged.
mkdir -p "$output_path"
output_path="$(cd "$output_path" && pwd)"

# DPK's launchers take input and output as a single python-literal argument
# rather than as two flags.
python -m "$module" \
  --data_local_config "{'input_folder': '$input_path', 'output_folder': '$output_path'}" \
  "$@"

# Register the output for the declared artifact id. Must start at the beginning
# of a line for the skypilot monitor's regex to capture it.
echo "LLMB_ARTIFACT_ID:${artifact_id} LLMB_ARTIFACT_PATH:${output_path}"
