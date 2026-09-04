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
# CONTRACT
#   dpk_run.sh --module <mod> --input-path <dir> --output-path <dir> \
#              --artifact-id <id> [--validate <transform>] [--] [transform flags...]
#
#   --module       python module to run with `python -m` (a DPK
#                  PythonTransformLauncher accepting --data_local_config).
#   --input-path   directory the transform reads. The caller resolves this from
#                  the declared input's staged path ($GB_INPUT_<name>).
#   --output-path  directory the transform writes. May be RELATIVE (the step's
#                  default is ./output); it is created and absolutized here.
#   --artifact-id  the declared output's name, used in the artifact marker.
#   --validate     transform name to look for a validator for; omitted or empty
#                  means no validation. See VALIDATION below.
#   --             everything after it is passed through to the transform
#                  verbatim as argv. Optional, but required if any transform flag
#                  could otherwise look like one of the options above.
#
# VALIDATION
# With --validate <t>, the script runs ./src/validate_<t>.py after the transform
# succeeds, if that file exists. The lookup is a RULE, not a table: the step
# derives the path the same way it derives the python module, so adding a
# validator for another transform is dropping in a file — no change here or in
# step.yaml.
#
# A missing validator is NOT an error: `validate: true` is a general request, and
# most transforms have no validator yet. It is announced on stdout rather than
# skipped silently, because a silent skip reads as "validation passed" to anyone
# looking at a green build.
#
# The validator is run BEFORE the artifact marker is emitted, so a failure fails
# the target and the output is never registered. It writes validation.json into
# the output directory, so the validation record travels with the data it
# validated (safe: the validators glob for data suffixes, never *.json).
#
# The caller is responsible for activating the venv (bare-node mode) and for
# exporting $GB_INPUT_<name> for each declared input.
set -euo pipefail

module=""
input_path=""
output_path=""
artifact_id=""
validate=""

# Parse only this script's own options, then hand the rest to the transform.
# An explicit `--` ends option parsing; so does the first unrecognized token, so
# a caller that omits `--` still works as long as transform flags come last.
while [ "$#" -gt 0 ]; do
  case "$1" in
    --module)       module="$2";       shift 2 ;;
    --input-path)   input_path="$2";   shift 2 ;;
    --output-path)  output_path="$2";  shift 2 ;;
    --artifact-id)  artifact_id="$2";  shift 2 ;;
    --validate)     validate="$2";     shift 2 ;;
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
# rather than as two flags: --data_local_config is declared `type=ast.literal_eval`
# (data_access_factory.py), so this string is parsed as PYTHON, not read as a path.
#
# The paths are therefore escaped for python's single-quoted string syntax, which
# is NOT the same fix as the shell escaping the step-template applies to args and
# to the GB_INPUT_ exports. A quote here survives the shell fine and then breaks
# literal_eval instead ("unterminated string literal"), so it must arrive as a
# backslash escape. Backslashes are doubled first, or a trailing one would escape
# the closing quote. An env:/// path is the build author's verbatim URI text, so
# this is reachable rather than theoretical.
esc_input=$(printf '%s' "$input_path" | sed "s/\\\\/\\\\\\\\/g; s/'/\\\\'/g")
esc_output=$(printf '%s' "$output_path" | sed "s/\\\\/\\\\\\\\/g; s/'/\\\\'/g")

python -m "$module" \
  --data_local_config "{'input_folder': '$esc_input', 'output_folder': '$esc_output'}" \
  "$@"

# Validate before registering, so a failure fails the target rather than
# publishing output that was just shown to be inconsistent. `set -e` carries the
# validator's non-zero exit; no marker is printed in that case.
if [ -n "$validate" ]; then
  validator="./src/validate_${validate}.py"
  if [ -f "$validator" ]; then
    echo "dpk: validating ${validate} output with ${validator}"
    # report dir == output dir, so validation.json ships with the data it
    # describes. --input enables the completeness pass (did every non-empty
    # source file produce output?), which the consistency pass cannot see.
    python "$validator" "$output_path" "$output_path" --input "$input_path"
  else
    # Deliberately not an error. `validate: true` is a general request and most
    # transforms have no validator yet — but say so, because a silent skip is
    # indistinguishable from "validation passed" on a green build.
    echo "dpk: validate requested, but no validator for transform '${validate}'" \
         "(expected ${validator}) — skipping"
  fi
fi

# Register the output for the declared artifact id. Must start at the beginning
# of a line for the skypilot monitor's regex to capture it.
#
# The marker is a SPACE-DELIMITED line consumed by a regex, and the path is then
# interpolated into a JSON string template by the monitor
# (builtins/monitors/skypilot/monitor.yaml). Two characters therefore cannot be
# carried, and both fail SILENTLY rather than loudly:
#   * a space in the artifact id — the monitor captures binding_id with [^ ]+, so
#     "a b" registers as "a", binding the wrong artifact id;
#   * a double quote in the path — it terminates the monitor's JSON string early and
#     corrupts the event.
# Neither can be escaped away here: the delimiter and the JSON template belong to the
# consumer. So refuse, naming the value, instead of registering something wrong.
case $artifact_id in
  *[[:space:]]*|*'"'*)
    echo "dpk: ERROR artifact id contains whitespace or a double quote:" \
         "'${artifact_id}'" >&2
    echo "dpk: the artifact marker is space-delimited, so the monitor would register" >&2
    echo "dpk: only the first word. Rename the declared output." >&2
    exit 1 ;;
esac
case $output_path in
  *'"'*)
    echo "dpk: ERROR output path contains a double quote: '${output_path}'" >&2
    echo "dpk: the monitor interpolates the path into JSON, which this would break." >&2
    exit 1 ;;
esac
echo "GB_ARTIFACT_ID:${artifact_id} GB_ARTIFACT_PATH:${output_path}"
