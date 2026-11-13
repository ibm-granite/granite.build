#!/usr/bin/env bash

# Copyright Granite.Build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-License-Identifier: Apache-2.0
#
# Run one mergekit-yaml merge and register the merged model. Invoked from the
# generated step.yaml's `run:` block and shipped to the node by
# `file_mounts: {src: src}`.
#
# Order: check paths -> write the recipe (mergekit_prepare.py: input names -> staged
# paths, args -> flags) -> merge into a staging dir beside <output> -> check the output -> move it
# into place -> print the artifact marker. Nothing is registered unless every step
# succeeded, and an existing output dir is never overwritten.
#
# CONTRACT
#   run.sh --venv <dir> \
#          --input-model_1 <path> --input-model_2 <path> --input-model_3 <path> \
#          --input-model_4 <path> --input-base_model <path> \
#          --merge-config-json <json> --args-json <json> \
#          --copy-tokenizer true|false --lazy-unpickle true|false \
#          --allow-crimes true|false --trust-remote-code true|false \
#          --output-model <path> [--run-id <id>]
#
#   --venv       ./venv from setup.sh, or "" when mk_image provides mergekit-yaml.
#   --input-*    staged path of a bound input, "" when that input is not bound.
#   --run-id     the target step run's id, the same on every retry of it (optional):
#                lets a retry register an output an earlier attempt of the same run
#                already moved into place, instead of refusing it as existing.
#   Every other option is required; the step-template passes all of them.
set -euo pipefail

# ASCII-exact pattern matching (a UTF-8 locale widens bracket expressions).
export LC_ALL=C

here="$(cd "$(dirname "$0")" && pwd)"
fail() {
  for line in "$@"; do echo "mergekit: ERROR $line" >&2; done
  exit 1
}

venv=""
merge_config_json=""
args_json=""
copy_tokenizer=""
lazy_unpickle=""
allow_crimes=""
trust_remote_code=""
output=""
run_id=""
inputs=()   # pairs: name path

while [ "$#" -gt 0 ]; do
  case "$1" in
    --venv)               venv="$2";               shift 2 ;;
    --input-*)            inputs+=("${1#--input-}" "$2"); shift 2 ;;
    --merge-config-json)  merge_config_json="$2";  shift 2 ;;
    --args-json)          args_json="$2";          shift 2 ;;
    --copy-tokenizer)     copy_tokenizer="$2";     shift 2 ;;
    --lazy-unpickle)      lazy_unpickle="$2";      shift 2 ;;
    --allow-crimes)       allow_crimes="$2";       shift 2 ;;
    --trust-remote-code)  trust_remote_code="$2";  shift 2 ;;
    --output-model)       output="$2";             shift 2 ;;
    --run-id)             run_id="$2";             shift 2 ;;
    *) fail "unknown argument '$1'." ;;
  esac
done

[ -n "$output" ] || fail "--output-model is empty."

if [ -n "$venv" ]; then
  [ -x "$venv/bin/python" ] || fail "no venv at '$venv' (did setup run?)."
  # shellcheck source=/dev/null
  . "$venv/bin/activate"
fi
command -v mergekit-yaml >/dev/null || fail "mergekit-yaml is not on PATH."
# The interpreter for the helper calls below: the venv's, or the image's. Many images
# have only python3, and a few only python. It must import yaml (PyYAML, a mergekit
# dependency), which mergekit_prepare.py and the recipe reads need.
py="$(command -v python3 || command -v python)" \
  || fail "no python3 or python on PATH (mk_image must provide one, with PyYAML)."
"$py" -c 'import yaml' 2>/dev/null \
  || fail "'$py' cannot import yaml (PyYAML); mk_image must provide it, as mergekit's own dependency."

# --- inputs ------------------------------------------------------------------------
# Absolute paths (MergeKit runs from the workdir, but a later reader may not), and each
# bound one must be a model or LoRA directory.
prepare_inputs=()
for ((i = 0; i < ${#inputs[@]}; i += 2)); do
  name="${inputs[i]}"; path="${inputs[i + 1]}"
  if [ -n "$path" ]; then
    [ -d "$path" ] || fail "input $name is not a directory: '$path'."
    path="$(cd "$path" && pwd)"
    # The path goes back into MergeKit's reference syntax, <model>[@<rev>][+<lora>],
    # where '@' would split off a revision and '+' a LoRA.
    case $path in
      *[@+]*) fail "input $name's staged path contains '@' or '+', which MergeKit would read as a revision or a LoRA: '$path'." \
                   "stage it under a path without those characters." ;;
    esac
    if [ ! -f "$path/config.json" ] && [ ! -f "$path/adapter_config.json" ]; then
      fail "input $name has no config.json or adapter_config.json: '$path'." \
           "bind a Hugging Face model or LoRA adapter directory."
    fi
  fi
  prepare_inputs+=("--input-$name" "$path")
done

# --- output ------------------------------------------------------------------------
# Absolute, without '.' or empty components: the staging dir is created in the output's
# parent and named after its last component, which a trailing '/.' would get wrong.
# '..' is refused rather than resolved: through a symlinked parent it names another
# directory.
case $output in /*) ;; *) output="$PWD/$output" ;; esac
case "/$output/" in */../*) fail "output path must not contain '..': '$output'." ;; esac
rest="${output#/}/"; norm=""
while [ -n "$rest" ]; do
  part="${rest%%/*}"; rest="${rest#*/}"
  case $part in ''|.) ;; *) norm="$norm/$part" ;; esac
done
output="$norm"
[ -n "$output" ] || fail "output path must not be '/'."
# The monitors read the marker path into JSON ('{ "path": "<path>" }').
case $output in
  *[\"\\]*|*[[:cntrl:]]*|*' ') fail "output path must not contain a '\"', a '\\' or a control character, or end in a space: '$output'." ;;
esac
# Overlap first: it names the input, where the "not empty" check below would not.
for ((i = 1; i < ${#prepare_inputs[@]}; i += 2)); do
  p="${prepare_inputs[i]}"
  [ -n "$p" ] || continue
  case "$output/" in "$p"/*) fail "output path is inside input '${prepare_inputs[i - 1]#--input-}': '$output'." ;; esac
  case "$p/" in "$output"/*) fail "input '${prepare_inputs[i - 1]#--input-}' is inside the output path: '$p'." ;; esac
done
# A symlink is refused even when it points at an empty dir: the final mv would move the
# staged model INTO the link's target while the marker named the link itself.
[ ! -L "$output" ] || fail "output path is a symlink: '$output'." \
  "set output_path to a real directory, or to a path that does not exist yet."
# A retry of a run whose earlier attempt finished the merge and moved it into place (the
# job was then lost, e.g. its node went away before the markers were read): register
# that output again rather than refuse it. The record is written beside the output, not
# in it (a push of the model dir must not carry it), before the output is moved into
# place, and names this run; any other run still gets the "already exists" refusal.
done_record="$(dirname "$output")/.$(basename "$output").mergekit-done"
if [ -n "$run_id" ] && [ -f "$output/config.json" ] && [ -f "$done_record" ] \
   && [ "$(head -n 1 "$done_record")" = "run $run_id" ]; then
  echo "mergekit: '$output' was written by an earlier attempt of this run; registering it."
  tail -n +2 "$done_record"
  exit 0
fi
if [ -e "$output" ]; then
  if [ ! -d "$output" ] || [ -n "$(ls -A "$output" 2>/dev/null || echo x)" ]; then
    fail "output path already exists and is not an empty directory: '$output'." \
         "give each build its own output_path."
  fi
fi
mkdir -p "$(dirname "$output")"

# --- recipe and flags -----------------------------------------------------------------
work="$PWD/.mergekit"
# Hub downloads (models a recipe names by Hub id) are cached on the environment's
# shared root, as setup.sh's uv cache is, so they download once per cluster.
# huggingface_hub locks its own files, though not reliably on every network filesystem;
# and the dir is created at this user's umask, so on a workdir shared between users it
# may need to be made group-writable. The LoRA merge cache stays per run: MergeKit reuses
# any dir already at a merged adapter's path, so a shared one could hand one run
# another's partial merge.
hf_cache="${GB_SHARED_WORKDIR:-$PWD}/.mergekit-hf-cache"
mkdir -p "$work" "$hf_cache" "$work/lora-cache"
recipe="$work/merge.yaml"
expect_file="$work/expect"
flags_text="$("$py" "$here/mergekit_prepare.py" \
  --merge-config-json "$merge_config_json" \
  --args-json "$args_json" \
  --recipe-out "$recipe" \
  --expect-out "$expect_file" \
  --copy-tokenizer "$copy_tokenizer" \
  --lazy-unpickle "$lazy_unpickle" \
  --allow-crimes "$allow_crimes" \
  --trust-remote-code "$trust_remote_code" \
  "${prepare_inputs[@]}")" || exit 1
flags=()
while IFS= read -r line; do [ -n "$line" ] && flags+=("$line"); done <<<"$flags_text"
# What the merged model must contain, as mergekit_prepare.py decided from the recipe
# and copy_tokenizer: "tokenizer" (MergeKit builds or copies one) and/or
# "chat_template" (one was asked for). One place decides; the checks below only read it.
want_tokenizer=""; want_template=""
while IFS= read -r line; do
  case $line in
    tokenizer)     want_tokenizer=x ;;
    chat_template) want_template=x ;;
  esac
done <"$expect_file"

# GPUs only when SkyPilot allocated some; MergeKit merges on the CPU otherwise.
device=()
if [ "${SKYPILOT_NUM_GPUS_PER_NODE:-0}" -gt 0 ] 2>/dev/null; then device=(--cuda); fi

echo "mergekit: recipe ($recipe):"
sed 's/^/  /' "$recipe"

# --- merge ---------------------------------------------------------------------------
# Staging: MergeKit writes into <output parent>/.mergekit-stage.XXXXXX/<output name>,
# then that inner dir is renamed onto <output>. Same parent, so the mv is a rename on
# one filesystem; and MergeKit sees the real name, which it puts in the model card's
# title (merge.py: generate_card(name=basename(out_path))). The wrapper is removed on
# any exit. mktemp -d makes it 0700; the inner dir is made at the umask, as a plain
# mkdir would, so a later step can read the model.
stage_root="$(mktemp -d "$(dirname "$output")/.mergekit-stage.XXXXXX")" \
  || fail "cannot create a staging dir next to '$output'."
trap 'rm -rf "$stage_root"' EXIT
stage="$stage_root/$(basename "$output")"
mkdir "$stage"
# MergeKit's stderr (its logging) also goes to a file, so the output check below can
# see whether it swallowed a tokenizer error. A pipeline through tee, not a process
# substitution, so the log is complete when the pipeline returns; PIPESTATUS keeps
# both exit codes. stdout (fd 3, a copy of ours) is left alone and not leaked to
# mergekit-yaml. The log is removed first, so a line from an earlier run in the same
# workdir can't be matched, and a log tee could not write fails the step below.
mk_log="$work/mergekit.log"
rm -rf "$mk_log"
exec 3>&1
set +e
mergekit-yaml "$recipe" "$stage" \
  --transformers-cache "$hf_cache" \
  --lora-merge-cache "$work/lora-cache" \
  "${device[@]+"${device[@]}"}" \
  "${flags[@]+"${flags[@]}"}" \
  2>&1 1>&3 3>&- | tee "$mk_log" >&2
statuses=("${PIPESTATUS[@]}")
set -e
exec 3>&-
rc=${statuses[0]}; tee_rc=${statuses[1]}
[ "$rc" -eq 0 ] || fail "mergekit-yaml failed (exit $rc); see the log above."
[ "$tee_rc" -eq 0 ] || fail "could not write MergeKit's log to '$mk_log' (tee exit $tee_rc), so the output can't be checked."

# --- check the output ---------------------------------------------------------------
[ -f "$stage/config.json" ] || fail "the merge wrote no config.json."
tensors="$("$py" - "$stage" <<'PY'
import json, os, struct, sys
d = sys.argv[1]; n = 0; files = 0
for f in sorted(os.listdir(d)):
    if f.endswith(".safetensors"):
        files += 1
        with open(os.path.join(d, f), "rb") as fh:
            size = struct.unpack("<Q", fh.read(8))[0]
            n += sum(1 for k in json.loads(fh.read(size)) if k != "__metadata__")
print(f"{files} {n}")
PY
)"
read -r shard_count tensor_count <<<"$tensors"
[ "${tensor_count:-0}" -gt 0 ] || fail "the merge wrote no tensors (${shard_count:-0} safetensors files)."

# Tokenizer and chat template: check what landed, rather than mirror MergeKit's rules.
# MergeKit COPIES a tokenizer (copy_tokenizer) inside a try that logs "Failed to copy
# tokenizer ... still successful" on any error, including a copy that fails partway; it
# BUILDS one (tokenizer_source / tokenizer) outside it; and chat_template: auto can
# silently set none. So:
#  1. MergeKit's own swallowed-error line fails the step, whichever files landed;
#  2. a tokenizer must be present whenever one was copied or built;
#  3. a chat template must be present whenever one was asked for, whatever the path.
# A chat template is saved as chat_template.jinja, chat_template.json, or a
# "chat_template" key in tokenizer_config.json, depending on the transformers version.
grep_rc=0; grep -q "Failed to copy tokenizer" "$mk_log" || grep_rc=$?
case $grep_rc in
  0) fail "MergeKit failed to copy the tokenizer (\"Failed to copy tokenizer\" above), though it reports the merge as successful." \
          "A common cause is a chat_template it rejects; the traceback above names it." \
          "Set copy_tokenizer: false to register the model without a tokenizer." ;;
  1) ;;
  *) fail "could not read MergeKit's log '$mk_log' (grep exit $grep_rc), so the output can't be checked." ;;
esac
# A tokenizer is tokenizer_config.json plus its vocabulary: tokenizer.json (fast), or a
# slow tokenizer's own file, which is all MergeKit's reserialize path (save_pretrained)
# writes for one with no fast version.
has_vocab=""
for f in tokenizer.json tokenizer.model spiece.model sentencepiece.bpe.model vocab.json vocab.txt tokenizer.tiktoken; do
  [ -f "$stage/$f" ] && { has_vocab=x; break; }
done
if [ -n "$want_tokenizer" ]; then
  if [ ! -f "$stage/tokenizer_config.json" ] || [ -z "$has_vocab" ]; then
    fail "the merge wrote no tokenizer, though one was expected (copy_tokenizer, or a tokenizer_source / tokenizer block)." \
         "MergeKit logs the cause above; a common one is a chat_template it rejects." \
         "Set copy_tokenizer: false (and no tokenizer_source / tokenizer) to register the model without a tokenizer."
  fi
fi
if [ -n "$want_template" ] && [ ! -f "$stage/chat_template.jinja" ] && [ ! -f "$stage/chat_template.json" ] \
   && ! "$py" -c 'import json, sys; sys.exit(0 if json.load(open(sys.argv[1])).get("chat_template") else 1)' "$stage/tokenizer_config.json" 2>/dev/null; then
  fail "merge_config.chat_template was set, but the merged model has no chat template." \
       "With chat_template: auto, MergeKit sets none when no input model has one."
fi

# An empty output dir (checked above) is removed so the mv renames onto it; if it is
# still there (it gained files since, or rmdir failed), mv would nest the stage inside.
method="$("$py" -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["merge_method"])' "$recipe")"
version="$("$py" -c 'from importlib.metadata import version; print(version("mergekit"))' 2>/dev/null || echo unknown)"
markers="GB_STEP_METADATA_KEY:merge_method GB_STEP_METADATA_VALUE:$method
GB_STEP_METADATA_KEY:mergekit_version GB_STEP_METADATA_VALUE:$version
GB_STEP_METADATA_KEY:tensor_count GB_STEP_METADATA_VALUE:$tensor_count
GB_ARTIFACT_ID:model GB_ARTIFACT_PATH:$output"
if [ -d "$output" ]; then rmdir "$output" 2>/dev/null || fail "could not replace the empty output dir: '$output'."; fi
# The record for a retry (above), just before the move: if the job dies between the two,
# the retry finds no output and merges again.
if [ -n "$run_id" ]; then
  printf 'run %s\n%s\n' "$run_id" "$markers" > "$done_record" \
    || fail "could not write '$done_record'."
fi
mv "$stage" "$output"
if [ ! -f "$output/config.json" ]; then
  # mv nested the stage inside $output (it reappeared after the rmdir): remove the nested
  # copy, which the EXIT trap (the stage wrapper) no longer covers, and the record, which
  # would otherwise let a retry register whatever is at $output.
  rm -rf "${output:?}/$(basename "$stage")"
  rm -f "$done_record"
  fail "the merged model did not land at '$output'."
fi
rm -rf "$stage_root"
trap - EXIT

echo "$markers"
