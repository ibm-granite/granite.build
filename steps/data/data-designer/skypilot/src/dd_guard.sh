#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Required-config guards for the data-designer step, invoked from the TOP of both the
# generated step.yaml's `setup:` and `run:` blocks and shipped to the node by
# `file_mounts: {src: src}`.
#
# Same shape and reasons as the dpk step's dpk_guard.sh: the checks live in a real
# file so they are shellcheck'd, `bash -n`'d and unit tested rather than Jinja, and
# they run in BOTH phases — `setup` because it carries the install and should refuse an
# invalid build before paying for it, `run` because a warm cluster can reach it without
# a fresh setup.
#
# What is NOT here: anything about the Data Designer config itself. That is checked by
# Data Designer's own validate() at the start of dd_run.py, before any model call; this
# script only knows whether a config was given at all (--has-config).
#
# CONTRACT
#   dd_guard.sh --output <name> --num-records <n> --resume <mode> \
#               --export-format <fmt> --min-records <n> --seed-path <dir> \
#               --dd-version <v> --dd-image <ref> \
#               --has-config <flag> --has-providers <flag> \
#               --local-runtime <r> --local-model <m> --local-port <p>
#
#   Every option is REQUIRED but may be EMPTY — that is what is being checked. The two
#   --has-* flags are non-empty when the template saw a non-empty value. The --local-*
#   options are data_designer_config.local_model's runtime, model and port.
set -euo pipefail

# Pattern matching below must be ASCII-exact (see dpk_guard.sh: a UTF-8 locale widens
# bracket expressions to accented letters).
export LC_ALL=C

output=""
num_records=""
resume=""
export_format=""
min_records=""
seed_path=""
dd_version=""
dd_image=""
has_config=""
has_providers=""
local_runtime=""
local_model=""
local_port=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --output)        output="$2";        shift 2 ;;
    --num-records)   num_records="$2";   shift 2 ;;
    --resume)        resume="$2";        shift 2 ;;
    --export-format) export_format="$2"; shift 2 ;;
    --min-records)   min_records="$2";   shift 2 ;;
    --seed-path)     seed_path="$2";     shift 2 ;;
    --dd-version)    dd_version="$2";    shift 2 ;;
    --dd-image)      dd_image="$2";      shift 2 ;;
    --has-config)    has_config="$2";    shift 2 ;;
    --has-providers) has_providers="$2"; shift 2 ;;
    --local-runtime) local_runtime="$2"; shift 2 ;;
    --local-model)   local_model="$2";   shift 2 ;;
    --local-port)    local_port="$2";    shift 2 ;;
    --)              shift; break ;;
    *)               break ;;
  esac
done

fail() {
  for line in "$@"; do echo "data-designer: $line" >&2; done
  exit 1
}

# --- config / providers -----------------------------------------------------------
if [ -z "$has_config" ]; then
  fail "ERROR data_designer_config.config is required." \
       "it is the Data Designer config (columns, model_configs, ...) the step generates from."
fi

# --- local model ------------------------------------------------------------------
case $local_runtime in
  ""|ollama) ;;
  *)
    fail "ERROR data_designer_config.local_model.runtime must be empty or 'ollama';" \
         "got '${local_runtime}'." ;;
esac

if [ -n "$local_runtime" ]; then
  if [ -z "$local_model" ]; then
    fail "ERROR data_designer_config.local_model.model is required when runtime is set," \
         "e.g. granite4:350m." ;
  fi
  case $local_port in
    ""|0*|*[!0-9]*)
      fail "ERROR data_designer_config.local_model.port must be a positive integer," \
           "got '${local_port}'." ;;
  esac
  # Checked after the digits-only case above, so the arithmetic is safe.
  if [ "$local_port" -gt 65535 ]; then
    fail "ERROR data_designer_config.local_model.port must be at most 65535, got '${local_port}'."
  fi
  # Not supported together, deliberately. With dd_image the step skips its own install,
  # so nothing adds the zstandard module Ollama's .tar.zst download needs to unpack, and
  # the image may lack python3 or curl too. Refused here rather than failing midway
  # through setup. To serve a model in your own image, bake Ollama into it and point
  # model_providers at http://127.0.0.1:<port>/v1 instead.
  if [ -n "$dd_image" ]; then
    fail "ERROR data_designer_config.local_model cannot be used together with dd_image." \
         "To serve a model in your own image, include Ollama in the image and point" \
         "model_providers at it (http://127.0.0.1:<port>/v1); see USAGE.md," \
         "\"A model on the node\"."
  fi
fi

# Required rather than defaulted: Data Designer's built-in providers (nvidia, openai,
# openrouter) read keys this step does not deliver, so falling back to them would fail
# at the first model call with an authentication error about a provider the build
# never named. A local_model supplies its own provider, so it satisfies this.
if [ -z "$has_providers" ] && [ -z "$local_runtime" ]; then
  fail "ERROR data_designer_config.model_providers is required." \
       "name at least one provider (endpoint + credential env var) for the" \
       "model_configs in the config to point at, or serve a model on the node with" \
       "local_model; see USAGE.md, \"Model providers\"."
fi

# --- output -----------------------------------------------------------------------
if [ -z "$output" ]; then
  fail "ERROR data_designer_config.output is required." \
       "it must name one of this target's declared outputs, and becomes the" \
       "artifact id the step registers."
fi

# The artifact marker is space-delimited and the monitor interpolates it into JSON, so
# whitespace or a double quote in the id would register the wrong artifact silently.
case $output in
  *[[:space:]]*|*'"'*)
    fail "ERROR data_designer_config.output contains whitespace or a double quote: '${output}'" \
         "the artifact marker is space-delimited; rename the declared output." ;;
esac

# --- counts -----------------------------------------------------------------------
case $num_records in
  ""|0*|*[!0-9]*)
    fail "ERROR data_designer_config.num_records must be a positive integer, got '${num_records}'." ;;
esac

case $min_records in
  "") ;;
  *[!0-9]*)
    fail "ERROR data_designer_config.min_records must be a non-negative integer or null," \
         "got '${min_records}'." ;;
esac

# Checked here, before the install: otherwise it is found only after the generation,
# which a min_records above num_records can never pass.
if [ -n "$min_records" ] && [ "$min_records" -gt "$num_records" ]; then
  fail "ERROR data_designer_config.min_records (${min_records}) is greater than" \
       "num_records (${num_records}); the run could never pass."
fi

# --- enums ------------------------------------------------------------------------
case $resume in
  never|always|if_possible|restart) ;;
  *)
    fail "ERROR data_designer_config.resume must be one of if_possible, always, restart," \
         "never; got '${resume}'." ;;
esac

case $export_format in
  ""|jsonl|csv|parquet) ;;
  *)
    fail "ERROR data_designer_config.export_format must be empty or one of jsonl, csv," \
         "parquet; got '${export_format}'." ;;
esac

# --- runtime ----------------------------------------------------------------------
if [ -z "$dd_image" ] && [ -z "$dd_version" ]; then
  fail "ERROR data_designer_config.dd_version is required when dd_image is not set." \
       "it is the data-designer release installed into the venv."
fi

# --- seed path --------------------------------------------------------------------
# A misspelled binding name does not fail at render: step config renders with
# PreserveUndefined, so the literal "{{ ... }}" arrives here instead. Refuse it before
# the install (same check as the dpk step's input_path).
case $seed_path in
  *'{{'*|*'{%'*)
    fail "ERROR the seed input path still contains an unrendered Jinja expression:" \
         "'${seed_path}'" \
         "the binding name is probably misspelled; it must be the target input 'seed'." ;;
esac
