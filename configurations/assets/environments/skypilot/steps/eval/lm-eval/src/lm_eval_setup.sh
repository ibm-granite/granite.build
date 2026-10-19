#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Dependency install for the eval/lm-eval step, called from the generated step.yaml's
# `setup:` block when lm_eval_config.install is true (shipped by `file_mounts`).
# SkyPilot syncs file mounts before `setup` runs, so ./src is present.
#
# CONTRACT
#   lm_eval_setup.sh --venv <dir> --torch-index-url <url|""> --index-url <url>
#                    -- <requirement>...
#
# Two install phases, because uv prefers an --extra-index-url over the primary index:
# mixing the torch index in as an extra would silently pick the wrong torch wheels.
#   1. torch alone, from --torch-index-url (skipped when it is empty, so torch then
#      comes from --index-url as a dependency of the lm-eval extra);
#   2. the requirements, from --index-url. torch is already satisfied, so uv keeps
#      the phase-1 build.
set -euo pipefail

venv=""
torch_index_url=""
index_url=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --venv)            venv="$2";            shift 2 ;;
    --torch-index-url) torch_index_url="$2"; shift 2 ;;
    --index-url)       index_url="$2";       shift 2 ;;
    --)                shift; break ;;
    *) echo "eval/lm-eval: ERROR internal: unknown setup option '$1'" >&2; exit 2 ;;
  esac
done

: "${venv:?--venv is required}"
: "${index_url:?--index-url is required}"

# uv isn't on a bare node or in python:3.12-slim. --break-system-packages is needed
# on PEP 668 interpreters and unknown to pip older than 23.0.1, so fall back.
if ! pip install --quiet --no-cache-dir --break-system-packages uv 2>/dev/null; then
  pip install --quiet --no-cache-dir uv
fi

# Inside the step's workdir, not $GB_SHARED_WORKDIR (team guidance).
export UV_CACHE_DIR="$PWD/.uv-cache"

uv venv "$venv"
# shellcheck source=/dev/null
. "$venv/bin/activate"

if [ -n "$torch_index_url" ]; then
  echo "eval/lm-eval: installing torch from $torch_index_url"
  uv pip install --quiet --index-url "$torch_index_url" torch
fi

if [ "$#" -gt 0 ]; then
  echo "eval/lm-eval: installing $* from $index_url"
  uv pip install --quiet --index-url "$index_url" "$@"
fi

python - <<'EOF'
import importlib.metadata as m
for d in ("lm_eval", "torch", "transformers"):
    try:
        print(f"eval/lm-eval: {d}=={m.version(d)}")
    except m.PackageNotFoundError:
        pass
EOF
