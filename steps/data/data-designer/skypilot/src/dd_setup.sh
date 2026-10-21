#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Bare-node dependency install for the data-designer step, invoked from the generated
# step.yaml's `setup:` block and shipped to the node by `file_mounts: {src: src}`.
#
# The same install as the dpk step's dpk_setup.sh, which carries the full rationale:
# SkyPilot syncs file mounts before `setup` (PROVISION -> SYNC_WORKDIR ->
# SYNC_FILE_MOUNTS -> SETUP), uv is bootstrapped with pip because a bare launcher node
# does not have it, `--break-system-packages` is needed on PEP 668 interpreters, and
# UV_CACHE_DIR is anchored at the environment's shared root so the cache is reused
# across runs and hard-links into the venv. Data Designer's wheel set is far lighter
# than DPK's heavy extras (no torch), but the cache still saves the re-download.
#
# Only invoked in bare-node mode. When config.data_designer_config.dd_image is set the
# step skips this entirely — the image is expected to already provide data-designer.
#
# CONTRACT
#   dd_setup.sh --venv <dir> --index-url <url> [--] [pip requirements...]
#
#   --venv        directory to create the virtualenv in (the step passes ./venv).
#   --index-url   package index passed to `uv pip install`.
#   --            everything after it is a pip requirement to install (the step
#                 passes data-designer==<dd_version>). Zero requirements is valid.
set -euo pipefail

venv=""
index_url=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --venv)       venv="$2";       shift 2 ;;
    --index-url)  index_url="$2";  shift 2 ;;
    --)           shift; break ;;
    *)            break ;;
  esac
done

: "${venv:?--venv is required}"
: "${index_url:?--index-url is required}"

if ! pip install --quiet --no-cache-dir --break-system-packages uv 2>/dev/null; then
  pip install --quiet --no-cache-dir uv
fi

export UV_CACHE_DIR="${GB_SHARED_WORKDIR:-$PWD}/.uv-cache"

uv venv "$venv"
# shellcheck source=/dev/null
. "$venv/bin/activate"

if [ "$#" -gt 0 ]; then
  uv pip install --quiet --index-url "$index_url" "$@"
fi
