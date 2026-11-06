#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Bare-node install for the mergekit step, invoked from the generated step.yaml's
# `setup:` block and shipped to the node by `file_mounts: {src: src}`.
#
# The same install as the data-designer step's dd_setup.sh (and dpk's dpk_setup.sh),
# except that it removes an existing venv first (below). Those carry the full rationale:
# SkyPilot syncs file mounts before `setup`, uv is bootstrapped with pip because a bare
# launcher node does not have it, `--break-system-packages` is needed on PEP 668
# interpreters, and UV_CACHE_DIR is anchored at the environment's shared root so the
# cache is reused across runs. That
# matters more here than for data-designer: mergekit pulls torch, and on Linux PyPI's
# torch pulls the CUDA wheels with it (about 2 GB). With a shared workdir (local SLURM
# and kind: /shared) that is once per cluster; where GB_SHARED_WORKDIR is unset (aws)
# the cache is the run's own workdir, so every run downloads it again.
#
# Only invoked when config.mergekit_config.mk_image is empty. With an image the step
# skips this entirely; the image is expected to provide mergekit-yaml.
#
# CONTRACT
#   setup.sh --venv <dir> --index-url <url> [--] [pip requirements...]
#
#   --venv        directory to create the virtualenv in (the step passes ./venv).
#   --index-url   package index passed to `uv pip install`.
#   --            everything after it is a pip requirement to install (the step
#                 passes mergekit==<mergekit_version>). Zero requirements is valid.
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

# Python 3.12, pinned: mergekit needs >= 3.10, and a fixed version keeps uv from
# picking the newest CPython (where torch may have no wheel yet) when the node has no
# suitable interpreter. uv downloads a managed CPython once into the shared cache then.
export UV_PYTHON_INSTALL_DIR="${GB_SHARED_WORKDIR:-$PWD}/.uv-python"
# Removed first: a retry or a re-run in the same workdir finds the last attempt's venv,
# possibly half-made, which `uv venv` refuses. rm rather than `uv venv --clear --force`:
# `pip install uv` keeps a uv already on the node, and those flags need uv 0.8 / 0.12.
# Starting fresh also drops a half-finished install; the shared uv cache makes the
# reinstall cheap.
rm -rf -- "$venv"
uv venv --python 3.12 "$venv"
# shellcheck source=/dev/null
. "$venv/bin/activate"

if [ "$#" -gt 0 ]; then
  uv pip install --quiet --index-url "$index_url" "$@"
fi
