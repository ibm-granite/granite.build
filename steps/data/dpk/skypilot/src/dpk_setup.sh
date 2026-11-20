#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
#
# Bare-node dependency install for the dpk step, invoked from the generated
# step.yaml's `setup:` block and shipped to the node by `file_mounts: {src: src}`.
#
# Safe to call from `setup` because SkyPilot syncs file mounts BEFORE running it:
# sky/execution.py's stage order is PROVISION -> SYNC_WORKDIR ->
# SYNC_FILE_MOUNTS -> SETUP -> PRE_EXEC -> EXEC, and `_execute` calls
# backend.sync_file_mounts() before backend.setup() unconditionally.
#
# Only invoked in bare-node mode. When config.dpk_config.dpk_image is set the step
# skips this entirely — the image is expected to already provide DPK.
#
# CONTRACT
#   dpk_setup.sh --venv <dir> --index-url <url> [--constraint <spec>]...
#                [--] [pip requirements...]
#
#   --venv        directory to create the virtualenv in (the step passes ./venv,
#                 relative to the working directory `run` also starts in).
#   --index-url   package index passed to `uv pip install`.
#   --constraint  one version constraint (e.g. "pyarrow<26"); repeatable. They are
#                 written one per line to <venv>/gb-constraints.txt and passed as
#                 `uv pip install --constraint <file>`, which bounds packages the
#                 requirements pull in without adding any. The file is left in the
#                 venv so a failed install can be diagnosed on the node. The step
#                 passes dpk_config.pip_constraints here (default "pyarrow<26",
#                 #653), after dpk_guard.sh has validated them; the same per-entry
#                 rule is repeated below as a backstop for direct callers. Zero
#                 constraints is valid and passes no --constraint.
#
#   Every option takes a value; one given last with none is refused (exit 2)
#   rather than tripping `set -u` on an unbound $2.
#   --            everything after it is a pip requirement to install. The step
#                 passes exactly one, the derived
#                 data-prep-toolkit-transforms[extra]==version; constraints never
#                 travel here, they have their own flag. Zero requirements is valid
#                 — the caller may want a bare venv — so the install is skipped
#                 rather than failing.
set -euo pipefail

venv=""
index_url=""
# Kept as a newline-joined string rather than an array: "${arr[@]}" on an empty
# array is an unbound-variable error under `set -u` in bash < 4.4 (macOS /bin/bash
# is 3.2), and these are written to a file one per line anyway.
constraints=""

# Each entry must be exactly one specifier line of the constraints file, which is
# requirements-file syntax: a leading '-' is an OPTION line (uv honours
# --extra-index-url, -c and -r there), a leading '#' a comment, a trailing '\' a
# continuation, an embedded CR/LF a second line, and an empty entry a blank line
# that is ignored. dpk_guard.sh refuses the same shapes with a config-level message;
# see its "PIP CONSTRAINTS" section. Leading whitespace is refused because the
# parser strips it before seeing the '-' or '#'.
bad_constraint() {
  case $1 in
    ""|*$'\n'*|*$'\r'*|[[:space:]]*|-*|\#*|*\\) return 0 ;;
  esac
  return 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --venv|--index-url|--constraint)
      if [ "$#" -lt 2 ]; then
        echo "dpk_setup.sh: $1 needs a value" >&2
        exit 2
      fi ;;
  esac
  case "$1" in
    --venv)       venv="$2";       shift 2 ;;
    --index-url)  index_url="$2";  shift 2 ;;
    --constraint)
      if bad_constraint "$2"; then
        echo "dpk_setup.sh: --constraint must be one version specifier (not empty," >&2
        echo "dpk_setup.sh: multi-line, or starting with '-', '#' or whitespace, or" >&2
        echo "dpk_setup.sh: ending with a backslash): '$2'" >&2
        exit 2
      fi
      constraints="${constraints}${2}"$'\n'
      shift 2 ;;
    --)           shift; break ;;
    *)            break ;;
  esac
done

: "${venv:?--venv is required}"
: "${index_url:?--index-url is required}"

# Installs go through uv, which DPK's own Dockerfile.python also uses. It
# resolves and installs far faster than pip, and it populates a venv by
# hard-linking from a shared cache instead of copying — which matters because a
# heavyweight extra like [pii-redactor] pulls ~125 packages (torch, flair,
# presidio) and a full copy costs ~6G per venv. uv is not preinstalled on a bare
# launcher node, so bootstrap it with pip first (same order as the DPK image).
#
# --break-system-packages is required on PEP 668 "externally managed" interpreters
# (Debian 12+, Ubuntu 23.04+, recent Fedora — increasingly the default), where a
# plain `pip install` into the system interpreter dies with
# `error: externally-managed-environment`. This is the FIRST command in `setup`
# under `set -eu`, so without it cluster bring-up fails before the venv exists.
# The flag is accepted from pip 23.0.1; older pip does not know it and there is no
# 668 marker to trip on, so try it first and fall back. --user is not a substitute:
# it is ignored under an active venv and still blocked on some 668 distros.
#
# Installing one leaf tool into the system interpreter is what the flag is for; the
# DPK dependencies it then resolves all land in the venv below, never system-wide.
#
# ON ASSUMING A BARE `pip`. Both lines below call `pip`, not `pip3` or
# `python3 -m pip`, and on a minimal image that has only the latter they would fail
# with 127 under `set -eu` — before the venv exists, and reporting
# "pip: command not found" rather than anything actionable. That is a real shape, and
# the reason it is accepted rather than worked around is that the guarantee comes from
# SKYPILOT, not from the image: SkyPilot provisions its own Python environment on the
# node and runs `setup` inside it, which is where `pip` comes from. Measured on the
# local Docker SLURM cluster, whose containers have NEITHER `pip` nor `pip3` in a
# plain login shell — yet every fixture installs DPK fine, because setup does not run
# in a plain login shell.
#
# So the dependency is on SkyPilot's environment contract. If a future endpoint ever
# breaks it, the fix is `python3 -m pip` (which cannot be assumed either — a
# python-less image has no bootstrap at all) or a preinstalled uv via `dpk_image`.
if ! pip install --quiet --no-cache-dir --break-system-packages uv 2>/dev/null; then
  pip install --quiet --no-cache-dir uv
fi

# UV_CACHE_DIR must be (a) on the same filesystem as the venv, or uv silently
# copies instead of hard-linking, and (b) STABLE ACROSS RUNS, or there is nothing
# to link from and the cache is pure overhead. A cache inside the per-run workdir
# satisfies (a) but not (b) — measured: the venv shrank 5.8G -> 5.5G while a fresh
# 6.2G cache appeared, doubling the footprint. So anchor it at the
# environment-level shared root when there is one, falling back to the per-run dir
# where there is not (e.g. aws, where each step is its own instance anyway).
export UV_CACHE_DIR="${GB_SHARED_WORKDIR:-$PWD}/.uv-cache"

uv venv "$venv"
# shellcheck source=/dev/null
. "$venv/bin/activate"

# Requirements arrive as real argv, so a version specifier containing characters
# the shell would otherwise split or glob (e.g. the "[extra]" in
# data-prep-toolkit-transforms[pii-redactor]==1.1.8) needs no re-quoting here.
# Skip the install when there are none: `uv pip install` with no arguments is an
# error, and a bare venv is a legitimate outcome.
if [ "$#" -gt 0 ]; then
  if [ -n "$constraints" ]; then
    constraints_file="$venv/gb-constraints.txt"
    printf '%s' "$constraints" > "$constraints_file"
    # A constraint that has gone stale (say, a DPK release that needs pyarrow>=26)
    # surfaces as uv's "No solution found", which names the package but not where
    # the bound came from. Point at the config key, keeping uv's exit status.
    rc=0
    uv pip install --quiet --index-url "$index_url" --constraint "$constraints_file" "$@" || rc=$?
    if [ "$rc" -ne 0 ]; then
      echo "dpk: the dependency install failed with these dpk_config.pip_constraints applied:" >&2
      sed 's/^/dpk:   /' "$constraints_file" >&2
      echo "dpk: if uv's error above names a constrained package, the constraint may conflict" >&2
      echo "dpk: with this dpk_version: override pip_constraints in the build ([] for none)." >&2
      exit "$rc"
    fi
  else
    uv pip install --quiet --index-url "$index_url" "$@"
  fi
fi
