#!/bin/bash
# Entry point for a bash step that runs a Python run.py in a dedicated venv.
#
# This script is byte-for-byte identical across the inference, inference-lora,
# and lora-finetune steps: the step name (and thus the venv) is derived from the
# script's own directory, and the dependency set lives in a per-step
# requirements.txt alongside run.py. Keep the three copies in sync.
#
# Shebang note: `#!/bin/bash` is an ABSOLUTE path on purpose, not
# `#!/usr/bin/env bash`. The nohup launcher runs steps with a clean env that has
# no PATH (see bash.py launch_nohup, which passes env= without inheriting
# os.environ), so any `env`-based resolution — `env bash` here, or `env python3`
# on run.py — can fail to find its interpreter. An absolute path the kernel
# resolves directly sidesteps that. /bin/bash is guaranteed on the deploy image.
#
# The launcher resolves the interpreter for us (gbserver runs on a pinned
# Python, requires-python >=3.11,<3.14) and passes its directory as
# LLMB_BASH_PYTHON_DIR. We prepend that onto a known-good default PATH so this
# script (and the venv it builds) use that interpreter — no re-discovery, and no
# risk of falling through to a host's system python (e.g. macOS /usr/bin/python3,
# which is 3.9 and too old for the step's deps).
set -eu
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STEP="$(basename "$SCRIPT_DIR")"

# Interpreter dir resolved by the launcher; prepend it to a known-good PATH.
export PATH="${LLMB_BASH_PYTHON_DIR:?command.sh: launcher must set LLMB_BASH_PYTHON_DIR}:/usr/local/bin:/usr/bin:/bin"

# Pick a stable, writable base for the cached venv WITHOUT relying on $HOME: the
# nohup launcher builds the job env from scratch (see bash.py launch_nohup) and
# does NOT pass HOME, so `set -u` would abort on it. It DOES always export
# LLMB_BASH_OUTPUT_DIR, shaped "<gb-home>/workdir/llm-build-<id>/.../outputs".
# Strip at "/workdir/" to recover the stable per-user GB home root, so the venv
# is cached across builds/reruns (not rebuilt per launch). Fall back to the
# output dir itself, then /tmp, if the shape is unexpected.
OUT="${LLMB_BASH_OUTPUT_DIR:-}"
case "$OUT" in
  */workdir/*) VENV_BASE="${OUT%%/workdir/*}/.gb-venvs" ;;
  ?*)          VENV_BASE="$OUT/.gb-venvs" ;;
  *)           VENV_BASE="${TMPDIR:-/tmp}/.gb-venvs" ;;
esac
mkdir -p "$VENV_BASE"

PY="$LLMB_BASH_PYTHON_DIR/python3"
[ -x "$PY" ] || PY="python3"  # fall back to PATH (which now leads with the dir above)

VENV="$VENV_BASE/$STEP"
if [ ! -x "$VENV/bin/python" ]; then
  echo "command.sh: creating venv at $VENV using $PY"
  "$PY" -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
fi

# Install (version-capped) deps into the venv. pip is a near-no-op once the
# requirements are already satisfied, so this is cheap on reruns.
echo "command.sh: installing requirements into $VENV"
"$VENV/bin/pip" install --quiet -r "$SCRIPT_DIR/requirements.txt"

echo "command.sh: launching run.py with $VENV/bin/python"
exec "$VENV/bin/python" "$SCRIPT_DIR/run.py"
