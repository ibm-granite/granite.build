#!/bin/sh
# Entry point for the lora-finetune bash step.
#
# The nohup launcher runs steps with a sanitized, PATH-less env, so a
# `#!/usr/bin/env python3` shebang on run.py is unreliable. Resolve a real
# interpreter by trying absolute paths then PATH (works on the container's 3.13
# and on a host's python3), build a dedicated venv once, and exec run.py in it.
# run.py's ensure_deps() then pip-installs (version-capped) deps INTO this venv
# on first run; the venv is cached for reruns.
set -eu
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

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

PY=""
for c in /usr/local/bin/python3.13 python3.13 python3; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
[ -n "$PY" ] || { echo "command.sh: no python3 interpreter found" >&2; exit 127; }

VENV="$VENV_BASE/lora-finetune"
if [ ! -x "$VENV/bin/python" ]; then
  echo "command.sh: creating venv at $VENV using $PY"
  "$PY" -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
fi

echo "command.sh: launching run.py with $VENV/bin/python"
exec "$VENV/bin/python" "$SCRIPT_DIR/run.py"
