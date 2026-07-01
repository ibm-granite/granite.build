#!/bin/bash
# Entry point: build a per-step venv and run run.py in it.
# Byte-for-byte identical across the inference, inference-lora, and lora-finetune
# steps (venv name derived from the script's dir); keep the three copies in sync.
#
# Absolute `#!/bin/bash` (not `env`): the launcher runs steps with a PATH-less env,
# so env-based interpreter resolution can fail. /bin/bash is guaranteed on the image.
set -eu
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STEP="$(basename "$SCRIPT_DIR")"

# The launcher passes its own (pinned >=3.11) Python dir; lead PATH with it so we
# don't fall through to a host's system python (e.g. macOS 3.9).
export PATH="${LLMB_BASH_PYTHON_DIR:?command.sh: launcher must set LLMB_BASH_PYTHON_DIR}:/usr/local/bin:/usr/bin:/bin"

# Cache the venv under the GB home (recovered from LLMB_BASH_OUTPUT_DIR, not $HOME
# which the launcher doesn't pass) so it persists across reruns.
OUT="${LLMB_BASH_OUTPUT_DIR:-}"
case "$OUT" in
  */workdir/*) VENV_BASE="${OUT%%/workdir/*}/.gb-venvs" ;;
  ?*)          VENV_BASE="$OUT/.gb-venvs" ;;
  *)           VENV_BASE="${TMPDIR:-/tmp}/.gb-venvs" ;;
esac
mkdir -p "$VENV_BASE"

PY="$LLMB_BASH_PYTHON_DIR/python3"
[ -x "$PY" ] || PY="python3"

VENV="$VENV_BASE/$STEP"
if [ ! -x "$VENV/bin/python" ]; then
  echo "command.sh: creating venv at $VENV using $PY"
  "$PY" -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
fi

# Install deps (near-no-op once satisfied, so cheap on reruns).
echo "command.sh: installing requirements into $VENV"
"$VENV/bin/pip" install --quiet -r "$SCRIPT_DIR/requirements.txt"

echo "command.sh: launching run.py with $VENV/bin/python"
exec "$VENV/bin/python" "$SCRIPT_DIR/run.py"
