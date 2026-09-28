#!/bin/bash
# autotune step entrypoint (Bash + Docker). Jinja-rendered with strict=True at
# run time, then executed. Stages:
#   S1  materialize the tuning config (inline config.autotune-config, else the
#       hpo_config input) to a YAML file and export AUTOTUNE_CONFIG_FILE
#   S2  provide a Python interpreter (venv on bash; image python on docker)
#   S3  optional SETUP_COMMAND update hook, run inside $FM_TUNE_ROOT
#   S4  exec run.py, which builds the main.py argv and runs fm-tune
#
# Absolute #!/bin/bash (not env): the launcher runs with a sanitized PATH-less
# env, so lead PATH with the launcher's pinned python dir (bash). Shell ${VAR}
# uses single braces and is NOT a Jinja token.
set -eu
# Prepend, do not replace: on docker LLMB_BASH_PYTHON_DIR is unset (it is a Bash-
# environment variable) and the image's own PATH is what carries its interpreter
# and CUDA tools, so clobbering it would strip conda/venv paths. The literal
# fallback only applies when the launcher hands us no PATH at all.
export PATH="${LLMB_BASH_PYTHON_DIR:+$LLMB_BASH_PYTHON_DIR:}${PATH:-/usr/local/bin:/usr/bin:/bin}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

: "${FM_TUNE_ROOT:?command.sh: FM_TUNE_ROOT must be set (fm-tune checkout on bash; image path on docker)}"
OUT="${LLMB_BASH_OUTPUT_DIR:?command.sh: LLMB_BASH_OUTPUT_DIR unset}"
mkdir -p "$OUT"

# --- S1: materialize the tuning config -------------------------------------
# Deliberately NOT under $OUT: $OUT is registered as the `custom` model artifact,
# and the tuning config should not ship inside the model.
CONFIG_DIR="$(dirname "$OUT")/.gb-autotune"
mkdir -p "$CONFIG_DIR"
CONFIG_FILE="$CONFIG_DIR/autotune_config.yaml"
{% if 'autotune-config' in config %}
echo '{{ config['autotune-config'] | to_yaml | b64encode }}' | base64 -d > "$CONFIG_FILE"
{% else %}
if [ -n "${LLMB_BASH_INPUT_HPO_CONFIG:-}" ]; then
  # `fileset` binds a directory as often as a single file: pick the sole YAML inside.
  if [ -d "$LLMB_BASH_INPUT_HPO_CONFIG" ]; then
    HPO_SRC="$(find "$LLMB_BASH_INPUT_HPO_CONFIG" -maxdepth 1 -type f \
      \( -name '*.yaml' -o -name '*.yml' \) | sort | head -n 1)"
    if [ -z "$HPO_SRC" ]; then
      echo "command.sh: no .yaml/.yml file in hpo_config dir $LLMB_BASH_INPUT_HPO_CONFIG" >&2
      exit 1
    fi
  else
    HPO_SRC="$LLMB_BASH_INPUT_HPO_CONFIG"
  fi
  cp "$HPO_SRC" "$CONFIG_FILE"
else
  echo "command.sh: no config.autotune-config block and no hpo_config input bound" >&2
  exit 1
fi
{% endif %}
export AUTOTUNE_CONFIG_FILE="$CONFIG_FILE"

# --- S2: interpreter --------------------------------------------------------
if [ "${BASH_BUILD_VENV:-true}" = "true" ]; then
  case "$OUT" in
    */workdir/*) VENV_BASE="${OUT%%/workdir/*}/.gb-venvs" ;;
    *)           VENV_BASE="${TMPDIR:-/tmp}/.gb-venvs" ;;
  esac
  mkdir -p "$VENV_BASE"
  export HF_HOME="$VENV_BASE/hf-cache"
  # FM_TUNE_ROOT may be a git remote (ssh/https/*.git) rather than a local
  # checkout: clone it once into VENV_BASE and repoint FM_TUNE_ROOT at the clone
  # so the editable install and run.py's cwd both operate on a real tree.
  # FM_TUNE_REF (optional) pins a branch or tag. Private repos need git creds on
  # the runner. On docker (BASH_BUILD_VENV=false) FM_TUNE_ROOT is a baked image
  # path, so this branch is skipped.
  case "$FM_TUNE_ROOT" in
    git@*|*://*|*.git)
      # Key the clone dir on the ref: a single shared dir would silently serve a
      # stale ref to any later build that asked for a different one.
      FM_TUNE_REF_SLUG="$(printf '%s' "${FM_TUNE_REF:-default}" | tr '/:@' '---')"
      FM_TUNE_SRC="$VENV_BASE/fm-tune-src-$FM_TUNE_REF_SLUG"
      if [ -d "$FM_TUNE_SRC/.git" ]; then
        # Refresh so a moving branch does not go stale; a network blip is not fatal
        # because the existing checkout is already at the requested ref.
        ( cd "$FM_TUNE_SRC" \
          && git fetch --depth 1 origin "${FM_TUNE_REF:-HEAD}" \
          && git checkout --force FETCH_HEAD ) \
          || echo "command.sh: warning: could not refresh $FM_TUNE_SRC, using as-is" >&2
      else
        # Clear any half-finished clone before retrying.
        rm -rf "$FM_TUNE_SRC"
        git clone --depth 1 ${FM_TUNE_REF:+--branch "$FM_TUNE_REF"} "$FM_TUNE_ROOT" "$FM_TUNE_SRC"
      fi
      export FM_TUNE_ROOT="$FM_TUNE_SRC"
      ;;
  esac
  PY="${LLMB_BASH_PYTHON_DIR:-}/python3"; [ -x "$PY" ] || PY=python3
  VENV="$VENV_BASE/autotune"
  [ -x "$VENV/bin/python" ] || { "$PY" -m venv "$VENV"; "$VENV/bin/pip" install --quiet --upgrade pip; }
  # fm-tune's main.py imports `ray` unconditionally; ray lives in fm-tune's `core`
  # and `full` extras, NOT the base package (which is just pyyaml). Install a
  # real extra via FM_TUNE_EXTRA (default `core`: light, ray+datasets; `full` adds
  # verl/vllm/flash-attn for GPU). Set FM_TUNE_EXTRA= (empty) for a base-only install.
  # BACKEND stays a runtime choice: run.py passes it to main.py --backend {torch,mlx}.
  # `-` not `:-`: an explicitly empty FM_TUNE_EXTRA must stay empty so the
  # documented base-only install below is reachable.
  FM_TUNE_EXTRA="${FM_TUNE_EXTRA-core}"
  if [ -n "$FM_TUNE_EXTRA" ]; then
    "$VENV/bin/pip" install --quiet -e "${FM_TUNE_ROOT}[${FM_TUNE_EXTRA}]"
  else
    "$VENV/bin/pip" install --quiet -e "$FM_TUNE_ROOT"
  fi
  PYTHON="$VENV/bin/python"
else
  PYTHON="$(command -v python3 || command -v python || true)"
  if [ -z "$PYTHON" ]; then
    echo "command.sh: no python3/python on PATH ($PATH); BASH_BUILD_VENV=false expects the image to provide one" >&2
    exit 1
  fi
fi

# Put the chosen interpreter's bin dir first so SETUP_COMMAND's `pip`/`python`
# resolve to the interpreter S4 actually execs, not the launcher's or the distro's.
PATH="$(dirname "$PYTHON"):$PATH"; export PATH

# --- S3: optional update hook ----------------------------------------------
if [ -n "${SETUP_COMMAND:-}" ]; then
  ( cd "$FM_TUNE_ROOT" && sh -c "$SETUP_COMMAND" )
fi

# --- S4: run ----------------------------------------------------------------
exec "$PYTHON" "$SCRIPT_DIR/run.py"
