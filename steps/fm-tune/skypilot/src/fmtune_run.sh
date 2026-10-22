#!/usr/bin/env bash
# Run one fm-tune (autotunex/src/fm-tune/main.py) tuning job.
#
# Shared by the bash and skypilot copies of the fm-tune step (byte-identical;
# steps/fm-tune/skypilot/test/test_fm_tune_script_copies_match.py enforces it).
# The launcher maps fm_tune_config and the input bindings to FMTUNE_* env vars;
# this script validates them, resolves the fm-tune source, builds or reuses a
# cached venv, writes the HPO config, runs main.py with --model_at_root, and on
# success registers FMTUNE_OUTPUT_DIR as the `model` output.
set -euo pipefail

die() { echo "fm-tune: $*" >&2; exit 2; }
log() { echo "[fm-tune] $*"; }

: "${FMTUNE_PYTHON:=}"
# Bash launcher env values reach the child unexpanded (bash.py _launch_env_layers
# passes them to create_subprocess_exec), so the bash template leaves these three
# empty and they default here from the LLMB_BASH_* builtins. SkyPilot sets them.
if [ -z "${FMTUNE_OUTPUT_DIR:-}" ] && [ -n "${LLMB_BASH_OUTPUT_DIR:-}" ]; then
  FMTUNE_OUTPUT_DIR="$LLMB_BASH_OUTPUT_DIR/model"   # a subdir: --model_at_root clears its output_dir, and job.log lives in LLMB_BASH_OUTPUT_DIR
fi
if [ -z "${FMTUNE_CACHE_DIR:-}" ] && [ -n "${LLMB_BASH_OUTPUT_DIR:-}" ]; then
  # The bash output dir is <GB_HOME>/workdir/llm-build-<id>/.../launch-<id>/outputs
  # (per build), so cache beside workdir/ to reuse the venv across builds.
  case "$LLMB_BASH_OUTPUT_DIR" in
    */workdir/*) FMTUNE_CACHE_DIR="${LLMB_BASH_OUTPUT_DIR%%/workdir/*}/.fm-tune-cache" ;;
    *) FMTUNE_CACHE_DIR="$LLMB_BASH_OUTPUT_DIR/../../.fm-tune-cache" ;;
  esac
fi
if [ -z "$FMTUNE_PYTHON" ]; then
  FMTUNE_PYTHON="${LLMB_BASH_PYTHON_DIR:+$LLMB_BASH_PYTHON_DIR/}python3"
fi
if [ -n "${LLMB_BASH_PYTHON_DIR:-}" ]; then
  export PATH="$LLMB_BASH_PYTHON_DIR:${PATH:-/usr/local/bin:/usr/bin:/bin}"
fi
: "${FMTUNE_SOURCE_REF:=main}"
: "${FMTUNE_SOURCE_SUBDIR:=autotunex/src/fm-tune}"
: "${FMTUNE_EXTRA:=core}"
: "${FMTUNE_TUNING_ALGO:=}"
: "${FMTUNE_RL_ALGO:=}"
: "${FMTUNE_RUN_NAME:=fm-tune}"
: "${FMTUNE_AUTOTUNE:=true}"
: "${FMTUNE_BACKEND:=}"
: "${FMTUNE_SEED:=}"
: "${FMTUNE_CONFIG_PRESET:=autotune.yaml}"
: "${FMTUNE_HPO_CONFIG_JSON:=}"
: "${FMTUNE_EXTRA_ARGS:=}"
: "${FMTUNE_SERVER_URL:=}"
: "${FMTUNE_JOB_ID:=}"
: "${FMTUNE_MODEL:=}"
: "${FMTUNE_BASE_MODEL_DIR:=}"
: "${FMTUNE_SOURCE_PATH:=}"
: "${FMTUNE_SOURCE_REPO:=}"

# ---- validate (before any clone or install) --------------------------------
[ -n "${FMTUNE_DATASET_DIR:-}" ] || die "FMTUNE_DATASET_DIR is empty (bind the dataset input)"
[ -n "${FMTUNE_OUTPUT_DIR:-}" ] || die "FMTUNE_OUTPUT_DIR is empty"
[ -n "${FMTUNE_CACHE_DIR:-}" ] || die "FMTUNE_CACHE_DIR is empty"
if { [ -n "$FMTUNE_SOURCE_PATH" ] && [ -n "$FMTUNE_SOURCE_REPO" ]; } \
   || { [ -z "$FMTUNE_SOURCE_PATH" ] && [ -z "$FMTUNE_SOURCE_REPO" ]; }; then
  die "set exactly one of source.path and source.repo"
fi
if [ -n "$FMTUNE_BASE_MODEL_DIR" ]; then
  model="$FMTUNE_BASE_MODEL_DIR"
elif [ -n "$FMTUNE_MODEL" ]; then
  model="$FMTUNE_MODEL"
else
  die "set fm_tune_config.model or bind the base_model input"
fi
[ -n "${FMTUNE_TRAIN_FILE:-}" ] || die "train_file is empty"
[ -n "${FMTUNE_VALIDATION_FILE:-}" ] || die "validation_file is empty"
train="$FMTUNE_DATASET_DIR/$FMTUNE_TRAIN_FILE"
val="$FMTUNE_DATASET_DIR/$FMTUNE_VALIDATION_FILE"
[ -e "$train" ] || die "train_file not found: $train"
[ -e "$val" ] || die "validation_file not found: $val"

# ---- source ----------------------------------------------------------------
mkdir -p "$FMTUNE_CACHE_DIR"
if [ -n "$FMTUNE_SOURCE_PATH" ]; then
  root="$FMTUNE_SOURCE_PATH"
else
  key="$(printf '%s@%s' "$FMTUNE_SOURCE_REPO" "$FMTUNE_SOURCE_REF" | cksum | cut -d' ' -f1)"
  clone="$FMTUNE_CACHE_DIR/src-$key"
  if [ ! -d "$clone/.git" ]; then
    log "cloning $FMTUNE_SOURCE_REPO@$FMTUNE_SOURCE_REF"
    rm -rf "$clone.tmp"
    git clone --quiet --depth 1 --branch "$FMTUNE_SOURCE_REF" "$FMTUNE_SOURCE_REPO" "$clone.tmp"
    mv "$clone.tmp" "$clone"
  fi
  root="$clone/$FMTUNE_SOURCE_SUBDIR"
fi
[ -f "$root/main.py" ] || die "no main.py under fm-tune source $root"

preset="$root/autotune/configs/$FMTUNE_CONFIG_PRESET"
if [ "$FMTUNE_HPO_CONFIG_JSON" = "{}" ] || [ -z "$FMTUNE_HPO_CONFIG_JSON" ]; then
  [ -f "$preset" ] || die "config_preset not found: $preset"
  config_file="$preset"
else
  mkdir -p "$FMTUNE_OUTPUT_DIR"
  config_file="$(cd "$FMTUNE_OUTPUT_DIR/.." && pwd)/fm_tune_hpo.json"
  printf '%s\n' "$FMTUNE_HPO_CONFIG_JSON" > "$config_file"
fi

# ---- venv ------------------------------------------------------------------
if [ "${FMTUNE_SKIP_INSTALL:-}" = "1" ]; then
  py="$FMTUNE_PYTHON"
else
  pyver="$("$FMTUNE_PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  "$FMTUNE_PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
    || die "fm-tune needs Python >= 3.10, $FMTUNE_PYTHON is $pyver"
  vkey="$( { printf '%s\n%s\n%s\n' "$root" "$FMTUNE_EXTRA" "$pyver"; cat "$root/pyproject.toml"; } | cksum | cut -d' ' -f1)"
  venv="$FMTUNE_CACHE_DIR/venv-$vkey"
  lock="$venv.lock"
  until mkdir "$lock" 2>/dev/null; do
    owner="$(cat "$lock/pid" 2>/dev/null || true)"
    if [ -n "$owner" ] && ! kill -0 "$owner" 2>/dev/null; then
      log "taking over stale venv lock from dead pid $owner"; rm -rf "$lock"; continue
    fi
    log "waiting for venv lock $lock"; sleep 5
  done
  echo $$ > "$lock/pid"
  trap 'rm -rf "$lock"' EXIT
  if [ ! -f "$venv/.installed" ]; then
    log "building venv $venv (extra: $FMTUNE_EXTRA)"
    rm -rf "$venv"
    "$FMTUNE_PYTHON" -m venv "$venv"
    "$venv/bin/python" -m pip install --quiet --upgrade pip
    "$venv/bin/python" -m pip install --quiet -e "$root[$FMTUNE_EXTRA]"
    touch "$venv/.installed"
  fi
  rm -rf "$lock"; trap - EXIT
  py="$venv/bin/python"
fi

# ---- run -------------------------------------------------------------------
args=(
  --config_file "$config_file"
  --train_file "$train"
  --validation_file "$val"
  --model_name_or_path "$model"
  --run_name "$FMTUNE_RUN_NAME"
  --output_dir "$FMTUNE_OUTPUT_DIR"
  --output_model_name "$FMTUNE_RUN_NAME"
  --model_at_root --cleanup --save_history
)
[ -n "$FMTUNE_TUNING_ALGO" ] && args+=(--tuning_algo "$FMTUNE_TUNING_ALGO")
[ -n "$FMTUNE_RL_ALGO" ] && args+=(--rl_algo "$FMTUNE_RL_ALGO")
[ -n "$FMTUNE_BACKEND" ] && args+=(--backend "$FMTUNE_BACKEND")
[ -n "$FMTUNE_SEED" ] && args+=(--seed "$FMTUNE_SEED")
[ "$FMTUNE_AUTOTUNE" = "false" ] && args+=(--no_autotune)
[ -n "$FMTUNE_SERVER_URL" ] && args+=(--autotunex_server_url "$FMTUNE_SERVER_URL")
[ -n "$FMTUNE_JOB_ID" ] && args+=(--job_id "$FMTUNE_JOB_ID")
if [ -n "$FMTUNE_EXTRA_ARGS" ]; then
  read -r -a extra <<< "$FMTUNE_EXTRA_ARGS"
  args+=("${extra[@]}")
fi

mkdir -p "$FMTUNE_OUTPUT_DIR"
log "running main.py (run_name=$FMTUNE_RUN_NAME)"
cd "$root"
status=0
"$py" main.py "${args[@]}" || status=$?
if [ "$status" -ne 0 ]; then
  echo "[fm-tune] main.py failed with exit code $status" >&2
  exit "$status"
fi
echo "GB_ARTIFACT_ID:model GB_ARTIFACT_PATH:$FMTUNE_OUTPUT_DIR"
