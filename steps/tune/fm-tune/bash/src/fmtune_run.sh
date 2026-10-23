#!/usr/bin/env bash
# Run one fm-tune (autotunex/src/fm-tune/main.py) tuning job.
#
# Shared by the bash and skypilot copies of the fm-tune step (byte-identical;
# steps/tune/fm-tune/skypilot/test/test_fm_tune_script_copies_match.py enforces it).
# The launcher maps fm_tune_config and the input bindings to FMTUNE_* env vars;
# this script validates them, resolves the fm-tune source, builds or reuses a
# cached venv, writes the HPO config, runs main.py with --model_at_root, and on
# success registers FMTUNE_OUTPUT_DIR as the `model` output.
set -euo pipefail

die() { echo "fm-tune: $*" >&2; exit 2; }
log() { echo "[fm-tune] $*"; }
# One cache lock at a time: a mkdir lock dir holding the owner's <host>:<pid>.
# Builds share the cache, so the clone and the venv are both built under one. A
# lock is stale only if its owner ran on this host and is dead; another host's pid
# can't be checked from here, so such a lock is waited on for FMTUNE_LOCK_TIMEOUT
# seconds and then reported, never taken over. A lock with no pid file (owner
# killed between mkdir and the write) is stale once older than the timeout.
: "${FMTUNE_LOCK_TIMEOUT:=1800}"
case "$FMTUNE_LOCK_TIMEOUT" in ''|*[!0-9]*) die "FMTUNE_LOCK_TIMEOUT must be a whole number of seconds" ;; esac
host="$(uname -n)"
held_lock=""
lock() {
  local owner start=$SECONDS
  while ! mkdir "$1" 2>/dev/null; do
    if [ ! -d "$1" ]; then
      mkdir "$1" 2>/dev/null && break  # released between the two mkdirs
      die "cannot create lock $1 (is $(dirname "$1") writable?)"
    fi
    owner="$(cat "$1/pid" 2>/dev/null || true)"
    if [ -z "$owner" ]; then
      # find -mmin (whole minutes, rounded up) is the same on GNU and BSD; stat is not.
      if [ -n "$(find "$1" -maxdepth 0 -mmin +$(( (FMTUNE_LOCK_TIMEOUT + 59) / 60 )) 2>/dev/null)" ]; then
        log "taking over stale lock $1 (no owner recorded)"; rm -rf "$1"; continue
      fi
    elif [ "${owner%:*}" = "$host" ]; then
      if ! kill -0 "${owner##*:}" 2>/dev/null; then
        log "taking over stale lock $1 from dead pid $owner"; rm -rf "$1"; continue
      fi
    elif [ $((SECONDS - start)) -ge "$FMTUNE_LOCK_TIMEOUT" ]; then
      die "lock $1 is held by $owner (another host) after ${FMTUNE_LOCK_TIMEOUT}s; remove it if no fm-tune build is running"
    fi
    log "waiting for lock $1"; sleep 5
  done
  echo "$host:$$" > "$1/pid"
  held_lock="$1"
  trap 'rm -rf "$held_lock"' EXIT
}
unlock() { rm -rf "$held_lock"; held_lock=""; trap - EXIT; }

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
    */workdir/*) FMTUNE_CACHE_DIR="${LLMB_BASH_OUTPUT_DIR%/workdir/*}/.fm-tune-cache" ;;
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
# main.py runs after `cd "$root"` (so does the venv python, under the cache dir),
# and the marker is read off-node: make every path absolute against the start dir
# before anything uses it. No realpath/readlink -f (macOS bash 3.2). Empty stays empty.
case $FMTUNE_OUTPUT_DIR in /*) ;; *) FMTUNE_OUTPUT_DIR="$PWD/$FMTUNE_OUTPUT_DIR" ;; esac
case $FMTUNE_CACHE_DIR in /*) ;; *) FMTUNE_CACHE_DIR="$PWD/$FMTUNE_CACHE_DIR" ;; esac
case $FMTUNE_DATASET_DIR in /*) ;; *) FMTUNE_DATASET_DIR="$PWD/$FMTUNE_DATASET_DIR" ;; esac
case $FMTUNE_SOURCE_PATH in ''|/*) ;; *) FMTUNE_SOURCE_PATH="$PWD/$FMTUNE_SOURCE_PATH" ;; esac
case $FMTUNE_BASE_MODEL_DIR in ''|/*) ;; *) FMTUNE_BASE_MODEL_DIR="$PWD/$FMTUNE_BASE_MODEL_DIR" ;; esac
[ "$FMTUNE_OUTPUT_DIR" = / ] || FMTUNE_OUTPUT_DIR="${FMTUNE_OUTPUT_DIR%/}"
# The monitors read the marker path into a JSON string ('{ "path": "<path>" }'):
# a '"' or a control character fails the parse, a '\' is read as an escape, and a
# newline also cuts the marker line short. The bash monitor rstrip()s each line, so
# a trailing space would be lost too.
case $FMTUNE_OUTPUT_DIR in
  *[\"\\]*|*[[:cntrl:]]*|*' ')
    die "output dir must not contain a '\"', a '\\' or a control character, or end in a space: $FMTUNE_OUTPUT_DIR" ;;
esac
if [ -n "$FMTUNE_BASE_MODEL_DIR" ]; then
  model="$FMTUNE_BASE_MODEL_DIR"
elif [ -n "$FMTUNE_MODEL" ]; then
  model="$FMTUNE_MODEL"
else
  die "set fm_tune_config.model or bind the base_model input"
fi
# run_name names the output model dir and is echoed in logs: keep it to a safe set
# (spelled out, not a range, so no locale can widen it).
case "$FMTUNE_RUN_NAME" in
  *[!ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-]*)
    die "run_name must match ^[A-Za-z0-9._-]+$" ;;
esac
# Both go to git as positional args; a leading '-' would be read as an option.
case "$FMTUNE_SOURCE_REPO" in -*) die "source.repo must not start with '-'" ;; esac
case "$FMTUNE_SOURCE_REF" in -*) die "source.ref must not start with '-'" ;; esac
[ -n "${FMTUNE_TRAIN_FILE:-}" ] || die "train_file is required"
[ -n "${FMTUNE_VALIDATION_FILE:-}" ] || die "validation_file is required"
# Both are joined onto the dataset dir: an absolute path or a '..' component
# (matched per component, so a name like a..b.jsonl is fine) would leave it.
case "/$FMTUNE_TRAIN_FILE/" in //*|*/../*)
  die "train_file must be a relative path inside the dataset (no leading '/', no '..' component): $FMTUNE_TRAIN_FILE" ;;
esac
case "/$FMTUNE_VALIDATION_FILE/" in //*|*/../*)
  die "validation_file must be a relative path inside the dataset (no leading '/', no '..' component): $FMTUNE_VALIDATION_FILE" ;;
esac
train="$FMTUNE_DATASET_DIR/$FMTUNE_TRAIN_FILE"
val="$FMTUNE_DATASET_DIR/$FMTUNE_VALIDATION_FILE"
[ -e "$train" ] || die "train_file not found: $train"
[ -e "$val" ] || die "validation_file not found: $val"
# On success --model_at_root deletes every other entry in the output dir, which is
# only safe when it holds nothing or a previous model (a file fm-tune's save leaves
# at the root). Refuse anything else now, not after training.
if [ -L "$FMTUNE_OUTPUT_DIR" ] && [ ! -e "$FMTUNE_OUTPUT_DIR" ]; then
  die "output dir $FMTUNE_OUTPUT_DIR is a dangling symlink"
fi
if [ -e "$FMTUNE_OUTPUT_DIR" ] && [ ! -d "$FMTUNE_OUTPUT_DIR" ]; then
  die "output dir $FMTUNE_OUTPUT_DIR exists and is not a directory"
fi
entries=""
if [ -d "$FMTUNE_OUTPUT_DIR" ]; then
  entries="$(ls -A "$FMTUNE_OUTPUT_DIR")" || die "cannot list output dir $FMTUNE_OUTPUT_DIR"
fi
if [ -n "$entries" ] \
   && [ ! -f "$FMTUNE_OUTPUT_DIR/config.json" ] \
   && [ ! -f "$FMTUNE_OUTPUT_DIR/adapter_config.json" ] \
   && [ ! -f "$FMTUNE_OUTPUT_DIR/model.safetensors" ] \
   && [ ! -f "$FMTUNE_OUTPUT_DIR/adapters.safetensors" ]; then
  die "output dir $FMTUNE_OUTPUT_DIR is not empty and is not a previous model output" \
      "(no config.json, adapter_config.json, model.safetensors or adapters.safetensors); point output_path at a new or empty directory"
fi
# A bound base model has config.json at its root, so it passes the check above.
if [ -n "$FMTUNE_BASE_MODEL_DIR" ] && [ "$FMTUNE_OUTPUT_DIR" -ef "$FMTUNE_BASE_MODEL_DIR" ]; then
  die "output dir $FMTUNE_OUTPUT_DIR is the same directory as the base_model input"
fi
if [ "$FMTUNE_OUTPUT_DIR" -ef "$FMTUNE_DATASET_DIR" ]; then
  die "output dir $FMTUNE_OUTPUT_DIR is the same directory as the dataset input"
fi

# ---- source ----------------------------------------------------------------
mkdir -p "$FMTUNE_CACHE_DIR"
if [ -n "$FMTUNE_SOURCE_PATH" ]; then
  root="$FMTUNE_SOURCE_PATH"
else
  key="$(printf '%s@%s' "$FMTUNE_SOURCE_REPO" "$FMTUNE_SOURCE_REF" | cksum | cut -d' ' -f1)"
  # src-<key> (keyed on repo@ref) is only a fetch mirror; nothing runs from it.
  # main.py runs from, and the venv pip -e installs, an immutable tree-<sha> per
  # commit, so a build refreshing the mirror never changes a running build's tree.
  clone="$FMTUNE_CACHE_DIR/src-$key"
  lock "$clone.lock"
  if [ ! -d "$clone/.git" ]; then
    log "cloning $FMTUNE_SOURCE_REPO@$FMTUNE_SOURCE_REF"
    rm -rf "$clone.tmp"
    git clone --quiet --depth 1 --branch "$FMTUNE_SOURCE_REF" -- "$FMTUNE_SOURCE_REPO" "$clone.tmp"
    mv "$clone.tmp" "$clone"
    # ^{commit}: an annotated tag's FETCH_HEAD is the tag object, so peel both
    # sides or one commit gets two trees (and two venvs).
    sha="$(git -C "$clone" rev-parse 'HEAD^{commit}')"
  else
    # The cache outlives builds: resolve the ref's current commit, not the first one.
    log "updating $FMTUNE_SOURCE_REPO@$FMTUNE_SOURCE_REF"
    git -C "$clone" fetch --quiet --depth 1 -- origin "$FMTUNE_SOURCE_REF" \
      || die "could not update the cached clone $clone to $FMTUNE_SOURCE_REF"
    sha="$(git -C "$clone" rev-parse 'FETCH_HEAD^{commit}')"
  fi
  tree="$FMTUNE_CACHE_DIR/tree-$sha"
  if [ ! -d "$tree" ]; then
    # git archive, not git worktree add: a plain directory with no .git link back
    # to the mirror, so pruning or re-cloning the mirror can never affect it.
    rm -rf "$tree.tmp"; mkdir -p "$tree.tmp"
    git -C "$clone" archive "$sha" | tar -x -C "$tree.tmp" \
      || die "could not extract $sha from $clone"
    mv "$tree.tmp" "$tree"
  fi
  unlock
  log "using $FMTUNE_SOURCE_REPO@$FMTUNE_SOURCE_REF = $sha"
  root="$tree/$FMTUNE_SOURCE_SUBDIR"
fi
[ -f "$root/main.py" ] || die "no main.py under fm-tune source $root"

preset="$root/autotune/configs/$FMTUNE_CONFIG_PRESET"
if [ "$FMTUNE_HPO_CONFIG_JSON" = "{}" ] || [ -z "$FMTUNE_HPO_CONFIG_JSON" ]; then
  [ -f "$preset" ] || die "config_preset not found: $preset"
  config_file="$preset"
else
  mkdir -p "$FMTUNE_OUTPUT_DIR"
  config_file="$(cd "$FMTUNE_OUTPUT_DIR/.." && pwd)/fm_tune_hpo.yaml"
  write_hpo_yaml=1  # converted after $py is known (below)
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
  lock "$venv.lock"
  if [ ! -f "$venv/.installed" ]; then
    log "building venv $venv (extra: $FMTUNE_EXTRA)"
    rm -rf "$venv"
    "$FMTUNE_PYTHON" -m venv "$venv"
    "$venv/bin/python" -m pip install --quiet --upgrade pip
    "$venv/bin/python" -m pip install --quiet -e "$root[$FMTUNE_EXTRA]"
    touch "$venv/.installed"
  fi
  unlock
  py="$venv/bin/python"
fi

# fm-tune reads --config_file with yaml.safe_load (YAML 1.1: a float needs a ".",
# so JSON's 5e-06 would load as a string). Re-emit the JSON as YAML; safe_dump
# writes 5.0e-06, which round-trips as a float.
if [ "${write_hpo_yaml:-}" = "1" ]; then
  printf '%s' "$FMTUNE_HPO_CONFIG_JSON" | "$py" -c '
import json, sys, yaml
sys.stdout.write(yaml.safe_dump(json.load(sys.stdin), sort_keys=False))
' > "$config_file" || die "hpo_config is not valid JSON (or could not be converted to YAML)"
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
  # read stops at the first newline, so join lines first. The ${x+...} form keeps
  # an empty array (whitespace-only input) from tripping set -u on bash 3.2.
  read -r -a extra <<< "${FMTUNE_EXTRA_ARGS//$'\n'/ }"
  args+=(${extra[@]+"${extra[@]}"})
fi

mkdir -p "$FMTUNE_OUTPUT_DIR"
# fm-tune's AutoTuneX bridge logs under $LOG_PATH (default ./logs, the source tree).
export LOG_PATH="${LOG_PATH:-$(cd "$FMTUNE_OUTPUT_DIR/.." && pwd)}"
# The venv's console scripts (ray, ...) must win over any others on PATH.
case "$py" in */*) export PATH="$(dirname "$py"):${PATH:-/usr/local/bin:/usr/bin:/bin}" ;; esac
log "running main.py (run_name=$FMTUNE_RUN_NAME)"
cd "$root"
status=0
"$py" main.py "${args[@]}" || status=$?
if [ "$status" -ne 0 ]; then
  echo "[fm-tune] main.py failed with exit code $status" >&2
  exit "$status"
fi
echo "GB_ARTIFACT_ID:model GB_ARTIFACT_PATH:$FMTUNE_OUTPUT_DIR"
