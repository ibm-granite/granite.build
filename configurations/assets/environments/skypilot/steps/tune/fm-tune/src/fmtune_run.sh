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
# One cache lock at a time: a mkdir lock dir whose pid file names the owner as
# <host>:<pid>. Builds share the cache, so the clone and tree extraction and the
# venv build each run under one; it is released before training. A lock is never
# taken over automatically: every takeover scheme left a window for two holders,
# and a stale lock needs a hard kill (-9, OOM, a lost node) within those minutes.
# A waiter polls every 5 s. If the owner ran on this host and is gone, it stops at
# once; otherwise (alive, another host, no owner recorded) it gives up after
# FMTUNE_LOCK_TIMEOUT seconds. Either way it names the lock: rm -rf it to clear it.
: "${FMTUNE_LOCK_TIMEOUT:=1800}"
# At most 9 digits: $(( )) would wrap a longer one.
case "$FMTUNE_LOCK_TIMEOUT" in ''|*[!0-9]*|??????????*)
  die "FMTUNE_LOCK_TIMEOUT must be a whole number of seconds (at most 9 digits)" ;; esac
FMTUNE_LOCK_TIMEOUT=$((10#$FMTUNE_LOCK_TIMEOUT))  # a leading 0 would read as octal in $(( ))
host="$(uname -n)"
held_lock=""
hpo_dir=""
# The one EXIT trap: release a held lock, remove the run's private hpo_config dir.
# unlock clears held_lock itself, so the trap never unlocks twice.
on_exit() { [ -z "$held_lock" ] || unlock; [ -z "$hpo_dir" ] || rm -rf "$hpo_dir"; }
trap on_exit EXIT
sq() { local q="'"; printf "'%s'" "${1//$q/$q\\$q$q}"; }  # shell-quote for the rm hint
gone() {  # kill -0 fails for a dead pid and for another user's live one: only ESRCH's text differs
  case "$1" in ''|*[!0-9]*) return 1 ;; esac
  case "$(LC_ALL=C; kill -0 "$1" 2>&1)" in *"No such process"*) return 0 ;; esac
  return 1
}
lock() {
  local owner start=$SECONDS
  until mkdir "$1" 2>/dev/null; do
    if [ ! -d "$1" ]; then
      mkdir "$1" 2>/dev/null && break  # released between the mkdir and this look
      [ -d "$1" ] || die "cannot create lock $1 (is $(dirname "$1") writable?)"
    fi
    owner="$(cat "$1/pid" 2>/dev/null || true)"
    # Re-read: an owner that released the lock and then exited is not stale.
    if [ "${owner%:*}" = "$host" ] && gone "${owner##*:}" \
       && [ "$(cat "$1/pid" 2>/dev/null)" = "$owner" ]; then
      die "lock $1 is held by $owner, which is no longer running; remove it with: rm -rf $(sq "$1") if no fm-tune build is running"
    fi
    if [ $((SECONDS - start)) -ge "$FMTUNE_LOCK_TIMEOUT" ]; then
      die "lock $1 is still held after ${FMTUNE_LOCK_TIMEOUT}s (owner: ${owner:-none recorded}); the owner may still be working (a venv build can take longer): raise FMTUNE_LOCK_TIMEOUT and re-run. Only if no fm-tune build is running, remove it with: rm -rf $(sq "$1")"
    fi
    log "waiting for lock $1"; sleep 5
  done
  # Only the holder writes pid, so a rename within the dir is atomic enough: no
  # waiter ever reads it half-written.
  { echo "$host:$$" > "$1/pid.new" && mv -f "$1/pid.new" "$1/pid"; } 2>/dev/null \
    || { rm -rf "$1" 2>/dev/null || true; die "cannot write the owner of lock $1"; }  # ours: else every run waits on it
  held_lock="$1"
}
unlock() {  # forget the lock first: a failed or interrupted rm must not unlock twice
  local l="$held_lock" g
  held_lock=""
  [ "$(cat "$l/pid" 2>/dev/null)" = "$host:$$" ] || return 0
  # Rename first so no new mkdir lands in a dir being removed; if a dir can't be renamed, remove it in place.
  g="$l.gone.$host.$$"; mv "$l" "$g" 2>/dev/null || g="$l"
  rm -rf "$g" 2>/dev/null || true
}

: "${FMTUNE_PYTHON:=}"
# Bash launcher env values reach the child unexpanded (bash.py _launch_env_layers
# passes them to create_subprocess_exec), so the bash template leaves these three
# empty and they default here from the LLMB_BASH_* builtins. SkyPilot sets them.
if [ -z "${FMTUNE_OUTPUT_DIR:-}" ] && [ -n "${LLMB_BASH_OUTPUT_DIR:-}" ]; then
  FMTUNE_OUTPUT_DIR="$LLMB_BASH_OUTPUT_DIR/model"   # a subdir: it must be empty to publish into, and job.log lives in LLMB_BASH_OUTPUT_DIR
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
: "${FMTUNE_CONFIG_NAME:=fm_tune_hpo}"
: "${FMTUNE_HPO_CONFIG_JSON:=}"
: "${FMTUNE_EXTRA_ARGS:=}"
: "${FMTUNE_SERVER_URL:=}"
: "${FMTUNE_JOB_ID:=}"
# With the bridge on and no job_id, register the run under its build id (gbserver sets
# GB_BUILD_ID in every env); the bridge's template path expects build_id == job_id.
[ -n "$FMTUNE_SERVER_URL" ] && [ -z "$FMTUNE_JOB_ID" ] && FMTUNE_JOB_ID="${GB_BUILD_ID:-}"
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
# Every trailing '/', but not a lone one: [ -L /tmp/ ] looks through the link, and the
# output dir names the registered model and its <dir>.logs sibling.
while [ "$FMTUNE_CACHE_DIR" != / ] && [ "${FMTUNE_CACHE_DIR%/}" != "$FMTUNE_CACHE_DIR" ]; do
  FMTUNE_CACHE_DIR="${FMTUNE_CACHE_DIR%/}"
done
# Staging is <output dir>.partial.*, which for a path ending in '/.' would land
# inside the output dir: drop every '.' and empty component (trailing '/'s too).
# A '..' is refused, not resolved: through a symlinked parent it names another dir.
case "/$FMTUNE_OUTPUT_DIR/" in */../*)
  die "output dir must not contain a '..' component: $FMTUNE_OUTPUT_DIR" ;; esac
rest="${FMTUNE_OUTPUT_DIR#/}/"; norm=""
while [ -n "$rest" ]; do
  part="${rest%%/*}"; rest="${rest#*/}"
  case $part in ''|.) ;; *) norm="$norm/$part" ;; esac
done
FMTUNE_OUTPUT_DIR="${norm:-/}"
case $FMTUNE_DATASET_DIR in /*) ;; *) FMTUNE_DATASET_DIR="$PWD/$FMTUNE_DATASET_DIR" ;; esac
case $FMTUNE_SOURCE_PATH in ''|/*) ;; *) FMTUNE_SOURCE_PATH="$PWD/$FMTUNE_SOURCE_PATH" ;; esac
case $FMTUNE_BASE_MODEL_DIR in ''|/*) ;; *) FMTUNE_BASE_MODEL_DIR="$PWD/$FMTUNE_BASE_MODEL_DIR" ;; esac
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
# config_name is a filename (<name>.yaml) and the config's name in AutoTuneX.
case "$FMTUNE_CONFIG_NAME" in
  *[!ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-]*)
    die "config_name must match ^[A-Za-z0-9._-]+$" ;;
esac
# Only these two: --no_autotune is passed for false alone, so a typo would run a sweep.
case "$FMTUNE_AUTOTUNE" in true|false) ;; *)
  die "autotune must be true or false, got '$FMTUNE_AUTOTUNE'" ;; esac
# Both go to git as positional args; a leading '-' would be read as an option.
case "$FMTUNE_SOURCE_REPO" in -*) die "source.repo must not start with '-'" ;; esac
case "$FMTUNE_SOURCE_REF" in -*) die "source.ref must not start with '-'" ;; esac
# Messages name the repo without its URL userinfo: a token in
# https://user:<token>@host/... would otherwise land in job.log.
repo_shown="$(printf '%s' "$FMTUNE_SOURCE_REPO" | sed -E 's#^([A-Za-z][A-Za-z0-9+.-]*://)[^/@]*@#\1***@#')"
# Exactly 40 hex characters is a commit SHA, lowercased before any use (spelled
# out, not a range: in a UTF-8 locale bash 3.2's [a-f] also matches 'B').
ref_sha=""
lc_ref="$(printf '%s' "$FMTUNE_SOURCE_REF" | tr ABCDEF abcdef)"
if [ "${#lc_ref}" -eq 40 ]; then
  case "$lc_ref" in *[!0123456789abcdef]*) ;; *) FMTUNE_SOURCE_REF="$lc_ref"; ref_sha=1 ;; esac
fi
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
# fm-tune's AutotunePipeline raises when both are none (case-insensitively), but only
# once training starts; an empty one means fm-tune's default, which is none too.
case "$(printf '%s:%s' "$FMTUNE_TUNING_ALGO" "$FMTUNE_RL_ALGO" | tr '[:upper:]' '[:lower:]')" in
  :|none:|:none|none:none)
    die "tuning_algo and rl_algo cannot both be empty or none; set tuning_algo (e.g. lora) or an online rl_algo (ppo, grpo, dapo)" ;;
  :dpo|:kto|none:dpo|none:kto)  # AUTOTUNE_OFFLINE_RL
    die "an offline rl_algo (dpo, kto) needs a tuning_algo (e.g. lora or sft)" ;;
esac
train="$FMTUNE_DATASET_DIR/$FMTUNE_TRAIN_FILE"
val="$FMTUNE_DATASET_DIR/$FMTUNE_VALIDATION_FILE"
[ -e "$train" ] || die "train_file not found: $train"
[ -e "$val" ] || die "validation_file not found: $val"
# main.py trains in a private staging dir beside the output dir (--model_at_root
# clears only that), and the model is renamed onto the output dir at the end, which
# succeeds only while it is missing or empty: nothing in it is ever deleted. These
# checks fail early, before any clone or install; the rename is the guarantee.
if [ -L "$FMTUNE_OUTPUT_DIR" ] && [ ! -e "$FMTUNE_OUTPUT_DIR" ]; then
  die "output dir $FMTUNE_OUTPUT_DIR is a dangling symlink"
fi
if [ -e "$FMTUNE_OUTPUT_DIR" ] && [ ! -d "$FMTUNE_OUTPUT_DIR" ]; then
  die "output dir $FMTUNE_OUTPUT_DIR exists and is not a directory"
fi
if [ -n "$FMTUNE_BASE_MODEL_DIR" ] && [ "$FMTUNE_OUTPUT_DIR" -ef "$FMTUNE_BASE_MODEL_DIR" ]; then
  die "output dir $FMTUNE_OUTPUT_DIR is the same directory as the base_model input"
fi
if [ "$FMTUNE_OUTPUT_DIR" -ef "$FMTUNE_DATASET_DIR" ]; then
  die "output dir $FMTUNE_OUTPUT_DIR is the same directory as the dataset input"
fi
entries=""
if [ -d "$FMTUNE_OUTPUT_DIR" ]; then
  entries="$(ls -A "$FMTUNE_OUTPUT_DIR")" || die "cannot list output dir $FMTUNE_OUTPUT_DIR"
fi
if [ -n "$entries" ]; then
  die "output dir $FMTUNE_OUTPUT_DIR is not empty; point output_path at a new or empty directory"
fi

# ---- source ----------------------------------------------------------------
# Builds run code from the cache (trees, venvs), so it must be the user's own and
# not writable by anyone else. A new one (and any missing parent) is made 0700; an
# existing one is checked, never chmod'ed. A symlink of the user's own (say, to a
# bigger disk) is followed and its target checked; one owned by someone else is
# refused, since in a shared sticky dir its owner can re-point it.
(umask 077 && mkdir -p "$FMTUNE_CACHE_DIR") || die "cannot create cache dir $FMTUNE_CACHE_DIR"
if [ -L "$FMTUNE_CACHE_DIR" ] && [ -z "$(find "$FMTUNE_CACHE_DIR" -maxdepth 0 -user "$(id -u)")" ]; then
  die "cache dir $FMTUNE_CACHE_DIR is a symlink owned by $(ls -ld "$FMTUNE_CACHE_DIR" | awk '{print $3}'), not $(id -un 2>/dev/null || id -u); set FMTUNE_CACHE_DIR to a dir of your own"
fi
if [ ! -O "$FMTUNE_CACHE_DIR" ]; then
  die "cache dir $FMTUNE_CACHE_DIR is owned by $(ls -ldL "$FMTUNE_CACHE_DIR" | awk '{print $3}'), not $(id -un 2>/dev/null || id -u); set FMTUNE_CACHE_DIR to a dir of your own"
fi
if [ -n "$(find -H "$FMTUNE_CACHE_DIR" -maxdepth 0 \( -perm -020 -o -perm -002 \))" ]; then
  die "cache dir $FMTUNE_CACHE_DIR is writable by other users ($(ls -ldL "$FMTUNE_CACHE_DIR" | awk '{print $1}')); run chmod go-w $FMTUNE_CACHE_DIR"
fi
if [ -n "$FMTUNE_SOURCE_PATH" ]; then
  root="$FMTUNE_SOURCE_PATH"
elif [ -n "$ref_sha" ] && [ -d "$FMTUNE_CACHE_DIR/tree-$FMTUNE_SOURCE_REF" ]; then
  # A commit never moves and its tree is moved into place whole: no lock, no git.
  log "using $repo_shown@$FMTUNE_SOURCE_REF = $FMTUNE_SOURCE_REF"
  root="$FMTUNE_CACHE_DIR/tree-$FMTUNE_SOURCE_REF/$FMTUNE_SOURCE_SUBDIR"
else
  key="$(printf '%s@%s' "$FMTUNE_SOURCE_REPO" "$FMTUNE_SOURCE_REF" | cksum | cut -d' ' -f1)"
  # src-<key> (keyed on repo@ref) is only a fetch mirror; nothing runs from it.
  # main.py runs from, and the venv pip -e installs, an immutable tree-<sha> per
  # commit, so a build refreshing the mirror never changes a running build's tree.
  clone="$FMTUNE_CACHE_DIR/src-$key"
  lock "$clone.lock"
  if [ ! -d "$clone/.git" ] && [ -z "$ref_sha" ]; then
    log "cloning $repo_shown@$FMTUNE_SOURCE_REF"
    rm -rf "$clone.tmp"
    git clone --quiet --depth 1 --branch "$FMTUNE_SOURCE_REF" -- "$FMTUNE_SOURCE_REPO" "$clone.tmp" \
      || die "could not clone $repo_shown at $FMTUNE_SOURCE_REF (source.ref must be a branch, a tag, or a full 40-character commit SHA)"
    mv "$clone.tmp" "$clone"
    # ^{commit}: an annotated tag's FETCH_HEAD is the tag object, so peel both
    # sides or one commit gets two trees (and two venvs).
    sha="$(git -C "$clone" rev-parse 'HEAD^{commit}')"
  elif [ ! -d "$clone/.git" ]; then
    # A SHA: fetch that one commit by id, not a branch.
    log "fetching $repo_shown@$FMTUNE_SOURCE_REF"
    rm -rf "$clone.tmp"
    git init -q "$clone.tmp"
    git -C "$clone.tmp" remote add -- origin "$FMTUNE_SOURCE_REPO"
    git -C "$clone.tmp" fetch --quiet --depth 1 -- origin "$FMTUNE_SOURCE_REF" \
      || die "could not fetch commit $FMTUNE_SOURCE_REF from $repo_shown (a commit SHA must be the full 40 characters of a commit in that repo)"
    # A tag object's id peels to another commit; a tree's does not peel at all.
    sha="$(git -C "$clone.tmp" rev-parse --verify --quiet 'FETCH_HEAD^{commit}')" && [ "$sha" = "$FMTUNE_SOURCE_REF" ] \
      || die "source.ref $FMTUNE_SOURCE_REF is not a commit in $repo_shown (it names a tag object or a tree)"
    mv "$clone.tmp" "$clone"
  else
    # The cache outlives builds: resolve the ref's current commit, not the first one.
    log "updating $repo_shown@$FMTUNE_SOURCE_REF"
    git -C "$clone" fetch --quiet --depth 1 -- origin "$FMTUNE_SOURCE_REF" \
      || die "could not update the cached clone $clone to $FMTUNE_SOURCE_REF"
    sha="$(git -C "$clone" rev-parse 'FETCH_HEAD^{commit}')"
    # A SHA's mirror outlives its tree-<sha> (deleted to free space): same check as above.
    if [ -n "$ref_sha" ] && [ "$sha" != "$FMTUNE_SOURCE_REF" ]; then
      die "source.ref $FMTUNE_SOURCE_REF is not a commit in $repo_shown (it names a tag object or a tree)"
    fi
  fi
  tree="$FMTUNE_CACHE_DIR/tree-$sha"
  if [ ! -d "$tree" ]; then
    # git archive, not git worktree add: a plain directory with no .git link back
    # to the mirror, so pruning or re-cloning the mirror can never affect it.
    # A per-run temp dir: a branch and its SHA hold different locks but share tree-<sha>.
    t="$tree.tmp.$host.$$"; rm -rf "$t"; mkdir -p "$t"
    git -C "$clone" archive "$sha" | tar -x -C "$t" \
      || die "could not extract $sha from $clone"
    if [ -d "$tree" ]; then rm -rf "$t"; else mv "$t" "$tree" 2>/dev/null || rm -rf "$t"; fi
  fi
  unlock
  log "using $repo_shown@$FMTUNE_SOURCE_REF = $sha"
  root="$tree/$FMTUNE_SOURCE_SUBDIR"
fi
[ -f "$root/main.py" ] || die "no main.py under fm-tune source $root"

preset="$root/autotune/configs/$FMTUNE_CONFIG_PRESET"
if [ "$FMTUNE_HPO_CONFIG_JSON" = "{}" ] || [ -z "$FMTUNE_HPO_CONFIG_JSON" ]; then
  [ -f "$preset" ] || die "config_preset not found: $preset"
  config_file="$preset"
else
  # Private to this run: builds whose output dirs share a parent would overwrite
  # each other's file. Named <config_name>.yaml, its name in AutoTuneX. The EXIT
  # trap removes it on every exit, a failed venv build or conversion included.
  # An explicit template: macOS mktemp -d ignores $TMPDIR, GNU's honours it.
  hpo_dir="$(mktemp -d "${TMPDIR:-/tmp}/fm-tune-hpo.XXXXXX")" \
    || die "cannot create a temp dir for hpo_config"
  config_file="$hpo_dir/$FMTUNE_CONFIG_NAME.yaml"
  write_hpo_yaml=1  # converted after $py is known (below)
fi

# ---- venv ------------------------------------------------------------------
if [ "${FMTUNE_SKIP_INSTALL:-}" = "1" ]; then
  py="$FMTUNE_PYTHON"
else
  pyver="$("$FMTUNE_PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  "$FMTUNE_PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
    || die "fm-tune needs Python >= 3.10, $FMTUNE_PYTHON is $pyver"
  # The interpreter's real path, arch and OS: a venv links to its base python, so
  # one built from another python (an upgrade, another node type) must not be reused.
  pyid="$("$FMTUNE_PYTHON" -c 'import os, platform, sys; print(os.path.realpath(sys.executable), platform.machine(), sys.platform)')"
  vkey="$( { printf '%s\n%s\n%s\n%s\n' "$root" "$FMTUNE_EXTRA" "$pyver" "$pyid"; cat "$root/pyproject.toml"; } | cksum | cut -d' ' -f1)"
  venv="$FMTUNE_CACHE_DIR/venv-$vkey"
  lock "$venv.lock"
  # A cached venv whose python no longer runs (its base interpreter changed in
  # place) is rebuilt: nothing can be using it.
  if [ ! -f "$venv/.installed" ] || ! "$venv/bin/python" -c '' 2>/dev/null; then
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
# The rename target: a symlinked output dir is published through its real dir (a
# rename onto the link itself fails), and staging sits beside it, on the same
# filesystem. mktemp makes it unique to this run and 0700; open it up to the umask.
target="$FMTUNE_OUTPUT_DIR"
[ -L "$target" ] && target="$(cd "$target" && pwd -P)"
mkdir -p "$(dirname "$target")" || die "cannot create the parent of output dir $target"
stage="$(mktemp -d "$target.partial.XXXXXX")" && chmod "$(umask -S)" "$stage" \
  || die "cannot create a staging dir beside output dir $target"
args=(
  --config_file "$config_file"
  --train_file "$train"
  --validation_file "$val"
  --model_name_or_path "$model"
  --run_name "$FMTUNE_RUN_NAME"
  --output_dir "$stage"
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

# fm-tune's AutoTuneX bridge logs under $LOG_PATH/logs (default ./logs, the source
# tree). Not inside staging, which --model_at_root clears on success, and not the
# output dir's parent, which builds with sibling output dirs share: <output dir>.logs.
export LOG_PATH="${LOG_PATH:-$FMTUNE_OUTPUT_DIR.logs}"
# The venv's console scripts (ray, ...) must win over any others on PATH.
case "$py" in */*) export PATH="$(dirname "$py"):${PATH:-/usr/local/bin:/usr/bin:/bin}" ;; esac
log "running main.py (run_name=$FMTUNE_RUN_NAME)"
cd "$root"
status=0
"$py" main.py "${args[@]}" || status=$?
if [ "$status" -ne 0 ]; then
  echo "[fm-tune] main.py failed with exit code $status; its files are kept in $stage" >&2
  exit "$status"
fi
# rename(2), not mv (which would move staging into an existing dir): atomic, and it
# fails on a non-empty target, such as another build's model published meanwhile.
"$py" -c 'import os, sys; os.rename(sys.argv[1], sys.argv[2])' "$stage" "$target" 2>/dev/null \
  || die "cannot move the model into output dir $target (another build may have written it meanwhile); the model is kept in $stage"
echo "GB_ARTIFACT_ID:model GB_ARTIFACT_PATH:$FMTUNE_OUTPUT_DIR"
