#!/usr/bin/env bash
# Remove old granite.build job-log directories (llm-build-<build_id>) from a
# directory on a SkyPilot environment's shared filesystem (on BlueVela,
# /proj/data-eng/llmb-read-write/builds/).
#
# gbserver keeps each build's job logs in <shared_workdir>/llm-build-<build_id>/,
# and teardown leaves them there on purpose, so they pile up. The script looks
# only at the directories directly inside DIR, exactly as given, and removes one
# only when all of these hold:
#   * its name starts with llm-build-
#   * it is owned by the current user, so other users' entries are never touched
#   * it was last modified more than DAYS days ago
# A directory whose name doesn't start with llm-build- is reported as ignored.
# An llm-build- directory kept because of its owner or age is only counted.
# Nothing is deleted unless --delete is given; the default is a dry run.
#
# Usage:
#   scripts/cleanup-build-dir.sh DAYS DIR            # dry run
#   scripts/cleanup-build-dir.sh --delete DAYS DIR   # really delete
#
#   DAYS  minimum age in days (an integer, at least 1)
#   DIR   the directory to clean (required), e.g.
#         /proj/data-eng/llmb-read-write/builds
#
# Run it on a login node as the account that ran the builds (e.g. granitebuild).
# Pick DAYS longer than your longest build: a directory's modification time only
# changes when an entry directly inside it is added or removed, so a long build
# writing only deeper down can look old while it is still running.
set -euo pipefail

PREFIX="llm-build-"
DELETE=false
ME=$(id -un)

# usage
# Print the header comment as help text.
usage() {
  sed -n '2,28p' "$0"
}

# die MESSAGE
# Print MESSAGE to stderr and exit with status 2.
die() {
  echo "cleanup-build-dir: $*" >&2
  exit 2
}

# older_than PATH DAYS
# Succeed when PATH was last modified more than DAYS days ago.
older_than() {
  [[ -n "$(find "$1" -maxdepth 0 -mtime "+$(($2 - 1))" -print)" ]]
}

# owned_by_me PATH
# Succeed when PATH is owned by the user running the script.
owned_by_me() {
  [[ -n "$(find "$1" -maxdepth 0 -user "$ME" -print)" ]]
}

# is_candidate PATH DAYS
# Succeed when PATH should be removed: owned by the current user and older than
# DAYS days. The caller has already checked the name prefix.
is_candidate() {
  owned_by_me "$1" && older_than "$1" "$2"
}

# remove_one PATH
# Remove PATH, or just list it in a dry run. Returns 1 if the removal failed.
remove_one() {
  local path=$1
  if ! $DELETE; then
    echo "would remove: $path"
  elif rm -rf -- "$path"; then
    echo "removed: $path"
  else
    echo "FAILED: $path" >&2
    return 1
  fi
}

# clean DIR DAYS
# Remove each candidate directly inside DIR, report each directory without the
# prefix, and count the llm-build- ones kept for owner or age; then print a
# summary. Returns 1 if any removal failed.
clean() {
  local dir=$1 days=$2 path removed=0 ignored=0 kept=0 failed=0
  while IFS= read -r path; do
    if [[ "${path##*/}" != "$PREFIX"* ]]; then
      echo "ignored: $path (name does not start with $PREFIX)"
      ignored=$((ignored + 1))
    elif ! is_candidate "$path" "$days"; then
      kept=$((kept + 1))
    elif remove_one "$path"; then
      removed=$((removed + 1))
    else
      failed=$((failed + 1))
    fi
  done < <(find "$dir" -mindepth 1 -maxdepth 1 -type d -print | sort)
  local tail="ignored $ignored, kept $kept (other owner or newer than $days days)"
  if $DELETE; then
    echo "removed $removed, $tail, failed $failed"
  else
    echo "would remove $removed, $tail (dry run; pass --delete to remove)"
  fi
  [[ $failed -eq 0 ]]
}

# parse_args ARGS...
# Set DELETE, DAYS and DIR from the command line.
parse_args() {
  local positional=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --delete)  DELETE=true ;;
      -h|--help) usage; exit 0 ;;
      -*)        die "unknown option: $1" ;;
      *)         positional+=("$1") ;;
    esac
    shift
  done
  [[ ${#positional[@]} -eq 2 ]] || { usage >&2; exit 2; }
  DAYS=${positional[0]}
  DIR=${positional[1]}
  [[ "$DAYS" =~ ^[0-9]+$ && "$DAYS" -ge 1 ]] || die "DAYS must be an integer >= 1, got: $DAYS"
  [[ -d "$DIR" ]] || die "not a directory: $DIR"
  DIR=$(cd "$DIR" && pwd -P)
}

main() {
  parse_args "$@"
  echo "searching: $DIR/$PREFIX*  older than: $DAYS days  user: $ME  delete: $DELETE"
  clean "$DIR" "$DAYS"
}

main "$@"
