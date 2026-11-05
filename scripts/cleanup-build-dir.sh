#!/usr/bin/env bash
usage() {
  cat <<'EOF'
Remove old granite.build build directories, whose names contain the build id
(a UUID), from a directory on a SkyPilot environment's shared filesystem. On
BlueVela that covers both the job-log directories in
/proj/data-eng/llmb-read-write/builds/ (llm-build-<uuid>) and the per-run
workdirs in /proj/data-eng/llmb-read-write/builds/builds/ (<uuid>).

Both pile up. Job logs are kept on purpose. Teardown removes only each target
run's builds/<uuid>/runs/<run_id>/, so every build leaves at least an empty
builds/<uuid>/runs/ behind (only the EFS shared-filesystem provider also
removes those parents). A build that is cancelled or killed before teardown,
or whose teardown fails (that failure is logged and ignored), leaves its whole
per-run workdir. The script looks only at the directories directly inside
DIR, exactly as given, and removes one only when all of these hold:
  * its name contains a UUID anywhere, with any prefix or suffix (e.g.
    87caf891-2393-4a16-895a-f50bf316fd65 or llm-build-87caf891-...)
  * it is owned by the current user, so other users' entries are never touched
  * nothing anywhere in its tree was modified in the last DAYS days
A directory whose name contains no UUID is reported as ignored. A matching
directory kept because of its owner or age is only counted.
Nothing is deleted unless --delete is given; the default is a dry run.

Usage:
  scripts/cleanup-build-dir.sh DAYS DIR            # dry run
  scripts/cleanup-build-dir.sh --delete DAYS DIR   # really delete

  DAYS  minimum age in days (an integer, at least 1)
  DIR   the directory to clean (required), e.g.
        /proj/data-eng/llmb-read-write/builds or .../builds/builds

Run it on a login node as the account that ran the builds (e.g. granitebuild).
Age is checked over the whole tree, not just the top directory, so a build
that is still running, or was restarted under the same build id with
`gb build restart` (which adds new runs inside an old directory), is kept.
A tree that can't be read in full is kept too.
EOF
}

set -euo pipefail

# A build id: a UUID in its usual 8-4-4-4-12 hex form, anywhere in the name.
# Not anchored, so a prefix or suffix (llm-build-<uuid>, <uuid>-old) still matches.
UUID_RE='[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'
DELETE=false
ME=$(id -un)

# die MESSAGE
# Print MESSAGE to stderr and exit with status 2.
die() {
  echo "cleanup-build-dir: $*" >&2
  exit 2
}

# older_than PATH DAYS
# Succeed when nothing in PATH's tree was modified in the last DAYS days.
# Fails (keeps the directory) if find can't read part of the tree.
older_than() {
  local newer
  newer=$(find "$1" -mtime "-$2" -print -quit 2>/dev/null) || return 1
  [[ -z "$newer" ]]
}

# owned_by_me PATH
# Succeed when PATH is owned by the user running the script.
owned_by_me() {
  [[ -n "$(find "$1" -maxdepth 0 -user "$ME" -print)" ]]
}

# is_candidate PATH DAYS
# Succeed when PATH should be removed: owned by the current user and older than
# DAYS days. The caller has already checked that the name contains a UUID.
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
# Remove each candidate directly inside DIR, report each directory whose name
# contains no UUID, and count the matching ones kept for owner or age; then
# print a summary. Returns 1 if any removal failed.
clean() {
  local dir=$1 days=$2 path removed=0 ignored=0 kept=0 failed=0
  while IFS= read -r path; do
    if [[ ! "${path##*/}" =~ $UUID_RE ]]; then
      echo "ignored: $path (name contains no build UUID)"
      ignored=$((ignored + 1))
    elif ! is_candidate "$path" "$days"; then
      kept=$((kept + 1))
    elif remove_one "$path"; then
      removed=$((removed + 1))
    else
      failed=$((failed + 1))
    fi
  done < <(find "$dir" -mindepth 1 -maxdepth 1 -type d -print | sort)
  local tail="ignored $ignored, kept $kept (other owner, or something in it newer than $days days)"
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
  echo "searching: $DIR/*<uuid>*  older than: $DAYS days  user: $ME  delete: $DELETE"
  clean "$DIR" "$DAYS"
}

main "$@"
