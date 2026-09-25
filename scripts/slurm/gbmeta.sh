#!/usr/bin/env bash
#
# gbmeta - pretty-print granite.build tracking metadata from a SLURM job.
#
# granite.build packs build-tracking fields into a job's SLURM --comment as a
# single whitespace-free "key=value;key=value" token (see
# src/gbserver/environment/_skypilot_metadata.py). That token is compact by
# necessity: SkyPilot emits "#SBATCH --comment=<value>" unquoted, so the stored
# value can contain no spaces. This helper expands it into an aligned,
# human-readable table for display only.
#
# Usage:
#   gbmeta.sh <jobid>            # look the comment up via scontrol, fall back to sacct
#   scontrol show job <id> -o | gbmeta.sh   # or pipe any text containing the comment
#   sacct -j <id> -o Comment%-200 -Pn | gbmeta.sh
#
# Exit status:
#   0  metadata found and printed
#   1  no granite.build metadata found
#   2  usage error

set -euo pipefail

usage() {
  echo "usage: gbmeta.sh <jobid>   |   <command producing a Comment> | gbmeta.sh" >&2
  exit 2
}

# fetch_comment JOBID
# Echo the raw --comment value for a SLURM job, trying the live controller
# (scontrol) first and the accounting DB (sacct, for finished jobs) second.
fetch_comment() {
  local jobid="$1" comment=""
  if command -v scontrol >/dev/null 2>&1; then
    # scontrol -o prints one line of space-separated Key=Value pairs; our
    # comment value has no spaces, so it ends at the next whitespace. Anchor on
    # a field boundary (start-of-line or a space) so the "Comment=" substring
    # inside AdminComment=/SystemComment= is not matched instead.
    comment="$(scontrol show job "$jobid" -o 2>/dev/null \
      | grep -oE '(^| )Comment=[^[:space:]]+' | head -n1 \
      | sed 's/^ //' | cut -d= -f2- || true)"
  fi
  if [ -z "$comment" ] && command -v sacct >/dev/null 2>&1; then
    comment="$(sacct -j "$jobid" -o Comment%-200 -Pn 2>/dev/null \
      | grep -m1 '.' || true)"
  fi
  printf '%s' "$comment"
}

# render COMMENT
# Print each ";"-separated key=value pair on its own aligned line. Splitting
# each pair at the FIRST "=" keeps values that themselves contain "=" or "://"
# intact (e.g. step_uri=space://steps/foo).
render() {
  local comment="$1"
  # If a wider line was piped (e.g. a scontrol -o row carrying AdminComment=,
  # Comment= and SystemComment= together), isolate just the real Comment value.
  # Anchor on a field boundary (start-of-line or a space) so the "Comment="
  # substring inside AdminComment=/SystemComment= is not matched instead.
  if printf '%s' "$comment" | grep -qE '(^| )Comment='; then
    comment="$(printf '%s' "$comment" \
      | grep -oE '(^| )Comment=[^[:space:]]+' | head -n1 \
      | sed 's/^ //' | cut -d= -f2-)"
  fi
  # Emit one aligned line per key=value pair; count them so an empty/"(null)"
  # comment reports "not found" rather than printing nothing and succeeding.
  printf '%s' "$comment" | tr ';' '\n' | awk -F= '
    NF >= 2 && $1 ~ /^[a-z_]+$/ {
      key = $1; sub(/^[^=]*=/, "", $0); printf "  %-11s %s\n", key ":", $0; n++
    }
    END { exit (n ? 0 : 1) }
  '
}

main() {
  local comment
  # Prefer an explicit job id argument over stdin: a redirected fd 0 (cron, CI,
  # a wrapping script) is not a tty, so keying on the arg first keeps
  # `gbmeta.sh <jobid>` working regardless of what stdin happens to be.
  if [ "$#" -ge 1 ]; then
    [ "$#" -eq 1 ] || usage
    comment="$(fetch_comment "$1")"
  elif [ ! -t 0 ]; then
    # No arg, but a comment (or a line containing it) is piped in.
    comment="$(cat)"
  else
    usage
  fi
  render "$comment" || { echo "no granite.build metadata found" >&2; exit 1; }
}

main "$@"
