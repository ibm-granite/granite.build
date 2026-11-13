#!/usr/bin/env bash

# Copyright Granite.Build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
# When invoked with a <jobid> it also prints the SLURM job name (the display
# name gbserver built from gb-<build>-<target>-<launch_id[:12]>, then truncated
# + user-hashed by SkyPilot's make_cluster_name_on_cloud). That line is omitted
# when the metadata is piped in, since there is no job to look the name up from.
#
# It also prints the launch_id (GB_SKYPILOT_LAUNCH_ID), which is NOT in the
# SLURM --comment. The full id is recovered by globbing the build's artifact
# tree from the build/target/step ids the comment does carry:
#   <root>/llm-build-<build_id>/target-*/target-run-<target_id>/
#          step-*/step-run-<step_id>/launch-<launch_id>
# If that directory is gone (cleaned up), it falls back to the 12-char prefix
# embedded in the SLURM job name (gb-...-<launch_id[:12]>-<userhash>), shown with
# a trailing "..." to mark it as a prefix.
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
  cat >&2 <<'USAGE'
usage: gbmeta.sh <jobid> [--all-steps | --no-steps]
       <command producing a Comment> | gbmeta.sh
  --all-steps  print every SLURM job step, not just the meaningful ones
  --no-steps   omit the per-step table (default is to show meaningful steps)
USAGE
  exit 2
}

# How much of the SLURM job-step table to print (jobid mode only).
STEP_MODE="filtered"   # filtered | all | none

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

# fetch_job_name JOBID
# Echo the SLURM job name, trying the live controller (scontrol) first and the
# accounting DB (sacct -X, one row per job) second. -Pn is parsable+unpadded so
# the name comes back with no trailing-space padding. Empty if neither knows it.
fetch_job_name() {
  local jobid="$1" name=""
  if command -v scontrol >/dev/null 2>&1; then
    name="$(scontrol show job "$jobid" -o 2>/dev/null \
      | grep -oE '(^| )JobName=[^[:space:]]+' | head -n1 \
      | sed 's/^ //;s/^JobName=//' || true)"
  fi
  if [ -z "$name" ] && command -v sacct >/dev/null 2>&1; then
    name="$(sacct -j "$jobid" -X -Pn -o JobName 2>/dev/null | grep -m1 '.' || true)"
  fi
  printf '%s' "$name"
}

# uid_to_name UID  -- resolve a numeric UID to its username (cached), echoing the
# UID unchanged if it doesn't resolve.
declare -A _UID_NAME_CACHE=()
uid_to_name() {
  local uid="$1" name
  case "$uid" in (*[!0-9]*|'') printf '%s' "$uid"; return 0 ;; esac
  if [ -n "${_UID_NAME_CACHE[$uid]:-}" ]; then printf '%s' "${_UID_NAME_CACHE[$uid]}"; return 0; fi
  name="$(getent passwd "$uid" 2>/dev/null | cut -d: -f1)"; [ -n "$name" ] || name="$uid"
  _UID_NAME_CACHE[$uid]="$name"; printf '%s' "$name"
}

# fetch_job_state JOBID
# Echo the job (allocation) State -- the `-X` row, i.e. the batch-script exit,
# NOT the workload step (see bvslurm.md "job state vs workload state"). sacct
# (accounting) is the source; "CANCELLED by <uid>" has its uid resolved to a
# username. Empty if sacct doesn't know the job.
fetch_job_state() {
  local jobid="$1" st=""
  command -v sacct >/dev/null 2>&1 || { return 0; }
  st="$(sacct -j "$jobid" -X -Pn -o State 2>/dev/null | grep -m1 '.' || true)"
  case "$st" in
    "CANCELLED by "*) st="CANCELLED by $(uid_to_name "${st#CANCELLED by }")" ;;
  esac
  printf '%s' "$st"
}

# Root of the shared build-artifact tree on /proj. Override with GB_BUILDS_ROOT
# if the deployment moves it.
GB_BUILDS_ROOT="${GB_BUILDS_ROOT:-/proj/data-eng/llmb-read-write/builds}"

# resolve_launch_id BUILD_ID TARGET_ID STEP_ID JOB_NAME
# Echo the full launch_id (GB_SKYPILOT_LAUNCH_ID). The id is not in the SLURM
# comment, so recover it from the artifact tree keyed on the ids that ARE in the
# comment. target-* / step-* are globbed because the path uses the *names*
# (target-train, step-custom_code_skypilot) while the comment carries the *ids*.
# Falls back to the 12-char prefix in the job name (…-<launch[:12]>-<userhash>),
# suffixed with "..." so a prefix is never mistaken for the full id.
resolve_launch_id() {
  local build_id="$1" target_id="$2" step_id="$3" job_name="$4" d lid=""
  if [ -n "$build_id" ] && [ -n "$target_id" ] && [ -n "$step_id" ]; then
    for d in "$GB_BUILDS_ROOT/llm-build-$build_id"/target-*/"target-run-$target_id"/step-*/"step-run-$step_id"/launch-*; do
      if [ -d "$d" ]; then
        lid="${d##*/launch-}"
        break
      fi
    done
  fi
  # Fallback: the SLURM job name is gb-...-<launch_id[:12]>-<userhash(8)>, so
  # launch_id[:12] is the final 12 chars once the -<userhash> group is dropped.
  # (launch_id[:12] can itself contain a dash, e.g. 6204a270-d2a, so take a
  # fixed 12-char tail, NOT the last dash-group.) Marked with "..." because it
  # is only the first 12 chars, and it is best-effort: if SkyPilot further
  # truncated the display name the chars before the userhash would be a
  # name-hash, not launch chars -- so the artifact glob above is preferred.
  if [ -z "$lid" ] && [ -n "$job_name" ]; then
    local no_hash="${job_name%-*}"      # drop -<userhash>
    if [ "${#no_hash}" -ge 12 ]; then
      lid="${no_hash: -12}..."          # final 12 chars = launch_id[:12]
    fi
  fi
  printf '%s' "$lid"
}

# extract_field COMMENT KEY  -- echo the value of one key=... in the comment.
extract_field() {
  printf '%s' "$1" | tr ';' '\n' | awk -F= -v k="$2" '$1==k { sub(/^[^=]*=/,""); print; exit }'
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

# print_steps JOBID  -- the SLURM JOB STEPS (the `.N`/.batch/sky-<n> rows under a
# job, NOT granite.build build steps) with per-step CPUs / GPUs / mem parsed out
# of AllocTRES. Default shows only the meaningful steps (named sky-*/batch/extern
# or anomalous CPU/state); --all-steps shows every one. NOTE a FAILED trailing
# enroot/bash/rm step is SkyPilot teardown, not the workload -- see bvslurm.md.
print_steps() {
  local jobid="$1"
  [ "$STEP_MODE" = "none" ] && return 0
  local raw
  raw="$(sacct -j "$jobid" -Pn \
          -o JobID,JobName,State,NCPUS,NNodes,Elapsed,AllocTRES 2>/dev/null || true)"
  [ -n "$raw" ] || { echo "  (no SLURM job-step accounting for $jobid)"; return 0; }

  printf '%s\n' "$raw" | awk -F'|' -v jobid="$jobid" -v mode="$STEP_MODE" '
    function tres(field, s,   re, v) {
      re = "(^|,)" field "=[^,]+"
      if (match(s, re)) { v = substr(s, RSTART, RLENGTH); sub(/.*=/, "", v); return v }
      return "-"
    }
    {
      id=$1; name=$2; state=$3; ncpus=$4; nnodes=$5; elapsed=$6; tr=$7
      suffix=id; sub(/^[^.]*/, "", suffix)
      if (suffix=="") { alloc_ncpus=ncpus; next }   # job row -> remember alloc CPUs
      sub(/ by [0-9]+.*$/, "", state)               # drop " by <uid>" tail
      gpu=tres("gres/gpu", tr); mem=tres("mem", tr)
      rows[++n]=sprintf("%-10s %-14s %-11s %5s %6s %4s %7s %10s",
                        suffix, substr(name,1,14), substr(state,1,11),
                        ncpus, nnodes, gpu, mem, elapsed)
      keep[n] = (name ~ /^sky/) || (name=="batch") || (name=="extern") \
                || (ncpus+0 != alloc_ncpus+0) || (state !~ /^COMPLETED/)
    }
    END {
      if (n==0) { print "  (no SLURM job steps recorded)"; exit 0 }
      shown=0
      for (i=1;i<=n;i++) if (mode=="all" || keep[i]) shown++
      if (mode=="all")
        printf "  SLURM job steps (%d, via sacct):\n", n
      else
        printf "  SLURM job steps (%d of %d shown; --all-steps for all):\n", shown, n
      printf "    %-10s %-14s %-11s %5s %6s %4s %7s %10s\n",
             "STEP","NAME","STATE","NCPUS","NNODES","GPU","MEM","ELAPSED"
      for (i=1;i<=n;i++)
        if (mode=="all" || keep[i]) print "    " rows[i]
    }'
}

main() {
  local comment job_name="" jobid=""
  # Parse args: a single <jobid> plus optional --all-steps/--no-steps. Flags may
  # appear in any order; anything else is the jobid (at most one).
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --all-steps) STEP_MODE="all";  shift ;;
      --no-steps)  STEP_MODE="none"; shift ;;
      -h|--help)   usage ;;
      -*)          echo "unknown option: $1" >&2; usage ;;
      *)           [ -z "$jobid" ] || usage; jobid="$1"; shift ;;
    esac
  done
  # Prefer an explicit job id over stdin: a redirected fd 0 (cron, CI, a wrapping
  # script) is not a tty, so keying on the arg first keeps `gbmeta.sh <jobid>`
  # working regardless of what stdin happens to be.
  local job_state=""
  if [ -n "$jobid" ]; then
    comment="$(fetch_comment "$jobid")"
    job_name="$(fetch_job_name "$jobid")"
    job_state="$(fetch_job_state "$jobid")"
  elif [ ! -t 0 ]; then
    # No arg, but a comment (or a line containing it) is piped in.
    comment="$(cat)"
  else
    usage
  fi
  render "$comment" || { echo "no granite.build metadata found" >&2; exit 1; }
  # Derive the launch_id (not in the comment) from the artifact tree, using the
  # ids the comment carries; works for piped input too. Falls back to the
  # job-name prefix when a jobid was given.
  local launch_id
  launch_id="$(resolve_launch_id \
    "$(extract_field "$comment" build_id)" \
    "$(extract_field "$comment" target_id)" \
    "$(extract_field "$comment" step_id)" \
    "$job_name")"
  [ -n "$launch_id" ] && printf '  %-11s %s\n' "launch_id:" "$launch_id"
  # Print the job name last so it sits with the rest of the table. Only when a
  # jobid was given (piped input has no job to resolve a name from) and the
  # lookup succeeded.
  [ -n "$job_name" ] && printf '  %-11s %s\n' "job_name:" "$job_name"
  # Job (allocation) State -- batch-script exit, not the workload step (the sky-*
  # step state is in the step table below). See bvslurm.md.
  [ -n "$job_state" ] && printf '  %-11s %s\n' "job_state:" "$job_state"
  # The per-step table needs a job to query; only available in jobid mode.
  [ -n "$jobid" ] && print_steps "$jobid"
  return 0
}

main "$@"
