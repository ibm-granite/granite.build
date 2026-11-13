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
# gbfind - find the SLURM job(s) for a granite.build build_id and print their
# tracking metadata (the reverse of gbmeta.sh, which goes jobid -> metadata).
#
# granite.build stores its tracking fields in each job's SLURM --comment as a
# whitespace-free "key=value;key=value" token that includes build_id=... (see
# src/gbserver/environment/_skypilot_metadata.py). SLURM job names do NOT
# contain the full build_id -- gbserver truncates the build name and splices in
# only the launch id (see gb-<build>-<target>-<launch_id[:12]> ...), so the only
# reliable way to go from a build_id to its job(s) is to scan the accounting
# comments. This script does that over a bounded time window.
#
# For each job it also prints the SLURM JOB STEPS (the `.N`/.batch/sky-<n> rows
# under a job -- NOT granite.build build steps) with their per-step CPUs / GPUs
# / memory / elapsed. SkyPilot runs the workload as nested `srun` steps inside
# the allocation, and each step requests its OWN --cpus-per-task, so the run
# step (sky-<n>) can hold fewer CPUs than the job (the historical cpu=1 bug). By
# default only the meaningful steps are shown -- the named sky-* steps, batch,
# and any step whose CPU count differs from the allocation or that did not
# COMPLETE -- since a containerized run emits hundreds of 0-second bookkeeping
# steps. Pass --all-steps to print every step, or --no-steps to skip the table.
#
# One build_id can map to MORE THAN ONE job (a relaunch/retry reuses the build
# but gets a new SLURM job). All matches are printed, newest first.
#
# Usage:
#   gbfind.sh <build_id> [--since <when>] [--until <when>]
#             [--all-steps | --no-steps]
#     <build_id>     full build uuid (as shown by gbmeta.sh / gbcli)
#     --since WHEN   sacct start of window (default: now-30days). SLURM/`sacct`
#                    time syntax: now-30days, 2026-11-01, 2026-11-01T00:00:00.
#     --until WHEN   sacct end of window   (default: now)
#     --all-steps    show every step row, not just the meaningful ones
#     --no-steps     omit the per-step table (metadata + job summary only)
#
# Note: sacct only returns jobs inside the window; widen --since for old builds.
#
# Exit status:
#   0  at least one job found and printed
#   1  no job found for that build_id in the window
#   2  usage error

set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: gbfind.sh <build_id> [--since <when>] [--until <when>] [--all-steps | --no-steps]
  <build_id>   full build uuid
  --since WHEN sacct window start (default now-30days; e.g. 2026-11-01, now-60days)
  --until WHEN sacct window end   (default now)
  --all-steps  print every SLURM step, not just the meaningful ones
  --no-steps   omit the per-step table
USAGE
  exit 2
}

BUILD_ID=""
SINCE="now-30days"
UNTIL="now"
STEP_MODE="filtered"   # filtered | all | none

while [ $# -gt 0 ]; do
  case "$1" in
    --since) [ $# -ge 2 ] || usage; SINCE="$2"; shift 2 ;;
    --until) [ $# -ge 2 ] || usage; UNTIL="$2"; shift 2 ;;
    --all-steps) STEP_MODE="all"; shift ;;
    --no-steps)  STEP_MODE="none"; shift ;;
    -h|--help) usage ;;
    -*) echo "unknown option: $1" >&2; usage ;;
    *) [ -z "$BUILD_ID" ] || usage; BUILD_ID="$1"; shift ;;
  esac
done

[ -n "$BUILD_ID" ] || usage
command -v sacct >/dev/null 2>&1 || { echo "sacct not found (run on a SLURM login node)" >&2; exit 2; }

# render_comment COMMENT  -- same aligned table as gbmeta.sh, split at first '='
# so values containing '=' or '://' (e.g. step_uri=space://...) stay intact.
render_comment() {
  printf '%s' "$1" | tr ';' '\n' | awk -F= '
    NF >= 2 && $1 ~ /^[a-z_]+$/ {
      key = $1; sub(/^[^=]*=/, "", $0); printf "  %-11s %s\n", key ":", $0
    }'
}

# extract_field COMMENT KEY  -- echo the value of one key=... in the comment.
extract_field() {
  printf '%s' "$1" | tr ';' '\n' | awk -F= -v k="$2" '$1==k { sub(/^[^=]*=/,""); print; exit }'
}

# Root of the shared build-artifact tree on /proj. Override with GB_BUILDS_ROOT
# if the deployment moves it.
GB_BUILDS_ROOT="${GB_BUILDS_ROOT:-/proj/data-eng/llmb-read-write/builds}"

# resolve_launch_id BUILD_ID TARGET_ID STEP_ID JOB_NAME
# Echo the full launch_id (GB_SKYPILOT_LAUNCH_ID). It is NOT in the SLURM comment,
# so recover it from the artifact tree keyed on the ids that ARE (build/target/
# step); target-*/step-* are globbed because the path uses the *names* while the
# comment carries the *ids*. Falls back to the 12-char prefix embedded in the job
# name (…-<launch[:12]>-<userhash>), marked with "..." so a prefix is never
# mistaken for the full id. Empty if neither source yields anything.
resolve_launch_id() {
  local build_id="$1" target_id="$2" step_id="$3" job_name="$4" d lid=""
  if [ -n "$build_id" ] && [ -n "$target_id" ] && [ -n "$step_id" ]; then
    for d in "$GB_BUILDS_ROOT/llm-build-$build_id"/target-*/"target-run-$target_id"/step-*/"step-run-$step_id"/launch-*; do
      if [ -d "$d" ]; then lid="${d##*/launch-}"; break; fi
    done
  fi
  if [ -z "$lid" ] && [ -n "$job_name" ]; then
    local no_hash="${job_name%-*}"      # drop -<userhash>
    if [ "${#no_hash}" -ge 12 ]; then lid="${no_hash: -12}..."; fi
  fi
  printf '%s' "$lid"
}

# uid_to_name UID  -- resolve a numeric UID to its username (cached), echoing the
# UID unchanged if it doesn't resolve (unknown/stale account).
declare -A _UID_NAME_CACHE=()
uid_to_name() {
  local uid="$1" name
  case "$uid" in (*[!0-9]*|'') printf '%s' "$uid"; return 0 ;; esac
  if [ -n "${_UID_NAME_CACHE[$uid]:-}" ]; then
    printf '%s' "${_UID_NAME_CACHE[$uid]}"; return 0
  fi
  name="$(getent passwd "$uid" 2>/dev/null | cut -d: -f1)"
  [ -n "$name" ] || name="$uid"
  _UID_NAME_CACHE[$uid]="$name"
  printf '%s' "$name"
}

# humanize_state STATE  -- SLURM reports "CANCELLED by <uid>"; rewrite the uid to
# the username (e.g. "CANCELLED by granitebuild"). Other states pass through.
humanize_state() {
  local state="$1"
  case "$state" in
    "CANCELLED by "*)
      printf 'CANCELLED by %s' "$(uid_to_name "${state#CANCELLED by }")"
      ;;
    *) printf '%s' "$state" ;;
  esac
}

# print_steps JOBID  -- per-step table with CPUs / GPUs / mem parsed out of
# AllocTRES. STEP_MODE decides how much is shown.
print_steps() {
  local jobid="$1"
  [ "$STEP_MODE" = "none" ] && return 0
  # One row per step (and the job + .batch/.extern rows). -Pn = parsable, no
  # padding, '|'-separated. NCPUS is the row's CPU count; AllocTRES carries
  # gres/gpu=/mem= per step. (NCPUS and AllocTRES cpu= are the same number, so we
  # show NCPUS and use the second slot for NNodes, which actually varies: a
  # step's CPU count is whole-allocation on a multi-node row, per-node on a
  # single-node step row.)
  local raw
  raw="$(sacct -j "$jobid" -Pn \
          -o JobID,JobName,State,NCPUS,NNodes,Elapsed,AllocTRES 2>/dev/null || true)"
  [ -n "$raw" ] || { echo "    (no SLURM job-step accounting for $jobid)"; return 0; }

  printf '%s\n' "$raw" | awk -F'|' -v jobid="$jobid" -v mode="$STEP_MODE" '
    function tres(field, s,   re, v) {
      # pull "<field>=<value>" out of the AllocTRES csv; "-" if absent
      re = "(^|,)" field "=[^,]+"
      if (match(s, re)) { v = substr(s, RSTART, RLENGTH); sub(/.*=/, "", v); return v }
      return "-"
    }
    {
      id=$1; name=$2; state=$3; ncpus=$4; nnodes=$5; elapsed=$6; tr=$7
      # step suffix: ".19" / ".batch"; the bare job id (no dot) is the alloc row
      suffix=id; sub(/^[^.]*/, "", suffix)
      if (suffix=="") { alloc_ncpus=ncpus; next }   # job row -> remember alloc CPUs
      # The compact table shows the bare state word; drop any " by <uid>" tail
      # (the canceller is already shown, resolved to a name, in the job header).
      sub(/ by [0-9]+.*$/, "", state)
      gpu=tres("gres/gpu", tr); mem=tres("mem", tr)
      rows[++n]=sprintf("%-10s %-14s %-11s %5s %6s %4s %7s %10s",
                        suffix, substr(name,1,14), substr(state,1,11),
                        ncpus, nnodes, gpu, mem, elapsed)
      # is this step "meaningful"? named sky-* / batch / extern, or anomalous
      keep[n] = (name ~ /^sky/) || (name=="batch") || (name=="extern") \
                || (ncpus+0 != alloc_ncpus+0) || (state !~ /^COMPLETED/)
    }
    END {
      if (n==0) { print "    (no SLURM job steps recorded)"; exit 0 }
      # Count what will show so the header can say "X of Y".
      shown=0
      for (i=1;i<=n;i++) if (mode=="all" || keep[i]) shown++
      # "SLURM job steps" (the srun steps inside the allocation) -- NOT
      # granite.build build steps. mode=="all" shows every one; otherwise only
      # the meaningful ones (named sky-*/batch/extern, or anomalous CPU/state).
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

# sky_step_status JOBID  -- echo the workload step's own state+exit, as a FACT:
# "<state> (exit <code>)" of the last `sky-<n>` run step, or empty if the job has
# none. We deliberately do NOT fold this and the job State into one "verdict":
# SLURM carries neither the gb build's true outcome nor a reliable one of its own
# here. The job State is just the batch script's exit, and SkyPilot's batch
# script runs teardown and exits 0 EVEN WHEN the workload step failed -- so a job
# can read COMPLETED over a FAILED sky-1 (e.g. 728171), or CANCELLED (sky down)
# over a COMPLETED sky-1 (e.g. the eval fan-outs). The two states genuinely mean
# different things, so we show BOTH and let the reader judge. (For an
# authoritative verdict use gbserver: `gb build status <build_id>`.)
sky_step_status() {
  local jobid="$1"
  sacct -j "$jobid" -Pn -o JobName,State,ExitCode 2>/dev/null \
    | awk -F'|' '$1 ~ /^sky-[0-9]+$/ { st=$2; ec=$3 }
                 END { if (st!="") printf "%s (exit %s)", st, ec }'
}

# Scan accounting comments for this build_id. -X = one row per job (no steps),
# -Pn = parsable, no header/padding, '|'-separated. Match the build_id only as a
# value of the build_id= key (anchored by ';' or line start) so a build_id that
# happened to appear elsewhere in the comment cannot false-match. A build can
# have MANY jobs: one per target/step (an eval fan-out), plus retries of any of
# them. Prepend the target name (parsed from the comment) as a sort key so the
# rows GROUP BY TARGET, then by JobID descending (newest launch of that target
# first); the target key is stripped back off before the rows are consumed.
# Group key is (target, step_uri), NOT target alone: one target can run several
# DIFFERENT steps (a pipeline, e.g. hfpull -> custom_code -> hfpush), which must
# NOT be lumped together. Jobs sharing the same (target, step_uri) are the
# repeated runs of one step (a retry, or parallel same-step jobs -- the SLURM
# comment carries no attempt field, so we don't assert which). Sort by target,
# then step_uri, then newest JobID first; the 3 sort keys are stripped back off.
rows="$(sacct -a -X -Pn --starttime="$SINCE" --endtime="$UNTIL" \
          -o JobID,JobName,State,Start,Elapsed,Comment 2>/dev/null \
        | awk -F'|' -v b="$BUILD_ID" '
            $6 ~ ("(^|;)build_id=" b "(;|$)") {
              t="~"; su="~"   # missing target/step_uri sort last
              if (match($6, /(^|;)target=[^;]+/))   { t=substr($6,RSTART,RLENGTH);  sub(/.*=/,"",t) }
              if (match($6, /(^|;)step_uri=[^;]+/)) { su=substr($6,RSTART,RLENGTH); sub(/.*=/,"",su) }
              jid=$1+0; printf "%s\t%s\t%010d\t%s\n", t, su, (10^10 - jid), $0
            }
          ' \
        | sort -t"$(printf '\t')" -k1,1 -k2,2 -k3,3 \
        | sed "s/^[^$(printf '\t')]*$(printf '\t')[^$(printf '\t')]*$(printf '\t')[^$(printf '\t')]*$(printf '\t')//" || true)"

if [ -z "$rows" ]; then
  echo "no SLURM job found for build_id $BUILD_ID in [$SINCE .. $UNTIL]" >&2
  echo "(widen the window with --since, e.g. --since now-90days)" >&2
  exit 1
fi

# short_step STEP_URI  -- last path segment of a step uri for compact display
# (space://steps/eval/unitxt -> unitxt; space://steps/hfpull -> hfpull).
short_step() {
  local s="${1##*/}"; [ -n "$s" ] && printf '%s' "$s" || printf '%s' "$1"
}

# Pre-pass: count jobs per (target, step_uri). A step that ran more than once
# (same target + step_uri in >1 job) is a repeated run -- a retry, or parallel
# same-step jobs; the comment has no attempt field so we label it neutrally
# ("run k/N") rather than asserting "retry". DIFFERENT step_uris under one
# target are pipeline stages, NOT repeats, and are counted separately.
declare -A _GRP_COUNT=()
while IFS='|' read -r _jid _jn _st _sr _el _cmt; do
  [ -n "$_jid" ] || continue
  _t="$(extract_field "$_cmt" target)";   [ -n "$_t" ]  || _t="(no target)"
  _su="$(extract_field "$_cmt" step_uri)"; [ -n "$_su" ] || _su="(no step)"
  _GRP_COUNT["$_t|$_su"]=$(( ${_GRP_COUNT["$_t|$_su"]:-0} + 1 ))
done <<EOF
$rows
EOF

# Walk the rows (already sorted target -> step_uri -> newest JobID first).
n=0
distinct_targets=0
last_target=""
declare -A _TGT_SEEN=()      # which targets we've counted
declare -A _GRP_SEEN=()      # occurrence index within a (target,step_uri) group
while IFS='|' read -r jobid jobname state start elapsed comment; do
  [ -n "$jobid" ] || continue
  n=$((n + 1))
  [ "$n" -gt 1 ] && echo
  local_target="$(extract_field "$comment" target)"
  [ -n "$local_target" ] || local_target="(no target)"
  local_step_uri="$(extract_field "$comment" step_uri)"
  [ -n "$local_step_uri" ] || local_step_uri="(no step)"
  if [ -z "${_TGT_SEEN[$local_target]:-}" ]; then
    distinct_targets=$((distinct_targets + 1)); _TGT_SEEN[$local_target]=1
  fi
  # "run k/N of <step>" tag when this step ran more than once. Rows newest-first,
  # so occurrence 1 is the latest run (run N), occurrence k is run N-k+1.
  gkey="$local_target|$local_step_uri"
  total=${_GRP_COUNT[$gkey]:-1}
  tag=""
  if [ "$total" -gt 1 ]; then
    _GRP_SEEN[$gkey]=$(( ${_GRP_SEEN[$gkey]:-0} + 1 ))
    runno=$(( total - ${_GRP_SEEN[$gkey]} + 1 ))
    if [ "${_GRP_SEEN[$gkey]}" -eq 1 ]; then
      tag="   [run $runno/$total of $(short_step "$local_step_uri"), latest]"
    else
      tag="   [run $runno/$total of $(short_step "$local_step_uri"), superseded]"
    fi
  fi
  state="$(humanize_state "$state")"
  # Show BOTH states, no inferred verdict (see sky_step_status). Header lists the
  # job state and, when the job has a workload step, the sky-* step state beside
  # it -- they often differ and mean different things (job State = batch-script
  # exit incl. teardown; sky-* = the workload srun's own fate).
  sky="$(sky_step_status "$jobid")"
  if [ -n "$sky" ]; then
    hdr_status="job $state | workload $sky"
  else
    hdr_status="job $state"
  fi
  printf '== %s / %s -> job %s  %s  (%s)%s\n' \
    "$local_target" "$(short_step "$local_step_uri")" "$jobid" "$hdr_status" "$elapsed" "$tag"
  render_comment "$comment"
  launch_id="$(resolve_launch_id \
    "$(extract_field "$comment" build_id)" \
    "$(extract_field "$comment" target_id)" \
    "$(extract_field "$comment" step_id)" \
    "$jobname")"
  [ -n "$launch_id" ] && printf '  %-11s %s\n' "launch_id:" "$launch_id"
  printf '  %-11s %s\n' "job_id:"    "$jobid"
  printf '  %-11s %s\n' "job_name:"  "$jobname"
  printf '  %-11s %s\n' "job_state:" "$state"           # batch-script exit (CANCELLED = sky down teardown)
  [ -n "$sky" ] && printf '  %-11s %s\n' "workload:" "$sky"   # sky-* run step's own state+exit
  printf '  %-11s %s\n' "started:"   "$start"
  printf '  %-11s %s\n' "elapsed:"   "$elapsed"
  print_steps "$jobid"
done <<EOF
$rows
EOF

# Trailer: jobs / distinct targets, and NAME any step that ran more than once
# (target/step xN) so repeats are visible without scanning every header.
if [ "$n" -ge 2 ]; then
  repeated=""
  for g in "${!_GRP_COUNT[@]}"; do
    if [ "${_GRP_COUNT[$g]}" -gt 1 ]; then
      t="${g%%|*}"; su="${g#*|}"
      repeated="$repeated ${t}/$(short_step "$su")(x${_GRP_COUNT[$g]})"
    fi
  done
  if [ -n "$repeated" ]; then
    echo "# $n jobs across $distinct_targets target(s); steps run more than once:${repeated}. grouped by target/step, newest first." >&2
  else
    echo "# $n jobs across $distinct_targets target(s), one run each; grouped by target/step, newest first." >&2
  fi
fi
exit 0
