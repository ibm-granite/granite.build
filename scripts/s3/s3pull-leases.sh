#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0

usage() {
  cat <<'EOF'
List or release the reader leases in a SkyPilot s3pull cache.

The s3pull step keeps s3:// / cos:// inputs in a cache shared across builds
(<shared_workdir>/cos_cache unless the store sets cache_path). Each pull records
a reader lease for its target run, so that no other build prunes or evicts the
version that run reads:

  <cache>/.<hash>.leases/<targetrun_id>   line 1: the run dir ($GB_BUILD_WORKDIR)
                                          line 2: the pinned version
                                          line 3: when it was taken (epoch)

A lease is live while its run dir exists; gbserver removes the run dir when the
build ends. A build that crashed before teardown therefore leaves a live lease,
and the input it pins is never deleted. Use `list` to find such leases and
`release` to drop the leases of a run you know is dead. Deleting the dead run's
directory has the same effect.

Usage:
  scripts/s3/s3pull-leases.sh list    --cache DIR [--lease-max-age-days N]
  scripts/s3/s3pull-leases.sh release --cache DIR TARGETRUN_ID

  DIR                    the s3pull cache directory (required)
  TARGETRUN_ID           the target run whose leases to remove (release only)
  --lease-max-age-days N treat leases older than N days as dead, as the store's
                         lease_max_age_days does (default 0: never), so that
                         `list` reports live/dead the way s3pull would

`release` takes each cache entry's lock first, so it never races a pull, prune
or eviction; it fails if a lock stays held for GB_S3PULL_LOCK_WAIT_SECONDS
(default 60).

Run it where the cache is mounted, e.g. a cluster login node. For the local
SLURM fixture, pipe it into a cluster container:
  docker exec -i slurm-c1 bash -s -- list --cache /shared/cos_cache < scripts/s3/s3pull-leases.sh
EOF
}

set -euo pipefail

LOCK_WAIT_SECONDS="${GB_S3PULL_LOCK_WAIT_SECONDS:-60}"

# Print the mtime (epoch) of path $1 (GNU, then BSD stat).
mtime() {
    stat -c %Y "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null
}

# True if lease file $1 is live, with s3pull's rule: its run dir exists (or is
# "-"), and it is not older than LEASE_MAX_AGE_DAYS (0 = never too old).
lease_live() {
    local file="$1" run_dir created
    run_dir="$(sed -n 1p "${file}" 2>/dev/null)" || return 1
    [ -n "${run_dir}" ] || return 1
    if [ "${LEASE_MAX_AGE_DAYS}" -gt 0 ]; then
        created="$(sed -n 3p "${file}")"
        [[ "${created}" =~ ^[0-9]+$ ]] || created="$(mtime "${file}")" || created=0
        [ "${created}" -ge $(( $(date +%s) - LEASE_MAX_AGE_DAYS * 86400 )) ] || return 1
    fi
    [ "${run_dir}" = "-" ] || [ -d "${run_dir}" ]
}

# Print one line per lease: entry, version, targetrun, live/dead, run dir.
list_leases() {
    local leases file entry state
    printf '%-66s %-28s %-38s %-5s %s\n' ENTRY VERSION TARGETRUN STATE RUN_DIR
    for leases in "${CACHE}"/.*.leases; do
        [ -d "${leases}" ] || continue
        entry="$(basename "${leases}")"
        entry="${entry#.}"
        entry="${entry%.leases}"
        for file in "${leases}"/*; do
            [ -f "${file}" ] || continue
            state=dead
            if lease_live "${file}"; then state=live; fi
            printf '%-66s %-28s %-38s %-5s %s\n' "${entry}" "$(sed -n 2p "${file}")" \
                "$(basename "${file}")" "${state}" "$(sed -n 1p "${file}")"
        done
    done
}

# Take entry <hash> $1's lock (waiting up to LOCK_WAIT_SECONDS), so a release
# never races a pull, prune or eviction of that entry.
lock_entry() {
    local lock="${CACHE}/.$1.lock" waited=0
    until mkdir "${lock}" 2>/dev/null; do
        if [ "${waited}" -ge "${LOCK_WAIT_SECONDS}" ]; then
            echo "ERROR: ${lock} is held by $(cat "${lock}/owner" 2>/dev/null || echo unknown); try again" >&2
            return 1
        fi
        sleep 1
        waited=$(( waited + 1 ))
    done
    echo "s3pull-leases.sh|pid:$$" > "${lock}/owner"
}

# Remove target run $1's lease from every entry, each under that entry's lock.
release_leases() {
    local id="$1" leases entry count=0
    for leases in "${CACHE}"/.*.leases; do
        [ -f "${leases}/${id}" ] || continue
        entry="$(basename "${leases}")"
        entry="${entry#.}"
        entry="${entry%.leases}"
        lock_entry "${entry}" || continue
        rm -f "${leases}/${id}"
        rmdir "${leases}" 2>/dev/null || true
        rm -rf "${CACHE}/.${entry}.lock"
        echo "released lease ${id} on ${entry}"
        count=$(( count + 1 ))
    done
    echo "${count} lease(s) released for ${id}"
}

main() {
    [ $# -ge 1 ] || { usage >&2; exit 2; }
    case "$1" in -h|--help) usage; exit 0 ;; esac
    local cmd="$1" id=""
    shift
    CACHE=""
    LEASE_MAX_AGE_DAYS=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --cache) CACHE="${2:-}"; shift 2 ;;
            --lease-max-age-days) LEASE_MAX_AGE_DAYS="${2:-}"; shift 2 ;;
            -h|--help) usage; exit 0 ;;
            *) id="$1"; shift ;;
        esac
    done
    [ -n "${CACHE}" ] && [ -d "${CACHE}" ] || { echo "ERROR: --cache must name an existing directory" >&2; exit 2; }
    [[ "${LEASE_MAX_AGE_DAYS}" =~ ^[0-9]+$ ]] || { echo "ERROR: --lease-max-age-days must be a non-negative integer" >&2; exit 2; }
    case "${cmd}" in
        list) list_leases ;;
        release)
            # A target-run id is a single path component; refuse anything else.
            [[ "${id}" =~ ^[A-Za-z0-9._-]+$ ]] && [ "${id}" != "." ] && [ "${id}" != ".." ] \
                || { echo "ERROR: release needs a target-run id" >&2; exit 2; }
            release_leases "${id}"
            ;;
        *) usage >&2; exit 2 ;;
    esac
}

main "$@"
