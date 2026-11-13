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

# teardown-s3.sh — Remove the local S3 (SeaweedFS) container and optionally its data volume
#
# Usage:
#   bash scripts/s3/teardown-s3.sh                # stop + remove container, keep data
#   bash scripts/s3/teardown-s3.sh --remove-data  # also remove the data volume

set -euo pipefail

# ── Configurable defaults (must match setup-s3.sh) ────────────────────
GB_S3_CONTAINER_NAME="${GB_S3_CONTAINER_NAME:-gb-s3}"
GB_S3_DATA_VOLUME="${GB_S3_DATA_VOLUME:-gb-s3-data}"

# ── Parse arguments ──────────────────────────────────────────────────────
REMOVE_DATA=false
for arg in "$@"; do
    case "$arg" in
        --remove-data) REMOVE_DATA=true ;;
        --help|-h)
            echo "Usage: bash scripts/s3/teardown-s3.sh [--remove-data]"
            echo "  --remove-data  Also remove the persistent data volume"
            exit 0
            ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ── Logging helpers ──────────────────────────────────────────────────────
log_skip()   { printf "  [SKIP]   %s\n" "$*"; }
log_remove() { printf "  [REMOVE] %s\n" "$*"; }
log_info()   { printf "  [INFO]   %s\n" "$*"; }

# ── Detect container runtime ─────────────────────────────────────────────
detect_container_cli() {
    if command -v docker >/dev/null 2>&1; then
        CONTAINER_CLI="docker"
    elif command -v podman >/dev/null 2>&1; then
        CONTAINER_CLI="podman"
    else
        echo "  [ERROR]  Neither docker nor podman found on PATH" >&2
        exit 1
    fi
    log_info "Using container runtime: ${CONTAINER_CLI}"
}

# ── Remove container ─────────────────────────────────────────────────────
remove_container() {
    local name="${GB_S3_CONTAINER_NAME}"

    if ! ${CONTAINER_CLI} container inspect "${name}" &>/dev/null; then
        log_skip "Container '${name}' does not exist"
        return
    fi

    ${CONTAINER_CLI} rm -f "${name}"
    log_remove "Container '${name}'"
}

# ── Remove data volume ───────────────────────────────────────────────────
remove_volume() {
    local vol="${GB_S3_DATA_VOLUME}"

    if ! ${CONTAINER_CLI} volume inspect "${vol}" &>/dev/null; then
        log_skip "Volume '${vol}' does not exist"
        return
    fi

    ${CONTAINER_CLI} volume rm "${vol}"
    log_remove "Volume '${vol}'"
}

# ── Main ─────────────────────────────────────────────────────────────────
main() {
    echo ""
    echo "================================================================"
    echo "  S3 Storage (SeaweedFS) Teardown"
    echo "================================================================"
    echo ""

    detect_container_cli
    remove_container

    if [[ "${REMOVE_DATA}" == "true" ]]; then
        remove_volume
    else
        log_info "Data volume '${GB_S3_DATA_VOLUME}' preserved (use --remove-data to remove)"
    fi

    echo ""
    echo "  Done."
    echo ""
}

main "$@"
