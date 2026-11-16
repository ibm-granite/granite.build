#!/usr/bin/env bash
# teardown-s3.sh — Remove the local S3 (SeaweedFS) container and optionally its data volume
#
# Usage:
#   bash scripts/s3/teardown-s3.sh                # stop + remove container, keep data
#   bash scripts/s3/teardown-s3.sh --remove-data  # also remove the data volume and
#                                                 # the space secrets file setup-s3.sh wrote

set -euo pipefail

# ── Configurable defaults (must match setup-s3.sh) ────────────────────
GB_S3_CONTAINER_NAME="${GB_S3_CONTAINER_NAME:-gb-s3}"
GB_S3_DATA_VOLUME="${GB_S3_DATA_VOLUME:-gb-s3-data}"
GB_S3_SECRETS_SPACE="${GB_S3_SECRETS_SPACE:-public}"
GB_S3_HOME_DIR="${GB_HOME_DIR:-${HOME}/.granite.build}"
GB_S3_SECRETS_FILE="${GB_S3_HOME_DIR/#\~/${HOME}}/space_secrets/${GB_S3_SECRETS_SPACE}/s3-local.yaml"

# ── Parse arguments ──────────────────────────────────────────────────────
REMOVE_DATA=false
for arg in "$@"; do
    case "$arg" in
        --remove-data) REMOVE_DATA=true ;;
        --help|-h)
            echo "Usage: bash scripts/s3/teardown-s3.sh [--remove-data]"
            echo "  --remove-data  Also remove the persistent data volume and the"
            echo "                 s3-local space secrets file written by setup-s3.sh"
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

# ── Remove space secrets file ────────────────────────────────────────────
remove_space_secrets() {
    local file="${GB_S3_SECRETS_FILE}"
    if [[ ! -f "${file}" ]]; then
        log_skip "Space secrets file ${file} does not exist"
        return
    fi
    rm -f "${file}"
    log_remove "Space secrets file ${file}"
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
        remove_space_secrets
    else
        log_info "Data volume '${GB_S3_DATA_VOLUME}' preserved (use --remove-data to remove)"
    fi

    echo ""
    echo "  Done."
    echo ""
}

main "$@"
