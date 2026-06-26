#!/usr/bin/env bash
# setup-minio.sh — Deploy MinIO for local S3-compatible object storage
#
# Deploys a MinIO container, creates the gb-checkpoints bucket, and prints
# AWS environment variables for SkyPilot / AWS CLI compatibility.
#
# Usage:
#   bash scripts/minio/setup-minio.sh
#
# The script is idempotent — safe to re-run. Existing resources are skipped.

set -euo pipefail

# ── Configurable defaults (override via env vars) ─────────────────────────
MINIO_CONTAINER_NAME="${MINIO_CONTAINER_NAME:-gb-minio}"
MINIO_ROOT_USER="${MINIO_ROOT_USER:-minioadmin}"
MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD:-minioadmin}"
MINIO_API_PORT="${MINIO_API_PORT:-9000}"
MINIO_CONSOLE_PORT="${MINIO_CONSOLE_PORT:-9001}"
MINIO_IMAGE="${MINIO_IMAGE:-quay.io/minio/minio:latest}"
MINIO_BUCKET="${MINIO_BUCKET:-gb-checkpoints}"
MINIO_DATA_VOLUME="${MINIO_DATA_VOLUME:-gb-minio-data}"

# ── Logging helpers (matches setup-skypilot.sh conventions) ───────────────
log_skip()   { printf "  [SKIP]   %s\n" "$*"; }
log_create() { printf "  [CREATE] %s\n" "$*"; }
log_info()   { printf "  [INFO]   %s\n" "$*"; }
log_error()  { printf "  [ERROR]  %s\n" "$*" >&2; }

# ── Detect container runtime (Docker or Podman) ──────────────────────────
detect_container_cli() {
    if command -v docker >/dev/null 2>&1; then
        CONTAINER_CLI="docker"
    elif command -v podman >/dev/null 2>&1; then
        CONTAINER_CLI="podman"
    else
        log_error "Neither docker nor podman found on PATH"
        exit 1
    fi
    log_info "Using container runtime: ${CONTAINER_CLI}"
}

# ── Container lifecycle ──────────────────────────────────────────────────
ensure_container_running() {
    local name="${MINIO_CONTAINER_NAME}"

    # Check if container exists (running or stopped)
    if ${CONTAINER_CLI} container inspect "${name}" &>/dev/null; then
        local state
        state="$(${CONTAINER_CLI} container inspect \
            --format '{{.State.Status}}' "${name}" 2>/dev/null || \
            ${CONTAINER_CLI} container inspect \
            --format '{{.State.Running}}' "${name}" 2>/dev/null || echo "unknown")"

        if [[ "${state}" == "running" || "${state}" == "true" ]]; then
            log_skip "Container '${name}' is already running"
            return
        fi

        # Container exists but is stopped — start it
        log_info "Container '${name}' exists but is stopped (state: ${state}), starting..."
        ${CONTAINER_CLI} start "${name}"
        log_create "Started existing container '${name}'"
        return
    fi

    # Container does not exist — create and run.
    # Pre-pull quietly so the `run` below finds the image locally and prints no
    # per-layer progress.  Best-effort: if the pull can't complete, fall through
    # to `run`, which uses a cached image or surfaces the error as before.
    ${CONTAINER_CLI} pull --quiet "${MINIO_IMAGE}" || true
    ${CONTAINER_CLI} run -d \
        --name "${name}" \
        -p "${MINIO_API_PORT}:9000" \
        -p "${MINIO_CONSOLE_PORT}:9001" \
        -e "MINIO_ROOT_USER=${MINIO_ROOT_USER}" \
        -e "MINIO_ROOT_PASSWORD=${MINIO_ROOT_PASSWORD}" \
        -v "${MINIO_DATA_VOLUME}:/data" \
        "${MINIO_IMAGE}" \
        server /data --console-address ":9001"
    log_create "Container '${name}' (API :${MINIO_API_PORT}, Console :${MINIO_CONSOLE_PORT})"
}

# ── Health check ─────────────────────────────────────────────────────────
wait_for_healthy() {
    local url="http://localhost:${MINIO_API_PORT}/minio/health/ready"
    local max_attempts=30
    local attempt=0

    log_info "Waiting for MinIO to be ready at ${url}..."
    while [ "${attempt}" -lt "${max_attempts}" ]; do
        if curl -sf "${url}" >/dev/null 2>&1; then
            log_info "MinIO is ready"
            return
        fi
        attempt=$((attempt + 1))
        sleep 1
    done

    log_error "MinIO did not become ready within ${max_attempts}s"
    exit 1
}

# ── Bucket creation ──────────────────────────────────────────────────────
create_bucket() {
    local name="${MINIO_CONTAINER_NAME}"
    local bucket="${MINIO_BUCKET}"
    local alias="local"

    # Configure mc alias inside the container
    ${CONTAINER_CLI} exec "${name}" \
        mc alias set "${alias}" http://localhost:9000 \
        "${MINIO_ROOT_USER}" "${MINIO_ROOT_PASSWORD}" >/dev/null 2>&1

    # Check if bucket exists
    if ${CONTAINER_CLI} exec "${name}" mc ls "${alias}/${bucket}" &>/dev/null; then
        log_skip "Bucket '${bucket}' already exists"
        return
    fi

    ${CONTAINER_CLI} exec "${name}" mc mb "${alias}/${bucket}"
    log_create "Bucket '${bucket}'"
}

# ── Summary ──────────────────────────────────────────────────────────────
print_summary() {
    echo ""
    echo "================================================================"
    echo "  MinIO S3 Storage — Ready"
    echo "================================================================"
    echo ""
    echo "  API endpoint:  http://localhost:${MINIO_API_PORT}"
    echo "  Console:       http://localhost:${MINIO_CONSOLE_PORT}"
    echo "  Bucket:        ${MINIO_BUCKET}"
    echo "  Container:     ${MINIO_CONTAINER_NAME}"
    echo ""
    echo "  Export these variables for AWS CLI / SkyPilot:"
    echo ""
    echo "    export AWS_ACCESS_KEY_ID=${MINIO_ROOT_USER}"
    echo "    export AWS_SECRET_ACCESS_KEY=${MINIO_ROOT_PASSWORD}"
    echo "    export AWS_ENDPOINT_URL=http://localhost:${MINIO_API_PORT}"
    echo ""
    echo "  Verify:"
    echo "    aws --endpoint-url http://localhost:${MINIO_API_PORT} s3 ls"
    echo ""
    echo "  Teardown:"
    echo "    bash scripts/minio/teardown-minio.sh"
    echo ""
    echo "================================================================"
}

# ── Main ─────────────────────────────────────────────────────────────────
main() {
    echo ""
    echo "================================================================"
    echo "  MinIO S3 Storage Setup"
    echo "================================================================"
    echo ""

    detect_container_cli
    ensure_container_running
    wait_for_healthy
    create_bucket
    print_summary
}

main "$@"
