#!/usr/bin/env bash
# setup-s3.sh — Deploy a local S3-compatible object store (SeaweedFS)
#
# Deploys a single-node SeaweedFS container (`weed mini`) serving the S3 API,
# creates the gb-checkpoints bucket, and prints AWS environment variables for
# SkyPilot / AWS CLI compatibility.
#
# Usage:
#   bash scripts/s3/setup-s3.sh
#
# Requires the AWS CLI on PATH (the repo .venv provides it via skypilot[aws]).
#
# Demo-only (scripts/demo-slurm.sh): do not wire into CI. The image comes from
# Docker Hub, whose anonymous pull rate limit a scheduled job would eventually hit.
# The script is idempotent — safe to re-run. Existing resources are skipped.

set -euo pipefail

# ── Configurable defaults (override via env vars) ─────────────────────────
GB_S3_CONTAINER_NAME="${GB_S3_CONTAINER_NAME:-gb-s3}"
GB_S3_ACCESS_KEY="${GB_S3_ACCESS_KEY:-gbadmin}"
GB_S3_SECRET_KEY="${GB_S3_SECRET_KEY:-gbadmin}"
GB_S3_PORT="${GB_S3_PORT:-9000}"
# Pinned so an upstream release cannot change behavior under the demo.
GB_S3_IMAGE="${GB_S3_IMAGE:-docker.io/chrislusf/seaweedfs:4.47}"
GB_S3_BUCKET="${GB_S3_BUCKET:-gb-checkpoints}"
GB_S3_DATA_VOLUME="${GB_S3_DATA_VOLUME:-gb-s3-data}"

ENDPOINT="http://localhost:${GB_S3_PORT}"

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

check_aws_cli() {
    if ! command -v aws >/dev/null 2>&1; then
        log_error "AWS CLI not found on PATH (activate the repo .venv, or install awscli)"
        exit 1
    fi
}

# ── Container lifecycle ──────────────────────────────────────────────────
ensure_container_running() {
    local name="${GB_S3_CONTAINER_NAME}"

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
    # Pre-pull quietly so the `run` below prints no per-layer progress.
    ${CONTAINER_CLI} pull --quiet "${GB_S3_IMAGE}"
    # `weed mini` runs master, volume, filer and S3 in one process; it creates
    # an admin identity from AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY. -ip pins
    # the internal component address to loopback so a restart (new container
    # IP) doesn't strand the volume registration. Unused gateways are disabled.
    ${CONTAINER_CLI} run -d \
        --name "${name}" \
        -p "${GB_S3_PORT}:9000" \
        -e "AWS_ACCESS_KEY_ID=${GB_S3_ACCESS_KEY}" \
        -e "AWS_SECRET_ACCESS_KEY=${GB_S3_SECRET_KEY}" \
        -v "${GB_S3_DATA_VOLUME}:/data" \
        "${GB_S3_IMAGE}" \
        mini -dir=/data -ip=127.0.0.1 -ip.bind=0.0.0.0 -s3.port=9000 \
        -webdav=false -admin.ui=false -s3.port.iceberg=0 -s3.port.lance=0
    log_create "Container '${name}' (S3 API :${GB_S3_PORT})"
}

# ── Health check ─────────────────────────────────────────────────────────
wait_for_healthy() {
    local url="${ENDPOINT}/healthz"
    local max_attempts=60
    local attempt=0

    log_info "Waiting for S3 endpoint to be ready at ${url}..."
    while [ "${attempt}" -lt "${max_attempts}" ]; do
        if curl -sf "${url}" >/dev/null 2>&1; then
            log_info "S3 endpoint is ready"
            return
        fi
        attempt=$((attempt + 1))
        sleep 1
    done

    log_error "S3 endpoint did not become ready within ${max_attempts}s"
    exit 1
}

# ── Bucket creation ──────────────────────────────────────────────────────
# Runs the AWS CLI against the local endpoint with only this store's
# credentials, so a caller's AWS profile / session token can't leak in.
local_aws() {
    env -u AWS_PROFILE -u AWS_SESSION_TOKEN \
        AWS_ACCESS_KEY_ID="${GB_S3_ACCESS_KEY}" \
        AWS_SECRET_ACCESS_KEY="${GB_S3_SECRET_KEY}" \
        aws --region us-east-1 --endpoint-url "${ENDPOINT}" "$@"
}

create_bucket() {
    local bucket="${GB_S3_BUCKET}"

    # `s3 mb` fails on an existing bucket, so check first.
    if local_aws s3api head-bucket --bucket "${bucket}" &>/dev/null; then
        log_skip "Bucket '${bucket}' already exists"
        return
    fi

    local_aws s3 mb "s3://${bucket}" >/dev/null
    log_create "Bucket '${bucket}'"
}

# ── Summary ──────────────────────────────────────────────────────────────
print_summary() {
    echo ""
    echo "================================================================"
    echo "  S3 Storage (SeaweedFS) — Ready"
    echo "================================================================"
    echo ""
    echo "  API endpoint:  ${ENDPOINT}"
    echo "  Bucket:        ${GB_S3_BUCKET}"
    echo "  Container:     ${GB_S3_CONTAINER_NAME}"
    echo ""
    echo "  Export these variables for AWS CLI / SkyPilot:"
    echo ""
    echo "    export AWS_ACCESS_KEY_ID=${GB_S3_ACCESS_KEY}"
    echo "    export AWS_SECRET_ACCESS_KEY=${GB_S3_SECRET_KEY}"
    echo "    export AWS_ENDPOINT_URL=${ENDPOINT}"
    echo ""
    echo "  Verify:"
    echo "    aws --endpoint-url ${ENDPOINT} s3 ls"
    echo ""
    echo "  Teardown:"
    echo "    bash scripts/s3/teardown-s3.sh"
    echo ""
    echo "================================================================"
}

# ── Main ─────────────────────────────────────────────────────────────────
main() {
    echo ""
    echo "================================================================"
    echo "  S3 Storage (SeaweedFS) Setup"
    echo "================================================================"
    echo ""

    detect_container_cli
    check_aws_cli
    ensure_container_running
    wait_for_healthy
    create_bucket
    print_summary
}

main "$@"
