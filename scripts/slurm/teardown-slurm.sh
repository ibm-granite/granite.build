#!/usr/bin/env bash
# teardown-slurm.sh -- Clean teardown of the Docker SLURM cluster.
#
# This script:
#   1. Stops and removes all SLURM Docker containers
#   2. Removes Docker volumes (shared filesystem, mysql data, munge)
#   3. Optionally removes the SSH key pair and SkyPilot SLURM config
#
# Usage:
#   bash scripts/slurm/teardown-slurm.sh           # teardown containers + volumes
#   bash scripts/slurm/teardown-slurm.sh --all      # also remove SSH key + sky config
#
# Environment variables:
#   DOCKER - Container runtime: docker or podman (default: auto-detect)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_KEY_PATH="${HOME}/.ssh/slurm_docker_key"
CLEAN_ALL=false

for arg in "$@"; do
    case "$arg" in
        --all) CLEAN_ALL=true ;;
        --help|-h)
            echo "Usage: bash scripts/slurm/teardown-slurm.sh [--all]"
            echo "  --all   Also remove SSH key and SkyPilot SLURM config"
            exit 0
            ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

# ---- Helpers ----

log()  { printf "\033[32m[SLURM]\033[0m %s\n" "$*"; }
warn() { printf "\033[33m[SLURM]\033[0m %s\n" "$*" >&2; }

detect_docker() {
    if [ -n "${DOCKER:-}" ]; then
        echo "$DOCKER"
    elif command -v docker &>/dev/null; then
        echo "docker"
    elif command -v podman &>/dev/null; then
        echo "podman"
    else
        warn "Neither docker nor podman found. Nothing to tear down."
        exit 0
    fi
}

DOCKER_CMD="$(detect_docker)"
COMPOSE_CMD="$DOCKER_CMD compose"

if ! $COMPOSE_CMD version &>/dev/null 2>&1; then
    if command -v docker-compose &>/dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        warn "No working 'docker compose' or 'docker-compose' found."
        exit 0
    fi
fi

# ---- Step 1: Stop containers and remove volumes ----

log "Stopping Docker SLURM cluster..."
$COMPOSE_CMD -f "$SCRIPT_DIR/docker-compose.yml" \
    --project-name slurm-dev down -v --remove-orphans 2>/dev/null || true

log "Containers and volumes removed."

# ---- Step 2: Remove SkyPilot SLURM SSH config ----

SLURM_SSH_CONFIG="${HOME}/.slurm/config"
if [ -f "$SLURM_SSH_CONFIG" ]; then
    log "Removing SkyPilot SLURM SSH config: $SLURM_SSH_CONFIG"
    rm -f "$SLURM_SSH_CONFIG"
fi

# ---- Step 3: Clean up SSH key (if --all) ----

if [ "$CLEAN_ALL" = true ]; then
    if [ -f "$SSH_KEY_PATH" ]; then
        log "Removing SSH key pair: $SSH_KEY_PATH"
        rm -f "$SSH_KEY_PATH" "${SSH_KEY_PATH}.pub"
    fi
fi

# ---- Step 4: Remove known hosts entry for localhost:SLURM_SSH_PORT ----

KNOWN_HOSTS="${HOME}/.ssh/known_hosts"
if [ -f "$KNOWN_HOSTS" ]; then
    ssh-keygen -R "[localhost]:${SLURM_SSH_PORT:-2222}" -f "$KNOWN_HOSTS" 2>/dev/null || true
fi

log "Teardown complete."
