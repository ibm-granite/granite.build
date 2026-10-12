#!/usr/bin/env bash
# teardown-postgres.sh -- Stop the local test PostgreSQL started by setup-postgres.sh.
#
# Removes the container (and with it all test data) and the generated env file and
# CA certificate. Safe to run when nothing is running.
#
# Usage:
#   bash scripts/postgres/teardown-postgres.sh
#
# Environment variables:
#   GBTEST_POSTGRES_DIR  - Directory setup-postgres.sh wrote to
#                          (default: <repo>/.cache/gbtest-postgres)
#   DOCKER               - Container runtime: docker or podman (default: auto-detect)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PG_DIR="${GBTEST_POSTGRES_DIR:-${REPO_ROOT}/.cache/gbtest-postgres}"
CONTAINER_NAME="gbtest-postgres"

if [[ -n "${DOCKER:-}" ]]; then
    RUNTIME="${DOCKER}"
elif command -v docker >/dev/null 2>&1; then
    RUNTIME=docker
elif command -v podman >/dev/null 2>&1; then
    RUNTIME=podman
else
    echo "[postgres-teardown] neither docker nor podman found; nothing to stop"
    RUNTIME=""
fi

if [[ -n "${RUNTIME}" ]]; then
    "${RUNTIME}" rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi
rm -rf "${PG_DIR}"
echo "[postgres-teardown] Removed ${CONTAINER_NAME} and ${PG_DIR}"
