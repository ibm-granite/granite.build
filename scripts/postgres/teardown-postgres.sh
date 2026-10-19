#!/usr/bin/env bash
# teardown-postgres.sh -- Stop the local test PostgreSQL started by setup-postgres.sh.
#
# Removes the container (and with it all test data) and the env file and CA
# certificate setup wrote. Only those files are deleted; their directories are
# removed only if left empty, so GBTEST_POSTGRES_DIR may safely point at a shared
# directory. Safe to run when nothing is running.
#
# Usage:
#   bash scripts/postgres/teardown-postgres.sh
#
# Environment variables:
#   GBTEST_POSTGRES_DIR  - Directory setup-postgres.sh wrote to
#                          (default: <repo>/.cache/gbtest-postgres)
#   DOCKER               - Container runtime: docker or podman (default: auto-detect)

set -euo pipefail

# PG_DIR, CONTAINER_NAME, detect_runtime, remove_generated_files.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RUNTIME="$(detect_runtime)"
if [[ -n "${RUNTIME}" ]]; then
    "${RUNTIME}" rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
else
    echo "[postgres-teardown] neither docker nor podman found; no container to stop"
fi
remove_generated_files
echo "[postgres-teardown] Removed ${CONTAINER_NAME} and its generated files in ${PG_DIR}"
