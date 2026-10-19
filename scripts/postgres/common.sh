#!/usr/bin/env bash
# common.sh -- Shared settings and helpers for setup-postgres.sh / teardown-postgres.sh.
#
# Sourced, not run. Defines the container name, the paths of the files setup
# generates, and helpers to pick the container runtime and remove those files.
#
# GBTEST_POSTGRES_DIR may point anywhere (default: <repo>/.cache/gbtest-postgres),
# so nothing here deletes it recursively: remove_generated_files deletes only the
# files setup writes, then removes the directories only if they are left empty.

POSTGRES_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${POSTGRES_SCRIPT_DIR}/../.." && pwd)"
PG_DIR="${GBTEST_POSTGRES_DIR:-${REPO_ROOT}/.cache/gbtest-postgres}"
CONTAINER_NAME="gbtest-postgres"
ENV_FILE="${PG_DIR}/postgres.env"
CERTS_DIR="${PG_DIR}/certs"
CA_FILE="${CERTS_DIR}/ca.crt"

# detect_runtime: print the container runtime to use (docker or podman), or
# nothing if neither is installed.
detect_runtime() {
    if [[ -n "${DOCKER:-}" ]]; then
        echo "${DOCKER}"
    elif command -v docker >/dev/null 2>&1; then
        echo docker
    elif command -v podman >/dev/null 2>&1; then
        echo podman
    fi
}

# remove_generated_files: delete the env file and CA cert setup wrote, then the
# certs dir and PG_DIR only if empty. Never deletes anything else in PG_DIR.
remove_generated_files() {
    rm -f "${ENV_FILE}" "${CA_FILE}"
    rmdir "${CERTS_DIR}" 2>/dev/null || true
    rmdir "${PG_DIR}" 2>/dev/null || true
}
