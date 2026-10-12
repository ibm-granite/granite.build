#!/usr/bin/env bash
# setup-postgres.sh -- Start a local, TLS-enabled PostgreSQL for the SQL storage tests.
#
# This script:
#   1. Generates a throwaway CA and a server certificate for localhost/127.0.0.1
#   2. Starts a PostgreSQL container published on 127.0.0.1 only, with a random
#      password and TLS required at TLSv1.3 or higher
#   3. Waits for the server, then creates the schema the tests use
#   4. Writes the GBSERVER_SQL_* connection settings (and GBTEST_LOCAL_POSTGRES=true)
#      to an env file readable only by you
#
# The tests load the env file themselves (test/libgbtest/local_postgres.py) in
# GB_ENVIRONMENT=STANDALONE runs, which `make extended-tests` sets.
# Re-running replaces the container and regenerates the password and certs.
# Stop it with scripts/postgres/teardown-postgres.sh.
#
# Security decisions (test-only, ephemeral server):
#   - Image: quay.io/sclorg/postgresql-16-c9s, Red Hat's public Software Collections
#     build (upstream of registry.redhat.io/rhel9/postgresql-16). Chosen over the
#     registry.redhat.io image because that one requires a Red Hat registry login,
#     which CI and fork PRs don't have.
#   - The port is bound to 127.0.0.1, never 0.0.0.0.
#   - The password is generated per run (never hardcoded) and lives only in the
#     0600 env file under the gitignored .cache/ directory.
#   - Clients connect with sslmode=verify-full against the generated CA, the same
#     TLS code path used against the hosted database. The CA is self-signed,
#     which is acceptable only because the server is local and test-only.
#
# Usage:
#   bash scripts/postgres/setup-postgres.sh
#
# Environment variables:
#   GBTEST_POSTGRES_PORT   - Host port on 127.0.0.1 (default: 55432)
#   GBTEST_POSTGRES_IMAGE  - Image (default: quay.io/sclorg/postgresql-16-c9s)
#   GBTEST_POSTGRES_DIR    - Where certs and the env file go
#                            (default: <repo>/.cache/gbtest-postgres)
#   DOCKER                 - Container runtime: docker or podman (default: auto-detect)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PG_PORT="${GBTEST_POSTGRES_PORT:-55432}"
PG_IMAGE="${GBTEST_POSTGRES_IMAGE:-quay.io/sclorg/postgresql-16-c9s}"
PG_DIR="${GBTEST_POSTGRES_DIR:-${REPO_ROOT}/.cache/gbtest-postgres}"
CONTAINER_NAME="gbtest-postgres"
PG_USER="gbtest"
PG_DBNAME="gbtest"
PG_SCHEMA="gbtest"
ENV_FILE="${PG_DIR}/postgres.env"
READY_TIMEOUT_SECONDS=90

# ---- Helpers ----

log() { echo "[postgres-setup] $*"; }
die() { echo "[postgres-setup] ERROR: $*" >&2; exit 1; }

# detect_runtime: print the container runtime to use (docker or podman).
detect_runtime() {
    if [[ -n "${DOCKER:-}" ]]; then
        echo "${DOCKER}"
    elif command -v docker >/dev/null 2>&1; then
        echo docker
    elif command -v podman >/dev/null 2>&1; then
        echo podman
    else
        die "neither docker nor podman found"
    fi
}

# generate_certs DIR: write ca.crt and server.{crt,key} (SAN localhost, 127.0.0.1) into DIR.
generate_certs() {
    local dir="$1"
    local ext="${dir}/server.ext"
    openssl req -x509 -new -nodes -newkey rsa:3072 -sha256 -days 30 \
        -subj "/CN=gbtest-postgres-ca" \
        -keyout "${dir}/ca.key" -out "${dir}/ca.crt" 2>/dev/null
    openssl req -new -nodes -newkey rsa:3072 -sha256 \
        -subj "/CN=localhost" \
        -keyout "${dir}/server.key" -out "${dir}/server.csr" 2>/dev/null
    printf 'subjectAltName=DNS:localhost,IP:127.0.0.1\nextendedKeyUsage=serverAuth\n' > "${ext}"
    openssl x509 -req -sha256 -days 30 -in "${dir}/server.csr" \
        -CA "${dir}/ca.crt" -CAkey "${dir}/ca.key" -CAcreateserial \
        -extfile "${ext}" -out "${dir}/server.crt" 2>/dev/null
    rm -f "${dir}/server.csr" "${dir}/ca.key" "${dir}/ca.srl" "${ext}"
}

# write_image_extensions DIR: lay out the sclorg image's extension hooks in DIR.
# The image sources postgresql-pre-start/*.sh before the server starts and loads
# postgresql-cfg/*.conf. The pre-start hook copies the key to a 0600 file owned by
# the server user, as PostgreSQL requires.
write_image_extensions() {
    local dir="$1"
    mkdir -p "${dir}/postgresql-pre-start" "${dir}/postgresql-cfg" "${dir}/certs"
    cat > "${dir}/postgresql-pre-start/10-tls.sh" <<'EOF'
mkdir -p /var/lib/pgsql/tls
cp /opt/app-root/src/certs/server.crt /var/lib/pgsql/tls/server.crt
cp /opt/app-root/src/certs/server.key /var/lib/pgsql/tls/server.key
chmod 600 /var/lib/pgsql/tls/server.key
EOF
    cat > "${dir}/postgresql-cfg/tls.conf" <<'EOF'
ssl = on
ssl_cert_file = '/var/lib/pgsql/tls/server.crt'
ssl_key_file = '/var/lib/pgsql/tls/server.key'
ssl_min_protocol_version = 'TLSv1.3'
EOF
}

# wait_ready RUNTIME: block until the server accepts TLS connections, or fail.
wait_ready() {
    local runtime="$1" waited=0
    until "${runtime}" exec "${CONTAINER_NAME}" \
            psql "host=127.0.0.1 dbname=${PG_DBNAME} sslmode=require" -tAc 'select 1' \
            >/dev/null 2>&1; do
        if (( waited >= READY_TIMEOUT_SECONDS )); then
            "${runtime}" logs --tail 30 "${CONTAINER_NAME}" >&2 || true
            die "PostgreSQL not ready after ${READY_TIMEOUT_SECONDS}s"
        fi
        sleep 2
        waited=$((waited + 2))
    done
}

# write_env_file PASSWORD: write the connection settings the tests read.
write_env_file() {
    local password="$1"
    umask 077
    cat > "${ENV_FILE}" <<EOF
# Generated by scripts/postgres/setup-postgres.sh -- local test database only.
GBTEST_LOCAL_POSTGRES=true
GBSERVER_SQL_SCHEME=postgresql+psycopg
GBSERVER_SQL_HOST=localhost
GBSERVER_SQL_PORT=${PG_PORT}
GBSERVER_SQL_DBNAME=${PG_DBNAME}
GBSERVER_SQL_SCHEMA=${PG_SCHEMA}
GBSERVER_SQL_USER=${PG_USER}
GBSERVER_SQL_PASSWD=${password}
GBSERVER_SQL_SSLROOT_CERT_FILE=${PG_DIR}/certs/ca.crt
EOF
}

# ---- Main ----

RUNTIME="$(detect_runtime)"
command -v openssl >/dev/null 2>&1 || die "openssl is required to generate the TLS certs"

log "Using ${RUNTIME}; image ${PG_IMAGE}; port 127.0.0.1:${PG_PORT}"
"${RUNTIME}" rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
rm -rf "${PG_DIR}"
mkdir -p "${PG_DIR}"
chmod 700 "${PG_DIR}"

EXT_DIR="${PG_DIR}/image-src"
write_image_extensions "${EXT_DIR}"
generate_certs "${EXT_DIR}/certs"
mkdir -p "${PG_DIR}/certs"
cp "${EXT_DIR}/certs/ca.crt" "${PG_DIR}/certs/ca.crt"

PASSWORD="$(openssl rand -hex 24)"

# Create, copy the extension hooks in (no bind mount, so file ownership and
# permissions inside the container don't depend on the host runtime), then start.
"${RUNTIME}" create --name "${CONTAINER_NAME}" \
    -p "127.0.0.1:${PG_PORT}:5432" \
    -e POSTGRESQL_USER="${PG_USER}" \
    -e POSTGRESQL_PASSWORD="${PASSWORD}" \
    -e POSTGRESQL_DATABASE="${PG_DBNAME}" \
    "${PG_IMAGE}" >/dev/null
"${RUNTIME}" cp "${EXT_DIR}/." "${CONTAINER_NAME}:/opt/app-root/src/"
rm -rf "${EXT_DIR}"   # the server key now exists only inside the container
"${RUNTIME}" start "${CONTAINER_NAME}" >/dev/null

log "Waiting for PostgreSQL to accept TLS connections..."
wait_ready "${RUNTIME}"

"${RUNTIME}" exec "${CONTAINER_NAME}" \
    psql "host=127.0.0.1 dbname=${PG_DBNAME} sslmode=require" -v ON_ERROR_STOP=1 -qc \
    "CREATE SCHEMA IF NOT EXISTS ${PG_SCHEMA} AUTHORIZATION ${PG_USER};" >/dev/null

write_env_file "${PASSWORD}"
log "Ready. Connection settings written to ${ENV_FILE}"
log "Test runs with GB_ENVIRONMENT=STANDALONE load them automatically."
