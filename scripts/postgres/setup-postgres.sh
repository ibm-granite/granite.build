#!/usr/bin/env bash
# setup-postgres.sh -- Start a local, TLS-enabled PostgreSQL for the SQL storage tests.
#
# This script:
#   1. Generates a throwaway CA and a server certificate for localhost/127.0.0.1
#   2. Starts a PostgreSQL container published on 127.0.0.1 only, with a random
#      password. TCP logins require TLS (TLSv1.3 or higher) and a password; the
#      image's default trust/plaintext pg_hba.conf rules are replaced
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
#   - The server rejects TCP logins without TLS and requires a SCRAM password for
#     every TCP login (hostssl scram-sha-256 / hostnossl reject); only the
#     container-local Unix socket uses peer auth. Clients connect with
#     sslmode=verify-full against the generated CA, the same TLS code path used
#     against the hosted database. The CA is self-signed, which is acceptable
#     only because the server is local and test-only.
#
# Usage:
#   bash scripts/postgres/setup-postgres.sh
#
# Environment variables:
#   GBTEST_POSTGRES_PORT   - Host port on 127.0.0.1 (default: 55432)
#   GBTEST_POSTGRES_IMAGE  - Image (default: quay.io/sclorg/postgresql-16-c9s)
#   GBTEST_POSTGRES_DIR    - Where the CA cert and env file go
#                            (default: <repo>/.cache/gbtest-postgres). May be a
#                            shared directory: only the files setup writes are
#                            ever replaced or deleted, never the directory itself.
#   DOCKER                 - Container runtime: docker or podman (default: auto-detect)

set -euo pipefail

# PG_DIR, CONTAINER_NAME, ENV_FILE, CERTS_DIR, CA_FILE, detect_runtime,
# remove_generated_files.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

PG_PORT="${GBTEST_POSTGRES_PORT:-55432}"
PG_IMAGE="${GBTEST_POSTGRES_IMAGE:-quay.io/sclorg/postgresql-16-c9s}"
PG_USER="gbtest"
PG_DBNAME="gbtest"
PG_SCHEMA="gbtest"
READY_TIMEOUT_SECONDS=90

# ---- Helpers ----

log() { echo "[postgres-setup] $*"; }
die() { echo "[postgres-setup] ERROR: $*" >&2; exit 1; }

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
# The image sources postgresql-pre-start/*.sh before the server first starts (and
# before it initializes the database) and loads postgresql-cfg/*.conf. The
# pre-start hook copies the key to a 0600 file owned by the server user, as
# PostgreSQL requires, and writes the pg_hba.conf that tls.conf points hba_file at.
# That file replaces the image's own, whose rules trust 127.0.0.1 with no
# password (reachable from the host under some container networking, e.g.
# rootless podman) and accept plaintext password logins.
write_image_extensions() {
    local dir="$1"
    mkdir -p "${dir}/postgresql-pre-start" "${dir}/postgresql-cfg" "${dir}/certs"
    cat > "${dir}/postgresql-pre-start/10-tls.sh" <<'EOF'
mkdir -p /var/lib/pgsql/tls
cp /opt/app-root/src/certs/server.crt /var/lib/pgsql/tls/server.crt
cp /opt/app-root/src/certs/server.key /var/lib/pgsql/tls/server.key
chmod 600 /var/lib/pgsql/tls/server.key
cat > /var/lib/pgsql/tls/pg_hba.conf <<'HBA'
local     all  all        peer
hostssl   all  all  all   scram-sha-256
hostnossl all  all  all   reject
HBA
EOF
    cat > "${dir}/postgresql-cfg/tls.conf" <<'EOF'
ssl = on
ssl_cert_file = '/var/lib/pgsql/tls/server.crt'
ssl_key_file = '/var/lib/pgsql/tls/server.key'
ssl_min_protocol_version = 'TLSv1.3'
hba_file = '/var/lib/pgsql/tls/pg_hba.conf'
EOF
}

# wait_ready RUNTIME PASSWORD: block until the app user can log in over TCP with
# TLS and its password, as the tests will, or fail. (The local socket alone could
# answer early, from the temporary server the image runs while initializing.)
# The password goes through the environment, never the command line.
wait_ready() {
    local runtime="$1" waited=0
    export PGPASSWORD="$2"
    until "${runtime}" exec -e PGPASSWORD "${CONTAINER_NAME}" \
            psql "host=127.0.0.1 dbname=${PG_DBNAME} user=${PG_USER} sslmode=require" \
            -tAc 'select 1' >/dev/null 2>&1; do
        if (( waited >= READY_TIMEOUT_SECONDS )); then
            "${runtime}" logs --tail 30 "${CONTAINER_NAME}" >&2 || true
            die "PostgreSQL not ready after ${READY_TIMEOUT_SECONDS}s"
        fi
        sleep 2
        waited=$((waited + 2))
    done
    unset PGPASSWORD
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
GBSERVER_SQL_SSLROOT_CERT_FILE=${CA_FILE}
EOF
}

# ---- Main ----

# Remove a previous run's files first (only those; PG_DIR may be a shared dir), so
# a setup that fails at any later step leaves no env file pointing at a server
# that isn't running. The env file is written only once setup succeeds.
remove_generated_files

RUNTIME="$(detect_runtime)"
[[ -n "${RUNTIME}" ]] || die "neither docker nor podman found"
command -v openssl >/dev/null 2>&1 || die "openssl is required to generate the TLS certs"

log "Using ${RUNTIME}; image ${PG_IMAGE}; port 127.0.0.1:${PG_PORT}"
"${RUNTIME}" rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
# A new dir is private; an existing one keeps its permissions (the env file is
# written 0600 regardless).
[[ -d "${PG_DIR}" ]] || mkdir -m 700 -p "${PG_DIR}"
mkdir -m 700 -p "${CERTS_DIR}"

# The image extension hooks and the server key are staged in a private (0700)
# temp dir, outside PG_DIR, and removed on exit. They go in its src/ subdir, which
# is what gets copied into the container.
EXT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/gbtest-postgres-src.XXXXXX")"
# On exit: remove the staging dir, and on failure also the half-created container
# and the files generated so far, so a failed setup leaves nothing behind.
cleanup() {
    local rc=$?
    rm -rf "${EXT_DIR}"
    if (( rc != 0 )); then
        "${RUNTIME}" rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
        remove_generated_files
    fi
}
trap cleanup EXIT
STAGE_DIR="${EXT_DIR}/src"
write_image_extensions "${STAGE_DIR}"
generate_certs "${STAGE_DIR}/certs"
cp "${STAGE_DIR}/certs/ca.crt" "${CA_FILE}"
# Make the staged files readable by any user. `docker cp` gives copied files
# root (or, with -a, the host's) ownership, and podman may map them differently;
# either way the server (uid 26) must be able to read them, so don't rely on
# ownership. The 0700 parent keeps the key private on the host, and the
# pre-start hook copies it to a 0600 file owned by the server user.
chmod -R a+rX "${STAGE_DIR}"

PASSWORD="$(openssl rand -hex 24)"

# Create, copy the extension hooks in (no bind mount, which would expose host
# ownership and permissions directly), then start. The password is passed by name
# (-e POSTGRESQL_PASSWORD, value read from this command's environment), so it never
# appears on a command line other local users can see in `ps`.
POSTGRESQL_PASSWORD="${PASSWORD}" "${RUNTIME}" create --name "${CONTAINER_NAME}" \
    -p "127.0.0.1:${PG_PORT}:5432" \
    -e POSTGRESQL_USER="${PG_USER}" \
    -e POSTGRESQL_PASSWORD \
    -e POSTGRESQL_DATABASE="${PG_DBNAME}" \
    "${PG_IMAGE}" >/dev/null
"${RUNTIME}" cp "${STAGE_DIR}/." "${CONTAINER_NAME}:/opt/app-root/src/"
rm -rf "${EXT_DIR}"   # our mktemp dir: the server key now exists only inside the container
"${RUNTIME}" start "${CONTAINER_NAME}" >/dev/null

log "Waiting for PostgreSQL to accept TLS logins..."
wait_ready "${RUNTIME}" "${PASSWORD}"

"${RUNTIME}" exec "${CONTAINER_NAME}" \
    psql "dbname=${PG_DBNAME}" -v ON_ERROR_STOP=1 -qc \
    "CREATE SCHEMA IF NOT EXISTS ${PG_SCHEMA} AUTHORIZATION ${PG_USER};" >/dev/null

write_env_file "${PASSWORD}"
log "Ready. Connection settings written to ${ENV_FILE}"
log "Test runs with GB_ENVIRONMENT=STANDALONE load them automatically."
