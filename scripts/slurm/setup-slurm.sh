#!/usr/bin/env bash
# setup-slurm.sh -- Bring up a Docker SLURM cluster and configure SkyPilot.
#
# This script:
#   1. Generates an SSH key pair for passwordless access to the login node
#   2. Starts the Docker SLURM cluster (slurmctld, c1..c4, mysql, slurmdbd)
#   3. Verifies SSH connectivity to slurmctld
#   4. Configures ~/.sky/config.yaml so SkyPilot discovers the SLURM cluster
#   5. Verifies the cluster is healthy (sinfo shows 4 compute nodes)
#
# Usage:
#   bash scripts/slurm/setup-slurm.sh
#
# Environment variables:
#   SLURM_SSH_PORT  - Host port for SSH to slurmctld (default: 2222)
#   SLURM_SSH_HOST  - Host address used to reach the published SSH port
#                     (default: 127.0.0.1). Pinned to IPv4 on purpose: the
#                     published port is bound on IPv4 (0.0.0.0), and on Linux
#                     `localhost` can resolve to ::1 first, which has no
#                     listener — so `localhost` is unreliable across Docker
#                     Desktop (macOS) vs native Linux runners.
#   SLURM_VERSION   - SLURM version / image tag (default: 25.11.4)
#   DOCKER          - Container runtime: docker or podman (default: auto-detect)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SSH_PORT="${SLURM_SSH_PORT:-2222}"
SLURM_SSH_HOST="${SLURM_SSH_HOST:-127.0.0.1}"
SSH_KEY_PATH="${HOME}/.ssh/slurm_docker_key"

# ---- Helpers ----

log()  { printf "\033[32m[SLURM]\033[0m %s\n" "$*"; }
warn() { printf "\033[33m[SLURM]\033[0m %s\n" "$*" >&2; }
err()  { printf "\033[31m[SLURM]\033[0m %s\n" "$*" >&2; exit 1; }

# Portable in-place sed (BSD on macOS requires the empty-string suffix).
sed_i() {
    if sed --version 2>/dev/null | grep -q GNU; then
        sed -i "$@"
    else
        sed -i '' "$@"
    fi
}

detect_docker() {
    if [ -n "${DOCKER:-}" ]; then
        echo "$DOCKER"
    elif command -v docker &>/dev/null; then
        echo "docker"
    elif command -v podman &>/dev/null; then
        echo "podman"
    else
        err "Neither docker nor podman found. Install one and retry."
    fi
}

DOCKER_CMD="$(detect_docker)"
COMPOSE_CMD="$DOCKER_CMD compose"

if ! $COMPOSE_CMD version &>/dev/null 2>&1; then
    if command -v docker-compose &>/dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        err "No working 'docker compose' or 'docker-compose' found."
    fi
fi

# ---- Step 1: Generate SSH key pair ----
# Must happen before docker compose up because the public key is
# bind-mounted into slurmctld via SSH_AUTHORIZED_KEYS.

if [ ! -f "$SSH_KEY_PATH" ]; then
    log "Generating SSH key pair at $SSH_KEY_PATH..."
    mkdir -p "$(dirname "$SSH_KEY_PATH")"
    ssh-keygen -t ed25519 -f "$SSH_KEY_PATH" -N "" -C "slurm-docker-dev"
else
    log "SSH key already exists at $SSH_KEY_PATH."
fi

# ---- Step 2: Detect GPU and configure SLURM ----
# Set SLURM_NO_GPU=1 to force the CPU-only path even when a GPU is present.

COMPOSE_FILES="-f $SCRIPT_DIR/docker-compose.yml"
HAS_GPU=false

# Materialize a fresh slurm.conf from the tracked template each run.
# All sed/awk mutations below operate on this generated (gitignored) file.
cp "$SCRIPT_DIR/slurm.conf.template" "$SCRIPT_DIR/slurm.conf"

if [ "${SLURM_NO_GPU:-0}" = "1" ] || ! nvidia-smi -L >/dev/null 2>&1; then
    log "No GPU detected (or SLURM_NO_GPU=1) — CPU-only cluster."
    # CPU-only: empty gres.conf, no GPU in node definitions
    cat > "$SCRIPT_DIR/gres.conf" <<'GRESEOF'
# No GPU resources available
GRESEOF
    sed_i 's/^NodeName=c1.*/NodeName=c1 CPUs=2 RealMemory=1024 State=UNKNOWN/' "$SCRIPT_DIR/slurm.conf"
    sed_i '/^GresTypes=/d' "$SCRIPT_DIR/slurm.conf"
else
    HAS_GPU=true
    log "GPU detected — enabling GPU passthrough on c1."
    COMPOSE_FILES="$COMPOSE_FILES -f $SCRIPT_DIR/docker-compose.gpu.yml"
    # GPU: enable nvidia auto-detection and add GRES to c1
    cat > "$SCRIPT_DIR/gres.conf" <<'GRESEOF'
AutoDetect=nvidia
GRESEOF
    sed_i 's/^NodeName=c1.*/NodeName=c1 CPUs=2 RealMemory=1024 Gres=gpu:1 State=UNKNOWN/' "$SCRIPT_DIR/slurm.conf"
    if ! grep -q '^GresTypes=' "$SCRIPT_DIR/slurm.conf"; then
        # Portable insert-before (BSD sed's `i\` syntax differs from GNU); use awk.
        awk '/^# ---- Compute nodes/ && !x { print "GresTypes=gpu\n"; x=1 } { print }' \
            "$SCRIPT_DIR/slurm.conf" > "$SCRIPT_DIR/slurm.conf.tmp" \
            && mv "$SCRIPT_DIR/slurm.conf.tmp" "$SCRIPT_DIR/slurm.conf"
    fi
fi

# ---- Step 3: Start the SLURM cluster ----

log "Starting Docker SLURM cluster..."
export SSH_AUTHORIZED_KEYS="${SSH_KEY_PATH}.pub"
export SLURM_SSH_PORT

$COMPOSE_CMD $COMPOSE_FILES \
    --project-name slurm-dev up -d

log "Waiting for SLURM cluster to become ready (may take 1-2 minutes)..."
timeout=240
elapsed=0
while true; do
    node_count=$($DOCKER_CMD exec slurm-slurmctld sinfo --noheader -N 2>/dev/null | wc -l || echo 0)
    if [ "$node_count" -ge 4 ]; then
        break
    fi
    sleep 5
    elapsed=$((elapsed + 5))
    if [ "$elapsed" -ge "$timeout" ]; then
        warn "Timed out after ${timeout}s. Current container status:"
        $COMPOSE_CMD -f "$SCRIPT_DIR/docker-compose.yml" --project-name slurm-dev ps
        warn "slurmctld logs:"
        $DOCKER_CMD logs slurm-slurmctld 2>&1 | tail -20
        for node in slurm-c1 slurm-c2 slurm-c3 slurm-c4; do
            warn "${node} logs:"
            $DOCKER_CMD logs "$node" 2>&1 | tail -10
        done
        err "Cluster not ready. Expected 4 compute nodes but found $node_count."
    fi
done
log "SLURM cluster is ready with $node_count compute nodes."

# ---- Step 3b: Install rsync and enable SSH on all nodes ----
# SkyPilot requires rsync for file transfer and SSH access to compute nodes
# for running setup commands after SLURM job allocation.

log "Installing rsync in SLURM containers..."
for node in slurm-slurmctld slurm-c1 slurm-c2 slurm-c3 slurm-c4; do
    if ! $DOCKER_CMD exec "$node" which rsync &>/dev/null; then
        $DOCKER_CMD exec "$node" dnf install -y -q rsync 2>/dev/null
    fi
done
log "rsync installed."

# Demote container-hostile `session required` PAM modules in the sshd stack on
# every node.  The stock image ships `session required pam_loginuid.so` (plus
# pam_selinux/pam_namespace); on a real Linux host (e.g. GitHub Actions) these
# fail inside the container, so sshd accepts the public key and then immediately
# closes the session ("Connection closed" with no shell).  On Docker Desktop's
# LinuxKit kernel they are effectively no-ops, which is why this only bites in
# CI.  Demoting them to `optional` lets the session proceed; PAM reads
# /etc/pam.d/sshd per-session, so slurmctld's already-running sshd picks this up
# without a restart, and the compute-node sshds started below inherit it.
# Args: $1 = container name.
relax_sshd_pam() {
    $DOCKER_CMD exec "$1" sed -i -E \
        's/^(session[[:space:]]+)required([[:space:]]+(pam_loginuid|pam_selinux|pam_namespace)\.so)/\1optional\2/' \
        /etc/pam.d/sshd 2>/dev/null || true
}

log "Relaxing container-hostile sshd PAM session modules..."
for node in slurm-slurmctld slurm-c1 slurm-c2 slurm-c3 slurm-c4; do
    relax_sshd_pam "$node"
done

log "Starting sshd on compute nodes..."
for node in slurm-c1 slurm-c2 slurm-c3 slurm-c4; do
    $DOCKER_CMD exec "$node" bash -c '
        ssh-keygen -A 2>/dev/null
        mkdir -p /root/.ssh && chmod 700 /root/.ssh
        /usr/sbin/sshd -D -e &
    '
done
log "sshd started on compute nodes."

# ---- Step 3c: Connect MinIO to slurm-net (if running) ----
# Allows SLURM containers to reach MinIO at gb-minio:9000 for S3 artifact push.

if $DOCKER_CMD container inspect gb-minio &>/dev/null 2>&1; then
    if ! $DOCKER_CMD network inspect slurm-net --format '{{range .Containers}}{{.Name}} {{end}}' 2>/dev/null | grep -q gb-minio; then
        $DOCKER_CMD network connect slurm-net gb-minio
        log "Connected MinIO (gb-minio) to slurm-net."
    else
        log "MinIO already connected to slurm-net."
    fi
else
    warn "MinIO container (gb-minio) not found. Run: bash scripts/minio/setup-minio.sh"
fi

# ---- Step 4: Verify SSH connectivity ----

# Print SSH/port diagnostics for the slurmctld login node.  Called when the
# connectivity check below fails so logs reveal *why* instead of just reporting
# the retry count: whether the host port is published, whether the TCP/SSH
# handshake reaches sshd, and — via the slurmctld container logs — any
# post-authentication/session failure (e.g. a PAM module rejecting the session
# after the key is accepted).  Every probe is best-effort (`|| true`) so the
# dump never masks the original failure.
dump_ssh_diagnostics() {
    warn "compose ps:"
    $COMPOSE_CMD -f "$SCRIPT_DIR/docker-compose.yml" --project-name slurm-dev ps || true
    warn "host port mapping for slurm-slurmctld:"
    $DOCKER_CMD port slurm-slurmctld || true
    warn "host listeners on ${SLURM_SSH_PORT}:"
    ss -tlnp 2>/dev/null | grep ":${SLURM_SSH_PORT}" \
        || echo "  (nothing listening on host port ${SLURM_SSH_PORT})"
    warn "slurmctld container logs (sshd runs with -e; post-auth failures land here):"
    $DOCKER_CMD logs slurm-slurmctld 2>&1 | tail -60 || true
    warn "verbose ssh attempt (last 40 lines):"
    ssh -vvv -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o ConnectTimeout=5 -i "$SSH_KEY_PATH" \
        -p "$SLURM_SSH_PORT" "root@${SLURM_SSH_HOST}" sinfo 2>&1 | tail -40 || true
}

log "Verifying SSH connectivity to slurmctld at ${SLURM_SSH_HOST}:${SLURM_SSH_PORT}..."
ssh_ok=false
for i in $(seq 1 10); do
    if ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
           -o ConnectTimeout=5 -i "$SSH_KEY_PATH" \
           -p "$SLURM_SSH_PORT" "root@${SLURM_SSH_HOST}" sinfo --noheader &>/dev/null; then
        ssh_ok=true
        break
    fi
    sleep 2
done
if [ "$ssh_ok" = false ]; then
    dump_ssh_diagnostics
    err "SSH to slurmctld failed after 10 attempts. Check SSH config and port $SLURM_SSH_PORT."
fi
log "SSH connectivity verified."

# ---- Step 5: Configure SkyPilot ----
# SkyPilot >=0.12 reads SLURM cluster SSH config from ~/.slurm/config.
# We manage only the slurm-docker block, leaving other entries untouched.

SLURM_SSH_CONFIG="${HOME}/.slurm/config"
MARKER_BEGIN="# BEGIN slurm-docker (managed by setup-slurm.sh)"
MARKER_END="# END slurm-docker"

log "Configuring SkyPilot SLURM SSH config at $SLURM_SSH_CONFIG..."
mkdir -p "$(dirname "$SLURM_SSH_CONFIG")"
touch "$SLURM_SSH_CONFIG"

# Remove existing slurm-docker block if present
sed_i "/$MARKER_BEGIN/,/$MARKER_END/d" "$SLURM_SSH_CONFIG"

# Append the managed block
cat >> "$SLURM_SSH_CONFIG" <<SSHEOF
$MARKER_BEGIN
Host slurm-docker
    HostName ${SLURM_SSH_HOST}
    User root
    Port ${SLURM_SSH_PORT}
    IdentityFile ${SSH_KEY_PATH}
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
$MARKER_END
SSHEOF

log "SkyPilot SLURM config written to $SLURM_SSH_CONFIG."

# ---- Step 6: Verify cluster health ----

log "Verifying SLURM cluster health..."
NODE_COUNT=$(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -i "$SSH_KEY_PATH" -p "$SLURM_SSH_PORT" "root@${SLURM_SSH_HOST}" \
    sinfo --noheader -N 2>/dev/null | wc -l)

if [ "$NODE_COUNT" -lt 4 ]; then
    warn "Expected 4 compute nodes but found $NODE_COUNT. Cluster may still be starting."
    warn "Run: ssh -i $SSH_KEY_PATH -p $SLURM_SSH_PORT root@${SLURM_SSH_HOST} sinfo"
else
    log "SLURM cluster is ready: $NODE_COUNT compute nodes."
fi

echo ""
log "Cluster status:"
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -i "$SSH_KEY_PATH" -p "$SLURM_SSH_PORT" "root@${SLURM_SSH_HOST}" sinfo

echo ""
log "Setup complete."
log ""
log "Quick reference:"
log "  SSH to login node:   ssh -F ~/.slurm/config slurm-docker"
log "  Run sinfo:           ssh -F ~/.slurm/config slurm-docker sinfo"
log "  Submit a test job:   ssh -F ~/.slurm/config slurm-docker sbatch --wrap 'hostname'"
log "  SkyPilot check:      sky check"
log "  Teardown:            bash $SCRIPT_DIR/teardown-slurm.sh"
