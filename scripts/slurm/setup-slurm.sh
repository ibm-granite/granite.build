#!/usr/bin/env bash
# setup-slurm.sh -- Bring up a Docker SLURM cluster and configure SkyPilot.
#
# This script:
#   1. Generates an SSH key pair for passwordless access to the login node
#   2. Starts the Docker SLURM cluster (slurmctld, c1..c4, mysql, slurmdbd)
#   3. Verifies SSH connectivity to slurmctld
#   4. Clears any stale gbserver-managed / legacy slurm-docker block from
#      ~/.slurm/config (so a leftover entry never trips gbserver's
#      refuse-on-conflict — no teardown needed just to refresh config)
#   5. Verifies the cluster is healthy (sinfo shows 4 compute nodes)
#
# It does NOT write the slurm-docker entry itself: gbserver materializes that
# from the inline `cluster_ssh_configs` in the Skypilot environment.yaml at build
# launch time. This script only provisions the cluster + SSH key and keeps
# ~/.slurm/config free of stale managed blocks.
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

# --quiet-pull suppresses the per-layer "Pulling/Extracting/Pull complete"
# progress (thousands of lines in CI); errors and the final status still print.
$COMPOSE_CMD $COMPOSE_FILES \
    --project-name slurm-dev up -d --quiet-pull

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

# Make the stock image's sshd PAM stack usable inside a container on a real
# Linux kernel.  Two failures surface only off Docker Desktop's LinuxKit kernel
# (hence "works on my Mac, fails in CI"):
#   * account phase — one of the `account required` modules (pam_sepermit /
#     pam_nologin / pam_unix, the last pulled in via `account include
#     password-auth`) denies root, so sshd logs "Access denied for user root by
#     PAM account configuration" right after the key is accepted.  We prepend
#     `account sufficient pam_permit.so` so the account phase short-circuits to
#     success before those modules run.
#   * session phase — `session required pam_loginuid/pam_selinux/pam_namespace`
#     fail in-container; we demote them to `optional`.
# Both edits are idempotent and best-effort, target /etc/pam.d/sshd which PAM
# reads per-session (so an already-running sshd picks them up with no restart),
# and only ever touch the ephemeral container — never the host's PAM config.
# Args: $1 = container name.
relax_sshd_pam() {
    $DOCKER_CMD exec "$1" sh -c '
        f=/etc/pam.d/sshd
        grep -q "^account[[:space:]]\+sufficient[[:space:]]\+pam_permit.so" "$f" \
            || sed -i "0,/^account/s//account    sufficient   pam_permit.so\n&/" "$f"
        sed -i -E "s/^(session[[:space:]]+)required([[:space:]]+(pam_loginuid|pam_selinux|pam_namespace)\.so)/\1optional\2/" "$f"
    ' 2>/dev/null || true
}

log "Relaxing container-hostile sshd PAM account/session modules..."
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

# Print SSH diagnostics for the slurmctld login node when the connectivity check
# below fails, so logs reveal *why* instead of just reporting the retry count.
# Two high-signal probes: the slurmctld container logs (sshd runs with -e, so any
# post-auth/PAM session failure lands there) and a verbose client-side attempt.
# Both are best-effort (`|| true`) so the dump never masks the original failure.
dump_ssh_diagnostics() {
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

# ---- Step 5: Clean stale SkyPilot SLURM SSH config ----
# This script no longer *writes* ~/.slurm/config — gbserver materializes the
# slurm-docker block from the inline `cluster_ssh_configs` in the Skypilot
# environment.yaml at build launch time (the script only provisions the cluster
# and the SSH key the inline IdentityFile references). But it does clear any
# leftover gbserver-managed or legacy `setup-slurm.sh` block on every run, so a
# stale entry can never trip gbserver's refuse-on-conflict and you never need a
# teardown just to refresh config. Unrelated `Host` entries are preserved.

SLURM_SSH_CONFIG="${HOME}/.slurm/config"

strip_block() {  # $1 = begin marker, $2 = end marker
    if [ -f "$SLURM_SSH_CONFIG" ] && grep -qF "$1" "$SLURM_SSH_CONFIG"; then
        log "Clearing stale managed block ('$1') from $SLURM_SSH_CONFIG"
        sed_i "/$1/,/$2/d" "$SLURM_SSH_CONFIG"
    fi
}

strip_block "# BEGIN gbserver-managed (cluster config)" "# END gbserver-managed"
strip_block "# BEGIN slurm-docker (managed by setup-slurm.sh)" "# END slurm-docker"

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
log "Quick reference (slurm SSH config is inlined in environment.yaml, not ~/.slurm/config):"
log "  SSH to login node:   ssh -i $SSH_KEY_PATH -p $SLURM_SSH_PORT root@${SLURM_SSH_HOST}"
log "  Run sinfo:           ssh -i $SSH_KEY_PATH -p $SLURM_SSH_PORT root@${SLURM_SSH_HOST} sinfo"
log "  Submit a test job:   ssh -i $SSH_KEY_PATH -p $SLURM_SSH_PORT root@${SLURM_SSH_HOST} sbatch --wrap 'hostname'"
log "  SkyPilot check:      sky check"
log "  Teardown:            bash $SCRIPT_DIR/teardown-slurm.sh"
