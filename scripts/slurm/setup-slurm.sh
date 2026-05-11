#!/usr/bin/env bash
# setup-slurm.sh -- Bring up a Docker SLURM cluster and configure SkyPilot.
#
# This script:
#   1. Starts the Docker SLURM cluster (slurmctld, c1, c2, mysql, slurmdbd, auth)
#   2. Generates an SSH key pair for passwordless access to the login node
#   3. Injects the public key into slurmctld's authorized_keys
#   4. Configures ~/.sky/config.yaml so SkyPilot discovers the SLURM cluster
#   5. Verifies the cluster is healthy (sinfo shows 2 compute nodes)
#
# Usage:
#   bash scripts/slurm/setup-slurm.sh
#
# Environment variables:
#   SLURM_SSH_PORT  - Host port for SSH to slurmctld (default: 2222)
#   SLURM_TAG       - Docker image tag for giovtorres/docker-slurm (default: latest)
#   DOCKER          - Container runtime: docker or podman (default: auto-detect)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SSH_PORT="${SLURM_SSH_PORT:-2222}"
SSH_KEY_PATH="${HOME}/.ssh/slurm_docker_key"
SKY_CONFIG="${HOME}/.sky/config.yaml"

# ---- Helpers ----

log()  { printf "\033[32m[SLURM]\033[0m %s\n" "$*"; }
warn() { printf "\033[33m[SLURM]\033[0m %s\n" "$*" >&2; }
err()  { printf "\033[31m[SLURM]\033[0m %s\n" "$*" >&2; exit 1; }

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

# Test that compose subcommand works; fall back to docker-compose binary
if ! $COMPOSE_CMD version &>/dev/null 2>&1; then
    if command -v docker-compose &>/dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        err "No working 'docker compose' or 'docker-compose' found."
    fi
fi

# ---- Step 1: Start the SLURM cluster ----

log "Starting Docker SLURM cluster..."
SLURM_SSH_PORT="$SLURM_SSH_PORT" \
    $COMPOSE_CMD -f "$SCRIPT_DIR/docker-compose.yml" \
    --project-name slurm-dev up -d

log "Waiting for slurmctld to become healthy..."
timeout=180
elapsed=0
while ! $DOCKER_CMD exec slurm-slurmctld sinfo --noheader &>/dev/null; do
    sleep 3
    elapsed=$((elapsed + 3))
    if [ "$elapsed" -ge "$timeout" ]; then
        err "Timed out waiting for slurmctld (${timeout}s). Check: $DOCKER_CMD logs slurm-slurmctld"
    fi
done
log "slurmctld is healthy."

# ---- Step 2: Generate SSH key pair ----

if [ ! -f "$SSH_KEY_PATH" ]; then
    log "Generating SSH key pair at $SSH_KEY_PATH..."
    mkdir -p "$(dirname "$SSH_KEY_PATH")"
    ssh-keygen -t ed25519 -f "$SSH_KEY_PATH" -N "" -C "slurm-docker-dev"
else
    log "SSH key already exists at $SSH_KEY_PATH."
fi

# ---- Step 3: Inject public key into slurmctld ----

log "Injecting SSH public key into slurmctld..."
PUB_KEY="$(cat "${SSH_KEY_PATH}.pub")"
$DOCKER_CMD exec slurm-slurmctld bash -c "
    mkdir -p /root/.ssh && chmod 700 /root/.ssh
    grep -qF '${PUB_KEY}' /root/.ssh/authorized_keys 2>/dev/null || \
        echo '${PUB_KEY}' >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
"

# Verify SSH connectivity
log "Verifying SSH connectivity to slurmctld..."
ssh_ok=false
for i in $(seq 1 10); do
    if ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
           -o ConnectTimeout=5 -i "$SSH_KEY_PATH" \
           -p "$SLURM_SSH_PORT" root@localhost sinfo --noheader &>/dev/null; then
        ssh_ok=true
        break
    fi
    sleep 2
done
if [ "$ssh_ok" = false ]; then
    err "SSH to slurmctld failed after 10 attempts. Check SSH config and port $SLURM_SSH_PORT."
fi
log "SSH connectivity verified."

# ---- Step 4: Configure SkyPilot ----

log "Configuring SkyPilot for SLURM cluster..."
mkdir -p "$(dirname "$SKY_CONFIG")"

# Build the SLURM stanza for ~/.sky/config.yaml using Python for safe YAML handling
python3 - "$SKY_CONFIG" "$SSH_KEY_PATH" "$SLURM_SSH_PORT" <<'PYEOF'
import sys, os

try:
    import yaml
except ImportError:
    # Fallback: install PyYAML if missing
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pyyaml"])
    import yaml

config_path = sys.argv[1]
ssh_key     = sys.argv[2]
ssh_port    = int(sys.argv[3])

# Load existing config or start fresh
if os.path.exists(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
else:
    config = {}

# Define the SLURM cluster configuration for SkyPilot
slurm_cluster = {
    "name": "slurm-docker",
    "ips": ["localhost"],
    "auth": {
        "ssh_user": "root",
        "ssh_private_key": ssh_key,
    },
    "ssh_port": ssh_port,
    "python": "/usr/bin/python3",
}

# Place under allowed_clouds or slurm section depending on SkyPilot version.
# SkyPilot >= 0.7 uses 'allowed_clouds' list and 'slurm' key.
if "slurm" not in config:
    config["slurm"] = {}
config["slurm"]["cluster"] = slurm_cluster

with open(config_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"Wrote SLURM config to {config_path}")
PYEOF

log "SkyPilot config written to $SKY_CONFIG."

# ---- Step 5: Verify cluster health ----

log "Verifying SLURM cluster health..."
NODE_COUNT=$(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -i "$SSH_KEY_PATH" -p "$SLURM_SSH_PORT" root@localhost \
    sinfo --noheader -N 2>/dev/null | wc -l)

if [ "$NODE_COUNT" -lt 2 ]; then
    warn "Expected 2 compute nodes but found $NODE_COUNT. Cluster may still be starting."
    warn "Run: ssh -i $SSH_KEY_PATH -p $SLURM_SSH_PORT root@localhost sinfo"
else
    log "SLURM cluster is ready: $NODE_COUNT compute nodes."
fi

# Show cluster status
echo ""
log "Cluster status:"
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -i "$SSH_KEY_PATH" -p "$SLURM_SSH_PORT" root@localhost sinfo

echo ""
log "Setup complete."
log ""
log "Quick reference:"
log "  SSH to login node:   ssh -i $SSH_KEY_PATH -p $SLURM_SSH_PORT root@localhost"
log "  Run sinfo:           ssh -i $SSH_KEY_PATH -p $SLURM_SSH_PORT root@localhost sinfo"
log "  Submit a test job:   ssh -i $SSH_KEY_PATH -p $SLURM_SSH_PORT root@localhost sbatch --wrap 'hostname'"
log "  SkyPilot check:      sky check"
log "  Teardown:            bash $SCRIPT_DIR/teardown-slurm.sh"
