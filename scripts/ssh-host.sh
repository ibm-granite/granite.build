#!/bin/bash
# SSH directly to the host VM (not the container) for a SkyPilot cluster.
# Usage: ./scripts/ssh-host.sh <cluster-name> [command...]

set -e

CLUSTER_NAME="${1:?Usage: $0 <cluster-name> [command...]}"
shift
COMMAND="$*"

SSH_CONFIG="$HOME/.sky/generated/ssh/$CLUSTER_NAME"

if [ ! -f "$SSH_CONFIG" ]; then
    echo "ERROR: SSH config not found: $SSH_CONFIG"
    exit 1
fi

HOST_IP=$(grep -o 'ProxyCommand.*[0-9]\+\.[0-9]\+\.[0-9]\+\.[0-9]\+' "$SSH_CONFIG" | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+\.[0-9]\+')
KEY_PATH=$(grep 'IdentityFile' "$SSH_CONFIG" | head -1 | awk '{print $2}')

if [ -z "$HOST_IP" ]; then
    echo "ERROR: Could not extract host IP from $SSH_CONFIG"
    exit 1
fi

if [ -z "$KEY_PATH" ]; then
    echo "ERROR: Could not extract SSH key path from $SSH_CONFIG"
    exit 1
fi

if [ -n "$COMMAND" ]; then
    ssh -i "$KEY_PATH" -p 22 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@"$HOST_IP" "$COMMAND"
else
    echo "Cluster:  $CLUSTER_NAME"
    echo "Host IP:  $HOST_IP"
    echo "SSH Key:  $KEY_PATH"
    echo "---"
    echo "Connecting to host (ubuntu@$HOST_IP:22)..."
    echo ""
    ssh -i "$KEY_PATH" -p 22 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@"$HOST_IP"
fi
