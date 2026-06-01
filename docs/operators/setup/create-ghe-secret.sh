#!/usr/bin/env bash
# Create a Kubernetes secret for GitHub Enterprise credentials in the SkyPilot namespace.
#
# Usage:
#   export GHE_TOKEN=ghp_xxxxx   # or the script reads from your environment
#   bash docs/setup/create-ghe-secret.sh
#
# The secret is created as:
#   Name:      ghe-credentials
#   Namespace: $SKYPILOT_NAMESPACE (default: skypilot)
#   Key:       token
#
# Idempotent — skips if the secret already exists.

set -euo pipefail

NAMESPACE="${SKYPILOT_NAMESPACE:-skypilot}"
SECRET_NAME="${GHE_SECRET_NAME:-ghe-credentials}"
SECRET_KEY="token"

if [[ -z "${GHE_TOKEN:-}" ]]; then
    echo "ERROR: GHE_TOKEN environment variable is not set."
    echo ""
    echo "Set it with:  export GHE_TOKEN=ghp_xxxxx"
    echo "Then re-run:  bash $0"
    exit 1
fi

if kubectl get secret "${SECRET_NAME}" -n "${NAMESPACE}" &>/dev/null; then
    echo "SKIP: Secret '${SECRET_NAME}' already exists in namespace '${NAMESPACE}'"
    echo "To recreate it, delete first:  kubectl delete secret ${SECRET_NAME} -n ${NAMESPACE}"
    exit 0
fi

kubectl create secret generic "${SECRET_NAME}" \
    --namespace="${NAMESPACE}" \
    --from-literal="${SECRET_KEY}=${GHE_TOKEN}"

echo "CREATED: Secret '${SECRET_NAME}' in namespace '${NAMESPACE}' (key: ${SECRET_KEY})"
