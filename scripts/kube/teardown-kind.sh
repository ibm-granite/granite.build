#!/usr/bin/env bash
# teardown-kind.sh -- Delete the local kind cluster created by setup-kind.sh.
#
# Runs `sky local down`, which deletes the kind cluster and, if its context was
# current, switches to the first remaining context in ~/.kube/config. Check which
# context that is before launching anything else (`kubectl config current-context`).
#
# Usage:
#   bash scripts/kube/teardown-kind.sh
#
# Environment variables:
#   KIND_CLUSTER_NAME - kind cluster name (default: skypilot)

set -euo pipefail

KIND_CLUSTER_NAME="${KIND_CLUSTER_NAME:-skypilot}"

if ! command -v sky >/dev/null 2>&1; then
  echo "teardown-kind: sky not found — activate the repo-root .venv (make venv)" >&2
  exit 1
fi

sky local down --name "${KIND_CLUSTER_NAME}"

echo "teardown-kind: current kube context is now '$(kubectl config current-context 2>/dev/null || echo none)'"
