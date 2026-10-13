#!/usr/bin/env bash
# teardown-kind.sh -- Delete the local kind cluster created by setup-kind.sh.
#
# Runs `sky local down`, which deletes the kind cluster and, if its context was
# current, switches to the first remaining context in ~/.kube/config. Check which
# context that is before launching anything else (`kubectl config current-context`).
#
# Then undoes what builds on skypilot/kubernetes/local leave in the global SkyPilot
# config: gbserver deep-merges the env's cloud_config into ~/.sky/config.yaml and
# never removes keys, so `kubernetes.allowed_contexts: [kind-<name>]` (which limits
# all SkyPilot Kubernetes use on this machine to the kind cluster) and that
# context's `context_configs` entry would otherwise stay. Only those keys are
# removed; see sky_config_unpin.py.
#
# Usage:
#   bash scripts/kube/teardown-kind.sh
#
# Environment variables:
#   KIND_CLUSTER_NAME - kind cluster name (default: skypilot)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIND_CLUSTER_NAME="${KIND_CLUSTER_NAME:-skypilot}"

if ! command -v sky >/dev/null 2>&1; then
  echo "teardown-kind: sky not found — activate the repo-root .venv (make venv)" >&2
  exit 1
fi

sky local down --name "${KIND_CLUSTER_NAME}"

# The venv's python has PyYAML (a SkyPilot dependency).
python "${SCRIPT_DIR}/sky_config_unpin.py" "kind-${KIND_CLUSTER_NAME}"

echo "teardown-kind: current kube context is now '$(kubectl config current-context 2>/dev/null || echo none)'"
