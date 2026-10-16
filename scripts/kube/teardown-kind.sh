#!/usr/bin/env bash
# teardown-kind.sh -- Delete the local kind cluster created by setup-kind.sh.
#
# Runs `sky local down`, which deletes the kind cluster and, if its context was
# current, switches to the first remaining context in ~/.kube/config. Check which
# context that is before launching anything else (`kubectl config current-context`).
#
# Then removes the cloud_config gbserver last wrote into the global SkyPilot config
# (skypilot_config.release_cloud_config). After a build on skypilot/kubernetes/local
# that is `kubernetes.allowed_contexts: [kind-<name>]`, which limits all SkyPilot
# Kubernetes use on this machine to the kind cluster, and that context's
# `context_configs` entry. gbserver's next build on another environment would
# replace them anyway (last writer wins); this covers plain `sky` use until then.
# Keys gbserver never wrote, and keys edited since, stay.
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

python -c 'from gbserver.environment.skypilot_config import release_cloud_config as r
print("teardown-kind: removed gbserver'"'"'s cloud_config from ~/.sky/config.yaml" if r() else "teardown-kind: no gbserver cloud_config in ~/.sky/config.yaml")'

echo "teardown-kind: current kube context is now '$(kubectl config current-context 2>/dev/null || echo none)'"
