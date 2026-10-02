#!/usr/bin/env bash
# setup-kind.sh -- Bring up a local kind Kubernetes cluster for SkyPilot.
#
# The Kubernetes counterpart of scripts/slurm/setup-slurm.sh: it stands up the
# local infrastructure the skypilot/kubernetes build tests need, and those tests
# self-skip unless it is up (libgbtest.kube.kind_cluster_reachable).
#
# This script:
#   1. Checks the prerequisites up front, reporting all of them at once rather
#      than one per run: what SkyPilot's create_cluster.sh needs (docker, kind,
#      kubectl — warning on Rancher Desktop's shim), plus socat and GNU netcat,
#      without which SkyPilot leaves the Kubernetes cloud disabled
#   2. Runs `sky local up --no-gpus`, which creates (or reuses) the kind cluster
#   3. Verifies the API server answers, runs `sky check kubernetes` (sky local up
#      skips it when the cluster already exists) and fails unless Kubernetes is
#      enabled, then prints each node's allocatable CPU and memory, which is what
#      a fixture's resources have to fit
#
# CONTEXT SWITCH. `sky local up` makes kind-<name> the CURRENT kube context. That
# is deliberate and is not undone here: SkyPilot launches into the current
# context, so restoring a previous one would send the next build to whatever
# cluster that was. The tests refuse to run unless the current context is the
# kind one. Switch back yourself (`kubectl config use-context <ctx>`) when done,
# or run scripts/kube/teardown-kind.sh.
#
# Usage:
#   bash scripts/kube/setup-kind.sh
#
# Environment variables:
#   KIND_CLUSTER_NAME - kind cluster name (default: skypilot, SkyPilot's own
#                       default; the kube context is kind-<name>). The tests read
#                       the context from GBTEST_SKY_KUBE_CONTEXT, so set that to
#                       kind-<name> when overriding this.
#
# Podman works. SkyPilot's create_cluster.sh calls `docker info` unconditionally,
# which podman's `docker` shim answers, and kind detects podman itself ("enabling
# experimental podman provider"; KIND_EXPERIMENTAL_PROVIDER=podman forces it). kind
# on podman needs a ROOTFUL machine (`podman machine set --rootful`). Verified on
# macOS with podman 5.5 + kind 0.33. On podman this script also lifts the node
# container's default process cap (see "Podman: lift the node container's process
# cap" below), which otherwise limits every pod to 307 processes and threads.
#
# Idempotent: re-running with the cluster already up reuses it.

set -euo pipefail

KIND_CLUSTER_NAME="${KIND_CLUSTER_NAME:-skypilot}"
CONTEXT="kind-${KIND_CLUSTER_NAME}"

# ---- 1. Prerequisites ----
missing=()
command -v sky >/dev/null 2>&1 ||
  missing+=("sky: activate the repo-root .venv (make venv) — skypilot[kubernetes] is a base dependency")
if ! command -v docker >/dev/null 2>&1; then
  missing+=("docker: install Docker Desktop, Rancher Desktop or Colima")
elif ! docker info >/dev/null 2>&1; then
  missing+=("docker: the CLI is installed but the daemon is not reachable — start it")
fi
command -v kind >/dev/null 2>&1 ||
  missing+=("kind: https://kind.sigs.k8s.io/docs/user/quick-start/#installation (e.g. brew install kind)")
# Present is not enough, and neither is `version --client`. Rancher Desktop's
# kubectl is a kuberlr shim that, on first CONTACT with a cluster, fetches a client
# matching the server version — from a retired bucket
# (storage.googleapis.com/kubernetes-release) that now 404s unless a compatible
# kubectl is already available to it. A client-only call never contacts a server,
# so it passes even when the shim is about to fail; the failure surfaces only once
# the kind cluster exists. Nothing before the cluster can test for that, so warn on
# the shim here and name it again if the readiness check below fails.
kubectl_hint=""
if ! command -v kubectl >/dev/null 2>&1; then
  missing+=("kubectl: https://kubernetes.io/docs/tasks/tools/")
elif ! kubectl version --client >/dev/null 2>&1; then
  missing+=("kubectl: '$(command -v kubectl)' is installed but does not run")
else
  case "$(command -v kubectl) -> $(readlink "$(command -v kubectl)" 2>/dev/null || true)" in
    *'.rd/bin'*|*'Rancher Desktop'*)
      kubectl_hint="kubectl resolves to Rancher Desktop's kuberlr shim ($(command -v kubectl)), which can fail fetching a client for a new cluster version; put a real kubectl first on PATH (brew install kubectl)"
      echo "setup-kind: WARNING ${kubectl_hint}" >&2 ;;
  esac
fi
# SkyPilot reaches pods through `kubectl port-forward` and needs socat plus GNU
# netcat for it; without them `sky check` reports Kubernetes disabled and every
# launch fails. The nc test mirrors SkyPilot's own
# (check_port_forward_mode_dependencies): macOS's BSD nc exits 1 on -h and names
# apple, and it runs whichever nc is FIRST on PATH, so a brew netcat must shadow
# /usr/bin/nc.
socat -V >/dev/null 2>&1 ||
  missing+=("socat: brew install socat (or apt install socat)")
nc_rc=0
nc_err="$(nc -h 2>&1 >/dev/null)" || nc_rc=$?
if ! command -v nc >/dev/null 2>&1; then
  missing+=("netcat: brew install netcat (or apt install netcat)")
elif [ "$nc_rc" -eq 1 ] && printf '%s' "$nc_err" | grep -qi apple; then
  missing+=("netcat: '$(command -v nc)' is macOS's BSD nc; SkyPilot needs GNU netcat — brew install netcat, and make sure it is ahead of /usr/bin on PATH")
elif [ "$nc_rc" -ne 0 ]; then
  missing+=("netcat: '$(command -v nc) -h' failed (exit $nc_rc)")
fi

if [ "${#missing[@]}" -gt 0 ]; then
  echo "setup-kind: missing prerequisites:" >&2
  for m in "${missing[@]}"; do
    echo "  * $m" >&2
  done
  exit 1
fi

previous_context="$(kubectl config current-context 2>/dev/null || true)"

# ---- 2. Create (or reuse) the cluster ----
echo "setup-kind: running 'sky local up --no-gpus --name ${KIND_CLUSTER_NAME}'"
sky local up --no-gpus --name "${KIND_CLUSTER_NAME}"

# ---- 3. Verify ----
current_context="$(kubectl config current-context 2>/dev/null || true)"
if [ "$current_context" != "$CONTEXT" ]; then
  echo "setup-kind: ERROR expected current kube context '${CONTEXT}', got '${current_context}'" >&2
  exit 1
fi

if ! kubectl --context "$CONTEXT" get --raw /readyz --request-timeout=10s >/dev/null; then
  echo "setup-kind: ERROR the API server for '${CONTEXT}' is not ready" >&2
  [ -z "$kubectl_hint" ] || echo "setup-kind: likely cause: ${kubectl_hint}" >&2
  exit 1
fi

# ---- Podman: lift the node container's process cap ----
# Podman starts every container with pids_limit=2048 (Docker sets none), and kind
# passes no limit of its own nor offers a config field for one. systemd inside the
# kind node then caps every unit and pod at DefaultTasksMax = 15% of that: 307.
# SkyPilot's runtime alone uses ~280 of those per pod, so anything multi-threaded
# in a step fails with EAGAIN — measured: the HF xet download in the inline hf
# pull panics "failed to spawn thread", peaking at exactly 307.
#
# Fixed per node rather than in podman's containers.conf, so other containers keep
# podman's default. `podman update` lifts the cap live, but systemd computes
# DefaultTasksMax only at boot, so the node is restarted. A restart changes the
# node container's IP; on this single-node cluster the API server and node object
# follow it (verified: Ready in ~10s, InternalIP updated).
if command -v podman >/dev/null 2>&1; then
  restarted=false
  for node in $(kind get nodes --name "${KIND_CLUSTER_NAME}" 2>/dev/null); do
    limit="$(podman inspect "$node" --format '{{.HostConfig.PidsLimit}}' 2>/dev/null || true)"
    case "$limit" in
      ""|-1|0) ;;  # not a podman container, or already unlimited
      *)
        echo "setup-kind: podman node '${node}' has pids_limit=${limit} (pods would get 15%); lifting it"
        podman update --pids-limit -1 "$node" >/dev/null
        podman restart "$node" >/dev/null
        restarted=true ;;
    esac
  done
  if [ "$restarted" = true ]; then
    ready=false
    for _ in $(seq 1 60); do
      if kubectl --context "$CONTEXT" get --raw /readyz --request-timeout=3s >/dev/null 2>&1; then
        ready=true
        break
      fi
      sleep 2
    done
    if [ "$ready" != true ] ||
      ! kubectl --context "$CONTEXT" wait --for=condition=Ready nodes --all --timeout=180s >/dev/null; then
      echo "setup-kind: ERROR the cluster did not come back Ready after restarting its podman node(s)." >&2
      echo "setup-kind: recreate it: bash scripts/kube/teardown-kind.sh && bash scripts/kube/setup-kind.sh" >&2
      exit 1
    fi
    echo "setup-kind: podman node(s) restarted without a process cap; cluster Ready"
  fi
fi

# `sky local up` runs `sky check` only when it CREATES the cluster, so on a re-run
# nothing has confirmed SkyPilot can use it. Run it here and require the cloud to
# be enabled; a disabled cloud otherwise surfaces as a failed launch much later.
check_out="$(sky check kubernetes 2>&1 || true)"
if ! printf '%s' "$check_out" | sed $'s/\033\\[[0-9;]*m//g' | grep -q "Kubernetes: enabled"; then
  echo "setup-kind: ERROR 'sky check kubernetes' did not report Kubernetes enabled:" >&2
  printf '%s\n' "$check_out" >&2
  exit 1
fi

echo "setup-kind: nodes (allocatable):"
kubectl --context "$CONTEXT" get nodes \
  -o custom-columns='NAME:.metadata.name,CPU:.status.allocatable.cpu,MEMORY:.status.allocatable.memory'

if [ -n "$previous_context" ] && [ "$previous_context" != "$CONTEXT" ]; then
  echo "setup-kind: NOTE current kube context switched from '${previous_context}' to '${CONTEXT}'."
  echo "setup-kind: SkyPilot launches into the current context; switch back with"
  echo "setup-kind:   kubectl config use-context ${previous_context}"
fi

echo "setup-kind: ready — skypilot/kubernetes builds now launch into '${CONTEXT}'."
