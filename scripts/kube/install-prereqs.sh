#!/usr/bin/env bash
# install-prereqs.sh -- Install the host tools setup-kind.sh needs beyond docker.
#
# Run by `make kube-setup` before setup-kind.sh, so `make extended-tests-setup`
# (and a developer's `make kube-setup`) works without hand-installing anything,
# the way slurm-setup needs only docker:
#
#   * kind, pinned and sha256-verified, into the repo venv's bin/ (no sudo). The
#     make targets activate the venv, so this kind shadows any other on PATH and
#     every run uses the version the tests are verified with.
#   * socat and GNU netcat, which SkyPilot needs for `kubectl port-forward`,
#     only when missing: with apt-get on Linux (non-interactive sudo, as on CI
#     runners) or Homebrew on macOS. macOS's own /usr/bin/nc is BSD netcat, which
#     SkyPilot rejects, so it counts as missing there.
#
# Anything it cannot install is left for setup-kind.sh's prerequisite check,
# which fails with an install hint. kubectl and docker are not installed here:
# both are part of a container toolchain the developer already chose.
#
# Usage:
#   bash scripts/kube/install-prereqs.sh
#
# Environment variables:
#   VENV - venv whose bin/ receives kind (default: <repo>/.venv)

set -euo pipefail

KIND_VERSION=v0.33.0
# From the release's published kind-<os>-<arch>.sha256sum files. Bumping
# KIND_VERSION means updating all four. (A case, not an associative array:
# macOS's /bin/bash is 3.2.)
kind_sha256() {
  case "$1" in
    linux-amd64) echo aee6151561422756b764a4ae28e7f44cda5af5a9eead3cc9985112b1de8d8e0d ;;
    linux-arm64) echo 20022bee6cfcd5086cb7234d218e3454e6090022f2a8f55d1fa7fcf42c3867a2 ;;
    darwin-amd64) echo 5a99f26f57246dc9319dd294803313197a0f34d33c525b3ea8b655db5916ece0 ;;
    darwin-arm64) echo 0c8c7dbe5e23594a198b786c4bc13dacc101fa6196b0cb0b23a1ca44e61f4b4f ;;
  esac
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${VENV:-${REPO_ROOT}/.venv}"

log() { echo "install-prereqs: $*"; }

# ---- kind ----
install_kind() {
  local os arch platform sha tmp
  case "$(uname -s)" in
    Linux) os=linux ;;
    Darwin) os=darwin ;;
    *) log "no pinned kind for $(uname -s); install kind yourself"; return 0 ;;
  esac
  case "$(uname -m)" in
    x86_64 | amd64) arch=amd64 ;;
    arm64 | aarch64) arch=arm64 ;;
    *) log "no pinned kind for $(uname -m); install kind yourself"; return 0 ;;
  esac
  platform="${os}-${arch}"
  sha="$(kind_sha256 "$platform")"

  if [ -x "${VENV}/bin/kind" ] &&
    "${VENV}/bin/kind" version 2>/dev/null | grep -q "kind ${KIND_VERSION} "; then
    return 0
  fi
  if [ ! -d "${VENV}/bin" ]; then
    log "no venv at ${VENV}; run 'make venv' first"
    return 1
  fi
  log "installing kind ${KIND_VERSION} (${platform}) into ${VENV}/bin"
  tmp="$(mktemp)"
  curl -fsSLo "$tmp" "https://kind.sigs.k8s.io/dl/${KIND_VERSION}/kind-${platform}"
  if command -v sha256sum >/dev/null 2>&1; then
    echo "${sha}  ${tmp}" | sha256sum -c - >/dev/null
  else
    echo "${sha}  ${tmp}" | shasum -a 256 -c - >/dev/null
  fi
  install -m 0755 "$tmp" "${VENV}/bin/kind"
  rm -f "$tmp"
}

# ---- socat + GNU netcat ----
# Same nc test as setup-kind.sh (and SkyPilot's check_port_forward_mode_dependencies).
gnu_nc_ok() {
  local rc=0 err
  command -v nc >/dev/null 2>&1 || return 1
  err="$(nc -h 2>&1 >/dev/null)" || rc=$?
  if [ "$rc" -eq 1 ] && printf '%s' "$err" | grep -qi apple; then
    return 1
  fi
  [ "$rc" -eq 0 ]
}

install_port_forward_tools() {
  local need_socat=0 need_nc=0
  socat -V >/dev/null 2>&1 || need_socat=1
  gnu_nc_ok || need_nc=1
  [ $((need_socat + need_nc)) -gt 0 ] || return 0

  case "$(uname -s)" in
    Linux)
      if command -v apt-get >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
        local pkgs=()
        [ "$need_socat" -eq 1 ] && pkgs+=(socat)
        # Ubuntu 24.04 has no `netcat` package candidate; netcat-openbsd's
        # `nc -h` exits 0, which SkyPilot's check accepts.
        [ "$need_nc" -eq 1 ] && pkgs+=(netcat-openbsd)
        log "installing ${pkgs[*]} with apt-get"
        sudo -n apt-get update -qq
        sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${pkgs[@]}"
      else
        log "socat/netcat missing and no non-interactive sudo apt-get; install them yourself"
      fi
      ;;
    Darwin)
      if command -v brew >/dev/null 2>&1; then
        local pkgs=()
        [ "$need_socat" -eq 1 ] && pkgs+=(socat)
        [ "$need_nc" -eq 1 ] && pkgs+=(netcat)
        log "installing ${pkgs[*]} with Homebrew"
        brew install "${pkgs[@]}"
      else
        log "socat/netcat missing and no Homebrew; install them yourself"
      fi
      ;;
  esac
}

install_kind
install_port_forward_tools
