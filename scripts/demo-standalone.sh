#!/bin/bash
# End-to-end demo: TRL fine-tuning + unitxt evaluation via gbcli standalone
#
# Quick start:
#   1. make demo-venv PYTHON=python3.13
#   2. source .venv/bin/activate
#   3. bash scripts/demo-standalone.sh
#
# The script auto-detects GPU vs CPU and builds the container image on first
# run if it doesn't exist yet.  Docker or Podman is detected automatically.
#
# Requirements:
#   - Docker or Podman with a running daemon
#   - For macOS Podman: VM should have at least 4 GB of RAM
#     (check with: podman machine inspect --format '{{.Resources.Memory}}')
#     Fine-tuning is memory-intensive; use --unitxt-only on low-memory systems.
#
# Usage:
#   bash scripts/demo-standalone.sh              # run both builds
#   bash scripts/demo-standalone.sh --trl-only   # TRL fine-tuning only
#   bash scripts/demo-standalone.sh --unitxt-only # unitxt evaluation only
#   GBSERVER_DEMO_CPU=1 bash scripts/demo-standalone.sh  # force CPU path

set -euo pipefail

# --- Parse arguments ---
RUN_TRL=true
RUN_UNITXT=true
for arg in "$@"; do
    case "$arg" in
        --trl-only)   RUN_UNITXT=false ;;
        --unitxt-only) RUN_TRL=false ;;
        --help|-h)
            echo "Usage: bash scripts/demo-standalone.sh [--trl-only | --unitxt-only]"
            echo "  --trl-only    Run only the TRL fine-tuning build"
            echo "  --unitxt-only Run only the unitxt evaluation build"
            exit 0
            ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

PORT=${GBSERVER_PORT:-8080}
SPACE_DIR="test-data/standalone-environments"
API="http://127.0.0.1:${PORT}/api/v1"
LOG_FILE="/tmp/gbserver-demo-$(date +%Y%m%d-%H%M%S).log"
export GB_ENVIRONMENT=STANDALONE
export GBSERVER_HOST="http://127.0.0.1:${PORT}"

# --- Detect container CLI (docker or podman) ---
if command -v docker >/dev/null 2>&1; then
    CONTAINER_CLI="docker"
elif command -v podman >/dev/null 2>&1; then
    CONTAINER_CLI="podman"
else
    echo "ERROR: Neither docker nor podman found on PATH"
    exit 1
fi

# --- Check Podman VM memory (macOS only) ---
if [ "$CONTAINER_CLI" = "podman" ] && command -v podman >/dev/null 2>&1; then
    VM_MEM=$(podman machine inspect --format '{{.Resources.Memory}}' 2>/dev/null || echo "")
    if [ -n "$VM_MEM" ] && [ "$VM_MEM" -gt 0 ] 2>/dev/null; then
        VM_MEM_GB=$((VM_MEM / 1024))
        if [ "$VM_MEM_GB" -lt 4 ]; then
            echo -e "\033[1;33m  ⚠ Podman VM has ${VM_MEM_GB} GB of RAM (${VM_MEM} MiB).\033[0m"
            echo -e "\033[1;33m    Fine-tuning requires at least 4 GB and may be OOM-killed.\033[0m"
            echo -e "\033[1;33m    Consider using --unitxt-only, or resize with:\033[0m"
            echo -e "\033[1;33m      podman machine set --memory 4096\033[0m"
        fi
    fi
fi

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
DIM='\033[2m'
NC='\033[0m'

step() { echo -e "\n${BLUE}=== $1 ===${NC}"; }
ok()   { echo -e "${GREEN}  ✓ $1${NC}"; }
warn() { echo -e "${YELLOW}  ⚠ $1${NC}"; }
hint() { echo -e "${DIM}  tip: $1${NC}"; }

cleanup() {
    if [ -n "${SERVER_PID:-}" ]; then
        step "Stopping server (PID ${SERVER_PID})"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        ok "Server stopped"
    fi
    rm -f ~/.llmb/llmb-server.db ~/.llmb/llmb-server.db.lck
}
trap cleanup EXIT

# --- 1. Clean state ---
step "Preparing clean environment"
pkill -f "gbserver standalone" 2>/dev/null || true
sleep 1
rm -rf ~/.llmb && mkdir -p ~/.llmb
rm -rf outputs
ok "Clean state"

# --- 2. Start server ---
step "Starting gbserver standalone on port ${PORT}"
gbserver standalone --space-dir "$SPACE_DIR" --port "$PORT" > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

for i in $(seq 1 20); do
    if curl -m 2 -s "${API}/spaces/" >/dev/null 2>&1; then
        ok "Server ready (PID ${SERVER_PID})"
        hint "tail -f ${LOG_FILE}"
        break
    fi
    if [ "$i" -eq 20 ]; then
        echo "ERROR: Server failed to start. Check ${LOG_FILE}"
        exit 1
    fi
    sleep 1
done

# --- 2b. Detect GPU ---
# Set GBSERVER_DEMO_CPU=1 to force the CPU path even when a GPU is present.
DOCKER_DIR="test-data/standalone-environments/docker"
if [ "${GBSERVER_DEMO_CPU:-0}" = "1" ] || ! nvidia-smi -L >/dev/null 2>&1; then
    TRL_BUILD="${SPACE_DIR}/builds/docker-trl-cpu.yaml"
    UNITXT_BUILD="${SPACE_DIR}/builds/docker-unitxt-cpu.yaml"
    DEMO_IMAGE="gbserver-test-trl-unitxt:cpu"
    DEMO_DOCKERFILE="${DOCKER_DIR}/Dockerfile.cpu.test"
    warn "Using CPU image (slower)"
else
    TRL_BUILD="${SPACE_DIR}/builds/docker-trl.yaml"
    UNITXT_BUILD="${SPACE_DIR}/builds/docker-unitxt.yaml"
    DEMO_IMAGE="gbserver-test-trl-unitxt:gpu"
    DEMO_DOCKERFILE="${DOCKER_DIR}/Dockerfile.test"
    ok "GPU detected — using CUDA image"
fi

ok "Container CLI: ${CONTAINER_CLI}"

# --- 2c. Build container image if missing ---
if ! ${CONTAINER_CLI} image exists "${DEMO_IMAGE}" 2>/dev/null \
   && ! ${CONTAINER_CLI} image inspect "${DEMO_IMAGE}" >/dev/null 2>&1; then
    step "Building ${DEMO_IMAGE} (first run only — may take a few minutes)"
    ${CONTAINER_CLI} build -t "${DEMO_IMAGE}" -f "${DEMO_DOCKERFILE}" "${DOCKER_DIR}"
    ok "Image built: ${DEMO_IMAGE}"
else
    ok "Image found: ${DEMO_IMAGE}"
fi

# --- 3. Submit builds ---
TRL_ID=""
UNITXT_ID=""

if [ "$RUN_TRL" = true ]; then
    step "Submitting TRL fine-tuning build (granite-4.0-350m)"
    TRL_OUTPUT=$(gb build start --skip-version-check --skip-validation \
        -f "${TRL_BUILD}" --format json -q 2>/dev/null)
    TRL_ID=$(echo "$TRL_OUTPUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['uuid'])")
    ok "TRL build submitted: ${TRL_ID}"
    hint "${CONTAINER_CLI} logs -f \$(${CONTAINER_CLI} ps -q --filter name=gb-trl-finetune)"
fi

if [ "$RUN_UNITXT" = true ]; then
    step "Submitting unitxt evaluation build (granite-4.0-350m)"
    UNITXT_OUTPUT=$(gb build start --skip-version-check --skip-validation \
        -f "${UNITXT_BUILD}" --format json -q 2>/dev/null)
    UNITXT_ID=$(echo "$UNITXT_OUTPUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['uuid'])")
    ok "unitxt build submitted: ${UNITXT_ID}"
    hint "${CONTAINER_CLI} logs -f \$(${CONTAINER_CLI} ps -q --filter name=gb-unitxt-eval)"
fi

# --- 4. Wait for completion with live event stream ---
step "Watching builds"
python3 - "$TRL_ID" "$UNITXT_ID" "$API" "$CONTAINER_CLI" <<'PYEOF'
import json, os, subprocess, sys, time, urllib.request

trl_id, unitxt_id, api, container_cli = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
builds = {}
if trl_id:
    builds[trl_id] = "trl"
if unitxt_id:
    builds[unitxt_id] = "unitxt"
seen_events = {bid: 0 for bid in builds}
prev_status = {bid: "" for bid in builds}
shown_containers = set()

GREEN  = "\033[0;32m"
YELLOW = "\033[1;33m"
CYAN   = "\033[0;36m"
DIM    = "\033[2m"
BOLD   = "\033[1m"
NC     = "\033[0m"

def fetch(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.loads(r.read())
    except Exception:
        return None

def get_status(bid):
    d = fetch(f"{api}/builds/{bid}/status2")
    if not d:
        return "?"
    return d.get("status", {}).get("build", {}).get("status", "?")

def running_containers():
    """Return set of running gb- container names."""
    try:
        out = subprocess.check_output(
            [container_cli, "ps", "--filter", "name=gb-", "--format", "{{.Names}}"],
            timeout=3, stderr=subprocess.DEVNULL
        ).decode().strip()
        return set(out.split("\n")) if out else set()
    except Exception:
        return set()

def poll_events(bid, label):
    d = fetch(f"{api}/builds/{bid}/events")
    if not d:
        return
    events = d.get("events", [])
    new_events = events[seen_events[bid]:]
    seen_events[bid] = len(events)
    for ev in new_events:
        be = ev.get("build_event", {})
        etype = be.get("type", "")
        payload = be.get("payload", {})
        meta = be.get("run_metadata", {})
        target = meta.get("target_name", "")

        if etype == "newartifact_in_environment_event":
            binding_id = payload.get("binding_id", "?")
            binding = payload.get("binding", {})
            path = binding.get("path", "") if isinstance(binding, dict) else str(binding)
            print(f"  {DIM}[{label}]{NC} {YELLOW}new artifact:{NC} {binding_id}  path={path}")

        elif etype == "artifact_pushed_event":
            uri = payload.get("uri", "?")
            binding_id = payload.get("binding_id", "?")
            print(f"  {DIM}[{label}]{NC} {GREEN}artifact pushed:{NC} {binding_id} -> {uri}")

        elif etype == "artifact_event":
            uri = payload.get("uri", "?")
            binding_id = payload.get("binding_id", "?")
            print(f"  {DIM}[{label}]{NC} {CYAN}artifact registered:{NC} {binding_id} -> {uri}")

        elif etype == "status_event":
            status = payload.get("status", "?")
            msg_type = meta.get("type", "")
            if msg_type == "TargetStep":
                step_uri = meta.get("targetstep_uri", "")
                step_name = step_uri.split("/")[-1] if "/" in step_uri else step_uri
                print(f"  {DIM}[{label}]{NC} step {step_name}: {status}")
            elif target:
                print(f"  {DIM}[{label}]{NC} target {target}: {status}")

if not builds:
    print("  No builds to watch.")
    sys.exit(0)

for iteration in range(120):
    statuses = {bid: get_status(bid) for bid in builds}
    elapsed = (iteration + 1) * 3

    # Status line
    parts = [f"{label}={statuses[bid]}" for bid, label in builds.items()]
    print(f"  [{elapsed:3d}s]  {'  '.join(parts)}")

    # Detect new containers
    for cname in running_containers() - shown_containers:
        shown_containers.add(cname)
        print(f"         {CYAN}container started:{NC} {BOLD}{cname}{NC}")
        print(f"         {DIM}tip: {container_cli} logs -f {cname}{NC}")

    # Poll and display new events
    for bid, label in builds.items():
        poll_events(bid, label)

    if all(s not in ("submitted", "pending", "running", "?") for s in statuses.values()):
        break
    time.sleep(3)
PYEOF

# --- 5. List builds ---
step "Build list"
gb build list --skip-version-check --show-all 2>&1 | grep -E "BUILD_ID|docker|Showing"

# --- 6. Build status ---
if [ -n "$TRL_ID" ]; then
    step "Build status — TRL fine-tuning"
    gb build status --skip-version-check "$TRL_ID" 2>&1
fi

if [ -n "$UNITXT_ID" ]; then
    echo ""
    step "Build status — unitxt evaluation"
    gb build status --skip-version-check "$UNITXT_ID" 2>&1
fi

# --- 7. Artifacts (API) ---
step "Registered artifacts (via REST API)"
ARTIFACTS=$(curl -m 5 -s "${API}/artifacts/" 2>/dev/null)
echo "$ARTIFACTS" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    arts = data if isinstance(data, list) else data.get('artifacts', data.get('items', []))
    if not arts:
        print('  (no artifacts found)')
    for a in arts:
        name = a.get('name', '?')
        uri = a.get('uri', '?')
        status = a.get('status', '?')
        build_id = a.get('created_by_build_id', '')[:8]
        print(f'  {name:<20s} {status:<10s} {uri}  (build {build_id}...)')
except Exception as e:
    print(f'  (error parsing artifacts: {e})')
"

# --- 8. Summary ---
echo ""
step "Demo complete"
[ -n "$TRL_ID" ] && ok "TRL fine-tuning:    ${TRL_ID}"
[ -n "$UNITXT_ID" ] && ok "unitxt evaluation:  ${UNITXT_ID}"
echo ""
echo "  Server log: ${LOG_FILE}"
echo "  To inspect: gb build status --skip-version-check --show-events <build-id>"
