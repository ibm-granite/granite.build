#!/bin/bash
# End-to-end demo: TRL fine-tuning + unitxt evaluation via SkyPilot on SLURM
#
# Quick start:
#   1. make g4os-skypilot-venv PYTHON=python3.13
#   2. source .venv/bin/activate
#   3. make minio-setup
#   4. make slurm-setup
#   5. bash scripts/demo-slurm.sh
#
# This script runs the same workloads as scripts/demo-standalone.sh but on
# a Docker SLURM cluster via SkyPilot instead of Docker containers.
#
# Requirements:
#   - Docker SLURM cluster running (make slurm-setup)
#   - MinIO running for artifact storage (make minio-setup, or auto-started)
#   - SkyPilot configured for SLURM (sky check shows Slurm: enabled)
#
# Usage:
#   bash scripts/demo-slurm.sh
#   bash scripts/demo-slurm.sh --trl-only
#   bash scripts/demo-slurm.sh --unitxt-only

set -euo pipefail

# --- Parse arguments ---
RUN_TRL=true
RUN_UNITXT=true
for arg in "$@"; do
    case "$arg" in
        --trl-only)   RUN_UNITXT=false ;;
        --unitxt-only) RUN_TRL=false ;;
        --help|-h)
            echo "Usage: bash scripts/demo-slurm.sh [--trl-only | --unitxt-only]"
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
LOG_FILE="/tmp/gbserver-slurm-demo-$(date +%Y%m%d-%H%M%S).log"
export GB_ENVIRONMENT=STANDALONE
export GBSERVER_HOST="http://127.0.0.1:${PORT}"
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-minioadmin}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-minioadmin}"

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

# --- 0. Pre-flight checks ---
step "Pre-flight checks"

if ! ssh -F ~/.slurm/config -o ConnectTimeout=5 slurm-docker sinfo --noheader &>/dev/null; then
    echo "ERROR: Docker SLURM cluster not reachable."
    echo "  Start it with: make slurm-setup"
    exit 1
fi
ok "SLURM cluster is running"

if ! sky check slurm 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -q "Slurm.*enabled"; then
    echo "ERROR: SkyPilot does not see SLURM."
    echo "  Check: sky check slurm"
    exit 1
fi
ok "SkyPilot SLURM enabled"

# Ensure MinIO is running (S3-compatible artifact store)
if ! curl -sf http://localhost:9000/minio/health/ready >/dev/null 2>&1; then
    step "Starting MinIO (S3 artifact store)"
    bash scripts/minio/setup-minio.sh
    ok "MinIO ready"
else
    ok "MinIO already running"
fi

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

# --- 3. Submit builds ---
TRL_BUILD="${SPACE_DIR}/builds/slurm-trl-cpu.yaml"
UNITXT_BUILD="${SPACE_DIR}/builds/slurm-unitxt-cpu.yaml"
TRL_ID=""
UNITXT_ID=""

if [ "$RUN_TRL" = true ]; then
    step "Submitting TRL fine-tuning build (granite-4.0-350m on SLURM)"
    TRL_OUTPUT=$(gb build start --skip-version-check --skip-validation \
        -f "${TRL_BUILD}" --format json -q 2>/dev/null)
    TRL_ID=$(echo "$TRL_OUTPUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['uuid'])")
    ok "TRL build submitted: ${TRL_ID}"
fi

if [ "$RUN_UNITXT" = true ]; then
    step "Submitting unitxt evaluation build (granite-4.0-350m on SLURM)"
    UNITXT_OUTPUT=$(gb build start --skip-version-check --skip-validation \
        -f "${UNITXT_BUILD}" --format json -q 2>/dev/null)
    UNITXT_ID=$(echo "$UNITXT_OUTPUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['uuid'])")
    ok "unitxt build submitted: ${UNITXT_ID}"
fi

# --- 4. Wait for completion with live event stream ---
# SLURM builds are slower due to setup phase (pip install), use longer timeout
step "Watching builds (SLURM setup phase may take 5-10 minutes on first run)"
python3 - "$TRL_ID" "$UNITXT_ID" "$API" <<'PYEOF'
import json, sys, time, urllib.request

trl_id, unitxt_id, api = sys.argv[1], sys.argv[2], sys.argv[3]
builds = {}
if trl_id:
    builds[trl_id] = "trl"
if unitxt_id:
    builds[unitxt_id] = "unitxt"
seen_events = {bid: 0 for bid in builds}

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

# Longer poll loop for SLURM (setup phase can be slow)
for iteration in range(300):
    statuses = {bid: get_status(bid) for bid in builds}
    elapsed = (iteration + 1) * 5

    # Status line
    parts = [f"{label}={statuses[bid]}" for bid, label in builds.items()]
    print(f"  [{elapsed:4d}s]  {'  '.join(parts)}")

    # Poll and display new events
    for bid, label in builds.items():
        poll_events(bid, label)

    if all(s not in ("submitted", "pending", "running", "?") for s in statuses.values()):
        break
    time.sleep(5)
PYEOF

# --- 5. Build status ---
if [ -n "$TRL_ID" ]; then
    step "Build status — TRL fine-tuning (SLURM)"
    gb build status --skip-version-check "$TRL_ID" 2>&1
fi

if [ -n "$UNITXT_ID" ]; then
    echo ""
    step "Build status — unitxt evaluation (SLURM)"
    gb build status --skip-version-check "$UNITXT_ID" 2>&1
fi

# --- 6. Artifacts (API) ---
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

# --- 7. Summary ---
echo ""
step "Demo complete (SLURM via SkyPilot)"
[ -n "$TRL_ID" ] && ok "TRL fine-tuning:    ${TRL_ID}"
[ -n "$UNITXT_ID" ] && ok "unitxt evaluation:  ${UNITXT_ID}"
echo ""
echo "  Server log: ${LOG_FILE}"
echo "  SLURM jobs: ssh -F ~/.slurm/config slurm-docker sacct"
