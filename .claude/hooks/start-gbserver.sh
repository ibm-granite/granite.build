#!/usr/bin/env bash
# SessionStart hook: ensure a standalone gbserver — which serves the bundled
# gbmcp MCP endpoint at /mcp — is running, so this repo's .mcp.json connects at
# session start instead of failing (a server that's down at launch is marked
# "failed" and can only be revived by a human /mcp Retry or a new session).
#
# The hook returns IMMEDIATELY: it re-execs itself fully detached (setsid+nohup,
# stdio redirected) and exits, so the slow first-run `make standalone-venv`
# (minutes) never blocks session startup. The detached gbserver keeps running
# after the hook returns AND after Claude exits, so the NEXT session finds it
# already up and connects with no race.
#
# First run only: the venv build won't finish before Claude's MCP client gives
# up (~7s), so the tools won't be there this session — run /mcp and Retry the
# gbmcp server (or start a new session). Every later session connects on launch.

set -u
PORT="${GBSERVER_PORT:-8080}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG="${TMPDIR:-/tmp}/gbserver-standalone-${PORT}.log"

# First (attached) invocation: bail if a server is already on this port, else
# relaunch self detached and return right away so the hook never blocks.
if [ "${_GB_HOOK_DETACHED:-}" != "1" ]; then
  pgrep -f "gbserver standalone --port ${PORT}" >/dev/null 2>&1 && exit 0
  _GB_HOOK_DETACHED=1 setsid nohup "$0" </dev/null >"$LOG" 2>&1 &
  exit 0
fi

# ---- detached worker (independent of Claude's shell/session) ----
cd "$REPO" || exit 1

# Build the standalone venv (gbserver + bundled gbmcp) if missing. Needs Python
# >=3.11 (the default 3.9 fails on sqlite_database).
if [ ! -x .venv/bin/gbserver ]; then
  PY=""
  for c in python3.13 python3.12 python3.11 python3; do
    if command -v "$c" >/dev/null 2>&1 \
        && "$c" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 11) else 1)' 2>/dev/null; then
      PY="$c"; break
    fi
  done
  [ -n "$PY" ] || { echo "start-gbserver: no Python >=3.11 found; install one, then \`make standalone-venv\`"; exit 1; }
  echo "start-gbserver: building standalone venv with $PY (first run; takes a few minutes)..."
  make standalone-venv PYTHON="$PY" || { echo "start-gbserver: make standalone-venv failed"; exit 1; }
fi

# Guard again in case a concurrent hook won the race while the venv built.
pgrep -f "gbserver standalone --port ${PORT}" >/dev/null 2>&1 && exit 0

# shellcheck disable=SC1091
source .venv/bin/activate
echo "start-gbserver: launching gbserver standalone on port ${PORT}..."
exec gbserver standalone --port "${PORT}" --space-dir configurations/spaces/local
