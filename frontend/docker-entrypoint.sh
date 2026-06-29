#!/bin/sh
set -e

# Start Python sidecar in background; exit container if it dies immediately
python -m gb_ui_backend &
SIDECAR_PID=$!

# Give it a moment to fail fast on misconfiguration
sleep 1
if ! kill -0 "$SIDECAR_PID" 2>/dev/null; then
  echo "ERROR: gb_ui_backend failed to start" >&2
  exit 1
fi

exec node_modules/.bin/next start --port 8000
