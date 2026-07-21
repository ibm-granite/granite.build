#!/usr/bin/env bash
# Build the gbmcp Claude Code plugin from the in-repo skills + docs.
#
# The plugin is the OUT-OF-REPO / pip distribution of the agent integration: it
# bundles the (flavor-agnostic) skills, an offline docs snapshot generated from
# docs/, a project-scope .mcp.json that registers gbmcp at /mcp, and a
# SessionStart hook that auto-starts a pip-installed gbserver. Installing the
# plugin gives a user the whole experience (MCP registered + skills + auto-start)
# with no checkout.
#
# Single source of truth = the repo's .claude/skills + docs/. Re-run this whenever
# those change so the plugin never drifts. Output is written OUTSIDE the repo by
# default (it's a build artifact, not repo source).
#
# Usage: scripts/build-gbmcp-plugin.sh [TARGET_DIR]
#   TARGET_DIR default: <repo>/../gbmcp-plugin

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-$REPO/../gbmcp-plugin}"
SKILLS_SRC="$REPO/.claude/skills"
DOCS_SRC="$REPO/docs"

echo "Building gbmcp plugin → $TARGET"
rm -rf "$TARGET"
mkdir -p "$TARGET/skills" "$TARGET/.claude-plugin" "$TARGET/hooks" "$TARGET/scripts"

# 1. Skills — copied verbatim (they're flavor-agnostic: they cover both a repo
#    checkout and a pip install, so they need no per-flavor rewrite).
for s in run-gbserver create-build create-step gb-docs; do
  cp -r "$SKILLS_SRC/$s" "$TARGET/skills/$s"
done

# 2. gb-docs offline snapshot — GENERATED from docs/ (never hand-maintained), so
#    a pip user with no checkout still has the docs. gb-docs reads references/
#    first, so an index path docs/<x> maps to references/<x>.
rm -rf "$TARGET/skills/gb-docs/references"
rsync -a --exclude '.git' "$DOCS_SRC/" "$TARGET/skills/gb-docs/references/"

# 3. plugin.json
cat > "$TARGET/.claude-plugin/plugin.json" <<'JSON'
{
  "name": "gbmcp",
  "version": "0.1.0",
  "description": "Granite.build MCP tools + agent skills for driving standalone gbserver builds from Claude Code. Bundles the gbmcp MCP registration (/mcp), the create-build / create-step / run-gbserver / gb-docs skills, an offline docs snapshot, and a SessionStart hook that auto-starts gbserver.",
  "author": {
    "name": "IBM Granite.build",
    "url": "https://github.com/ibm-granite/granite.build"
  },
  "homepage": "https://github.com/ibm-granite/granite.build",
  "keywords": ["granite.build", "gbserver", "mcp", "llm-build", "standalone"]
}
JSON

# 4. .mcp.json — registers the mounted gbmcp endpoint (auto-loaded from root).
cat > "$TARGET/.mcp.json" <<'JSON'
{
  "mcpServers": {
    "gbmcp": {
      "type": "http",
      "url": "http://127.0.0.1:${GBSERVER_PORT:-8080}/mcp/"
    }
  }
}
JSON

# 5. SessionStart hook config (auto-loaded from hooks/hooks.json).
cat > "$TARGET/hooks/hooks.json" <<'JSON'
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup",
        "hooks": [
          {
            "type": "command",
            "command": "\"${CLAUDE_PLUGIN_ROOT}/scripts/start-gbserver.sh\"",
            "timeout": 20,
            "statusMessage": "Ensuring gbserver (gbmcp /mcp) is up..."
          }
        ]
      },
      {
        "matcher": "resume",
        "hooks": [
          {
            "type": "command",
            "command": "\"${CLAUDE_PLUGIN_ROOT}/scripts/start-gbserver.sh\"",
            "timeout": 20,
            "statusMessage": "Ensuring gbserver (gbmcp /mcp) is up..."
          }
        ]
      }
    ]
  }
}
JSON

# 6. The hook script (PIP flavor: gbserver is expected on PATH via
#    `pip install 'granite.build[standalone]'` — no make, no checkout, no
#    --space-dir since the packaged space is auto-discovered). Detaches so the
#    hook returns immediately and the daemon survives the session.
cat > "$TARGET/scripts/start-gbserver.sh" <<'SH'
#!/usr/bin/env bash
# SessionStart hook (gbmcp plugin): ensure a standalone gbserver — which serves
# the gbmcp MCP endpoint at /mcp — is running, so the plugin's .mcp.json connects
# at session start (a server down at launch is marked "failed" and can only be
# revived by a human /mcp Retry or a new session).
#
# Returns IMMEDIATELY: re-execs itself detached (setsid+nohup) and exits, so it
# never blocks session startup. The daemon keeps running after the hook returns
# AND after Claude exits, so the NEXT session finds it already up (no race).
# First run: if gbserver isn't up before Claude's MCP client gives up (~7s), run
# /mcp and Retry the gbmcp server (or start a new session).
set -u
PORT="${GBSERVER_PORT:-8080}"
LOG="${TMPDIR:-/tmp}/gbserver-standalone-${PORT}.log"

if [ "${_GB_HOOK_DETACHED:-}" != "1" ]; then
  pgrep -f "gbserver standalone --port ${PORT}" >/dev/null 2>&1 && exit 0
  _GB_HOOK_DETACHED=1 setsid nohup "$0" </dev/null >"$LOG" 2>&1 &
  exit 0
fi

# ---- detached worker (independent of Claude's shell/session) ----
if ! command -v gbserver >/dev/null 2>&1; then
  echo "start-gbserver: gbserver not on PATH — install with: pip install 'granite.build[standalone]'"
  exit 1
fi
pgrep -f "gbserver standalone --port ${PORT}" >/dev/null 2>&1 && exit 0
echo "start-gbserver: launching gbserver standalone on port ${PORT}..."
exec gbserver standalone --port "${PORT}"
SH
chmod +x "$TARGET/scripts/start-gbserver.sh"

echo "Done. Plugin layout:"
find "$TARGET" -maxdepth 2 -not -path '*/references/*' | sort
echo "gb-docs references generated: $(find "$TARGET/skills/gb-docs/references" -type f | wc -l | tr -d ' ') files (docs/: $(find "$DOCS_SRC" -type f | wc -l | tr -d ' '))"
