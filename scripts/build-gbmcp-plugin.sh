#!/usr/bin/env bash
# Build the gbmcp Claude Code plugin from the in-repo skills + docs.
#
# The plugin is the OUT-OF-REPO / pip distribution of the agent integration: it
# bundles the (flavor-agnostic) skills, an offline docs snapshot generated from
# docs/, and a stdio .mcp.json that registers gbmcp. Installing the plugin gives
# a user the skills + the gbmcp tools with no checkout; the agent brings the
# gbserver backend up itself via the gbserver_start tool (no hook needed).
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
mkdir -p "$TARGET/skills" "$TARGET/.claude-plugin"

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
  "description": "Granite.build MCP tools + agent skills for driving standalone gbserver builds from Claude Code. Bundles the gbmcp stdio MCP server, the create-build / create-step / run-gbserver / gb-docs skills, and an offline docs snapshot.",
  "author": {
    "name": "IBM Granite.build",
    "url": "https://github.com/ibm-granite/granite.build"
  },
  "homepage": "https://github.com/ibm-granite/granite.build",
  "keywords": ["granite.build", "gbserver", "mcp", "llm-build", "standalone"]
}
JSON

# 4. .mcp.json — stdio launch of the pip-installed gbmcp (auto-loaded from root).
#    GB_ENVIRONMENT pins standalone (else gbcli freezes PROD at import); the port
#    is shared between the tools' GBSERVER_HOST and gbserver_start's launch port.
cat > "$TARGET/.mcp.json" <<'JSON'
{
  "mcpServers": {
    "gbmcp": {
      "type": "stdio",
      "command": "gbmcp",
      "env": {
        "GB_ENVIRONMENT": "STANDALONE",
        "GBSERVER_PORT": "${GBSERVER_PORT:-8080}",
        "GBSERVER_HOST": "http://127.0.0.1:${GBSERVER_PORT:-8080}"
      }
    }
  }
}
JSON

echo "Done. Plugin layout:"
find "$TARGET" -maxdepth 2 -not -path '*/references/*' | sort
echo "gb-docs references generated: $(find "$TARGET/skills/gb-docs/references" -type f | wc -l | tr -d ' ') files (docs/: $(find "$DOCS_SRC" -type f | wc -l | tr -d ' '))"
