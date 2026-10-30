# Image setup for the `mcpmark` family: MCPMark's harness and MCP-server
# runtimes (README.md). Everything fetched here is pinned by commit and/or
# sha256 and verified; the task STATE payloads (filesystem zips, postgres
# backups) are NOT baked -- the harness downloads them at run time from
# storage.mcpmark.ai (license unstated; granite_evals.benchmarks.mcpmark
# records digests of what it got).

# MCPMark (github.com/eval-sys/mcpmark, Apache-2.0) at the pinned commit.
# The codeload tarball's sha256 is provenance only (GitHub regenerates
# tarball bytes); the enforced pin is the digest of the extracted tree, the
# same scheme as tree_digest() in mcpmark/src/mcpmark.py -- which checks the
# .granite-commit marker written below, and whose tests check these pins
# agree with the module's (mcpmark/tests/test_mcpmark.py).
COMMIT=cd45b7f57923b9b3985467f5139927575f83141c
TARBALL_SHA256=cd1e67b1684e9cfc2354c023f1b3f61fc3d690af8dd5b69cd5e5fe1e6cf753c5
TREE_SHA256=f825fd368508329f7e51a1f512828e36ff996506fbe09207ab31449f0cde8c23

DEST=/opt/mcpmark/$COMMIT
mkdir -p "$DEST"
curl -fsSL --retry 5 -o /tmp/mcpmark.tar.gz \
    "https://codeload.github.com/eval-sys/mcpmark/tar.gz/$COMMIT"
echo "$TARBALL_SHA256  /tmp/mcpmark.tar.gz" | sha256sum -c -
tar -xzf /tmp/mcpmark.tar.gz -C "$DEST" --strip-components=1
rm -f /tmp/mcpmark.tar.gz

# The enforced pin: the extracted tree must digest to TREE_SHA256 (sha256 over
# every file's sorted posix relpath, NUL, and the file's own sha256 digest --
# the same function as tree_digest() in the module).
GOT=$(python3 - "$DEST" <<'EOF'
import hashlib, sys
from pathlib import Path
root = Path(sys.argv[1])
h = hashlib.sha256()
for p in sorted(root.rglob("*")):
    if p.is_file():
        h.update(p.relative_to(root).as_posix().encode())
        h.update(b"\0")
        h.update(hashlib.sha256(p.read_bytes()).digest())
print(h.hexdigest())
EOF
)
[ "$GOT" = "$TREE_SHA256" ] || { echo "mcpmark tree digest $GOT != $TREE_SHA256" >&2; exit 1; }
echo "$COMMIT" > "$DEST/.granite-commit"

# The harness runs from the checkout (its wheel packages only src/ and
# tasks/, not the repo-root pipeline.py) with its OWN venv: it hard-pins
# litellm==1.80.0 against the nemoskills/tbench camps (see [tool.uv]
# conflicts in src/core/pyproject.toml), and declares tooling (pixi, pipx)
# no other family wants. Upstream ships no pip lock (pixi.lock is conda), so
# the resolved set is frozen next to the venv for the record. UV_NO_CONFIG is
# set by the Dockerfile: this project's [tool.uv] settings must not leak here.
uv venv "$DEST/.venv"
uv pip install --python "$DEST/.venv/bin/python" "$DEST"
uv pip freeze --python "$DEST/.venv/bin/python" > "$DEST/granite-freeze.txt"

# Playwright's chromium for the playwright MCP server (upstream's Dockerfile
# installs it from both the Python and the npm side -- the revisions differ).
# PLAYWRIGHT_BROWSERS_PATH and NPM_CONFIG_CACHE are the Dockerfile's ENVs.
"$DEST/.venv/bin/playwright" install chromium
timeout 900 npx -y playwright install chromium

# postgres-mcp (the postgres MCP server, driven as `pipx run` by the harness)
# with mcp constrained <2: an unconstrained resolve installs mcp 2.0.0, which
# dropped mcp.server.fastmcp and fails every postgres task in ~3 s with
# "Connection closed" -- reading as a model failure. The constraint file is
# also exported (UV_CONSTRAINT/PIP_CONSTRAINT) by the module at run time, so
# even a cold re-resolve stays mcp<2. The `pipx run` smoke is the exact
# command the harness and the module's preflight use: it must answer --help.
export PIPX_HOME=/opt/pipx PIPX_BIN_DIR=/usr/local/bin
mkdir -p "$PIPX_HOME"
printf 'mcp<2\n' > "$PIPX_HOME/mcp_constraints.txt"
PIPX_CONSTRAINTS="$PIPX_HOME/mcp_constraints.txt"
UV_CONSTRAINT="$PIPX_CONSTRAINTS" PIP_CONSTRAINT="$PIPX_CONSTRAINTS" \
    "$DEST/.venv/bin/pipx" install postgres-mcp==0.3.0
UV_CONSTRAINT="$PIPX_CONSTRAINTS" PIP_CONSTRAINT="$PIPX_CONSTRAINTS" \
    "$DEST/.venv/bin/pipx" run postgres-mcp==0.3.0 --help > /dev/null

# Pre-warm the npx MCP servers the filesystem and playwright tasks drive, at
# the versions the harness pins (mcpmark_agent.py at the commit): the npm
# cache (NPM_CONFIG_CACHE) then serves them at run time. They speak MCP over
# stdio, so a closed stdin ends them; any exit is fine, the cache is the point.
timeout 300 npx -y @modelcontextprotocol/server-filesystem@2025.12.18 < /dev/null > /dev/null 2>&1 || true
timeout 300 npx -y @playwright/mcp@0.0.68 < /dev/null > /dev/null 2>&1 || true

# a+rX everywhere (the job user is not root); the two run-time download
# targets (a browser revision the pinned servers may want, npm's cache) stay
# writable so a cold path can self-heal instead of failing a task.
chmod -R a+rX /opt/mcpmark /opt/pipx
chmod -R a+rwX /opt/ms-playwright /opt/npm-cache
du -sh "$DEST" /opt/ms-playwright /opt/pipx /opt/npm-cache
