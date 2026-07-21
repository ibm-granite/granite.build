---
name: run-gbserver
description: Bring up the Granite.build standalone gbserver (which serves the bundled gbmcp MCP tools at /mcp) and confirm the mcp__gbmcp__* tools are available. Use when asked to run or set up gbserver / the MCP server, or when the gbmcp tools aren't responding.
allowed-tools: Bash(make *) Bash(source *) Bash(gbserver *) Bash(pgrep *) Bash(curl *) mcp__gbmcp__info_health mcp__gbmcp__build_start mcp__gbmcp__build_status mcp__gbmcp__build_log mcp__gbmcp__build_list mcp__gbmcp__build_describe mcp__gbmcp__build_job_log
---

# Run gbserver — it serves the gbmcp MCP tools at `/mcp`

`gbserver standalone` serves the REST API and the bundled **gbmcp** MCP endpoint at `/mcp` on one port. Start gbserver and the `mcp__gbmcp__*` tools come alive — you drive builds through them (`build_start`, `build_status`, `build_job_log`, …). If `info_health()` already returns, gbserver is up — just use the tools.

## 1. Start gbserver (a SessionStart hook usually does this for you)

This project ships a **SessionStart hook** that auto-starts gbserver, so it's often already up or coming up. **Check first** — don't double-start:
```
pgrep -f "gbserver standalone --port ${GBSERVER_PORT:-8080}" >/dev/null && echo "already up — reuse it"
```

If it's not running, start it. **Install once** (only if `gbserver` isn't available):
- *Repo checkout:* `make standalone-venv PYTHON=python3.12` builds gbserver + gbmcp from the **current source** (so you run the code you're editing), then `source .venv/bin/activate`. Needs Python ≥3.11 (3.9 fails on `sqlite_database`).
- *pip install (no checkout):* `gbserver` is already on `PATH` — nothing to build (`pip install 'granite.build[standalone]'` if it isn't).

**Start it as a background process** (Bash tool with `run_in_background: true`; not `&`/`nohup`), honoring `GBSERVER_PORT` (don't hardcode 8080):
```
gbserver standalone --port ${GBSERVER_PORT:-8080} > /tmp/gbserver.log 2>&1
```
In a **checkout**, activate the venv and point at the in-repo space in the same call — prefix `source .venv/bin/activate &&` and add `--space-dir configurations/spaces/local`. In a **pip install** the packaged space is auto-discovered (no `--space-dir`). (If Bash calls are network-sandboxed per call, also set `dangerouslyDisableSandbox: true` so later calls can reach the port.)

## 2. Confirm the tools are available

1. **Endpoint up:** `curl -s -o /dev/null -w '%{http_code}\n' "http://127.0.0.1:${GBSERVER_PORT:-8080}/mcp/"` → **406** means up (a bare GET lacks the SSE `Accept` header). Poll until 406.
2. **Tools connected:** call `info_health()` (or any `mcp__gbmcp__*` tool). If it returns, you're set.

**If the endpoint is up (406) but no `mcp__gbmcp__*` tool will call, the MCP connection didn't attach this session — and you cannot force it yourself.** Claude Code binds MCP connections at session start:
- If gbmcp was *mid-connect* at launch, `WaitForMcpServers` waits for that attach to finish — try it once.
- But if **gbserver was down when the session started** (the usual case right after you start it), there was no connection attempt to wait on: `WaitForMcpServers` just keeps reporting "failed to connect," and starting the server now does **not** retroactively attach it. **Stop and tell the user to open `/mcp` and Retry the `gbmcp` server (or start a new session)** — that reconnect is a human action, the only way to attach a server that was dead at launch. Don't fall back to the `gb` CLI to do the build — once connected, stay on the MCP tools.

## Debugging

- A build's real stdout: `build_job_log(build_id)` — the on-disk `job.log` tail. Prefer it over `build_log` / `build_status` (status events only).
- gbserver startup issues: read `/tmp/gbserver.log`. For setup/troubleshooting docs, use the `gb-docs` skill.
