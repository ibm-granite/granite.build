# gbmcp — Granite.build MCP server (bundled, standalone)

`gbmcp` is the [FastMCP](https://github.com/jlowin/fastmcp) server that exposes Granite.build to AI agents (Claude Code, etc.). It is **bundled into the `granite.build` distribution** (this `src/gbmcp` package) and is **standalone-only**: it ships the tools that work against a local `gbserver standalone` backend and nothing else.

## How it's served: a stdio process the MCP client launches

gbmcp runs as a **stdio** server: the MCP client (Claude Code, via a project `.mcp.json`) spawns `gbmcp` as a subprocess and speaks JSON-RPC over stdio. There is **no port and no HTTP endpoint** in the normal flow — the client owns the process, so the tools are available as soon as the session starts, whether or not the backend is running.

- Entry point: `gbmcp = "gbmcp.server:main"` (`pyproject.toml`) runs `mcp.run(transport="stdio")` — stdio is the only transport.
- Backend: the build and secret tools call a **separate `gbserver standalone`** over REST at `GBSERVER_HOST` (set in `.mcp.json` `env`). gbmcp's launch does *not* start gbserver — bring it up with the `gbserver_start` tool (below). `build_job_log` reads the local `job.log` directly, so it needs no REST.
- Auth: **none.** gbmcp is built with no auth verifier and talks to an unauthenticated localhost gbserver (`get_github_token()` returns `None`).
- stdout is reserved for the JSON-RPC stream; all logs/banner go to stderr (`show_banner=False`).

## Managing the gbserver backend

Because gbmcp is a separate process, its tools respond even when gbserver is down — manage the backend through tools:

- `gbserver_status()` — is it running **and** reachable?
- `gbserver_start()` — launch `gbserver standalone` and wait until ready (idempotent). It finds the `gbserver` executable next to the running interpreter (same distribution), so the one-time install (`pip install 'granite.build[standalone]'`, or a checkout's `make standalone-venv`) is the only prerequisite.
- `gbserver_stop()` — stop it.

### Lifecycle — a long-lived dev service

`gbserver standalone` is **not just a build backend**: the same process also serves the **web dashboard** (`http://127.0.0.1:<port>`) and the REST/analytics API. So it's treated like a dev server, not a per-build worker:

- The agent **starts it on demand** (`gbserver_start`) and **leaves it running** after a build — so you can view results in the dashboard between runs, and the next build reuses the warm server.
- The agent **stops it only when you ask** (`gbserver_stop`); it never stops it unprompted (that would tear down a dashboard you may be viewing). When it wraps up, it tells you the server is still running (with the dashboard URL) and lets you say when to stop it.
- `gbserver_start` launches the server **detached** (`start_new_session`), so it **persists after the Claude Code session ends**, until `gbserver_stop` or a reboot. That's intentional (a dev service stays up); stop it for a clean slate.

## Monitoring a build

Ensure `gbserver_status()` is `ready`, then poll **`build_status(build_id)`**; done when `details.status` is `success` / `failed` / `cancelled` (lowercase; `submitted → pending → running → success`). Then `build_job_log(build_id)` for the output.

## The toolset (18, standalone-only)

| Group | Tools |
|---|---|
| **gbserver** | `gbserver_status`, `gbserver_start`, `gbserver_stop` |
| **Builds** | `build_start`, `build_list`, `build_status`, `build_describe`, `build_log`, `build_job_log`, `build_cancel` |
| **Secrets** | `secret_list`, `secret_get`, `secret_create`, `secret_update`, `secret_delete` |
| **Info** | `info_health`, `info_version`, `info_gb_version` |

`build_job_log` is the primary debugging tool in standalone — it returns the on-disk `job.log` (the workload's real stdout/stderr), since there is no gbserver REST file surface locally.

## Packaging (`pyproject.toml`)

- `[tool.setuptools.packages.find].include` includes `"gbmcp*"`.
- `[project.optional-dependencies]`: `mcp = ["fastmcp>=3.4,<4", "httpx>=0.27"]`. The `standalone` extra pulls `granite.build[mcp]`, so a standalone install includes gbmcp by default.
- `[project.scripts]`: `gbmcp = "gbmcp.server:main"` (stdio).

Install: `pip install 'granite.build[standalone]'` (Python ≥3.11; the default 3.9 fails on `sqlite_database`), or in a checkout `make standalone-venv PYTHON=python3.13`.

## Run

In a repo checkout the root [`.mcp.json`](../../.mcp.json) registers `gbmcp` (stdio) for project scope — Claude Code auto-discovers it; approve it once. Then let the agent call `gbserver_start` to bring the backend up and drive builds. Nothing to start by hand, no endpoint to register, no reconnect.

To register manually elsewhere:
```bash
claude mcp add gbmcp \
  --env GB_ENVIRONMENT=STANDALONE --env GBSERVER_HOST=http://127.0.0.1:8080 \
  -- gbmcp
```
`GB_ENVIRONMENT=STANDALONE` is required (else gbcli freezes PROD defaults at import); `GBSERVER_HOST` / `GBSERVER_PORT` point the tools at the gbserver they drive.

## Test

```bash
# import smoke test (stdout must stay clean for stdio)
GB_ENVIRONMENT=STANDALONE python -c "import gbmcp.server; print('ok', hasattr(gbmcp.server, 'mcp'))"

# stdio handshake: initialize -> tools/list
GB_ENVIRONMENT=STANDALONE python - <<'EOF'
import json, subprocess
p = subprocess.Popen(["gbmcp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
send = lambda o: (p.stdin.write(json.dumps(o) + "\n"), p.stdin.flush())
send({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"t","version":"1"}}})
print("server:", json.loads(p.stdout.readline())["result"]["serverInfo"]["name"])
send({"jsonrpc":"2.0","method":"notifications/initialized"})
send({"jsonrpc":"2.0","id":2,"method":"tools/list"})
while (line := p.stdout.readline()):
    msg = json.loads(line)
    if msg.get("id") == 2:
        print("tools:", len(msg["result"]["tools"])); break
p.terminate()
EOF
```

## What's here vs. removed

gbmcp ships only the tools that work against a local gbserver. Deleted from the source (no backend in standalone, or meaningless here):

- **Non-standalone / niche build tools** — `build_lineage`, `build_validate`, `build_diff`, `build_events`, `build_status_batch`, `build_init`, `build_update`.
- **Remote/prod groups** — `docs_*`, `admin_log`, `artifact_*`, `template_*`, `step_*`, cross-build cache search (`build_leaderboard`/`search`/`compare`/`search_yaml`), `gb_dashboard` (`build_search_errors`/`get_ai_analysis`/`investigate`/`k8s_status`), `cos` (`build_check_cos_path`), `flight_plan` (`plan_*`), `sandbox_*`, gbserver-REST `build_files_*`.
- The GHE OAuth variant and client scripts (`server_oauth.py`, `client*.py`, `smoke_test.py`, `services/ghe_auth.py`).

The **gbserver lifecycle tools** (`gbserver_status/start/stop`) are central to the stdio model: gbmcp runs *outside* gbserver now, so starting/stopping the backend from a tool is safe and meaningful (they were removed when gbmcp was mounted *inside* gbserver, where `stop` would have killed its own host). There is no runtime tool pruning. Verified end-to-end with a real stdio `initialize` → `tools/list` handshake.
