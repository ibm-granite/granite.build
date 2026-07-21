# gbmcp — Granite.build MCP server (bundled, standalone)

`gbmcp` is the [FastMCP](https://github.com/jlowin/fastmcp) server that exposes Granite.build to AI agents (Claude Code, etc.). It is **bundled into the `granite.build` distribution** (this `src/gbmcp` package) and is **standalone-only**: it ships the tools that work against a local `gbserver standalone` backend and nothing else.

## How it's served: mounted in `gbserver standalone`

`gbserver standalone` mounts the gbmcp FastMCP app at **`/mcp`** on its own port — one process, one port, `GB_ENVIRONMENT=STANDALONE`. There is **no separate `gbmcp` process** in the normal flow.

- Mount point: `src/gbserver/api/root_api.py` — guarded (`is_standalone()` + `try: import gbmcp.server`), mounted before the `/` static mount, its FastMCP lifespan driven via `on_event` startup/shutdown (fail-safe: a gbmcp hiccup degrades `/mcp` only, never crashes gbserver).
- Auth: **none.** gbserver's `AuthMiddleware` exempts `/mcp` (`_PUBLIC_PATH_PREFIXES` in `src/gbserver/api/auth.py`), and gbmcp is constructed with **no auth verifier** in standalone — so the client needs no `Authorization` header. (`get_github_token()` returns `None`; the local gbserver accepts unauthenticated localhost.)
- Backend target: `gbserver standalone` auto-sets `GBSERVER_HOST` to its own port (`command_standalone.py`) so the mounted tools' `GBClient` reaches it on any port (it otherwise defaults to `:8080`).
- A standalone `gbmcp` **console script** (`gbmcp.server:main`, HTTP on `GBMCP_PORT`, default 8000) also exists for running it as a separate process (point `GBSERVER_HOST` at its gbserver).

## Monitoring a build

Poll **`build_status(build_id)`**; done when `details.status` is `success` / `failed` / `cancelled` (lowercase; `submitted → pending → running → success`). Then `build_job_log(build_id)` for the output.

## The toolset (17, standalone-only)

| Group | Tools |
|---|---|
| **Builds** | `build_start`, `build_list`, `build_status`, `build_describe`, `build_log`, `build_job_log`, `build_cancel` |
| **Space** | `space_list` |
| **Secrets** | `secret_list`, `secret_get`, `secret_create`, `secret_update`, `secret_delete` |
| **Info** | `info_health`, `info_version`, `info_gb_version`, `info_gb_environment` |

`build_job_log` is the primary debugging tool in standalone — it returns the on-disk `job.log` (the workload's real stdout/stderr), since there is no gbserver REST file surface locally.

## Packaging (`pyproject.toml`)

- `[tool.setuptools.packages.find].include` includes `"gbmcp*"`.
- `[project.optional-dependencies]`: `mcp = ["fastmcp", "httpx>=0.27", "sqlalchemy[asyncio]>=2.0", "asyncpg>=0.29", "boto3"]`. The `standalone` extra pulls `granite.build[mcp]`, so a standalone install gets the MCP server by default; `[mcp]` on its own is the standalone-process option.
- `[project.scripts]`: `gbmcp = "gbmcp.server:main"`.

Install: `pip install 'granite.build[standalone]'` already includes the MCP server (use Python ≥3.11; the default 3.9 fails on `sqlite_database`). Once merged to `@stable`, `pip install "granite.build[standalone] @ git+https://github.com/ibm-granite/granite.build.git@stable"`.

## Run

**Mounted (recommended):** the MCP endpoint comes up with the server.
```bash
gbserver standalone --port 8080                 # serves REST + MCP at /mcp
claude mcp add --scope project --transport http gbmcp "http://127.0.0.1:8080/mcp/"
```

> In a repo checkout you can skip the `claude mcp add` step: the root [`.mcp.json`](../../.mcp.json) registers `gbmcp` at `http://127.0.0.1:${GBSERVER_PORT:-8080}/mcp/` for project scope, and Claude Code auto-discovers it — just approve it once. `gbserver standalone` also prints the exact endpoint URL on startup.

**Standalone process (alternative):**
```bash
GB_ENVIRONMENT=STANDALONE GBMCP_PORT=8000 gbmcp
claude mcp add --scope project --transport http gbmcp "http://127.0.0.1:8000/mcp/"
```

> No `--header` / auth needed (standalone has no verifier). The `claude mcp add` port must match the server's port. Tools are namespaced by the registered name, i.e. `mcp__gbmcp__*`.

## Test

```bash
# import smoke test
GB_ENVIRONMENT=STANDALONE python -c "import gbmcp.server; print('import OK; mcp app:', hasattr(gbmcp.server, 'mcp'))"

# HTTP: core + MCP endpoint (server on :PORT)
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8080/api/v1               # 200
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8080/mcp/                  # 406 (endpoint live; a bare GET lacks the SSE Accept)

# Full MCP handshake + tool list (no auth header needed)
A=(-H "Content-Type: application/json" -H "Accept: application/json, text/event-stream")
SID=$(curl -s -D - -o /dev/null "${A[@]}" -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"c","version":"1"}}}' http://127.0.0.1:8080/mcp/ | tr -d '\r' | awk -F': ' 'tolower($1)=="mcp-session-id"{print $2}')
curl -s -o /dev/null "${A[@]}" -H "mcp-session-id: $SID" -d '{"jsonrpc":"2.0","method":"notifications/initialized"}' http://127.0.0.1:8080/mcp/
curl -s "${A[@]}" -H "mcp-session-id: $SID" -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' http://127.0.0.1:8080/mcp/ | grep -o '"name":"[^"]*"'
```

## What was removed vs. the standalone-separate `gbmcp` (and why)

Everything below was deleted from the source (not merely pruned at runtime), because it has no backend in standalone or is meaningless when mounted inside gbserver:

- **gbserver lifecycle** (`gbserver_start/stop/status/logs`) — gbmcp runs *inside* gbserver; managing the host process is nonsensical, and `_stop` would kill the server hosting the tool. (`tools/gbserver/lifecycle.py` + `utils/gbserver_process.py` deleted; the one helper still used, `tail_lines`, was inlined into `build_job_log.py`.)
- **Non-standalone / niche build tools** — `build_lineage`, `build_validate`, `build_diff`, `build_events`, `build_status_batch`, `build_init`, `build_update`.
- **Remote/prod groups** — `docs_*`, `admin_log`, `artifact_*`, `template_*`, `step_*`, cross-build cache search (`build_leaderboard`/`search`/`compare`/`search_yaml`), `gb_dashboard` (`build_search_errors`/`get_ai_analysis`/`investigate`/`k8s_status`), `cos` (`build_check_cos_path`), `flight_plan` (`plan_*`), `sandbox_*`, gbserver-REST `build_files_*`.
- The GHE OAuth variant and client scripts (`server_oauth.py`, `client*.py`, `smoke_test.py`, `services/ghe_auth.py`).

The source contains **only standalone-usable tools** — there is no runtime tool pruning (`utils/lifespan.py` just initializes the build cache + telemetry DB). The `mcp.run(transport="http")` server is stateful-session by default; the mounted-in-gbserver path drives its session-manager lifespan and has been verified end-to-end with a real MCP `initialize` → `tools/list` handshake.
