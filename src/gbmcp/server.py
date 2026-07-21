# NOTE: load_dotenv() MUST run before importing gbcli/gbmcp modules. gbcli's
# gbconstants resolves GBSERVER_INSTANCE/GB_ENVIRONMENT at import time, so any
# gbcli import that lands before .env is loaded freezes the PROD defaults for
# the process's lifetime — silently routing STANDALONE traffic to PROD.
from dotenv import load_dotenv

load_dotenv()

from pathlib import Path

from fastmcp import FastMCP
from fastmcp.server.providers import FileSystemProvider
from fastmcp.utilities.logging import get_logger

from gbcli.utils.cli_config import configureGBWorkingEnv
from gbmcp.services.telemetry.middleware import TelemetryMiddleware
from gbmcp.utils.lifespan import lifespan

logger = get_logger(__name__)

MCP_INSTRUCTIONS = """
This is the MCP server for Granite.build (a.k.a. LLM.build) in **standalone** mode. gbmcp is bundled into gbserver and served at /mcp by `gbserver standalone` — the backend is up whenever these tools respond.

## Typical workflow
Author a build.yaml (see the create-build guidance), then:
  build_start(file_content) -> build_status(build_id) -> build_log / build_job_log(build_id)
Use build_list to find builds and build_describe to inspect a build's definition.

## Tools

### Builds
- build_start(file_content, space, params): submit a build.yaml (as text) to run; returns a build_id
- build_list: list builds (show_all=True by default; all_user, username, page_size/page_index to paginate)
- build_status(build_id): current status — fast (details/targets/error)
- build_describe(build_id): full build definition YAML + metadata
- build_log(build_id): the build's gbserver log; use tail=N for the last N lines
- build_job_log(build_id): the on-disk job.log — the workload's REAL stdout/stderr (prints, tracebacks, success markers). The primary debugging artifact in standalone.
- build_cancel(build_id): cancel a running build

### Space
- space_list: list available spaces

### Secrets (build-time auth, e.g. an HF token)
- secret_list(space): list secret names (never values)
- secret_get(secret_name, space): returns the gbcli command to reveal a secret's value; the value is shown only in the user's terminal, never returned to the agent
- secret_create(secret_name, space) / secret_update(secret_name, space): return a gbcli command with a <secret-value> placeholder — the user fills the value in their terminal; do not fill it in or ask for it
- secret_delete(secret_name, space): delete a secret

### Info
- info_health: health check
- info_version: gbmcp version
- info_gb_version: Granite.build client/server version
- info_gb_environment: current environment (STANDALONE)

## Output filtering
The build read tools (build_status / build_describe / build_log / build_job_log) accept server-side filters to reduce response size:
- grep="pattern": filter lines by regex (flags: -Cn/-An/-Bn/-i/-v/-F/-w/-x/-c/-n/-o/-mN; e.g. grep="-C2 -i error")
- wc=True: return only line/char counts (gauge size first)
- head=N / tail=N: first/last N lines (for build_log, head/tail control what the API returns)
Strategy: wc=True first, then grep/tail to fetch only the relevant portion.

## Notes
- build_id is a UUID; use build_list to resolve a partial ID (short hash) to a full UUID.
- When a build fails or "succeeds" without doing anything, read build_job_log — that's where the workload's real output lives, not build_status.
- A build step runs with a clean env (no PATH/HOME); step scripts set PATH from $LLMB_BASH_PYTHON_DIR and build their own venv.
- HF auth: for hf:// model/dataset inputs, set HF_TOKEN in the gbserver environment — validate/download hit the HF Hub API and can rate-limit (HTTP 429) without a token. (hf:// -> https:// for browsing; models also drop the 'models/' segment.)
- space is an optional project/namespace scope; omit it to use the default space.
- if a tool fails unexpectedly, call info_gb_version or space_list first to verify connectivity.
"""

configureGBWorkingEnv()

mcp = FastMCP(
    name="gbmcp",
    instructions=MCP_INSTRUCTIONS,
    website_url="https://pages.github.ibm.com/granite-dot-build/",
    providers=[
        FileSystemProvider(root=Path(__file__).parent / "tools"),
    ],
    # No auth verifier: gbmcp is standalone-only and mounted inside gbserver
    # (whose AuthMiddleware exempts /mcp), so no bearer token is required.
    lifespan=lifespan,
    middleware=[TelemetryMiddleware()],
)


def main() -> None:
    """Console-script entry point (``gbmcp``). Serves the MCP over HTTP.

    Port comes from ``GBMCP_PORT`` (default 8000) so a bundled install can be
    pointed at a per-job port without code changes.
    """
    import os

    port = int(os.environ.get("GBMCP_PORT", "8000"))
    mcp.run(transport="http", port=port, stateless_http=True)


if __name__ == "__main__":
    main()
