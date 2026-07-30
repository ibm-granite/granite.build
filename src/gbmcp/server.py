# NOTE: load_dotenv() MUST run before importing gbcli/gbmcp modules. gbcli's
# gbconstants resolves GBSERVER_INSTANCE/GB_ENVIRONMENT at import time, so any
# gbcli import that lands before .env is loaded freezes the PROD defaults for
# the process's lifetime — silently routing STANDALONE traffic to PROD.
from dotenv import load_dotenv

load_dotenv()

# gbmcp is standalone-only, so set up its environment here — BEFORE the gbcli
# import below freezes GB_ENVIRONMENT / GBSERVER_INSTANCE at import time. Pin
# STANDALONE so gbmcp never silently routes to the PROD server (the default when
# GB_ENVIRONMENT is unset). GBSERVER_PORT is the single source of truth for the
# port; GBSERVER_HOST is derived from it, so a lone GBSERVER_PORT override
# retargets both the build tools (client) and the gbserver we launch.
import os

os.environ["GB_ENVIRONMENT"] = "STANDALONE"
os.environ.setdefault("GBSERVER_PORT", "8080")
os.environ.setdefault(
    "GBSERVER_HOST", f"http://127.0.0.1:{os.environ['GBSERVER_PORT']}"
)

from pathlib import Path

from fastmcp import FastMCP
from fastmcp.server.providers import FileSystemProvider
from fastmcp.utilities.logging import get_logger

from gbcli.utils.cli_config import configureGBWorkingEnv

logger = get_logger(__name__)

MCP_INSTRUCTIONS = """
This is the MCP server for Granite.build (a.k.a. LLM.build) in **standalone** mode. gbmcp runs as a local process; the gbserver backend it drives is a *separate* process reached over REST. These tools responding does NOT mean the backend is up — check gbserver_status, and gbserver_start it if needed, before build tools.

## Typical workflow
Ensure the backend is up (gbserver_status; gbserver_start if not), author a build.yaml (see the create-build guidance), then:
  build_start(file_content) -> build_status(build_id) -> build_log / build_job_log(build_id)
Use build_list to find builds and build_describe to inspect a build's definition.

## Tools

### gbserver backend (standalone process management)
- gbserver_status: is the local gbserver process up AND reachable — call before build tools if unsure
- gbserver_start: start gbserver standalone if not running; waits until ready (idempotent)
- gbserver_stop: stop the local gbserver for this port (idempotent)
Lifecycle: gbserver standalone also serves the web dashboard + REST API, so treat it as a long-lived dev service — start it when needed and LEAVE IT RUNNING (the dashboard stays viewable at http://127.0.0.1:<port> and builds reuse the warm server). gbserver_stop only when the user asks; never stop it unprompted.

### Builds
- build_start(file_content, space, params): submit a build.yaml (as text) to run; returns a build_id
- build_list: list builds (show_all=True by default; all_user, username, page_size/page_index to paginate)
- build_status(build_id): current status — fast (details/targets/error)
- build_describe(build_id): full build definition YAML + metadata
- build_log(build_id): the build's gbserver log; use tail=N for the last N lines
- build_job_log(build_id): the on-disk job.log — the workload's REAL stdout/stderr (prints, tracebacks, success markers). The primary debugging artifact in standalone.
- build_cancel(build_id): cancel a running build

### Secrets (build-time auth, e.g. an HF token)
- secret_list(space): list secret names (never values)
- secret_get(secret_name, space): returns the gbcli command to reveal a secret's value; the value is shown only in the user's terminal, never returned to the agent
- secret_create(secret_name, space) / secret_update(secret_name, space): return a gbcli command with a <secret-value> placeholder — the user fills the value in their terminal; do not fill it in or ask for it
- secret_delete(secret_name, space): delete a secret

### Info
- info_health: health check
- info_version: gbmcp version
- info_gb_version: Granite.build client/server version

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
- if a build tool fails or times out, call gbserver_status; if the backend isn't running, gbserver_start it. Leave it running afterward (it also serves the dashboard); when you finish, tell the user it's still running (dashboard at http://127.0.0.1:<port>) and to say when they want it stopped — gbserver_stop only if they ask.
"""

configureGBWorkingEnv()

mcp = FastMCP(
    name="gbmcp",
    instructions=MCP_INSTRUCTIONS,
    website_url="https://github.com/ibm-granite/granite.build",
    providers=[
        FileSystemProvider(root=Path(__file__).parent / "tools"),
    ],
    # No auth verifier: gbmcp is a local stdio process talking to an
    # unauthenticated localhost gbserver.
)


def main() -> None:
    """Console-script entry point (``gbmcp``).

    Runs over stdio — the MCP client (Claude Code, via ``.mcp.json``) launches
    gbmcp as a subprocess and speaks JSON-RPC over stdio, so stdout must stay
    pure (FastMCP logs + banner already go to stderr; show_banner=False also
    skips its startup version-check network call).
    """
    mcp.run(transport="stdio", show_banner=False)


if __name__ == "__main__":
    main()
