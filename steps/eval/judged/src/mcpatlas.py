"""MCP-Atlas (Granite: agentic-general, metric "pass rate").

Scale AI's MCP-Atlas (github.com/scaleapi/mcp-atlas, MIT; arXiv 2602.00933):
500 tool-use tasks over 36 real MCP servers, each with an expert list of claims
the final answer must contain. The headline is upstream's ``pass_rate_0.75``:
the fraction of tasks whose claim coverage (fulfilled 1, partially 0.5, not 0,
averaged over the task's claims) is at least 0.75.

What runs, all pinned:

- Data: ``ScaleAI/MCP-Atlas`` (cc-by-4.0) at :data:`DATASET_REVISION`, the
  ``MCP-Atlas.parquet`` file checked against :data:`PARQUET_SHA256`.
- Environment: upstream's agent-environment image (:data:`IMAGE`, a FastAPI app
  on port 1984 that fronts the MCP servers: ``/list-tools``, ``/call-tool``,
  ``/enabled-servers``). Under enroot it shares the host network, so the run
  holds the node lock ``port-<env_port>``; ``env_url=`` uses one started
  elsewhere instead (``docker run -p 1984:1984 ...`` on a laptop).
- Agent: a Python port of upstream's harness loop at :data:`COMMIT`
  (services/agent-harness ``agent-eval.ts`` + ``litellm-strategy.ts``): the
  prompt as the only message, no system prompt, the task's ENABLED_TOOLS from
  /list-tools as OpenAI function tools (``strict: false``), no sampling
  parameters, at most 256 turns / 100 tool calls, tool errors fed back to the
  model, 60 s per tool call, 1800 s per task (run_eval.py), and the final
  answer = the last assistant message with content.
- Scoring: upstream's ``services/scoring/score_claims.py`` claim extraction,
  per-claim judge prompt and JSON schema (ported verbatim below; with
  ``MCPATLAS_UPSTREAM=<checkout at COMMIT>`` tests/test_mcpatlas.py checks the
  file's sha256 and that the port matches it), one
  judge call per claim at temperature 0. Empty and ``ERROR:`` responses score
  0 without a judge call; responses are cut at 500,000 characters.

Subsets (``subset=``). 16 of the 36 servers need API keys or seeded SaaS
accounts (Airtable, Notion, Slack, MongoDB, Google Workspace, GitHub, Brave,
Exa, ...), which a Granite run does not have. The 20 key-less servers still call
public internet APIs (Wikipedia, arXiv, PubMed, OSM, DuckDuckGo, ...), so the
sandbox needs outbound HTTPS.

- ``keyless`` (default, 30 tasks): every ENABLED_TOOLS server is key-less, so
  the agent sees exactly upstream's tool list.
- ``keyless-gt`` (89 tasks): the reference trajectory uses only key-less
  servers; distractor tools on key-gated servers are missing from the list
  (flagged in results.json).
- ``all`` (500 tasks): every server; all keys of :data:`KEYED_SERVERS` must be
  in the environment (they are passed to the sandbox, where the model's code
  can read them, as upstream's ``.env`` is).

Each subset's task ids are pinned by count and sha256 (:data:`SUBSETS`). 85 tasks
also list distractor tools of four servers the public environment does not ship
(rijksmuseum-server, f1-mcp-server, anili, balldontlie; no reference trajectory
calls them); upstream's harness cannot offer them either, and ``keyless``
excludes those tasks.

Gold modes (no model, no GPU): ``agent=gold`` answers each task with its own
expert claims, one per line, which the judge must find covered (pass rate ~1);
``agent=replay`` also starts the environment and replays the reference
trajectory's tool calls, reporting how many fail (``details.replay``): a check
of servers and egress before a model run.

Deviations from upstream (also in ``details.deviations``):

- Judge: ``aws/claude-sonnet-5`` on the configured LiteLLM gateway (upstream:
  ``gemini-3.1-pro-preview``), metered (budget, 402).
- A judge call that fails (after ``judge_retries``) or whose reply is not the
  schema's JSON leaves the claim unjudged: the task is left out and retried on
  resume (registry.failure_policy), where upstream scores it not_fulfilled.
- An infrastructure failure of a generation (model endpoint unreachable or 5xx
  after retries, environment down) is not saved and is retried on resume;
  upstream records an empty answer (score 0). A 4xx from the model (e.g. the
  context is full) ends the trajectory as upstream's loop does.
- Only text blocks of tool results are kept (upstream's schema rejects other
  blocks; ``met-museum_get-museum-object`` is called with ``returnImage=false``
  as upstream does).

Layout: ``<output_dir>/repeat-<r>/<task>.json`` (trajectory and answer) and
``<task>.judgments[-<tag>].json`` (per-claim verdicts); finished ones are
skipped on restart. Splittable: ``--phase generate`` needs the model and the
environment, ``--phase score`` only the judge.

Options: ``subset``, ``agent=model|gold|replay``, ``env_url``, ``env_port=1984``,
``image``, ``sandbox`` (enroot), ``env_start_timeout=1200``,
``allow_offline=false``, ``concurrency=5``, ``max_turns=256``,
``max_tool_calls=100``, ``tool_timeout=60``, ``task_timeout=1800``,
``llm_timeout=600``, ``enable_thinking``, ``temperature`` / ``top_p`` /
``max_tokens`` (unset: generation_config), ``extra_llm_params=<json>``,
``threshold=0.75``, the judge options of JudgedBenchmark (``judge_model``,
``judge_base_url``, ``judge_api_key_env``, ``judge_thinking``, ...),
``judge_workers=8``, ``judge_parallel=4``, ``judge_retries=3``, ``judge_tag``,
``max_failed_frac=0.05``.
"""

from __future__ import annotations

import ast
import concurrent.futures
import hashlib
import json
import logging
import math
import os
import re
import shlex
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

from granite_evals import data
from granite_evals.benchmarks.judge_general import (
    JudgedBenchmark,
    _count,
    _write_json,
    total_usage,
)
from granite_evals.registry import failure_policy, pass_at_k_record, register

log = logging.getLogger(__name__)

REPO = "scaleapi/mcp-atlas"
COMMIT = "e3d543158a944f4b7c98f2f4094ce37ce9243a18"
# Upstream files ported here (checked by tests/test_mcpatlas.py against a checkout).
UPSTREAM_FILES = {
    "services/scoring/score_claims.py": "fff3ca8c8175b3e85449126ba6cd7f55a8462ffe7d57abe3b5b1a0ec60cefbcc",
    "LICENSE": "07f1426f81577330f3b49f3c1ac1d1de4b6a072890e41161aa92a648e43e5ed1",
}

DATASET = "ScaleAI/MCP-Atlas"
DATASET_REVISION = "8c563b55d7c967755f474299848049834d624617"
PARQUET = "MCP-Atlas.parquet"
PARQUET_SHA256 = "2d7bc052f14cbcb3b8294293481053f7111d256f9c9deaa96f3ff632d19958d0"
N_TASKS = 500

# run_eval.py's DEFAULT_SANDBOX_IMAGE; the tag's OCI index digest when pinned.
IMAGE = "ghcr.io/scaleapi/mcp-atlas:1.2.7"
IMAGE_DIGEST = "sha256:24e6ed3534916afe2c6825382da159a30e23516ef612be5d074fd96a74f9184c"
ENV_PORT = 1984
ENV_LOG = "/tmp/mcp-atlas-env.log"

# mcp_client.py DEFAULT_SERVERS: the servers that need no key.
KEYLESS_SERVERS = (
    "arxiv", "calculator", "cli-mcp-server", "clinicaltrialsgov-mcp-server", "context7", "ddg-search",
    "desktop-commander", "fetch", "filesystem", "git", "mcp-code-executor", "mcp-server-code-runner",
    "memory", "met-museum", "open-library", "osm-mcp-server", "pubmed", "weather", "whois", "wikipedia",
)  # fmt: skip
# mcp_server_template.json: the variables each key-gated server needs.
KEYED_SERVERS = {
    "airtable": ("AIRTABLE_API_KEY",),
    "alchemy": ("ALCHEMY_API_KEY",),
    "brave-search": ("BRAVE_API_KEY",),
    "e2b-server": ("E2B_API_KEY",),
    "exa": ("EXA_API_KEY",),
    "github": ("GITHUB_TOKEN",),
    "google-maps": ("GOOGLE_MAPS_API_KEY",),
    "google-workspace": ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "GOOGLE_REFRESH_TOKEN"),
    "lara-translate": ("LARA_ACCESS_KEY_ID", "LARA_ACCESS_KEY_SECRET"),
    "mongodb": ("MONGODB_CONNECTION_STRING",),
    "national-parks": ("NPS_API_KEY",),
    "notion": ("NOTION_TOKEN",),
    "oxylabs": ("OXYLABS_USERNAME", "OXYLABS_PASSWORD"),
    "slack": ("SLACK_MCP_XOXC_TOKEN", "SLACK_MCP_XOXD_TOKEN"),
    "twelvedata": ("TWELVE_DATA_API_KEY",),
    "weather-data": ("WEATHER_API_KEY",),
}
ALL_SERVERS = tuple(sorted({*KEYLESS_SERVERS, *KEYED_SERVERS}))
# agent-environment main.py TOOL_NAME_MAPPINGS (old names in the data).
TOOL_NAME_MAPPINGS = {"brave_brave_web_search": "brave-search_brave_web_search"}

# subset -> (task count, sha256 of the sorted task ids joined by ",") at DATASET_REVISION.
SUBSETS = {
    "keyless": (30, "9b67c39ddde6e6875a18fdab4a2b882209d363ef05afb350060c65b7e11cf2e0"),
    "keyless-gt": (89, "cccf28bb29814b5dfeaeecd2d2154bc2b89cc1b2f9d1349a010917aec236850d"),
    "all": (N_TASKS, "ca08f440cf6140b6dc3ac8fe4a5173cd688ed10f655c3a4fcd5ffec51a1cdba8"),
}

# Harness limits (agent-eval.ts, sandbox-client.ts, config.ts, run_eval.py).
MAX_TURNS, MAX_TOOL_CALLS = 256, 100
TOOL_TIMEOUT_S, LLM_TIMEOUT_S, TASK_TIMEOUT_S = 60, 600, 1800
# Waits (s) before retrying an LLM call: 429 (litellm-strategy: 2^i s, up to 5 times),
# 5xx (agent-eval: 10 s, 3 attempts), timeout (15 s). Tests set them to 0.
RETRY_WAIT = {"429": 1.0, "5xx": 10.0, "timeout": 15.0}
MAX_429_RETRIES, MAX_LLM_ATTEMPTS = 5, 3
MAX_RESPONSE_CHARS = 500_000
COVERAGE = {"fulfilled": 1.0, "partially_fulfilled": 0.5, "not_fulfilled": 0.0}

DEVIATIONS = [
    "judge: aws/claude-sonnet-5 on the configured LiteLLM gateway (upstream gemini-3.1-pro-preview), metered",
    "a failed or unparseable judge call leaves its task unscored and retried on resume "
    "(upstream: the claim is not_fulfilled)",
    "an infrastructure failure of a generation is retried on resume (upstream: empty answer, score 0)",
    "only text blocks of tool results are passed to the model",
]


# ---------------------------------------------------------------------------
# Ported from services/scoring/score_claims.py at COMMIT (MIT, Scale AI)
# ---------------------------------------------------------------------------


def clean_claim_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r'^[-*•·◦‣⁃]\s*', '', text)
    text = re.sub(r'^\d+[.)]\s*', '', text)
    text = text.replace('“', '"')
    text = re.sub(r'[”"]', '"', text)
    text = text.replace('‘', "'")
    text = text.replace('’', "'")
    text = text.replace('–', '-')
    text = text.replace('—', '-')
    text = text.replace('…', '...')
    text = re.sub(r'[.\s]*["\']+ $', '', text)
    text = re.sub(r'["\']+\.*$', '', text)
    text = text.strip(' \t\n\r')
    return text


def extract_claims(claim_blob) -> list[str]:
    if claim_blob is None:
        return []
    if isinstance(claim_blob, list):
        return [c for c in (clean_claim_text(str(x)) for x in claim_blob) if c and len(c) > 3]
    if not isinstance(claim_blob, str):
        claim_blob = str(claim_blob)
    claim_blob = claim_blob.strip()
    if not claim_blob:
        return []
    if claim_blob.startswith('[') and claim_blob.endswith(']'):
        try:
            parsed_list = json.loads(claim_blob)
            if isinstance(parsed_list, list):
                return [c for c in (clean_claim_text(str(x)) for x in parsed_list) if c and len(c) > 3]
        except (json.JSONDecodeError, ValueError):
            try:
                parsed_list = ast.literal_eval(claim_blob)
                if isinstance(parsed_list, list):
                    return [c for c in (clean_claim_text(str(x)) for x in parsed_list) if c and len(c) > 3]
            except (ValueError, SyntaxError):
                pass
    for sep in ["\n•", "\n-", "\n*", "\n1.", "\n2.", ";", "||"]:
        if sep in claim_blob:
            claims = [c for c in (clean_claim_text(p) for p in claim_blob.split(sep)) if c and len(c) > 3]
            if claims:
                return claims
    return [c for c in (clean_claim_text(line) for line in claim_blob.strip().split('\n')) if c and len(c) > 3]


def get_single_claim_evaluation_schema():
    return {
        "type": "object",
        "properties": {
            "claim_text": {"type": "string"},
            "coverage_outcome": {
                "type": "string",
                "enum": ["fulfilled", "partially_fulfilled", "not_fulfilled"]
            },
            "justification": {"type": "string"},
            "confidence_level": {
                "type": "number"
            }
        },
        "required": ["claim_text", "coverage_outcome", "justification", "confidence_level"]
    }


def claim_evaluation_prompt(claim: str, response: str) -> str:
    """CoverageEvaluator._get_single_claim_evaluation_prompt."""
    return f"""You are evaluating how well a model's response addresses a specific expert-defined claim.
SCORING CRITERIA:
- fulfilled: Claim is completely and accurately addressed. The response covers all key details.
- partially_fulfilled: Claim is partially addressed. The response covers some but not all key details.
- not_fulfilled: Claim is not addressed. The response does not include any key details.
NUMERICAL COMPARISON GUIDELINES:
- For numerical values, use reasonable approximation thresholds:
  * Exact match NOT required for decimals
  * Values within 5% of the claimed number are considered matching
  * For percentages, ±1 percentage points is acceptable
  * Round to appropriate significant figures based on context
- Consider the precision appropriate to the domain:
  * Scientific measurements may need higher precision
  * General statistics/estimates can have looser matching
  * Financial figures should match to reasonable business precision (e.g., millions/billions don't need exact cents)
- If a number is expressed differently but mathematically equivalent (e.g., "0.5" vs "50%" vs "half"), consider it a match
CLAIM TO EVALUATE:
{claim}
MODEL RESPONSE TO ANALYZE:
{response}
INSTRUCTIONS:
1. Determine if the core requirement of the claim is met in the response
2. Check if all key components from the claim appear substantively in the response
   - For numerical values, apply the flexible matching guidelines above
   - Focus on whether the same magnitude and meaning are conveyed
3. Assign the appropriate coverage_outcome
4. Provide specific justification referencing what was/wasn't covered
   - When numbers differ slightly, note if they're within acceptable range
5. Provide a confidence level (0.0-1.0) for your assessment
Be rigorous but fair in your assessment. Focus on whether the response conveys the same information as the claim, not on exact numerical precision unless precision is critical to the claim's meaning."""


def skipped_response(response: str | None) -> bool:
    """evaluate_dataframe_async: no judge call, coverage 0."""
    return not response or not response.strip() or str(response).startswith("ERROR:")


def truncate_response(response: str) -> str:
    if len(response) > MAX_RESPONSE_CHARS:
        return response[:MAX_RESPONSE_CHARS] + "\n\n[TRUNCATED — original response was too long]"
    return response


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


def tool_names(value: Any) -> list[str]:
    """run_eval.py parse_enabled_tools: a JSON list of names or {"name": ...} objects."""
    if isinstance(value, str):
        try:
            value = json.loads(value) if value.strip() else []
        except json.JSONDecodeError:
            return [t.strip() for t in value.split(",") if t.strip()]
    out = []
    for t in value or []:
        if isinstance(t, str):
            out.append(t)
        elif isinstance(t, dict) and t.get("name"):
            out.append(t["name"])
    return out


def server_of(tool: str) -> str:
    """The MCP server a tool belongs to (agent-environment prefixes ``<server>_``)."""
    tool = TOOL_NAME_MAPPINGS.get(tool, tool)
    if tool.startswith("MongoDB_"):
        return "mongodb"
    for name in sorted(ALL_SERVERS, key=len, reverse=True):
        if tool.startswith(name + "_"):
            return name
    return "?"


def trajectory_tool_calls(row: dict) -> list[dict]:
    traj = row.get("TRAJECTORY") or "[]"
    traj = json.loads(traj) if isinstance(traj, str) else traj
    return [c for m in traj if m.get("role") == "assistant" for c in (m.get("tool_calls") or [])]


def task_servers(row: dict, *, by: str) -> set[str]:
    if by == "gt":
        return {server_of(c["function"]["name"]) for c in trajectory_tool_calls(row)}
    return {server_of(t) for t in tool_names(row.get("ENABLED_TOOLS"))}


def in_subset(row: dict, subset: str) -> bool:
    if subset == "all":
        return True
    by = "gt" if subset == "keyless-gt" else "enabled"
    return task_servers(row, by=by) <= set(KEYLESS_SERVERS)


def ids_digest(ids) -> str:
    return hashlib.sha256(",".join(sorted(ids)).encode()).hexdigest()


def gold_response(row: dict) -> str:
    """The task's expert claims, one per line (agent=gold)."""
    return "\n".join(extract_claims(row.get("GTFA_CLAIMS")))


def coverage(outcomes: list[str]) -> float:
    """CoverageEvaluator.evaluate: mean of the claim scores, rounded to 3 places."""
    return round(sum(COVERAGE.get(o, 0.0) for o in outcomes) / len(outcomes), 3) if outcomes else 0.0


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def parse_judgement(content: str | None) -> dict | None:
    """The judge's claim_evaluation JSON (fences or text around it tolerated)."""
    text = (content or "").strip()
    for candidate in (text, *re.findall(r"\{.*\}", text, flags=re.DOTALL)):
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and "coverage_outcome" in obj:
            return obj
    return None


# ---------------------------------------------------------------------------
# Environment (the agent-environment image)
# ---------------------------------------------------------------------------


class InfraError(RuntimeError):
    """A failure that is not the model's: not saved, retried on resume."""


class AtlasEnv:
    """HTTP client of a running agent-environment (``/list-tools``, ``/call-tool``)."""

    def __init__(self, url: str, *, transport=None):
        import httpx

        self.url = url.rstrip("/")
        self._http = httpx.Client(base_url=self.url, transport=transport, timeout=180)
        self._tools: list[dict] | None = None
        self._lock = threading.Lock()

    def close(self) -> None:
        self._http.close()

    def health(self) -> bool:
        try:
            r = self._http.get("/health", timeout=10)
            return r.status_code == 200 and r.json().get("status") == "health_and_client_connection_ok"
        except Exception:
            return False

    def enabled_servers(self) -> dict:
        r = self._http.get("/enabled-servers", timeout=300)
        r.raise_for_status()
        return r.json()

    def list_tools(self) -> list[dict]:
        with self._lock:
            if self._tools is None:
                r = self._http.post("/list-tools", timeout=180)
                if r.status_code != 200:
                    raise InfraError(f"list-tools: HTTP {r.status_code}")
                self._tools = r.json()
            return self._tools

    def call_tool(self, name: str, args: dict, timeout: float) -> list[dict]:
        """sandbox-client.ts callTool: the content blocks the model sees."""
        import httpx

        try:
            r = self._http.post("/call-tool", json={"tool_name": name, "tool_args": args or {}}, timeout=timeout)
        except httpx.TimeoutException:
            return [{"type": "text", "text": f"Tool call timed out after {int(timeout)}s"}]
        except httpx.HTTPError as e:
            raise ToolCallError(f"Failed to call tool {name}: {e}") from e
        if r.status_code != 200:
            return [{"type": "text", "text": r.text}]
        body = r.json()
        if not (isinstance(body, list) and any(b.get("type") == "text" and str(b.get("text", "")).strip()
                                               for b in body)):
            return [{"type": "text", "text": "success"}]
        return [{"type": "text", "text": b["text"]} for b in body if b.get("type") == "text"]


class ToolCallError(RuntimeError):
    pass


class EnvSandbox:
    """The agent-environment image in a granite sandbox, serving on the host's
    network (enroot) at ``127.0.0.1:<port>``."""

    def __init__(self, image: str, *, port: int, env: dict[str, str], backend: str = ""):
        from granite_evals import sandbox

        backend = backend or os.environ.get("GRANITE_EVALS_SANDBOX", "enroot")
        if backend != "enroot":
            raise SystemExit(
                f"mcpatlas: sandbox={backend} gives the environment its own network; start "
                f"`{backend} run -d -p {port}:1984 {image}` yourself and pass --option env_url=http://localhost:{port}"
            )
        self.port = port
        self.box = sandbox.make_sandbox(image, backend=backend, env=env)

    def start(self) -> None:
        self.box.start()
        # The image's ENTRYPOINT (envsubst of the server config) and CMD, run from the
        # venv directly (``uv run`` may try to sync) and bound to loopback.
        cmd = (
            "export HOME=/root PATH=/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
            " && unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV PYTHONPATH UV_TOOL_DIR UV_TOOL_BIN_DIR XDG_DATA_HOME XDG_BIN_HOME && cd /agent-environment"
            f" && (setsid nohup ./entrypoint.sh .venv/bin/python -m uvicorn agent_environment.main:app"
            f" --host 127.0.0.1 --port {int(self.port)} > {ENV_LOG} 2>&1 < /dev/null &)"
        )
        r = self.box.execute(cmd, timeout=120)
        if r.returncode != 0:
            raise InfraError(f"starting the MCP-Atlas environment failed: {r.output[-2000:]}")

    def log_tail(self, n: int = 40) -> str:
        return self.box.execute(f"tail -n {int(n)} {shlex.quote(ENV_LOG)}", timeout=30).output

    def close(self) -> None:
        self.box.close()


# ---------------------------------------------------------------------------
# Agent loop (agent-eval.ts runMcpAgent + litellm-strategy.ts)
# ---------------------------------------------------------------------------


def function_tools(tools: list[dict]) -> list[dict]:
    """_transformToolCalls."""
    return [
        {"type": "function", "function": {"name": t["name"], "description": t.get("description"),
                                          "parameters": {**(t.get("inputSchema") or {})}, "strict": False}}
        for t in tools
    ]  # fmt: skip


def assistant_message(raw: dict) -> dict:
    """AssistantMessageSchema: role, content, tool_calls, reasoning_content."""
    msg: dict[str, Any] = {"role": "assistant", "content": raw.get("content")}
    if raw.get("tool_calls"):
        msg["tool_calls"] = [
            {"id": c["id"], "type": "function",
             "function": {"name": c["function"]["name"], "arguments": c["function"].get("arguments") or ""}}
            for c in raw["tool_calls"]
        ]  # fmt: skip
    if raw.get("reasoning_content") is not None:
        msg["reasoning_content"] = raw["reasoning_content"]
    return msg


def pruned_args(name: str, args: dict) -> dict:
    if name == "met-museum_get-museum-object":
        return {**args, "returnImage": False}
    return args


class ModelClient:
    """POST /chat/completions with upstream's retries; raises InfraError when the
    endpoint is unreachable or keeps failing, returns None on another 4xx."""

    def __init__(self, base_url: str, model: str, extra: dict, *, timeout: float, transport=None):
        import httpx

        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.extra = model, extra
        self._http = httpx.Client(transport=transport, timeout=timeout)

    def close(self) -> None:
        self._http.close()

    def complete(self, messages: list[dict], tools: list[dict]) -> tuple[dict | None, dict, str]:
        import httpx

        payload = {"model": self.model, "messages": messages, "tools": tools, **self.extra}
        n429 = attempts = 0
        while True:
            try:
                r = self._http.post(self.url, json=payload, headers={"Authorization": "Bearer EMPTY"})
            except httpx.TimeoutException:
                attempts += 1
                if attempts < MAX_LLM_ATTEMPTS:
                    time.sleep(RETRY_WAIT["timeout"])
                    continue
                raise InfraError("model call timed out")
            except httpx.HTTPError as e:
                raise InfraError(f"model endpoint unreachable: {e}") from e
            if r.status_code == 200:
                body = r.json()
                return assistant_message(body["choices"][0]["message"]), body.get("usage") or {}, ""
            if r.status_code == 429 and n429 < MAX_429_RETRIES:
                time.sleep(RETRY_WAIT["429"] * 2**n429)
                n429 += 1
                continue
            if r.status_code in (500, 502, 503) or r.status_code == 429:
                attempts += 1
                if attempts < MAX_LLM_ATTEMPTS:
                    time.sleep(RETRY_WAIT["5xx"])
                    continue
                raise InfraError(f"model endpoint HTTP {r.status_code}")
            # Any other error ends the trajectory, as upstream's loop does.
            return None, {}, f"HTTP {r.status_code}: {r.text[:500]}"


def run_agent(row: dict, env: AtlasEnv, model: ModelClient, *, max_turns: int, max_tool_calls: int,
              tool_timeout: float, task_timeout: float, clock: Callable[[], float] = time.monotonic) -> dict:
    enabled = set(tool_names(row.get("ENABLED_TOOLS")))
    available = [t for t in env.list_tools() if t["name"] in enabled]
    tools = function_tools(available)
    messages: list[dict] = [{"role": "user", "content": row["PROMPT"]}]
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    stop, error, tool_calls, turns = "max_turns", "", 0, 0
    deadline = clock() + task_timeout
    for _ in range(max_turns):
        if max_tool_calls and tool_calls >= max_tool_calls:
            stop = "max_tool_calls"
            break
        if clock() > deadline:
            stop = "timeout"
            break
        msg, u, error = model.complete(messages, tools)
        turns += 1
        for k in usage:
            usage[k] += int(u.get(k) or 0)
        if msg is None:
            stop = "llm_error"
            break
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            stop = "completed"
            break
        for call in calls:
            if max_tool_calls and tool_calls >= max_tool_calls:
                stop = "max_tool_calls"
                break
            tool_calls += 1
            name = call["function"]["name"]
            try:
                args = pruned_args(name, json.loads(call["function"]["arguments"] or "null"))
                content = env.call_tool(name, args, tool_timeout)
            except Exception as e:  # fed back to the model so it can recover
                content = [{"type": "text", "text": "Error: " + (str(e) or type(e).__name__).split("\n")[0]}]
            messages.append({"role": "tool", "content": content, "tool_call_id": call["id"]})
        if stop == "max_tool_calls":
            break
    if stop == "timeout":
        # run_eval.py's client timeout: the task's answer is the error.
        response = f"ERROR: timeout after {int(task_timeout)}s"
    else:
        response = next((m["content"] for m in reversed(messages)
                         if m["role"] == "assistant" and m.get("content")), "")  # fmt: skip
    return {
        "response": response,
        "stop": stop,
        "error": error,
        "turns": turns,
        "tool_calls": tool_calls,
        "tools_offered": len(tools),
        "tools_missing": sorted(enabled - {t["name"] for t in available}),
        "usage": usage,
        "messages": messages,
    }


def replay(row: dict, env: AtlasEnv, tool_timeout: float) -> dict:
    """Call the reference trajectory's tools in order (agent=replay)."""
    failed = []
    calls = trajectory_tool_calls(row)
    for c in calls:
        name = c["function"]["name"]
        try:
            content = env.call_tool(name, pruned_args(name, json.loads(c["function"]["arguments"] or "null")),
                                    tool_timeout)
            text = "".join(b["text"] for b in content)
            if text.startswith(('{"detail"', "Tool call timed out")):
                failed.append({"tool": name, "error": text[:300]})
        except Exception as e:
            failed.append({"tool": name, "error": str(e)[:300]})
    return {"calls": len(calls), "failed": failed}


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@register
class MCPAtlas(JudgedBenchmark):
    id = "mcpatlas"
    metric = "pass rate (claim coverage >= 0.75)"
    extra = "judged"  # httpx + openai (the judge); the environment is its own image
    harness_packages = ("openai", "httpx")
    dataset = DATASET
    dataset_revision = DATASET_REVISION
    splittable = True
    _transport: ClassVar[Any] = None  # tests: an httpx transport for the model and the environment

    @property
    def agent(self) -> str:
        agent = self.opt("agent", "model")
        if agent not in ("model", "gold", "replay"):
            raise SystemExit("mcpatlas: agent must be model, gold or replay")
        return agent

    @property
    def subset(self) -> str:
        s = self.opt("subset", "keyless")
        if s not in SUBSETS:
            raise SystemExit(f"mcpatlas: subset must be one of {', '.join(SUBSETS)}")
        return s

    def needs_server(self) -> bool:
        return self.agent == "model"

    def score_needs_server(self) -> bool:
        return self.judge_model == "self"

    # -- data ------------------------------------------------------------------

    def load_rows(self) -> tuple[list[dict], str, str | None]:
        source, revision = self.dataset_source()
        if source == DATASET and revision == DATASET_REVISION:
            from huggingface_hub import hf_hub_download

            path = Path(hf_hub_download(DATASET, PARQUET, repo_type="dataset", revision=revision))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != PARQUET_SHA256:
                raise SystemExit(f"mcpatlas: {PARQUET} sha256 {digest}, expected {PARQUET_SHA256}")
            import pyarrow.parquet as pq

            return pq.read_table(path).to_pylist(), source, revision
        if source.endswith(".parquet") and Path(source).exists():
            import pyarrow.parquet as pq

            return pq.read_table(source).to_pylist(), source, revision
        return data.load_split(source, revision=revision, split="train"), source, revision

    def select(self, rows: list[dict], pinned: bool) -> list[dict]:
        subset = [r for r in rows if in_subset(r, self.subset)]
        count, digest = SUBSETS[self.subset]
        if pinned:
            got = ids_digest(r["TASK"] for r in subset)
            if len(subset) != count or got != digest:
                raise SystemExit(f"mcpatlas: subset {self.subset} is {len(subset)} tasks (sha256 {got}); "
                                 f"pinned {count} ({digest})")
        return data.take(subset, self.config.limit, key="TASK")

    # -- entry point -----------------------------------------------------------

    def run(self, base_url: str, served_model_name: str) -> dict[str, Any]:
        rows, source, revision = self.load_rows()
        pinned = source == DATASET and revision == DATASET_REVISION
        tasks = self.select(rows, pinned)
        jobs = [(row, r) for r in range(self.repeats) for row in tasks]
        log.info("mcpatlas: %d tasks (subset %s) x %d repeats, agent=%s", len(tasks), self.subset,
                 self.repeats, self.agent)
        gens = self._generate_all(jobs, base_url, served_model_name)
        info = {"dataset": source, "dataset_revision": revision, "subset": self.subset,
                "tasks": [t["TASK"] for t in tasks], **self._describe()}  # fmt: skip
        if not self.scoring:
            failed = sum(g is None for g in gens)
            if failed:
                log.warning("mcpatlas: %d/%d generations failed; rerun --phase generate to retry", failed, len(gens))
            return {"n": len(gens) - failed, "generations_failed": failed, "generations_total": len(gens),
                    "stops": _count(g["stop"] for g in gens if g), **info}  # fmt: skip
        judge = self.make_judge(base_url, served_model_name)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.opt("judge_workers", 8)) as pool:
            judged = list(pool.map(lambda jg: self._judge(jg[0][0], jg[0][1], jg[1], judge), zip(jobs, gens)))
        return self._aggregate(jobs, gens, judged, judge, info)

    def _describe(self) -> dict[str, Any]:
        out = {
            "agent": self.agent,
            "harness": f"github.com/{REPO}@{COMMIT} (agent loop ported; score_claims.py prompt/schema)",
            "image": self.opt("image", IMAGE),
            "image_digest": IMAGE_DIGEST if self.opt("image", IMAGE) == IMAGE else None,
            "servers": self.servers(),
            "limits": {"max_turns": self.opt("max_turns", MAX_TURNS),
                       "max_tool_calls": self.opt("max_tool_calls", MAX_TOOL_CALLS),
                       "tool_timeout_s": self.opt("tool_timeout", TOOL_TIMEOUT_S),
                       "task_timeout_s": self.opt("task_timeout", TASK_TIMEOUT_S)},
            "request": {"extra": self.llm_extra(), "defaults": "generation_config of the checkpoint"},
            "deviations": DEVIATIONS + (["keyless-gt: ENABLED_TOOLS on key-gated servers are not offered"]
                                        if self.subset == "keyless-gt" else []),
        }  # fmt: skip
        return out

    def servers(self) -> list[str]:
        return list(ALL_SERVERS) if self.subset == "all" else list(KEYLESS_SERVERS)

    def llm_extra(self) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        for key, cast in (("temperature", float), ("top_p", float), ("max_tokens", int)):
            if key in self.config.options:
                extra[key] = cast(self.config.options[key])
        if "enable_thinking" in self.config.options:
            extra["chat_template_kwargs"] = {"enable_thinking": self.opt("enable_thinking", True)}
        if raw := self.opt("extra_llm_params", ""):
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise SystemExit("mcpatlas: extra_llm_params must be a JSON object")
            extra.update(parsed)
        return extra

    # -- generation ------------------------------------------------------------

    def _path(self, row: dict, r: int, suffix: str = "") -> Path:
        return self.config.output_dir / f"repeat-{r}" / f"{row['TASK']}{suffix}.json"

    def _generate_all(self, jobs, base_url: str, served: str) -> list[dict | None]:
        done: dict[int, dict | None] = {}
        todo = []
        for i, (row, r) in enumerate(jobs):
            path = self._path(row, r)
            if path.exists():
                done[i] = json.loads(path.read_text())
            elif self.agent == "gold" or not self.generating:
                done[i] = self._gold(row, r) if self.agent == "gold" else self._missing(row, r)
            else:
                todo.append(i)
        if todo:
            with self._environment() as env:
                model = None
                if self.agent == "model":
                    model = ModelClient(base_url, served, self.llm_extra(), transport=self._transport,
                                        timeout=self.opt("llm_timeout", float(LLM_TIMEOUT_S)))
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=self.opt("concurrency", 5)) as pool:
                        for i, g in zip(todo, pool.map(lambda i: self._generate(*jobs[i], env, model), todo)):
                            done[i] = g
                finally:
                    if model is not None:
                        model.close()
        return [done[i] for i in range(len(jobs))]

    def _missing(self, row: dict, r: int) -> None:
        log.warning("mcpatlas: %s", self.not_generated(f"{row['TASK']} repeat {r}"))

    def _gold(self, row: dict, r: int) -> dict:
        out = {"task": row["TASK"], "repeat": r, "agent": "gold", "response": gold_response(row),
               "stop": "gold", "messages": []}  # fmt: skip
        path = self._path(row, r)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(path, out)
        return out

    def _generate(self, row: dict, r: int, env: AtlasEnv, model: ModelClient | None) -> dict | None:
        started = time.time()
        try:
            if self.agent == "replay":
                out = {"task": row["TASK"], "repeat": r, "agent": "replay", "response": gold_response(row),
                       "stop": "replay", "replay": replay(row, env, self.opt("tool_timeout", float(TOOL_TIMEOUT_S))),
                       "messages": []}  # fmt: skip
            else:
                out = {"task": row["TASK"], "repeat": r, "agent": "model", **run_agent(
                    row, env, model,
                    max_turns=self.opt("max_turns", MAX_TURNS),
                    max_tool_calls=self.opt("max_tool_calls", MAX_TOOL_CALLS),
                    tool_timeout=self.opt("tool_timeout", float(TOOL_TIMEOUT_S)),
                    task_timeout=self.opt("task_timeout", float(TASK_TIMEOUT_S)),
                )}  # fmt: skip
        except Exception as e:  # infrastructure: not saved, retried on resume
            log.warning("mcpatlas: %s repeat %d failed (%s: %s); retried on resume", row["TASK"], r,
                        type(e).__name__, e)
            return None
        out["elapsed_s"] = round(time.time() - started, 1)
        path = self._path(row, r)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(path, out)
        log.info("mcpatlas: %s repeat %d: %s, %s tool calls", row["TASK"], r, out["stop"], out.get("tool_calls"))
        return out

    def _environment(self):
        """The running agent-environment: ``env_url``, or the image in a sandbox."""
        bench = self

        class _Ctx:
            def __enter__(self):
                self.box = self.lock = None
                if bench.agent == "gold":
                    return None
                url = bench.opt("env_url", "")
                if not url:
                    url = self._start()
                self.env = AtlasEnv(url, transport=bench._transport)
                try:
                    bench._wait_ready(self.env, self)
                except BaseException:
                    self.__exit__(None, None, None)
                    raise
                return self.env

            def _start(self) -> str:
                from granite_evals.sandbox.nodelock import node_locks

                port = bench.opt("env_port", ENV_PORT)
                self.lock = node_locks([f"port-{port}"], what="mcpatlas environment")
                self.lock.__enter__()
                self.box = EnvSandbox(bench.opt("image", IMAGE), port=port, env=bench.sandbox_env(),
                                      backend=bench.opt("sandbox", ""))
                self.box.start()
                return f"http://127.0.0.1:{port}"

            def __exit__(self, *exc):
                if getattr(self, "env", None) is not None:
                    self.env.close()
                if self.box is not None:
                    self.box.close()
                if self.lock is not None:
                    self.lock.__exit__(None, None, None)

        return _Ctx()

    def sandbox_env(self) -> dict[str, str]:
        """ENABLED_SERVERS for the subset, and for ``all`` every server's keys."""
        env = {"ENABLED_SERVERS": ",".join(self.servers())}
        if self.subset == "all":
            missing = [v for vs in KEYED_SERVERS.values() for v in vs if not os.environ.get(v, "").strip()]
            if missing:
                raise SystemExit("mcpatlas: subset=all needs these variables (accounts of the key-gated "
                                 f"MCP servers): {', '.join(missing)}")  # fmt: skip
            env.update({v: os.environ[v] for vs in KEYED_SERVERS.values() for v in vs})
        return env

    def _wait_ready(self, env: AtlasEnv, ctx) -> None:
        deadline = time.monotonic() + self.opt("env_start_timeout", 1200.0)
        while not env.health():
            if time.monotonic() > deadline:
                tail = ctx.box.log_tail() if ctx.box is not None else ""
                raise SystemExit(f"mcpatlas: the environment at {env.url} did not come up\n{tail}")
            time.sleep(self.opt("env_poll", 5.0))
        status = env.enabled_servers()
        offline = sorted(name for name, state in status.get("servers", []) if state != "OK")
        log.info("mcpatlas: environment up, %s/%s servers online", status.get("online"), status.get("total"))
        needed = set(self.servers())
        if offline and needed & set(offline) and not self.opt("allow_offline", False):
            raise SystemExit(f"mcpatlas: MCP servers offline: {', '.join(offline)} (check egress; "
                             "--option allow_offline=true runs without their tools)")  # fmt: skip
        self.offline_servers = offline

    # -- judging ---------------------------------------------------------------

    def _judge(self, row: dict, r: int, gen: dict | None, judge) -> dict:
        claims = extract_claims(row.get("GTFA_CLAIMS"))
        if gen is None:
            return {"status": "not_generated", "outcomes": None}
        if not claims:
            return {"status": "no_claims", "outcomes": None}
        if skipped_response(gen["response"]):
            return {"status": "skipped", "outcomes": ["not_fulfilled"] * len(claims)}
        tag = self.opt("judge_tag", "")
        path = self._path(row, r, f".judgments-{tag}" if tag else ".judgments")
        done: dict[str, dict] = json.loads(path.read_text()) if path.exists() else {}
        response = truncate_response(gen["response"])
        todo = [i for i in range(len(claims)) if str(i) not in done]

        def one(i: int) -> None:
            request = {
                "messages": [{"role": "user", "content": claim_evaluation_prompt(claims[i], response)}],
                "temperature": 0.0,
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "claim_evaluation", "schema": get_single_claim_evaluation_schema()}},
            }  # fmt: skip
            for attempt in range(self.opt("judge_retries", 3)):
                try:
                    completion = judge.create(**request)
                except Exception as e:
                    log.warning("mcpatlas: judge %s claim %d failed: %s", row["TASK"], i, e)
                    if getattr(e, "status_code", None) == 402:  # budget exhausted
                        return
                    continue
                verdict = parse_judgement(completion.choices[0].message.content)
                if verdict is not None:
                    done[str(i)] = {"claim": claims[i], "outcome": verdict.get("coverage_outcome"),
                                    "justification": verdict.get("justification", ""),
                                    "usage": judge.last_usage()}  # fmt: skip
                    return
                log.warning("mcpatlas: judge %s claim %d: unparseable verdict (attempt %d)", row["TASK"], i, attempt)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.opt("judge_parallel", 4)) as inner:
            list(inner.map(one, todo))
        if todo:
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_json(path, done)
        if len(done) < len(claims):
            return {"status": "judge_failed", "outcomes": None, "claims_unjudged": len(claims) - len(done)}
        return {"status": "judged", "outcomes": [done[str(i)]["outcome"] for i in range(len(claims))],
                "usage": [d["usage"] for d in done.values() if d.get("usage")]}  # fmt: skip

    def _aggregate(self, jobs, gens, judged, judge, info) -> dict[str, Any]:
        threshold = self.opt("threshold", 0.75)
        per_task: dict[str, list[float]] = {}
        failed = 0
        usage: list[dict] = []
        statuses = []
        for (row, _r), gen, j in zip(jobs, gens, judged):
            statuses.append(j["status"])
            usage += j.get("usage") or []
            if j["status"] == "no_claims":  # upstream: coverage None, not a valid score
                continue
            if j["outcomes"] is None:
                failed += 1
                continue
            per_task.setdefault(row["TASK"], []).append(coverage(j["outcomes"]))
        policy = failure_policy(self.id, "items", failed, len(jobs), self.opt("max_failed_frac", 0.05))
        scores = [c for cs in per_task.values() for c in cs]
        if not scores:
            raise SystemExit("mcpatlas: nothing was scored")

        def rate(t: float) -> float:
            return sum(c >= t for c in scores) / len(scores)

        out: dict[str, Any] = {
            "value": rate(threshold),
            "n": len(scores),
            **policy,
            "metric_note": f"pass@1[avg-of-{self.repeats}]: fraction of (task, repeat) items with "
                           f"claim coverage >= {threshold} (upstream pass_rate_0.75)",
            "threshold": threshold,
            "pass_rate_0.50": rate(0.5),
            "pass_rate_0.75": rate(0.75),
            "mean_coverage": round(sum(scores) / len(scores), 4),
            "tasks_scored": len(per_task),
            "statuses": _count(statuses),
            "stops": _count(g["stop"] for g in gens if g),
            **info,
        }  # fmt: skip
        k = self.repeats
        if k > 1:  # over the tasks with all k repeats scored
            full = {t: cs for t, cs in per_task.items() if len(cs) == k}
            pk = (sum(pass_at_k(k, sum(c >= threshold for c in cs), k) for cs in full.values())
                  / len(full)) if full else None  # fmt: skip
            out["pass_at_k"] = pass_at_k_record(k, out["value"], pk, len(full),
                                                f"pass@{k}: a repeat with coverage >= {threshold}, tasks with all k scored")
        else:
            out["pass_at_k"] = pass_at_k_record(1, out["value"], out["value"], len(scores))
        replays = [g["replay"] for g in gens if g and g.get("replay")]
        if replays:
            out["replay"] = {"tool_calls": sum(x["calls"] for x in replays),
                             "failed": sum(len(x["failed"]) for x in replays),
                             "failed_tools": _count(f["tool"] for x in replays for f in x["failed"])}  # fmt: skip
        offline = getattr(self, "offline_servers", None)
        if offline is not None:
            out["offline_servers"] = offline
        out.update(judge.describe())
        out["judge_request"] = ("score_claims.py: one call per claim, temperature 0, response_format "
                                "json_schema claim_evaluation (minus judge_dropped_params)")  # fmt: skip
        out["judge_usage"] = total_usage(usage, len(scores))
        log.info("mcpatlas: pass rate %.3f (>= %.2f), mean coverage %.3f over %d items",
                 out["value"], threshold, out["mean_coverage"], len(scores))
        return out
