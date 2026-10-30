"""MCP-Atlas adapter tests: a fake policy model and a fake agent-environment
(httpx MockTransport), a fake judge client; no network, no containers.

Optional checks against upstream:

- ``MCPATLAS_UPSTREAM=<checkout of scaleapi/mcp-atlas at mcpatlas.COMMIT>``:
  the scorer file's sha256, and the ported claim extraction / prompt / schema
  against upstream's own functions;
- ``MCPATLAS_PARQUET=<MCP-Atlas.parquet at mcpatlas.DATASET_REVISION>``: the
  file's sha256 and the pinned subset counts and digests.
"""

import ast
import hashlib
import json
import os
import re
import types
from pathlib import Path

import pytest

pytest.importorskip("openai")
httpx = pytest.importorskip("httpx")
pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")

from granite_evals import meter  # noqa: E402
from granite_evals.benchmarks import judge_general as jg  # noqa: E402
from granite_evals.benchmarks import mcpatlas as ma  # noqa: E402
from granite_evals.registry import RunConfig, get  # noqa: E402

# -- fakes ------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _placeholder_judge_endpoint(monkeypatch):
    """The module has no built-in judge URL (sites pass judge_base_url); tests that take the default path need one."""
    monkeypatch.setattr(jg, "DEFAULT_JUDGE_BASE_URL", "https://judge.example.test/v1")



@pytest.fixture(autouse=True)
def no_waits(monkeypatch):
    monkeypatch.setattr(ma, "RETRY_WAIT", {"429": 0.0, "5xx": 0.0, "timeout": 0.0})


TOOLS = [
    {"name": "wikipedia_search", "description": "search", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "calculator_calculate", "description": "calc", "inputSchema": {"type": "object"}},
    {"name": "met-museum_get-museum-object", "description": "obj", "inputSchema": {"type": "object"}},
    {"name": "notion_search", "description": "n", "inputSchema": {"type": "object"}},
]


def row(task, claims, tools=("wikipedia_search", "calculator_calculate"), traj=None, prompt="Q?"):
    return {
        "TASK": task,
        "ENABLED_TOOLS": json.dumps(list(tools)),
        "PROMPT": prompt,
        "GTFA_CLAIMS": json.dumps(claims),
        "TRAJECTORY": json.dumps(traj or [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "wikipedia_search", "arguments": "{}"}}]},
            {"role": "tool", "content": "x", "tool_call_id": "c1"},
        ]),
    }  # fmt: skip


def write_parquet(path: Path, rows: list[dict]) -> str:
    pq.write_table(pa.Table.from_pylist(rows), path)
    return str(path)


class FakeWorld:
    """The model endpoint and the agent-environment behind one MockTransport."""

    def __init__(self, script=None, tool=None, servers=None):
        self.script = script or (lambda messages, tools: {"role": "assistant", "content": "final answer"})
        self.tool = tool or (lambda name, args: httpx.Response(200, json=[{"type": "text", "text": f"{name} ok"}]))
        self.servers = servers if servers is not None else [[s, "OK"] for s in ma.KEYLESS_SERVERS]
        self.model_requests: list[dict] = []
        self.tool_calls: list[tuple[str, dict]] = []
        self.model_status: list[int] = []  # statuses to return before succeeding

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/chat/completions"):
            body = json.loads(request.content)
            self.model_requests.append(body)
            if self.model_status:
                return httpx.Response(self.model_status.pop(0), text="busy")
            msg = self.script(body["messages"], body["tools"])
            return httpx.Response(200, json={"choices": [{"message": msg}],
                                             "usage": {"prompt_tokens": 10, "completion_tokens": 2}})  # fmt: skip
        if path == "/health":
            return httpx.Response(200, json={"status": "health_and_client_connection_ok"})
        if path == "/enabled-servers":
            online = sum(s[1] == "OK" for s in self.servers)
            return httpx.Response(200, json={"servers": self.servers, "total": len(self.servers),
                                             "online": online, "offline": len(self.servers) - online})  # fmt: skip
        if path == "/list-tools":
            return httpx.Response(200, json=TOOLS)
        if path == "/call-tool":
            body = json.loads(request.content)
            self.tool_calls.append((body["tool_name"], body["tool_args"]))
            return self.tool(body["tool_name"], body["tool_args"])
        return httpx.Response(404)


def tool_call(name, args, cid="t1"):
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": cid, "type": "function", "function": {"name": name, "arguments": args}}]}  # fmt: skip


class FakeJudgeClient:
    def __init__(self, outcome):
        self.outcome, self.requests = outcome, []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    def create(self, **request):
        self.requests.append(request)
        prompt = request["messages"][0]["content"]
        claim = prompt.split("CLAIM TO EVALUATE:\n", 1)[1].split("\nMODEL RESPONSE TO ANALYZE:", 1)[0]
        response = prompt.split("MODEL RESPONSE TO ANALYZE:\n", 1)[1].split("\nINSTRUCTIONS:", 1)[0]
        out = self.outcome(claim, response)
        if isinstance(out, Exception):
            raise out
        content = out if out.startswith(("{", "not json", "```")) else json.dumps(
            {"claim_text": claim, "coverage_outcome": out, "justification": "j", "confidence_level": 0.9})
        msg = types.SimpleNamespace(content=content, reasoning_content=None)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=None)


def bench_for(tmp_path, rows, monkeypatch, world=None, judge=None, **cfg):
    options = {"env_url": "http://atlas-env", **cfg.pop("options", {})}
    config = RunConfig(model="m", output_dir=tmp_path / "out", dataset=write_parquet(tmp_path / "d.parquet", rows),
                       options=options, **cfg)  # fmt: skip
    bench = get("mcpatlas")(config)
    if world is not None:
        monkeypatch.setattr(ma.MCPAtlas, "_transport", httpx.MockTransport(world))
    client = FakeJudgeClient(judge or (lambda claim, response: "fulfilled" if claim in response else "not_fulfilled"))
    monkeypatch.setattr(ma.MCPAtlas, "make_judge", lambda self, base_url, served, cache_split=None: jg.Judge(
        base_url="http://judge/v1", model=jg.DEFAULT_JUDGE_MODEL, api_key="k", is_self=False, client=client))
    return bench, client


# -- the scorer port ----------------------------------------------------------


def test_extract_claims_formats():
    assert ma.extract_claims(None) == []
    assert ma.extract_claims(["- The sky is blue.", "ok", "2) Water boils at 100 C"]) == [
        "The sky is blue.", "Water boils at 100 C"]
    assert ma.extract_claims('["“A quoted claim”", "x"]') == ['"A quoted claim']
    assert ma.extract_claims("['python list one', 'python list two']") == ["python list one", "python list two"]
    assert ma.extract_claims("first claim here; second claim here") == ["first claim here", "second claim here"]
    assert ma.extract_claims("intro line\n• bullet one\n• bullet two") == ["intro line", "bullet one", "bullet two"]
    assert ma.extract_claims("line one is long\nline two is long") == ["line one is long", "line two is long"]


def test_coverage_and_skips():
    assert ma.coverage(["fulfilled", "partially_fulfilled", "not_fulfilled"]) == 0.5
    assert ma.coverage(["fulfilled", "weird"]) == 0.5
    assert ma.skipped_response("") and ma.skipped_response("ERROR: boom") and not ma.skipped_response("ok")
    long = "x" * (ma.MAX_RESPONSE_CHARS + 5)
    assert ma.truncate_response(long).endswith("[TRUNCATED — original response was too long]")
    prompt = ma.claim_evaluation_prompt("CLAIM", "RESP")
    assert "CLAIM TO EVALUATE:\nCLAIM\nMODEL RESPONSE TO ANALYZE:\nRESP\nINSTRUCTIONS:" in prompt
    assert ma.parse_judgement('```json\n{"coverage_outcome": "fulfilled"}\n```') == {"coverage_outcome": "fulfilled"}
    assert ma.parse_judgement("nope") is None
    assert ma.pass_at_k(2, 1, 2) == 1.0 and ma.pass_at_k(2, 0, 2) == 0.0


def test_servers_and_subsets():
    assert ma.server_of("osm-mcp-server_geocode") == "osm-mcp-server"
    assert ma.server_of("weather-data_current") == "weather-data"
    assert ma.server_of("weather_get_forecast") == "weather"
    assert ma.server_of("brave_brave_web_search") == "brave-search"
    assert ma.server_of("MongoDB_find") == "mongodb"
    assert ma.server_of("f1-mcp-server_get_driver_info") == "?"
    keyless = row("a", ["c1 claim"])
    gt_only = row("b", ["c1 claim"], tools=("wikipedia_search", "notion_search"))
    keyed = row("c", ["c1 claim"], tools=("notion_search",), traj=[tool_call("notion_search", "{}")])
    assert [ma.in_subset(r, "keyless") for r in (keyless, gt_only, keyed)] == [True, False, False]
    assert [ma.in_subset(r, "keyless-gt") for r in (keyless, gt_only, keyed)] == [True, True, False]
    assert all(ma.in_subset(r, "all") for r in (keyless, gt_only, keyed))
    assert set(ma.KEYLESS_SERVERS).isdisjoint(ma.KEYED_SERVERS) and len(ma.ALL_SERVERS) == 36


# -- the agent loop -------------------------------------------------------------


def test_agent_loop_offers_enabled_tools_and_feeds_back_results(tmp_path, monkeypatch):
    def script(messages, tools):
        n = sum(m["role"] == "assistant" for m in messages)
        if n == 0:
            return {**tool_call("wikipedia_search", '{"q": "x"}', "a"), "reasoning_content": "think"}
        if n == 1:
            return tool_call("calculator_calculate", "{bad json", "b")
        if n == 2:
            return tool_call("met-museum_get-museum-object", '{"id": 1}', "c")
        return {"role": "assistant", "content": "the answer", "extra_field": "dropped"}

    world = FakeWorld(script)
    env = ma.AtlasEnv("http://atlas-env", transport=httpx.MockTransport(world))
    model = ma.ModelClient("http://policy/v1", "served", {}, timeout=5, transport=httpx.MockTransport(world))
    r = row("t", ["claim"], tools=("wikipedia_search", "calculator_calculate", "met-museum_get-museum-object",
                                   "f1-mcp-server_x"))  # fmt: skip
    out = ma.run_agent(r, env, model, max_turns=10, max_tool_calls=10, tool_timeout=5, task_timeout=100)
    assert out["stop"] == "completed" and out["response"] == "the answer" and out["turns"] == 4
    assert out["tools_offered"] == 3 and out["tools_missing"] == ["f1-mcp-server_x"]
    first = world.model_requests[0]
    assert set(first) == {"model", "messages", "tools"}  # no sampling parameters, no thinking flag
    assert first["messages"] == [{"role": "user", "content": "Q?"}]
    assert first["tools"][0] == {"type": "function", "function": {
        "name": "wikipedia_search", "description": "search", "parameters": {"type": "object", "properties": {}},
        "strict": False}}  # fmt: skip
    msgs = out["messages"]
    assert msgs[1]["reasoning_content"] == "think" and "extra_field" not in msgs[-1]
    assert msgs[2] == {"role": "tool", "content": [{"type": "text", "text": "wikipedia_search ok"}],
                       "tool_call_id": "a"}  # fmt: skip
    assert msgs[4]["content"][0]["text"].startswith("Error: ")
    assert world.tool_calls == [("wikipedia_search", {"q": "x"}),
                                ("met-museum_get-museum-object", {"id": 1, "returnImage": False})]  # fmt: skip
    assert out["usage"] == {"prompt_tokens": 40, "completion_tokens": 8}


def test_tool_results_follow_the_harness(monkeypatch):
    def tool(name, args):
        if name == "a":
            return httpx.Response(500, json={"detail": "server down"})
        if name == "b":
            return httpx.Response(200, json=[{"type": "text", "text": "  "}])
        if name == "c":
            return httpx.Response(200, json=[{"type": "image", "data": "..."}, {"type": "text", "text": "t"}])
        raise httpx.ReadTimeout("slow")

    env = ma.AtlasEnv("http://atlas-env", transport=httpx.MockTransport(FakeWorld(tool=tool)))
    assert env.call_tool("a", {}, 5) == [{"type": "text", "text": '{"detail":"server down"}'}]
    assert env.call_tool("b", {}, 5) == [{"type": "text", "text": "success"}]
    assert env.call_tool("c", {}, 5) == [{"type": "text", "text": "t"}]
    assert env.call_tool("d", {}, 60) == [{"type": "text", "text": "Tool call timed out after 60s"}]


def test_model_errors_retry_end_or_raise():
    world = FakeWorld()
    world.model_status = [503, 429]
    model = ma.ModelClient("http://policy/v1", "s", {"temperature": 0.5}, timeout=5, transport=httpx.MockTransport(world))
    msg, _, err = model.complete([{"role": "user", "content": "q"}], [])
    assert msg["content"] == "final answer" and not err and len(world.model_requests) == 3
    assert world.model_requests[0]["temperature"] == 0.5
    world.model_status = [400]
    msg, _, err = model.complete([], [])
    assert msg is None and err.startswith("HTTP 400")
    world.model_status = [502, 502, 502]
    with pytest.raises(ma.InfraError):
        model.complete([], [])

    def down(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(ma.InfraError):
        ma.ModelClient("http://p/v1", "s", {}, timeout=5, transport=httpx.MockTransport(down)).complete([], [])


def test_limits_and_timeout():
    world = FakeWorld(lambda messages, tools: tool_call("wikipedia_search", "{}"))
    env = ma.AtlasEnv("http://e", transport=httpx.MockTransport(world))
    model = ma.ModelClient("http://p/v1", "s", {}, timeout=5, transport=httpx.MockTransport(world))
    out = ma.run_agent(row("t", ["c"]), env, model, max_turns=50, max_tool_calls=3, tool_timeout=5, task_timeout=99)
    assert out["stop"] == "max_tool_calls" and out["tool_calls"] == 3 and out["response"] == ""
    out = ma.run_agent(row("t", ["c"]), env, model, max_turns=2, max_tool_calls=0, tool_timeout=5, task_timeout=99)
    assert out["stop"] == "max_turns" and out["turns"] == 2
    ticks = iter(range(0, 10_000, 1000))
    out = ma.run_agent(row("t", ["c"]), env, model, max_turns=50, max_tool_calls=100, tool_timeout=5,
                       task_timeout=1500, clock=lambda: next(ticks))  # fmt: skip
    assert out["stop"] == "timeout" and out["response"].startswith("ERROR: timeout")


def test_llm_extra_options(tmp_path):
    def extra(**options):
        return get("mcpatlas")(RunConfig(model="m", output_dir=tmp_path, options=options)).llm_extra()

    assert extra() == {}
    assert extra(enable_thinking="false", temperature="0.6", extra_llm_params='{"top_k": 20}') == {
        "temperature": 0.6, "chat_template_kwargs": {"enable_thinking": False}, "top_k": 20}


# -- the benchmark ------------------------------------------------------------------


def test_gold_scores_every_claim_without_model_or_environment(tmp_path, monkeypatch):
    rows = [row("b", ["Claim beta one", "Claim beta two"]), row("a", ["Claim alpha"]), row("c", ["Claim gamma"])]

    def no_network(request):
        pytest.fail(f"gold touched {request.url}")

    bench, client = bench_for(tmp_path, rows, monkeypatch, world=no_network,
                              options={"agent": "gold", "env_url": ""}, limit=2)  # fmt: skip
    assert not bench.needs_server() and not bench.serves()
    out = bench.run("http://policy/v1", "served")
    assert out["value"] == 1.0 and out["n"] == 2 and out["tasks"] == ["a", "b"]  # first N by id
    assert len(client.requests) == 3
    req = client.requests[0]
    assert req["model"] == jg.DEFAULT_JUDGE_MODEL and req["temperature"] == 0.0
    assert req["response_format"]["json_schema"]["name"] == "claim_evaluation"
    assert (tmp_path / "out/repeat-0/a.json").exists() and (tmp_path / "out/repeat-0/a.judgments.json").exists()
    assert out["deviations"] and out["agent"] == "gold" and out["subset"] == "keyless"


def test_generate_then_score_and_resume(tmp_path, monkeypatch):
    rows = [row("t1", ["Paris is the capital", "It has 2 million people"]), row("t2", ["Answer is 42 exactly"])]
    answers = {"Q1": "Paris is the capital", "Q2": "Answer is 42 exactly"}
    rows[0]["PROMPT"], rows[1]["PROMPT"] = "Q1", "Q2"
    world = FakeWorld(lambda messages, tools: {"role": "assistant", "content": answers[messages[0]["content"]]})
    gen, client = bench_for(tmp_path, rows, monkeypatch, world=world, phase="generate", repeats=2)
    out = gen.run("http://policy/v1", "served")
    assert out["n"] == 4 and out["generations_failed"] == 0 and not client.requests and "value" not in out
    assert len(world.model_requests) == 4
    saved = json.loads((tmp_path / "out/repeat-1/t1.json").read_text())
    assert saved["response"] == "Paris is the capital" and saved["stop"] == "completed"

    def offline(request):
        pytest.fail(f"score phase touched {request.url}")

    score, client = bench_for(tmp_path, rows, monkeypatch, world=offline, phase="score", repeats=2,
                              options={"env_url": ""})  # fmt: skip
    assert not score.serves()
    out = score.run("http://policy/v1", "served")
    # t1: coverage 0.5 (< 0.75, >= 0.5) twice; t2: 1.0 twice
    assert out["value"] == 0.5 and out["pass_rate_0.50"] == 1.0 and out["mean_coverage"] == 0.75
    assert out["pass_at_k"] == {"k": 2, "pass_at_1": 0.5, "pass_at_k": 0.5, "n": 2,
                                "how": "pass@2: a repeat with coverage >= 0.75, tasks with all k scored"}
    assert out["n"] == 4
    assert len(client.requests) == 6
    again, client = bench_for(tmp_path, rows, monkeypatch, world=offline, phase="score", repeats=2,
                              options={"env_url": ""})  # fmt: skip
    assert again.run("http://p/v1", "s")["value"] == 0.5 and not client.requests  # judgments resumed


def test_score_phase_without_generations_fails(tmp_path, monkeypatch):
    bench, _ = bench_for(tmp_path, [row("t", ["claim one"])], monkeypatch, world=FakeWorld(), phase="score")
    with pytest.raises(SystemExit):
        bench.run("http://p/v1", "s")


def test_infrastructure_failures_are_not_saved(tmp_path, monkeypatch):
    world = FakeWorld()
    world.model_status = [502] * 3
    rows = [row("t", ["claim one"]), row("u", ["claim two"])]
    bench, _ = bench_for(tmp_path, rows, monkeypatch, world=world, phase="generate", options={"concurrency": "1"})
    out = bench.run("http://p/v1", "s")
    assert out["generations_failed"] == 1 and out["n"] == 1
    assert len(list((tmp_path / "out/repeat-0").glob("*.json"))) == 1


def test_judge_failures_are_dropped_and_counted(tmp_path, monkeypatch):
    rows = [row(f"t{i}", [f"claim number {i}"]) for i in range(4)]
    err = Exception("Error code: 503")
    world = FakeWorld(lambda messages, tools: {"role": "assistant", "content": "claim number 0 claim number 1"})
    bench, client = bench_for(tmp_path, rows, monkeypatch, world=world, judge=lambda c, r: (
        err if "3" in c else "not json" if "2" in c else "fulfilled" if c in r else "not_fulfilled"),
        options={"max_failed_frac": "0.5"})  # fmt: skip
    out = bench.run("http://p/v1", "s")
    assert out["n"] == 2 and out["value"] == 1.0 and out["statuses"]["judge_failed"] == 2
    assert len(client.requests) == 2 + 3 + 3  # unparseable and failed verdicts are retried judge_retries times
    strict, _ = bench_for(tmp_path, rows, monkeypatch, world=world, judge=lambda c, r: err)
    with pytest.raises(SystemExit):
        strict.run("http://p/v1", "s")


def test_error_responses_score_zero_without_judge(tmp_path, monkeypatch):
    world = FakeWorld(lambda messages, tools: {"role": "assistant", "content": ""})
    bench, client = bench_for(tmp_path, [row("t", ["claim one", "claim two"])], monkeypatch, world=world)
    out = bench.run("http://p/v1", "s")
    assert out["value"] == 0.0 and out["mean_coverage"] == 0.0 and not client.requests


def test_offline_servers_stop_the_run(tmp_path, monkeypatch):
    servers = [[s, "OK"] for s in ma.KEYLESS_SERVERS if s != "wikipedia"] + [["wikipedia", "ERROR_NOT_ONLINE"]]
    bench, _ = bench_for(tmp_path, [row("t", ["claim one"])], monkeypatch, world=FakeWorld(servers=servers))
    with pytest.raises(SystemExit, match="wikipedia"):
        bench.run("http://p/v1", "s")
    ok, _ = bench_for(tmp_path, [row("t", ["claim one"])], monkeypatch, world=FakeWorld(servers=servers),
                      options={"allow_offline": "true"})  # fmt: skip
    assert ok.run("http://p/v1", "s")["offline_servers"] == ["wikipedia"]


def test_replay_reports_failing_tools(tmp_path, monkeypatch):
    world = FakeWorld(tool=lambda name, args: httpx.Response(500, json={"detail": "nope"}))
    bench, _ = bench_for(tmp_path, [row("t", ["Claim one here"])], monkeypatch, world=world,
                         options={"agent": "replay"})  # fmt: skip
    assert not bench.needs_server()
    out = bench.run("http://p/v1", "s")
    assert out["replay"] == {"tool_calls": 1, "failed": 1, "failed_tools": {"wikipedia_search": 1}}
    assert out["value"] == 1.0  # the answer is the gold claims


def test_subset_all_needs_every_key_and_never_echoes_values(tmp_path, monkeypatch):
    names = [v for vs in ma.KEYED_SERVERS.values() for v in vs]
    for name in names:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NOTION_TOKEN", "value-not-to-print")
    bench = get("mcpatlas")(RunConfig(model="m", output_dir=tmp_path, options={"subset": "all"}))
    with pytest.raises(SystemExit) as e:
        bench.sandbox_env()
    assert "AIRTABLE_API_KEY" in str(e.value) and "value-not-to-print" not in str(e.value)
    assert "NOTION_TOKEN" not in str(e.value)
    for name in names:
        monkeypatch.setenv(name, "v")
    env = bench.sandbox_env()
    assert env["ENABLED_SERVERS"].count(",") == 35 and env["NOTION_TOKEN"] == "v"
    keyless = get("mcpatlas")(RunConfig(model="m", output_dir=tmp_path)).sandbox_env()
    assert keyless == {"ENABLED_SERVERS": ",".join(ma.KEYLESS_SERVERS)}


def test_pinned_subset_digest_is_checked(tmp_path):
    bench = get("mcpatlas")(RunConfig(model="m", output_dir=tmp_path))
    with pytest.raises(SystemExit, match="pinned 30"):
        bench.select([row("a", ["claim one"])], pinned=True)
    assert len(bench.select([row("a", ["claim one"])], pinned=False)) == 1


def test_podman_needs_env_url():
    with pytest.raises(SystemExit, match="env_url"):
        ma.EnvSandbox(ma.IMAGE, port=1984, env={}, backend="podman")


def test_default_judge_is_metered(tmp_path, monkeypatch):
    monkeypatch.setattr(jg, "DEFAULT_JUDGE_BASE_URL", "https://judge.example.test/v1")  # no built-in endpoint: use a placeholder
    monkeypatch.setenv("GRANITE_EVALS_JUDGE_API_KEY", "k")
    seen = []
    monkeypatch.setattr(meter, "metered", lambda url, role: seen.append((url, role)) or "http://127.0.0.1:1/v1")
    j = get("mcpatlas")(RunConfig(model="m", output_dir=tmp_path)).make_judge("http://p/v1", "s")
    assert seen == [(jg.DEFAULT_JUDGE_BASE_URL, "judge")] and j.model == "aws/claude-sonnet-5"


def test_registered_metadata():
    cls = type(get("mcpatlas")(RunConfig(model="m", output_dir=Path("/tmp/x"))))
    assert cls.splittable and cls.extra == "judged" and cls.dataset_revision == ma.DATASET_REVISION


# -- against upstream (optional) ------------------------------------------------------


def _upstream_functions(path: Path) -> dict:
    tree = ast.parse(path.read_text())
    ns: dict = {"re": re, "json": json, "ast": ast, "List": list}
    wanted = {"clean_claim_text", "extract_claims", "get_single_claim_evaluation_schema"}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            exec(compile(ast.Module([node], []), str(path), "exec"), ns)
        if isinstance(node, ast.ClassDef) and node.name == "CoverageEvaluator":
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "_get_single_claim_evaluation_prompt":
                    exec(compile(ast.Module([fn], []), str(path), "exec"), ns)
    return ns


@pytest.mark.skipif(not os.environ.get("MCPATLAS_UPSTREAM"), reason="MCPATLAS_UPSTREAM not set")
def test_port_matches_upstream():
    root = Path(os.environ["MCPATLAS_UPSTREAM"])
    for rel, digest in ma.UPSTREAM_FILES.items():
        assert hashlib.sha256((root / rel).read_bytes()).hexdigest() == digest, rel
    up = _upstream_functions(root / "services/scoring/score_claims.py")
    assert up["get_single_claim_evaluation_schema"]() == ma.get_single_claim_evaluation_schema()
    assert up["_get_single_claim_evaluation_prompt"](None, "C", "R") == ma.claim_evaluation_prompt("C", "R")
    blobs = [None, "", "  ", ["a", "- bullet claim", "“q” long"], '["json one", "json two"]',
             "['lit one', 'lit two']", "[not a list]", "a; bb; long enough", "x || long claim || y",
             "intro\n- one claim\n- two claim", "1. first claim\n2. second claim", "plain line one\nline two",
             "ends with quote”.", "dash – and — and …", 12345678]  # fmt: skip
    parquet = os.environ.get("MCPATLAS_PARQUET")
    if parquet:
        blobs += [r["GTFA_CLAIMS"] for r in pq.read_table(parquet).to_pylist()]
    for blob in blobs:
        assert ma.extract_claims(blob) == up["extract_claims"](blob), blob


@pytest.mark.skipif(not os.environ.get("MCPATLAS_PARQUET"), reason="MCPATLAS_PARQUET not set")
def test_pinned_dataset_subsets():
    path = Path(os.environ["MCPATLAS_PARQUET"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == ma.PARQUET_SHA256
    rows = pq.read_table(path).to_pylist()
    assert len(rows) == ma.N_TASKS
    for subset, (count, digest) in ma.SUBSETS.items():
        ids = [r["TASK"] for r in rows if ma.in_subset(r, subset)]
        assert (len(ids), ma.ids_digest(ids)) == (count, digest), subset
    assert all(ma.gold_response(r) for r in rows)
