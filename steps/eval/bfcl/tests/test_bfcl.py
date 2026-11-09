"""BFCL v4 adapter tests: result parsing, --limit selection, resume pruning and
the MCP web-search backend, with fakes (no model, no network)."""

import json
import subprocess
import sys

import httpx
import pytest

from granite_evals.benchmarks import bfcl
from granite_evals.registry import get

SEARCH_TEXT = (
    "Found 2 results for: tallest building in Europe\n\n"
    "1. List of tallest buildings in Europe - Wikipedia\n"
    "   https://en.wikipedia.org/wiki/List_of_tallest_buildings_in_Europe\n"
    "   The Lakhta Center in Saint Petersburg is the tallest skyscraper in Europe.\n"
    "   As of 2026, ...\n\n"
    "2. Margherita Hut - Wikipedia\n"
    "   https://en.wikipedia.org/wiki/Margherita_Hut\n\n"
)


def test_registered_with_suite_metric():
    cls = get("bfcl-v4")
    assert cls is bfcl.BFCLv4
    assert cls.metric == "overall_accuracy accuracy"
    assert cls.extra == "bfcl" and cls.harness_packages == ("bfcl-eval",)
    assert cls.dataset.startswith("https://pypi.org/") and cls.dataset_revision.startswith("sha256:")


def test_parse_search_text_maps_to_bfcl_shape():
    hits = bfcl.parse_search_text(SEARCH_TEXT)
    assert hits == [
        {
            "title": "List of tallest buildings in Europe - Wikipedia",
            "href": "https://en.wikipedia.org/wiki/List_of_tallest_buildings_in_Europe",
            "body": "The Lakhta Center in Saint Petersburg is the tallest skyscraper in Europe. As of 2026, ...",
        },
        {"title": "Margherita Hut - Wikipedia", "href": "https://en.wikipedia.org/wiki/Margherita_Hut", "body": ""},
    ]
    assert bfcl.parse_search_text("Found 0 results for: zzz\n\n") == []


def test_select_ids_first_n_per_category_with_memory_prereqs():
    entries = {
        "simple_python": [{"id": f"simple_python_{i}"} for i in range(5)],
        "memory_kv": [
            {"id": "memory_kv_prereq_0-a-0", "depends_on": []},
            {"id": "memory_kv_prereq_1-a-1", "depends_on": ["memory_kv_prereq_0-a-0"]},
            {"id": "memory_kv_prereq_2-b-0", "depends_on": []},
            {"id": "memory_kv_0-a-0", "depends_on": ["memory_kv_prereq_0-a-0", "memory_kv_prereq_1-a-1"]},
            {"id": "memory_kv_1-b-0", "depends_on": ["memory_kv_prereq_2-b-0"]},
        ],
    }
    ids = bfcl.select_ids(entries, 1)
    assert ids["simple_python"] == ["simple_python_0"]
    assert ids["memory_kv"] == ["memory_kv_prereq_0-a-0", "memory_kv_prereq_1-a-1", "memory_kv_0-a-0"]
    full = bfcl.select_ids(entries, None)
    assert len(full["memory_kv"]) == 5 and len(full["simple_python"]) == 5


def test_prune_infra_errors_keeps_model_failures(tmp_path):
    f = tmp_path / "BFCL_v4_simple_python_result.json"
    lines = [
        {"id": "a", "result": [{"f": "{}"}]},
        {"id": "b", "result": "Error during inference: Connection error."},
        {"id": "c", "result": "Error during inference: Expecting value: line 1 column 1"},
    ]
    f.write_text("".join(json.dumps(x) + "\n" for x in lines))
    assert bfcl.prune_infra_errors(f) == 1
    assert [json.loads(x)["id"] for x in f.read_text().splitlines()] == ["a", "c"]
    assert bfcl.prune_infra_errors(tmp_path / "missing.json") == 0


def test_read_overall_and_category_scores(tmp_path):
    csv_path = tmp_path / "data_overall.csv"
    cols = ["Rank", "Overall Acc", "Model", *bfcl.OVERALL_GROUPS.values()]
    vals = ["1", "52.41%", "m", "80.00%", "70.00%", "30.00%", "N/A", "10.00%", "90.00%", "85.00%"]
    csv_path.write_text(",".join(cols) + "\n" + ",".join(vals) + "\n")
    overall = bfcl.read_overall(csv_path)
    assert overall["overall"] == pytest.approx(0.5241)
    assert overall["groups"]["web_search"] is None and overall["groups"]["memory"] == pytest.approx(0.1)

    d = tmp_path / "score" / "granite-fc" / "agentic" / "memory" / "kv"
    d.mkdir(parents=True)
    (d / "BFCL_v4_memory_kv_score.json").write_text(
        json.dumps({"accuracy": 0.5, "correct_count": 1, "total_count": 2}) + "\n" + json.dumps({"id": "x"}) + "\n"
    )
    scores = bfcl.read_category_scores(tmp_path / "score" / "granite-fc")
    assert scores == {"memory_kv": {"accuracy": 0.5, "correct_count": 1, "total_count": 2}}


def _mcp_transport(log, *, throttle_first=0, sse=False):
    state = {"throttled": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        log.append((body.get("method"), request.headers.get("mcp-session-id")))
        if body["method"] == "initialize":
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": {}},
                                  headers={"mcp-session-id": "s1"})
        if body["method"] == "notifications/initialized":
            return httpx.Response(202)
        assert request.headers["mcp-session-id"] == "s1"
        assert body["params"] == {"name": "google_pse_search",
                                  "arguments": {"query": "q", "num_results": 3}}
        if state["throttled"] < throttle_first:
            state["throttled"] += 1
            return httpx.Response(429, text="slow down")
        msg = {"jsonrpc": "2.0", "id": body["id"], "result": {"content": [{"type": "text", "text": SEARCH_TEXT}]}}
        if sse:
            return httpx.Response(200, text=f"event: message\ndata: {json.dumps(msg)}\n\n",
                                  headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=msg)

    return httpx.MockTransport(handler)


@pytest.mark.parametrize("sse", [False, True])
def test_mcp_search_handshake_and_retry(sse, monkeypatch):
    monkeypatch.setattr(bfcl.time, "sleep", lambda s: None)
    log = []
    s = bfcl.MCPSearch("http://mcp/mcp", client=httpx.Client(transport=_mcp_transport(log, throttle_first=2, sse=sse)))
    hits = s.search("q", 3)
    assert len(hits) == 2 and hits[0]["href"].startswith("https://en.wikipedia.org/")
    assert log[0] == ("initialize", None) and log[1] == ("notifications/initialized", "s1")
    assert s.stats["throttled"] == 2 and s.stats["errors"] == 0
    s.search("q", 3)  # the session is reused
    assert sum(1 for m, _ in log if m == "initialize") == 1


def test_search_engine_query_replacement_shape(monkeypatch):
    class FakeSearch:
        def __init__(self):
            self.calls = []

        def search(self, query, n):
            self.calls.append((query, n))
            return bfcl.parse_search_text(SEARCH_TEXT)

    class API:
        show_snippet = True

    fake = FakeSearch()
    query = bfcl.make_search_engine_query(fake)
    api = API()
    assert query(api, "q", max_results=50, region="de-de") == bfcl.parse_search_text(SEARCH_TEXT)
    assert fake.calls == [("q", 10)]  # the MCP tool caps num_results at 10
    api.show_snippet = False
    assert query(api, "q", max_results=1) == [{"title": "List of tallest buildings in Europe - Wikipedia",
                                               "href": "https://en.wikipedia.org/wiki/List_of_tallest_buildings_in_Europe"}]

    class Broken:
        def search(self, query, n):
            raise RuntimeError("down")

    assert "error" in bfcl.make_search_engine_query(Broken())(api, "q")


def test_mcp_search_gives_up_with_error(monkeypatch):
    monkeypatch.setattr(bfcl.time, "sleep", lambda s: None)
    transport = httpx.MockTransport(lambda r: httpx.Response(503))
    s = bfcl.MCPSearch("http://mcp/mcp", client=httpx.Client(transport=transport), max_attempts=3)
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        s.search("q", 3)
    assert s.stats["errors"] == 1


def test_run_collects_child_results(tmp_path, monkeypatch):
    """run(): the child is faked; results come from BFCL's score files."""
    from granite_evals.registry import RunConfig

    def fake_child(cmd, env, check):
        root = tmp_path / "out" / "bfcl"
        cfg = json.loads((root / bfcl.CHILD_CONFIG).read_text())
        assert cfg["sampling"] == {"temperature": 0.5} and cfg["limit"] == 3
        assert env["OPENAI_BASE_URL"] == "http://srv/v1" and env["BFCL_PROJECT_ROOT"] == str(root)
        d = root / "score" / bfcl.REGISTRY_NAME / "non_live"
        d.mkdir(parents=True)
        (d / "BFCL_v4_simple_python_score.json").write_text(
            json.dumps({"accuracy": 2 / 3, "correct_count": 2, "total_count": 3}) + "\n")
        cols = ["Rank", "Overall Acc", "Model", *bfcl.OVERALL_GROUPS.values()]
        (root / "score" / "data_overall.csv").write_text(
            ",".join(cols) + "\n" + ",".join(["N/A", "12.34%", "m"] + ["N/A"] * 7) + "\n")
        (root / bfcl.SEARCH_STATS).write_text(json.dumps({"calls": 0}))

        class P:
            returncode = 0

        return P()

    monkeypatch.setattr(bfcl.subprocess, "run", fake_child)
    b = bfcl.BFCLv4(RunConfig(model="m", output_dir=tmp_path / "out", limit=3, options={"temperature": "0.5"}))
    out = b.run("http://srv/v1", "m")
    assert out["value"] == pytest.approx(0.1234) and out["n"] == 3
    pak = {"k": 1, "pass_at_1": pytest.approx(0.1234), "pass_at_k": pytest.approx(0.1234), "n": 3}
    assert out["pass_at_k"] == {**pak, "how": "BFCL Overall Acc, one sample"}
    assert out["web_search_backend"]["url"] == bfcl.DEFAULT_SEARCH_MCP_URL
    assert out["per_category"]["simple_python"]["correct_count"] == 2


@pytest.fixture
def harness_env(tmp_path, monkeypatch):
    """What the parent sets for the BFCL child (a fake local endpoint, no network)."""
    monkeypatch.setenv("BFCL_PROJECT_ROOT", str(tmp_path / "bfcl"))
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "EMPTY")
    pytest.importorskip("bfcl_eval")


def test_handler_sends_only_configured_sampling(harness_env):
    handler_cls = bfcl._make_handler({}, {"done": 0, "total": 0})
    h = handler_cls(model_name="served", temperature=0.001, registry_name="granite-fc", is_fc_model=True)
    sent = {}
    h.generate_with_backoff = lambda **kw: sent.update(kw) or ("resp", 0.1)
    h._query_FC({"message": [{"role": "user", "content": "hi"}], "tools": []})
    assert sent == {"messages": [{"role": "user", "content": "hi"}], "model": "served"}

    handler_cls = bfcl._make_handler({"temperature": 0.7, "max_tokens": 100}, {"done": 0, "total": 0})
    h = handler_cls(model_name="served", temperature=0.001, registry_name="granite-fc", is_fc_model=True)
    h.generate_with_backoff = lambda **kw: sent.update(kw) or ("resp", 0.1)
    h._query_FC({"message": [], "tools": [{"type": "function"}]})
    assert sent["temperature"] == 0.7 and sent["max_tokens"] == 100 and sent["tools"]


def test_handler_strips_reasoning_from_history(harness_env):
    from openai.types.chat import ChatCompletion

    resp = ChatCompletion.model_validate({
        "id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None, "reasoning": "think",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"a\": 1}"}}]}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    })
    h = bfcl._make_handler({}, {"done": 0, "total": 0})(
        model_name="m", temperature=0, registry_name="granite-fc", is_fc_model=True)
    data = h._parse_query_response_FC(resp)
    assert data["model_responses"] == [{"f": "{\"a\": 1}"}]
    assert data["reasoning_content"] == "think"
    hist = data["model_responses_message_for_chat_history"]
    assert hist["role"] == "assistant" and "reasoning" not in hist and hist["tool_calls"][0]["id"] == "c1"


def test_web_search_patch_target_exists(harness_env):
    import inspect

    from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.web_search import WebSearchAPI

    params = list(inspect.signature(WebSearchAPI.search_engine_query).parameters)
    assert params == ["self", "keywords", "max_results", "region"]


# -- generate / score phases ---------------------------------------------------


def _write_results(root, ids):
    d = root / "result" / bfcl.REGISTRY_NAME / "non_live"
    d.mkdir(parents=True, exist_ok=True)
    (d / "BFCL_v4_simple_python_result.json").write_text(
        "".join(json.dumps({"id": i, "result": [{"f": "{}"}]}) + "\n" for i in ids))


def _fake_child(tmp_path, seen, *, results=("simple_python_0", "simple_python_1")):
    """The BFCL child by phase: generate writes ids and results, score writes scores
    (or exits MISSING_EXIT when a selected id has no result), as child_main does."""

    def run(cmd, env, check):
        root = tmp_path / "out" / "bfcl"
        cfg = json.loads((root / bfcl.CHILD_CONFIG).read_text())
        seen.append((cfg["phase"], env.get("OPENAI_BASE_URL")))
        (root / bfcl.TEST_IDS_FILE).write_text(json.dumps(
            {"simple_python": ["simple_python_0", "simple_python_1"], "memory_kv": ["memory_kv_prereq_0-a-0"]}))
        code = 0
        if cfg["phase"] in ("all", "generate"):
            _write_results(root, results)
            (root / bfcl.SEARCH_STATS).write_text(json.dumps({"calls": 4}))
        if cfg["phase"] == "score" and bfcl.missing_results(root):
            code = bfcl.MISSING_EXIT
        elif cfg["phase"] in ("all", "score"):
            d = root / "score" / bfcl.REGISTRY_NAME / "non_live"
            d.mkdir(parents=True, exist_ok=True)
            (d / "BFCL_v4_simple_python_score.json").write_text(
                json.dumps({"accuracy": 0.5, "correct_count": 1, "total_count": 2}) + "\n")
            cols = ["Rank", "Overall Acc", "Model", *bfcl.OVERALL_GROUPS.values()]
            (root / "score" / "data_overall.csv").write_text(
                ",".join(cols) + "\n" + ",".join(["1", "5.00%", "m"] + ["N/A"] * 7) + "\n")
        return type("P", (), {"returncode": code})()

    return run


def test_missing_results_ignores_memory_prereqs(tmp_path):
    root = tmp_path / "bfcl"
    root.mkdir()
    (root / bfcl.TEST_IDS_FILE).write_text(json.dumps(
        {"simple_python": ["simple_python_0", "simple_python_1"], "memory_kv": ["memory_kv_prereq_0-a-0"]}))
    assert bfcl.scored_ids(root) == ["simple_python_0", "simple_python_1"]
    assert bfcl.missing_results(root) == ["simple_python_0", "simple_python_1"]
    _write_results(root, ["simple_python_1"])
    assert bfcl.missing_results(root) == ["simple_python_0"]


def test_generate_then_score(tmp_path, monkeypatch):
    from granite_evals.registry import RunConfig

    seen = []
    monkeypatch.setattr(bfcl.subprocess, "run", _fake_child(tmp_path, seen))
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    gen = bfcl.BFCLv4(RunConfig(model="m", output_dir=tmp_path / "out", limit=2, phase="generate")).run("http://srv/v1", "m")
    assert "value" not in gen and (gen["n"], gen["missing"]) == (2, 0) and gen["web_search"] == {"calls": 4}
    assert not (tmp_path / "out" / "bfcl" / "score").exists()
    out = bfcl.BFCLv4(RunConfig(model="m", output_dir=tmp_path / "out", limit=2, phase="score")).run("", "m")
    # score: no endpoint in the child's environment; the search stats are generation's
    assert seen == [("generate", "http://srv/v1"), ("score", None)]
    assert out["value"] == pytest.approx(0.05) and out["n"] == 2
    assert out["web_search_backend"]["calls"] == 4


def test_score_with_a_missing_generation_fails(tmp_path, monkeypatch):
    from granite_evals.registry import RunConfig

    monkeypatch.setattr(bfcl.subprocess, "run", _fake_child(tmp_path, [], results=["simple_python_1"]))
    gen = bfcl.BFCLv4(RunConfig(model="m", output_dir=tmp_path / "out", limit=2, phase="generate")).run("http://s", "m")
    assert (gen["n"], gen["missing"]) == (1, 1)
    with pytest.raises(RuntimeError, match="no generation for 1 test ids .first: simple_python_0"):
        bfcl.BFCLv4(RunConfig(model="m", output_dir=tmp_path / "out", limit=2, phase="score")).run("", "m")


def test_child_score_phase_evaluates_saved_results_only(tmp_path, harness_env):
    """The real child in the score phase: no search, no generation, BFCL's evaluation
    on the saved result file (endpoint and search server unreachable: never called).
    A subprocess, as in a run: bfcl_eval fixes its paths from BFCL_PROJECT_ROOT at import."""
    root = tmp_path / "bfcl"
    cfg = {"served_model_name": "m", "categories": ["simple_python"], "limit": 1, "num_threads": 1,
           "sampling": {}, "search_mcp_url": "http://127.0.0.1:9/mcp", "include_input_log": False, "phase": "score"}
    root.mkdir(exist_ok=True)
    (root / bfcl.CHILD_CONFIG).write_text(json.dumps(cfg))
    child = [sys.executable, "-m", "granite_evals.benchmarks.bfcl", str(root / bfcl.CHILD_CONFIG)]
    assert subprocess.run(child, check=False).returncode == bfcl.MISSING_EXIT
    _write_results(root, ["simple_python_0"])
    subprocess.run(child, check=True)
    assert bfcl.read_category_scores(root / "score" / bfcl.REGISTRY_NAME)["simple_python"]["total_count"] == 1
    assert (root / "score" / "data_overall.csv").exists()


def test_repeats_unsupported(tmp_path):
    from granite_evals.registry import RunConfig

    b = bfcl.BFCLv4(RunConfig(model="m", output_dir=tmp_path, repeats=2))
    with pytest.raises(SystemExit, match="single-run metric"):
        b.run("http://srv/v1", "m")
