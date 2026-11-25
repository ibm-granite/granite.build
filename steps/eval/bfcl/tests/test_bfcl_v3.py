"""BFCL v3 adapter tests: registration, the stdlib-only child module, result
parsing and the phases with a faked child; the real v3 child (BFCL's own
evaluator in the side venv) when that venv is present (GRANITE_BFCL_V3_PYTHON,
or the image's /opt/bfcl-v3), skipped otherwise. No model, no network."""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from granite_evals.benchmarks import bfcl, bfcl_v3, bfcl_v3_child
from granite_evals.registry import RunConfig, get

V3_PYTHON = os.environ.get(bfcl_v3.PYTHON_ENV) or bfcl_v3.DEFAULT_PYTHON
needs_v3_venv = pytest.mark.skipif(not Path(V3_PYTHON).exists(), reason=f"no BFCL v3 side venv at {V3_PYTHON}")


def test_registered_with_suite_metric():
    cls = get("bfcl-v3")
    assert cls is bfcl_v3.BFCLv3
    assert cls.metric == "overall_accuracy accuracy" and cls.extra == "bfcl" and cls.splittable
    # The harness lives in the side venv: the job venv's bfcl-eval is v4's, never v3's.
    assert cls.harness_packages == ()
    assert cls.dataset == "https://pypi.org/project/bfcl-eval/2025.8.6.2/"
    assert cls.dataset_revision.startswith("sha256:")


def test_child_imports_only_stdlib_at_module_level():
    """The child runs under the side venv's python -I: it must not import
    granite_evals (or anything the side venv lacks) at module level."""
    tree = ast.parse(bfcl_v3.CHILD.read_text())
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            top.add((node.module or "").split(".")[0])
    assert top <= {"__future__", "json", "os", "re", "sys", "pathlib", "types", "typing"}, top


def test_select_ids_first_n_per_category():
    entries = {"simple": [{"id": f"simple_{i}"} for i in range(5)], "irrelevance": [{"id": "irrelevance_0"}]}
    assert bfcl_v3_child.select_ids(entries, 2) == {"simple": ["simple_0", "simple_1"], "irrelevance": ["irrelevance_0"]}
    assert len(bfcl_v3_child.select_ids(entries, None)["simple"]) == 5


def _write_results(root, ids, category="simple", registry=bfcl_v3_child.REGISTRY_NAMES["fc"]):
    d = root / "result" / registry
    d.mkdir(parents=True, exist_ok=True)
    (d / f"BFCL_v3_{category}_result.json").write_text(
        "".join(json.dumps({"id": i, "result": [{"f": "{}"}]}) + "\n" for i in ids))


def _write_scores(root, overall="40.00%"):
    d = root / "score" / bfcl_v3_child.REGISTRY_NAMES["fc"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "BFCL_v3_simple_score.json").write_text(json.dumps({"accuracy": 0.5, "correct_count": 1, "total_count": 2}) + "\n")
    cols = ["Rank", "Overall Acc", "Model", *bfcl_v3.OVERALL_GROUPS.values()]
    (root / "score" / "data_overall.csv").write_text(
        ",".join(cols) + "\n" + ",".join(["1", overall, "m", "50.00%", "N/A", "N/A", "N/A", "N/A"]) + "\n")
    (root / "score" / "data_non_live.csv").write_text("Rank,Model,Non_Live Overall Acc\n1,m,25.00%\n")


def test_read_overall_and_category_scores(tmp_path):
    _write_scores(tmp_path)
    out = bfcl_v3.read_overall(tmp_path / "score")
    assert out["overall"] == pytest.approx(0.40)
    assert out["groups"]["non_live_ast"] == pytest.approx(0.5) and out["groups"]["non_live"] == pytest.approx(0.25)
    assert out["groups"]["live"] is None
    per = bfcl_v3.read_category_scores(tmp_path / "score" / bfcl_v3_child.REGISTRY_NAMES["fc"])
    assert per == {"simple": {"accuracy": 0.5, "correct_count": 1, "total_count": 2}}


def test_prune_and_missing_use_the_flat_v3_layout(tmp_path):
    root = tmp_path / "bfcl"
    root.mkdir()
    (root / bfcl_v3_child.TEST_IDS_FILE).write_text(json.dumps({"simple": ["simple_0", "simple_1"]}))
    assert bfcl_v3_child.missing_results(root) == ["simple_0", "simple_1"]
    _write_results(root, ["simple_1"])
    assert bfcl_v3_child.missing_results(root) == ["simple_0"]
    assert [p.name for p in bfcl_v3_child.result_files(root)] == ["BFCL_v3_simple_result.json"]


@pytest.fixture
def v3_python(tmp_path, monkeypatch):
    python = tmp_path / "v3python"
    python.write_text("")
    monkeypatch.setenv(bfcl_v3.PYTHON_ENV, str(python))
    return str(python)


def _fake_child(tmp_path, seen, *, results=("simple_0", "simple_1")):
    def run(cmd, env, check):
        root = tmp_path / "out" / "bfcl"
        cfg = json.loads((root / bfcl_v3.CHILD_CONFIG).read_text())
        seen.append((cmd[:3], cfg["phase"], env.get("OPENAI_BASE_URL"), cfg["sampling"]))
        (root / bfcl_v3_child.TEST_IDS_FILE).write_text(json.dumps({"simple": ["simple_0", "simple_1"]}))
        (root / bfcl_v3_child.HARNESS_VERSIONS).write_text(json.dumps({"bfcl-eval": "2025.8.6.2"}))
        code = 0
        if cfg["phase"] in ("all", "generate"):
            _write_results(root, results)
        if cfg["phase"] == "score" and bfcl_v3_child.missing_results(root):
            code = bfcl_v3_child.MISSING_EXIT
        elif cfg["phase"] in ("all", "score"):
            _write_scores(root, "5.00%")
        return type("P", (), {"returncode": code})()

    return run


def test_generate_then_score(tmp_path, monkeypatch, v3_python):
    seen = []
    monkeypatch.setattr(bfcl_v3.subprocess, "run", _fake_child(tmp_path, seen))
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    opts = {"temperature": "0.001"}
    gen = bfcl_v3.BFCLv3(RunConfig(model="m", output_dir=tmp_path / "out", limit=2, phase="generate", options=opts))
    out = gen.run("http://srv/v1", "m")
    assert "value" not in out and (out["n"], out["missing"]) == (2, 0)
    assert out["harness_versions"] == {"bfcl-eval": "2025.8.6.2"}
    out = bfcl_v3.BFCLv3(RunConfig(model="m", output_dir=tmp_path / "out", limit=2, phase="score")).run("", "m")
    child = [v3_python, "-I", str(bfcl_v3.CHILD)]
    assert seen == [(child, "generate", "http://srv/v1", {"temperature": 0.001}), (child, "score", None, {})]
    assert out["value"] == pytest.approx(0.05) and out["n"] == 2 and out["partial"]
    assert out["harness"] == "bfcl-eval==2025.8.6.2" and out["per_category"]["simple"]["total_count"] == 2
    assert out["pass_at_k"]["how"] == "BFCL v3 Overall Acc, one sample"


def test_score_with_a_missing_generation_fails(tmp_path, monkeypatch, v3_python):
    monkeypatch.setattr(bfcl_v3.subprocess, "run", _fake_child(tmp_path, [], results=["simple_1"]))
    gen = bfcl_v3.BFCLv3(RunConfig(model="m", output_dir=tmp_path / "out", limit=2, phase="generate")).run("http://s", "m")
    assert (gen["n"], gen["missing"]) == (1, 1)
    with pytest.raises(RuntimeError, match="no generation for 1 test ids .first: simple_0"):
        bfcl_v3.BFCLv3(RunConfig(model="m", output_dir=tmp_path / "out", limit=2, phase="score")).run("", "m")


def test_generate_prunes_infra_errors_first(tmp_path, monkeypatch, v3_python):
    d = tmp_path / "out" / "bfcl" / "result" / bfcl_v3_child.REGISTRY_NAMES["fc"]
    d.mkdir(parents=True)
    (d / "BFCL_v3_simple_result.json").write_text(
        json.dumps({"id": "simple_0", "result": "Error during inference: Connection error."}) + "\n"
        + json.dumps({"id": "simple_1", "result": "Error during inference: bad JSON"}) + "\n")
    monkeypatch.setattr(bfcl_v3.subprocess, "run", lambda cmd, env, check: type("P", (), {"returncode": 1})())
    with pytest.raises(RuntimeError, match="exited with 1"):
        bfcl_v3.BFCLv3(RunConfig(model="m", output_dir=tmp_path / "out", phase="generate")).run("http://s", "m")
    assert [json.loads(line)["id"] for line in (d / "BFCL_v3_simple_result.json").read_text().splitlines()] == ["simple_1"]


def test_needs_the_side_venv(tmp_path, monkeypatch):
    monkeypatch.setenv(bfcl_v3.PYTHON_ENV, str(tmp_path / "nope"))
    with pytest.raises(SystemExit, match="no BFCL v3 interpreter"):
        bfcl_v3.BFCLv3(RunConfig(model="m", output_dir=tmp_path)).run("http://s", "m")


def test_repeats_unsupported(tmp_path, v3_python):
    with pytest.raises(SystemExit, match="single-run metric"):
        bfcl_v3.BFCLv3(RunConfig(model="m", output_dir=tmp_path, repeats=2)).run("http://s", "m")


# -- the real child, in the side venv ------------------------------------------------


def _v3(code: str, root: Path) -> str:
    env = {**os.environ, "BFCL_PROJECT_ROOT": str(root), "OPENAI_API_KEY": "EMPTY",
           "OPENAI_BASE_URL": "http://127.0.0.1:9/v1"}
    return subprocess.run([V3_PYTHON, "-I", "-c", code], env=env, check=True, capture_output=True, text=True).stdout


def _possible_answers(root: Path, category: str, n: int) -> list[dict]:
    """The first ``n`` possible answers of a v3 category, read in the side venv."""
    code = (
        "import json,sys\n"
        "from bfcl_eval.constants.eval_config import POSSIBLE_ANSWER_PATH as P\n"
        f"lines=(P/'BFCL_v3_{category}.json').read_text().splitlines()\n"
        "print(json.dumps([json.loads(line) for line in lines]))\n"
    )
    return sorted(json.loads(_v3(code, root)), key=lambda a: int(a["id"].rsplit("_", 1)[1]))[:n]


def _gold_results(root: Path, category: str, n: int) -> list[str]:
    """FC-shaped results built from the possible answers (first value of every
    parameter), which BFCL's AST checker accepts."""
    answers = _possible_answers(root, category, n)
    d = root / "result" / bfcl_v3_child.REGISTRY_NAMES["fc"]
    d.mkdir(parents=True, exist_ok=True)
    with (d / f"BFCL_v3_{category}_result.json").open("a") as f:
        for a in answers:
            # As an FC model sends them: OpenAI tool names cannot hold ".", BFCL sends "_".
            calls = [{name.replace(".", "_"): json.dumps({k: v[0] for k, v in params.items() if v[0] != ""})}
                     for gt in a["ground_truth"] for name, params in gt.items()]
            f.write(json.dumps({"id": a["id"], "result": calls}) + "\n")
    return [a["id"] for a in answers]


@needs_v3_venv
def test_real_child_scores_with_v3_formula(tmp_path):
    root = tmp_path / "bfcl"
    root.mkdir()
    cfg = {"served_model_name": "m", "categories": ["simple", "irrelevance"], "limit": 2, "num_threads": 1,
           "sampling": {}, "include_input_log": False, "phase": "score"}
    (root / bfcl_v3.CHILD_CONFIG).write_text(json.dumps(cfg))
    child = [V3_PYTHON, "-I", str(bfcl_v3.CHILD), str(root / bfcl_v3.CHILD_CONFIG)]
    env = {**os.environ, "BFCL_PROJECT_ROOT": str(root), "OPENAI_API_KEY": "EMPTY"}
    assert subprocess.run(child, env=env, check=False).returncode == bfcl_v3_child.MISSING_EXIT
    _gold_results(root, "simple", 2)
    d = root / "result" / bfcl_v3_child.REGISTRY_NAMES["fc"]
    (d / "BFCL_v3_irrelevance_result.json").write_text(  # one abstention (correct), one call (wrong)
        json.dumps({"id": "irrelevance_0", "result": "I cannot help with that."}) + "\n"
        + json.dumps({"id": "irrelevance_1", "result": [{"f": "{}"}]}) + "\n")
    subprocess.run(child, env=env, check=True)
    per = bfcl_v3.read_category_scores(root / "score" / bfcl_v3_child.REGISTRY_NAMES["fc"])
    assert per["simple"] == {"accuracy": 1.0, "correct_count": 2, "total_count": 2}
    assert per["irrelevance"] == {"accuracy": 0.5, "correct_count": 1, "total_count": 2}
    out = bfcl_v3.read_overall(root / "score")
    # v3: Overall = mean(live, non-live, multi-turn); non-live = unweighted mean of
    # simple (mean of python/java/js), multiple, parallel, parallel multiple and
    # irrelevance; missing categories count as 0.
    simple = (1.0 + 0 + 0) / 3
    non_live = (simple + 0 + 0 + 0 + 0.5) / 5
    assert out["groups"]["non_live"] == pytest.approx(non_live, abs=5e-5)
    assert out["overall"] == pytest.approx(non_live / 3, abs=5e-5)
    versions = json.loads((root / bfcl_v3_child.HARNESS_VERSIONS).read_text())
    assert versions["bfcl-eval"] == bfcl_v3.HARNESS_VERSION


@needs_v3_venv
def test_real_handler_sends_served_name_and_only_configured_sampling(tmp_path):
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import json, bfcl_v3_child as c\n"
        "sent = {}\n"
        "h = c.make_handler('served', %s, {'done': 0, 'total': 0})('granite-fc', 0.001)\n"
        "h.generate_with_backoff = lambda **kw: sent.update(kw) or ('r', 0.1)\n"
        "h._query_FC({'message': [{'role': 'user', 'content': 'hi'}], 'tools': []})\n"
        "print(json.dumps(sent))\n"
    )
    src = str(bfcl_v3.CHILD.parent)
    assert json.loads(_v3(code % (src, "{}"), tmp_path)) == {
        "messages": [{"role": "user", "content": "hi"}], "model": "served"}
    sent = json.loads(_v3(code % (src, "{'temperature': 0.7}"), tmp_path))
    assert sent["temperature"] == 0.7 and sent["model"] == "served"


def test_list_shows_both_versions():
    out = subprocess.run([sys.executable, "-m", "granite_evals.cli", "list"], capture_output=True, text=True, check=True)
    assert "bfcl-v3" in out.stdout and "bfcl-v4" in out.stdout


# -- handler=legacy-granite -------------------------------------------------------

TEMPLATE = Path(__file__).parent / "data" / "granite-4.0-micro-b77de16e-chat_template.jinja"
"""The Granite 4.0 chat template, verbatim: huggingface.co/ibm-granite/granite-4.0-micro,
chat_template.jinja at revision b77de16e64f1bdb2ad943402334ec3a76947ead4 (Apache-2.0)."""

FUNCS = [
    {"name": "math.factorial", "description": "Factorial.",
     "parameters": {"type": "dict", "properties": {"number": {"type": "integer"}}, "required": ["number"]}},
    {"name": "ls", "description": "List files.", "parameters": {"type": "dict", "properties": {}}},
]
CONVERSATIONS = {
    "user only": [{"role": "user", "content": "What is 5!?"}],
    "system first": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "5!?"}],
    "multi-turn": [
        {"role": "user", "content": "List, then compute 5!."},
        {"role": "assistant", "content": "", "tool_calls": [
            {"name": "ls", "arguments": {}}, {"name": "math.factorial", "arguments": {"number": 5}}]},
        {"role": "tool", "name": "ls", "content": "a.txt"},
        {"role": "tool", "name": "math.factorial", "content": "120"},
        {"role": "assistant", "content": "Done: a.txt, 120."},
        {"role": "user", "content": "And with a string argument?"},
        {"role": "assistant", "content": "Sure.", "tool_calls": [
            {"function": {"name": "math.factorial", "arguments": "{\"number\": 3}"}}]},
        {"role": "tool", "name": "math.factorial", "content": "6"},
    ],
}


def _render_template(messages, functions):
    jinja2 = pytest.importorskip("jinja2")
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = json.dumps  # as BFCL's Granite handlers encode the tools (ASCII-escaped)
    return env.from_string(TEMPLATE.read_text()).render(messages=messages, tools=functions, add_generation_prompt=True)


@pytest.mark.parametrize("name", sorted(CONVERSATIONS))
@pytest.mark.parametrize("with_tools", [True, False])
def test_granite4_prompt_is_the_public_template(name, with_tools):
    functions = FUNCS if with_tools else []
    messages = CONVERSATIONS[name]
    assert bfcl_v3_child.granite4_prompt(messages, functions) == _render_template(messages, functions)


def test_granite4_prompt_shape():
    out = bfcl_v3_child.granite4_prompt(CONVERSATIONS["system first"], FUNCS[:1])
    assert out.startswith("<|start_of_role|>system<|end_of_role|>Be brief.\n\nYou are a helpful assistant with access")
    assert '\n{"name": "math.factorial", "description": "Factorial.", ' in out
    assert out.endswith("<|start_of_role|>user<|end_of_role|>5!?<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>")
    assert out.count("<|start_of_role|>system") == 1


def test_extract_tool_calls_and_reasoning():
    text = ('<tool_call>\n{"name": "math.factorial", "arguments": {"number": 5}}\n</tool_call>\n'
            '<tool_call>\nnot json\n</tool_call>\n<tool_call>\n{"name": "ls"}\n</tool_call>')
    assert bfcl_v3_child.extract_tool_calls(text) == [
        {"name": "math.factorial", "arguments": {"number": 5}}, {"name": "ls", "arguments": {}}]
    assert bfcl_v3_child.extract_tool_calls("I can't do that.") == []
    assert bfcl_v3_child.strip_reasoning("<think>\nhm\n</think>\n\nanswer") == "answer"
    assert bfcl_v3_child.strip_reasoning("answer") == "answer"


def test_max_new_tokens_is_bfcls_prompt_mode_budget():
    assert bfcl_v3_child.max_new_tokens(131072, 1000) == 4096
    assert bfcl_v3_child.max_new_tokens(4096, 1000) == 4096 - 1000 - 2
    assert bfcl_v3_child.max_new_tokens(1000, 999) == 1000


def test_vllm_tokens_reads_count_and_context(monkeypatch):
    t = bfcl_v3_child.VLLMTokens("http://srv:8000/v1", "m")
    seen = []

    def fake_json(url, body=None):
        seen.append((url, body))
        return {"data": [{"id": "x", "max_model_len": 1}, {"id": "m", "max_model_len": 8192}]} if body is None else {"count": 7}

    monkeypatch.setattr(t, "_json", fake_json)
    assert t.context_length() == 8192 and t.context_length() == 8192
    assert t.count("p") == 7
    assert seen == [("http://srv:8000/v1/models", None),
                    ("http://srv:8000/tokenize", {"model": "m", "prompt": "p", "add_special_tokens": False})]


def test_handler_option_selects_registry_and_is_validated(tmp_path, monkeypatch, v3_python):
    seen = []

    def run(cmd, env, check):
        root = tmp_path / "out" / "bfcl"
        cfg = json.loads((root / bfcl_v3.CHILD_CONFIG).read_text())
        seen.append(cfg["handler"])
        (root / bfcl_v3_child.TEST_IDS_FILE).write_text(json.dumps({"simple": ["simple_0"]}))
        _write_results(root, ["simple_0"], registry=bfcl_v3_child.REGISTRY_NAMES[cfg["handler"]])
        return type("P", (), {"returncode": 0})()

    monkeypatch.setattr(bfcl_v3.subprocess, "run", run)
    cfg = RunConfig(model="m", output_dir=tmp_path / "out", phase="generate", options={"handler": "legacy-granite"})
    out = bfcl_v3.BFCLv3(cfg).run("http://s", "m")
    assert seen == ["legacy-granite"] and (out["n"], out["missing"], out["handler"]) == (1, 0, "legacy-granite")
    with pytest.raises(SystemExit, match="unknown handler 'chat'"):
        bfcl_v3.BFCLv3(RunConfig(model="m", output_dir=tmp_path / "o2", options={"handler": "chat"})).run("http://s", "m")


@needs_v3_venv
def test_real_legacy_handler_request(tmp_path):
    """The legacy-granite handler in the side venv: the prompt it sends for a BFCL
    entry, the served name, BFCL's token budget, and no sampling unless set."""
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import json, bfcl_v3_child as c\n"
        "H = c.make_legacy_granite_handler('served', %s, {'done': 0, 'total': 0})\n"
        "h = H('granite-legacy', 0.001)\n"
        "h.tokens.context_length = lambda: 8192\n"
        "h.tokens.count = lambda p: 100\n"
        "sent = {}\n"
        "class R: pass\n"
        "h.client.completions.create = lambda **kw: sent.update(kw) or R()\n"
        "entry = {'id': 'simple_0', 'question': [[{'role': 'user', 'content': 'hi'}]], 'function': %s}\n"
        "data = h._pre_query_processing_prompting(entry)\n"
        "data['message'] = entry['question'][0]\n"
        "h._query_prompting(data)\n"
        "sent.pop('timeout')\n"
        "print(json.dumps({'sent': sent, 'funcs': data['function'], 'ast': h.decode_ast(%r)}))\n"
    )
    src = str(bfcl_v3.CHILD.parent)
    call = '<tool_call>\n{"name": "math.factorial", "arguments": {"number": 5}}\n</tool_call>'
    out = json.loads(_v3(code % (src, "{}", json.dumps(FUNCS[:1]), call), tmp_path))
    # The functions as BFCL pre-processes them for the category (a Python-syntax note).
    assert "Python 3 syntax" in out["funcs"][0]["description"]
    prompt = bfcl_v3_child.granite4_prompt([{"role": "user", "content": "hi"}], out["funcs"])
    assert out["sent"] == {"model": "served", "prompt": prompt, "max_tokens": 4096}
    assert out["ast"] == [{"math.factorial": {"number": 5}}]
    out = json.loads(_v3(code % (src, "{'temperature': 0.001}", json.dumps(FUNCS[:1]), call), tmp_path))
    assert out["sent"]["temperature"] == 0.001


@needs_v3_venv
def test_real_child_scores_legacy_text_results(tmp_path):
    """BFCL v3's evaluator on legacy-granite results: <tool_call> text, functions
    under their own (dotted) names."""
    root = tmp_path / "bfcl"
    root.mkdir()
    cfg = {"served_model_name": "m", "categories": ["simple"], "limit": 2, "num_threads": 1, "sampling": {},
           "include_input_log": False, "phase": "score", "handler": "legacy-granite"}
    (root / bfcl_v3.CHILD_CONFIG).write_text(json.dumps(cfg))
    answers = _possible_answers(root, "simple", 2)
    d = root / "result" / bfcl_v3_child.REGISTRY_NAMES["legacy-granite"]
    d.mkdir(parents=True)
    lines = []
    for a in answers:
        (name, params), = a["ground_truth"][0].items()
        args = {k: v[0] for k, v in params.items() if v[0] != ""}
        text = "<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>"
        lines.append(json.dumps({"id": a["id"], "result": text}))
    (d / "BFCL_v3_simple_result.json").write_text("\n".join(lines) + "\n")
    env = {**os.environ, "BFCL_PROJECT_ROOT": str(root), "OPENAI_API_KEY": "EMPTY"}
    subprocess.run([V3_PYTHON, "-I", str(bfcl_v3.CHILD), str(root / bfcl_v3.CHILD_CONFIG)], env=env, check=True)
    per = bfcl_v3.read_category_scores(root / "score" / bfcl_v3_child.REGISTRY_NAMES["legacy-granite"])
    assert per["simple"] == {"accuracy": 1.0, "correct_count": 2, "total_count": 2}


@pytest.fixture
def fake_vllm():
    """A stand-in OpenAI-compatible server: /v1/models, /tokenize, /v1/completions
    and /v1/chat/completions, every answer a refusal (no tool call), every request kept.
    ``fail[path]`` is a list of HTTP codes the next requests to ``path`` get instead,
    and ``delay[path]`` seconds each of them waits first."""
    import http.server
    import threading
    import time

    seen, fail, delay = [], {}, {}

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _failed(self):
            time.sleep(delay.get(self.path, 0))
            if fail.get(self.path):
                self._send({"detail": "injected"}, fail[self.path].pop(0))
                return True
            return False

        def do_GET(self):
            seen.append((self.path, None))
            if not self._failed():
                self._send({"object": "list", "data": [{"id": "m", "object": "model", "max_model_len": 32768}]})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, req))
            if self._failed():
                return
            usage = {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
            if self.path == "/tokenize":
                return self._send({"count": 10, "max_model_len": 32768, "tokens": []})
            if self.path == "/v1/completions":
                return self._send({"id": "c", "object": "text_completion", "created": 0, "model": "m", "usage": usage,
                                   "choices": [{"index": 0, "text": "I cannot help.", "finish_reason": "stop"}]})
            return self._send({"id": "c", "object": "chat.completion", "created": 0, "model": "m", "usage": usage,
                               "choices": [{"index": 0, "finish_reason": "stop",
                                            "message": {"role": "assistant", "content": "I cannot help."}}]})

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield SimpleNamespace(base_url=f"http://127.0.0.1:{server.server_port}/v1", seen=seen, fail=fail, delay=delay)
    server.shutdown()


@pytest.fixture
def quick_retries(monkeypatch):
    monkeypatch.setattr(bfcl_v3_child.VLLMTokens, "RETRY_WAIT", 0.0)
    monkeypatch.setattr(bfcl_v3_child.VLLMTokens, "TIMEOUT", 0.5)


def test_vllm_tokens_retries_transient_failures(fake_vllm, quick_retries):
    fake_vllm.fail["/tokenize"] = [503, 500]
    assert bfcl_v3_child.VLLMTokens(fake_vllm.base_url, "m").count("p") == 10
    assert [p for p, _ in fake_vllm.seen] == ["/tokenize"] * 3


def _closed_port_url():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{s.getsockname()[1]}/v1"


@pytest.mark.parametrize("failure", ["503", "refused", "timeout"])
def test_vllm_tokens_failures_are_retried_on_resume(tmp_path, fake_vllm, quick_retries, failure):
    """A vLLM hiccup that outlasts the retries is an infrastructure error: its
    "Error during inference" record is pruned before the next generate."""
    base_url = fake_vllm.base_url
    if failure == "503":
        fake_vllm.fail["/tokenize"] = [503] * 3
    elif failure == "timeout":
        fake_vllm.delay["/tokenize"] = 1.0
    else:
        base_url = _closed_port_url()
    with pytest.raises(bfcl_v3_child.EndpointError) as err:
        bfcl_v3_child.VLLMTokens(base_url, "m").count("p")
    assert "(3 attempts)" in str(err.value)
    result = tmp_path / "BFCL_v3_simple_result.json"
    result.write_text(json.dumps({"id": "simple_0", "result": f"Error during inference: {err.value}"}) + "\n")
    assert bfcl.prune_infra_errors(result) == 1, str(err.value)


def test_vllm_tokens_does_not_retry_a_missing_endpoint(fake_vllm, quick_retries):
    import urllib.error

    fake_vllm.fail["/tokenize"] = [404]
    with pytest.raises(urllib.error.HTTPError, match="404"):
        bfcl_v3_child.VLLMTokens(fake_vllm.base_url, "m").count("p")
    assert len(fake_vllm.seen) == 1


@pytest.mark.parametrize("path", ["/tokenize", "/v1/models"])
def test_vllm_tokens_probe_fails_loudly_without_vllms_endpoints(fake_vllm, quick_retries, path):
    fake_vllm.fail[path] = [404]
    with pytest.raises(SystemExit, match="handler=legacy-granite needs vLLM's /tokenize"):
        bfcl_v3_child.VLLMTokens(fake_vllm.base_url, "m").probe()
    fake_vllm.fail.clear()
    bfcl_v3_child.VLLMTokens(fake_vllm.base_url, "m").probe()


@needs_v3_venv
def test_real_legacy_child_refuses_a_server_without_tokenize(tmp_path, fake_vllm):
    """An OpenAI-compatible server that is not vLLM: the generating child fails at
    startup and writes no results (no wrong answers, no plausible score)."""
    fake_vllm.fail["/tokenize"] = [404] * 10
    root = tmp_path / "bfcl"
    root.mkdir()
    cfg = {"served_model_name": "m", "categories": ["simple", "irrelevance"], "limit": 1, "num_threads": 1,
           "sampling": {}, "include_input_log": False, "phase": "all", "handler": "legacy-granite"}
    (root / bfcl_v3.CHILD_CONFIG).write_text(json.dumps(cfg))
    env = {**os.environ, "BFCL_PROJECT_ROOT": str(root), "OPENAI_API_KEY": "EMPTY", "OPENAI_BASE_URL": fake_vllm.base_url}
    proc = subprocess.run([V3_PYTHON, "-I", str(bfcl_v3.CHILD), str(root / bfcl_v3.CHILD_CONFIG)], env=env,
                          capture_output=True, text=True, check=False)
    assert proc.returncode == 1 and "needs vLLM's /tokenize" in proc.stderr, proc.stderr[-2000:]
    assert [p for d in ("result", "score") for p in (root / d).rglob("*") if p.is_file()] == []
    assert {p for p, _ in fake_vllm.seen} <= {"/v1/models", "/tokenize"}


@needs_v3_venv
@pytest.mark.parametrize("handler", ["fc", "legacy-granite"])
def test_real_child_generates_and_scores_end_to_end(tmp_path, fake_vllm, handler):
    """BFCL v3's own generation loop (single- and multi-turn) and evaluation, with
    each handler, against the stand-in server: every entry gets a result and a score."""
    base_url = fake_vllm.base_url
    seen = fake_vllm.seen  # filtered to POSTs below
    root = tmp_path / "bfcl"
    root.mkdir()
    cfg = {"served_model_name": "m", "categories": ["simple", "multi_turn_base"], "limit": 1, "num_threads": 2,
           "sampling": {}, "include_input_log": False, "phase": "all", "handler": handler}
    (root / bfcl_v3.CHILD_CONFIG).write_text(json.dumps(cfg))
    env = {**os.environ, "BFCL_PROJECT_ROOT": str(root), "OPENAI_API_KEY": "EMPTY", "OPENAI_BASE_URL": base_url}
    subprocess.run([V3_PYTHON, "-I", str(bfcl_v3.CHILD), str(root / bfcl_v3.CHILD_CONFIG)], env=env, check=True)
    assert bfcl_v3_child.missing_results(root, handler) == []
    per = bfcl_v3.read_category_scores(root / "score" / bfcl_v3_child.REGISTRY_NAMES[handler])
    assert set(per) == {"simple", "multi_turn_base"} and all(s["total_count"] == 1 for s in per.values())
    gets = {p for p, r in seen if r is None}
    assert gets == (set() if handler == "fc" else {"/v1/models"})  # max_model_len, read once at startup
    assert [p for p, _ in seen].count("/v1/models") == len(gets)
    seen = [(p, r) for p, r in seen if r is not None]
    paths = {p for p, _ in seen}
    if handler == "fc":
        assert paths == {"/v1/chat/completions"}
        assert all(r["model"] == "m" and "temperature" not in r for _, r in seen)
    else:
        assert paths == {"/v1/completions", "/tokenize"}
        prompts = [r["prompt"] for p, r in seen if p == "/v1/completions"]
        assert all(p.startswith("<|start_of_role|>system<|end_of_role|>You are a helpful assistant with access")
                   and p.endswith("<|start_of_role|>assistant<|end_of_role|>") for p in prompts)
        # Multi-turn: later turns carry the earlier answers in the prompt.
        assert any(p.count("<|start_of_role|>assistant<|end_of_role|>I cannot help.") >= 1 for p in prompts)
        assert all(r["max_tokens"] == 4096 and "temperature" not in r for p, r in seen if p == "/v1/completions")
