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

import pytest

from granite_evals.benchmarks import bfcl_v3, bfcl_v3_child
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
    assert top <= {"__future__", "json", "os", "sys", "pathlib", "types", "typing"}, top


def test_select_ids_first_n_per_category():
    entries = {"simple": [{"id": f"simple_{i}"} for i in range(5)], "irrelevance": [{"id": "irrelevance_0"}]}
    assert bfcl_v3_child.select_ids(entries, 2) == {"simple": ["simple_0", "simple_1"], "irrelevance": ["irrelevance_0"]}
    assert len(bfcl_v3_child.select_ids(entries, None)["simple"]) == 5


def _write_results(root, ids, category="simple"):
    d = root / "result" / bfcl_v3.REGISTRY_NAME
    d.mkdir(parents=True, exist_ok=True)
    (d / f"BFCL_v3_{category}_result.json").write_text(
        "".join(json.dumps({"id": i, "result": [{"f": "{}"}]}) + "\n" for i in ids))


def _write_scores(root, overall="40.00%"):
    d = root / "score" / bfcl_v3.REGISTRY_NAME
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
    per = bfcl_v3.read_category_scores(tmp_path / "score" / bfcl_v3.REGISTRY_NAME)
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
    d = tmp_path / "out" / "bfcl" / "result" / bfcl_v3.REGISTRY_NAME
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


def _gold_results(root: Path, category: str, n: int) -> list[str]:
    """FC-shaped results built from the possible answers (first value of every
    parameter), which BFCL's AST checker accepts."""
    code = (
        "import json,sys\n"
        "from bfcl_eval.constants.eval_config import POSSIBLE_ANSWER_PATH as P\n"
        f"lines=(P/'BFCL_v3_{category}.json').read_text().splitlines()\n"
        "print(json.dumps([json.loads(line) for line in lines]))\n"
    )
    answers = sorted(json.loads(_v3(code, root)), key=lambda a: int(a["id"].rsplit("_", 1)[1]))[:n]
    d = root / "result" / bfcl_v3.REGISTRY_NAME
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
    d = root / "result" / bfcl_v3.REGISTRY_NAME
    (d / "BFCL_v3_irrelevance_result.json").write_text(  # one abstention (correct), one call (wrong)
        json.dumps({"id": "irrelevance_0", "result": "I cannot help with that."}) + "\n"
        + json.dumps({"id": "irrelevance_1", "result": [{"f": "{}"}]}) + "\n")
    subprocess.run(child, env=env, check=True)
    per = bfcl_v3.read_category_scores(root / "score" / bfcl_v3.REGISTRY_NAME)
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
